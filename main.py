import asyncio
import logging
import tempfile

from aiohttp import web
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from core.common_types import PendingQuestionRequest, QuestionAnswer
from core.config import settings
from core.openai_api import (
    get_model_answer,
    resume_model_answer,
    transcribe_audio,
    verify_default_model_available,
)
from core.question_handlers import (
    handle_question_text,
    register_question_handlers,
)
from core.question_state import PendingQuestion, get_question_store
from core.user_questions import UserQuestionService
from core.state_and_commands import (
    TELEGRAM_BOT_TOKEN,
    append_user_image_to_session,
    add_location_button,
    add_user,
    get_all_histories,
    get_last_session,
    get_local_time,
    get_notes_text,
    get_user_image_edit_session,
    info,
    list_users,
    remove_user,
    reply_service_text,
    reply_text,
    reset,
    send_service_notification,
    set_bot_version,
    set_session_info,
    start,
)
from core.tool_helpers import (
    describe_image_provenance,
    generated_image_from_session_dict,
    get_session_images,
    normalize_image_for_session,
)
from utils.elastic import get_all_user_notes
from utils.openrouter_client import OpenRouterService
from utils.sql import get_admins, in_user_list, init_db
from utils.yandex_maps import get_address

version = settings.application.version


# URL вебхука
WEBHOOK_URL = settings.server.webhook_url

logger = logging.getLogger(__name__)

def get_system_message():
    # Сообщение системы для пользователя
    local_time = get_local_time()
    system_message = {
        "role": "system",
        "content": settings.application.system_prompt.replace("{local_time}", str(local_time)),
    }
    return system_message


def log_image_settings() -> None:
    """Пишет несекретную сводку конфигурации маршрутов изображений."""

    media = settings.media
    logger.info(
        "Image configuration: mode=%s generation_model=%s generation_fast_model=%s "
        "edit_model=%s edit_fast_model=%s resolution=%s reference_pro_threshold=%s "
        "fallback_enabled=%s",
        media.image_routing_mode,
        media.image_generation_model,
        media.image_generation_fast_model,
        media.image_edit_model,
        media.image_edit_fast_model,
        media.image_default_resolution,
        media.image_multi_reference_pro_threshold,
        media.image_fallback_enabled,
    )

max_history_length = settings.application.max_history_length

user_histories=get_all_histories()

administrators_ids = get_admins()

semaphore = asyncio.Semaphore(settings.application.concurrency_limit)

async def get_bot_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_message: str | None,
    openrouter_service: OpenRouterService,
)->tuple[str|None, str|None]:
    try:
        user=update.effective_user
        if user is None:
            return None,None
        
        # Проверяем наличие активной visual session
        session = await get_user_image_edit_session(user.id)
        
        # Получение истории сообщений пользователя
        session_images = get_session_images(session) if session is not None else []
        if session_images:
            try:
                recent_session_images = session_images
                if len(session_images) > settings.media.max_session_images:
                    recent_session_images = session_images[-settings.media.max_session_images:]
                start_index = len(session_images) - len(recent_session_images) + 1
                images = [
                    generated_image_from_session_dict(image_dict)
                    for image_dict in recent_session_images
                ]
                image_labels = ", ".join(
                    f"№{index}"
                    for index in range(start_index, len(session_images) + 1)
                )
                provenance_summary = describe_image_provenance(
                    recent_session_images,
                    start_index=start_index,
                    all_images=session_images,
                )
                visual_user_text = (
                    f"{user_message}\n\n"
                    f"Вложения visual session пронумерованы по порядку: {image_labels}. "
                    "Для инструментов генерации можно указать номера картинок; "
                    "без номеров используются все."
                )
                if provenance_summary:
                    visual_user_text += f"\n\n{provenance_summary}"
                stored_history = _trim_history(
                    await user_histories.get(user.id, [])
                    + [{"role": "user", "content": visual_user_text}]
                )
                history = await user_histories.get(user.id, [])
                history = list(history)
                history.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": visual_user_text,
                        },
                        *[
                            {"type": "input_image", "image": image}
                            for image in images
                        ],
                    ],
                })
                history = _trim_history(history)
            except Exception as e:
                logger.error(
                    "Ошибка при обработке запроса с изображением: type=%s",
                    type(e).__name__,
                )
                return "Извините, произошла ошибка при обработке вашего запроса c изображением.", None
        else:
            # Получаем или создаем историю сообщений для пользователя
            history = await get_history(user.id, user_message)
            stored_history = history
    
   
        system_message= get_system_message()
        logger.info(
            "Подготовка запроса модели: history_items=%s, has_visual_session=%s",
            len(history),
            session is not None,
        )
        
        
        m_a = await get_model_answer(
            update,
            context,
            [system_message] + history,
            openrouter_service=openrouter_service,
            user_text=user_message if isinstance(user_message, str) else None,
        )

        # Работа приостановлена до ответа пользователя: вопрос отправляется
        # отдельным сообщением с кнопками, а текущий цикл модели не
        # продолжается и не повторяет уже выполненные действия. История
        # пользователя уже содержит исходный запрос, поэтому она только
        # сохраняется без изменений.
        if m_a.pending_question is not None:
            await user_histories.set(user.id, stored_history)
            await _ask_user_question(
                update,
                context,
                m_a.pending_question,
                m_a.pending_continuation,
            )
            return None, f"Токенов: использовано - {m_a.ctx_token}, сгенерировано - {m_a.completion_token}."

        # Добавляем дополнительную информацию в текстовую историю пользователя.
        if m_a.additional_system_messages is not None:
            for message in m_a.additional_system_messages:
                stored_history.append(message)
                logger.info("В историю добавлена системная информация: type=%s", type(message).__name__)
        # Добавляем ответ бота в историю
        if m_a.bot_reply is not None:
            stored_history.append({"role": "assistant", "content": m_a.bot_reply})
        
        # Обновляем историю пользователя
        await user_histories.set(user.id, stored_history)
        logger.info("История пользователя обновлена: items=%s", len(stored_history))
        
        return m_a.bot_reply, f"Токенов: использовано - {m_a.ctx_token}, сгенерировано - {m_a.completion_token}."

    except Exception as e:
        logger.error("Ошибка при обращении к OpenAI API: type=%s", type(e).__name__)
        return "Извините, произошла ошибка при обработке вашего запроса.", None

async def _ask_user_question(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    request,
    continuation,
) -> None:
    """Отправляет вопрос пользователю и сохраняет контекст продолжения.

    Обработка текущего обновления на этом заканчивается: цикл модели
    продолжится отдельным обновлением после ответа пользователя.
    """

    service = _get_question_service(context)
    try:
        await service.ask(update, request, continuation=continuation)
    except Exception as error:
        logger.error(
            "Не удалось отправить вопрос пользователю: type=%s",
            type(error).__name__,
            exc_info=True,
        )
        await reply_service_text(
            update,
            "Не удалось задать уточняющий вопрос. Повторите запрос.",
        )


def _get_question_service(context: ContextTypes.DEFAULT_TYPE) -> UserQuestionService:
    """Возвращает сервис вопросов из данных приложения."""

    application = getattr(context, "application", None)
    bot_data = getattr(application, "bot_data", None)
    if isinstance(bot_data, dict):
        service = bot_data.get("question_service")
        if isinstance(service, UserQuestionService):
            return service
    return UserQuestionService(store=get_question_store())


async def _continue_after_answer(
    record: PendingQuestion,
    answer: QuestionAnswer,
    update: Update | None,
    context: ContextTypes.DEFAULT_TYPE | None,
) -> None:
    """Продолжает прерванную задачу после ответа пользователя.

    Продолжение выполняется вне блокировки состояния: долгая генерация не
    мешает приёму других обновлений пользователя.
    """

    if record.continuation is None:
        logger.info(
            "У вопроса %s нет контекста продолжения: работа не возобновляется",
            record.question_id,
        )
        return

    if update is None or context is None:
        logger.info(
            "Нет Telegram-обновления для продолжения вопроса %s",
            record.question_id,
        )
        return

    openrouter_service = context.application.bot_data.get("openrouter_service")
    if not isinstance(openrouter_service, OpenRouterService):
        logger.error("OpenRouter service отсутствует в application.bot_data")
        await reply_service_text(
            update,
            "Не удалось продолжить задачу: media-сервис временно недоступен.",
        )
        return

    answer_text = await resume_model_answer(
        update,
        context,
        record.continuation,
        answer,
        openrouter_service=openrouter_service,
        question=PendingQuestionRequest(
            text=record.text,
            options=tuple(record.options),
            input_hint=record.input_hint,
            answer_mode=record.answer_mode,
            validator=record.validator,
        ),
    )

    if answer_text.bot_reply:
        # Фактический результат продолжения попадает в историю ровно один раз.
        user = update.effective_user
        if user is not None:
            history = list(await user_histories.get(user.id, []))
            for message in answer_text.additional_system_messages or ():
                history.append(message)
            history.append({"role": "assistant", "content": answer_text.bot_reply})
            await user_histories.set(user.id, _trim_history(history))
        await send_big_text(update, answer_text.bot_reply)
    if answer_text.pending_question is not None:
        await _ask_user_question(
            update,
            context,
            answer_text.pending_question,
            answer_text.pending_continuation,
        )


async def get_history(user_id, user_message):
    history = await user_histories.get(user_id, [])
    # Добавляем новое сообщение пользователя в историю
    history.append({"role": "user", "content": user_message})

    return _trim_history(history)


def _trim_history(history: list[dict]) -> list[dict]:
    """Ограничивает историю последними сообщениями, сохраняя системные записи."""

    # Разделяем историю на системные и прочие сообщения
    system_messages = [m for m in history if m["role"] == "system"]
    other_messages = [m for m in history if m["role"] != "system"]

    # Вычисляем, сколько сообщений можно оставить из не-системных, 
    # чтобы общая длина не превысила max_history_length
    allowed_other_count = max_history_length - len(system_messages)
    if allowed_other_count < 0:
        # Если системных сообщений даже больше, чем max_history_length,
        # то их не трогаем, но это означает, что ограничение фактически недостижимо
        allowed_other_count = 0

    # Если нужно урезать список прочих сообщений, оставляем последние allowed_other_count
    if allowed_other_count == 0:
        other_messages = []
    elif len(other_messages) > allowed_other_count:
        other_messages = other_messages[-allowed_other_count:]

    # Снова объединяем историю, сохранив системные сообщения на месте
    history = system_messages + other_messages
    return history

async def send_big_text(update: Update, text_to_send):
    if update.effective_user is None:
        return
    user = update.effective_user
    max_message_length = settings.telegram.max_message_length
    if len(text_to_send) > max_message_length:
        messages = [
            text_to_send[i:i + max_message_length]
            for i in range(0, len(text_to_send), max_message_length)
        ]
        for msg in messages:
            await reply_text(update,msg)
    else:
        await reply_text(update,text_to_send)
    history = await user_histories.get(user.id, [])
    if len(history) in settings.application.history_warning_thresholds:
        await reply_service_text(update, 
f"""Чтобы уменьшить количество затрачиваемых токенов, не забывайте сбрасывать контекст (историю) беседы с помощью команды /reset или командой из меню. 
Кроме того, бот в своих ответах учитывает предыдущие {max_history_length} сообщений. И это влияет на ответ. 
Это было {len(history)} сообщение. 
"""
)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if update.message is None:
        return
    if user is None:
        return
    user_message = update.message.text
    if not await in_user_list(user):
        await not_authorized_message(update, user)
        return
    # Активный вопрос перехватывает текст до нового обращения к модели.
    if await handle_question_text(update, context, user_message):
        return
    return await handle_message_inner(update, context, user_message)

async def handle_message_inner(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_message,
):
    user = update.effective_user
    if user is None:
        return
    openrouter_service = context.application.bot_data.get("openrouter_service")
    if not isinstance(openrouter_service, OpenRouterService):
        logger.error("OpenRouter service отсутствует в application.bot_data")
        await reply_service_text(update, "Media-сервис временно недоступен.")
        return
    bot_reply, token_service_message = await get_bot_reply(
        update,
        context,
        user_message,
        openrouter_service,
    )
    if bot_reply is None or len(bot_reply) == 0:
        return
    await send_big_text(update, bot_reply)
    await set_session_info(user)
    if token_service_message is not None:
        await reply_service_text(update, token_service_message)


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка голосовых сообщений и распознавание текста через OpenAI Whisper API."""
    user = update.effective_user
    if update.message is None:
        return
    if user is None:
        return
    voice = update.message.voice

    if not voice:
        await reply_service_text(update,"Что-то пошло не так. Голосовое сообщение не найдено.")
        return
    
    if not await in_user_list(user):
        await not_authorized_message(update, user)
        return
    
    # Получение файла голосового сообщения
    file = await context.bot.get_file(voice.file_id)

    # Использование временного файла для хранения голосового сообщения
    with tempfile.NamedTemporaryFile(delete=True, suffix=".ogg") as temp_file:
        await file.download_to_drive(temp_file.name)  # Загрузка файла
        logger.info(f"Временный файл загружен: {temp_file.name}")

        try:
            # Распознавание речи с использованием OpenAI
            recognized_text=transcribe_audio(temp_file.name)
            
            if recognized_text=="":
                 await reply_service_text(update,"Произошла ошибка при распознавании вашего сообщения.")
                 return
            await send_big_text(update, f"Распознаный текст: \n {recognized_text}")
            # Распознанный голос — тоже ответ на активный вопрос.
            if await handle_question_text(update, context, recognized_text):
                return
            await handle_message_inner(update, context, recognized_text)
            if recognized_text:
                logger.info("Распознанный текст от пользователя %s: length=%s", user.id, len(recognized_text))
        except Exception as e:
            logger.error(f"Ошибка при распознавании текста через OpenAI: {e}")
            await reply_service_text(update,"Произошла ошибка при распознавании вашего сообщения.")

async def not_authorized_message(update, user):
    await reply_service_text(update,f"Извините, у вас нет доступа к этому боту. Пользователь {user}")
    logger.error(f"Нет доступа: {user}. Допустимые пользователи: {administrators_ids}")

async def set_telegram_webhook(application:Application):
    await application.bot.set_webhook(WEBHOOK_URL)

# Обработка геолокации
async def location_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if update.message is None:
            return
        if update.effective_user is None:
            return
        if update.message.location:
            history = await user_histories.get(update.effective_user.id, [])
            latitude = update.message.location.latitude
            longitude = update.message.location.longitude
            address = await get_address(latitude,longitude)
            location_message = f"Твои координаты: Широта: {latitude} Долгота: {longitude}\nАдрес: {address}"
            history.append({"role": "system", "content": location_message})
            await user_histories.set(update.effective_user.id, history)
            await reply_service_text(update,location_message)
            await handle_message_inner(update, context, "Геолокация отправлена. Внимательно проанализируйте историю и постарайтесь ответить на ранее заданный вопрос или вызовите следующую функцию, необходимую для ответа.")
    except Exception as e:
        logger.error(f"Ошибка в обработчике геолокации: {e}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if update.message is None:
            return
        if update.effective_user is None:
            return
        user = update.effective_user
        if not await in_user_list(user):
            await not_authorized_message(update, user)
            return
        # Фотография сама по себе не закрывает активный вопрос и не меняет
        # подготовленные референсы видео: она только пополняет сессию.
        photo = update.message.photo
        document = getattr(update.message, "document", None)
        document_mime_type = getattr(document, "mime_type", None)
        if photo:
            image_file = await photo[-1].get_file()
        elif (
            document is not None
            and isinstance(document_mime_type, str)
            and document_mime_type.startswith("image/")
        ):
            image_file = await document.get_file()
        else:
            await reply_service_text(update, "В сообщении не найдено изображение.")
            return

        image_content = bytes(await image_file.download_as_bytearray())
        image_dict = normalize_image_for_session(image_content)

        image_number = await append_user_image_to_session(
            user.id,
            image_dict,
            "uploaded",
        )

        if update.message.caption and update.message.caption.strip():
            # Подпись к фотографии — тоже текст из исходного чата: при активном
            # вопросе она считается ответом на него.
            if await handle_question_text(update, context, update.message.caption):
                return
            await handle_message_inner(update, context, update.message.caption)
        else:
            await reply_service_text(
                update,
                f"Изображение №{image_number} добавлено в сессию "
                f"(всего: {image_number}). Можешь задать вопрос по картинкам, "
                "попросить их объединить/изменить или создать на их основе видео.",
            )
    except Exception as e:
        await reply_service_text(update,"Ошибка при загрузке изображения")
        logger.error(f"Ошибка в обработчике изображений: {e}")

async def show_notes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
            return
    if update.effective_user is None:
            return
    if await in_user_list(update.effective_user):

        documents = get_all_user_notes(update.effective_user.id)
        if len(documents) == 0:
            await reply_service_text(update,"Заметки не найдены.")
            return
        answer, system_message_body = get_notes_text(documents)
        history = await user_histories.get(update.effective_user.id, [])
        history.append({"role": "system", "content": system_message_body})
        await user_histories.set(update.effective_user.id, history)
        
        await reply_service_text(update,answer)
    else:
        await reply_service_text(update,"У вас нет прав на эту команду.")



async def main():
    set_bot_version(version)
    settings.require_runtime_secrets()
    log_image_settings()
    init_db()
    await verify_default_model_available()
    openrouter_service = OpenRouterService.from_settings(settings)

    async with openrouter_service:
        # Инициализация приложения с увеличенными таймаутами для загрузки изображений
        request = HTTPXRequest(
            connection_pool_size=settings.telegram.connection_pool_size,
            read_timeout=settings.telegram.read_timeout,
            write_timeout=settings.telegram.write_timeout,
            connect_timeout=settings.telegram.connect_timeout,
            pool_timeout=settings.telegram.pool_timeout,
        )
        application = (
            ApplicationBuilder()
            .token(TELEGRAM_BOT_TOKEN)
            .request(request)
            .build()
        )
        application.bot_data["openrouter_service"] = openrouter_service

        # Сервис вопросов: функция продолжения подключается явно и знает,
        # как возобновить прерванный цикл инструментов после ответа.
        question_service = register_question_handlers(
            application,
            UserQuestionService(store=get_question_store()),
        )
        question_service.set_continuation(_continue_after_answer)
        logger.info("Сервис вопросов зарегистрирован: %s", type(question_service).__name__)

        # Добавление обработчиков
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
        application.add_handler(MessageHandler(filters.LOCATION, location_handler))
        application.add_handler(
            MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_photo)
        )
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("list", list_users))
        application.add_handler(CommandHandler("add", add_user))
        application.add_handler(CommandHandler("remove", remove_user))
        application.add_handler(CommandHandler("reset", reset))
        application.add_handler(CommandHandler("last_session", get_last_session))
        application.add_handler(CommandHandler("info", info))
        application.add_handler(CommandHandler("location", add_location_button))
        application.add_handler(CommandHandler("show_notes", show_notes))
        application.add_handler(CommandHandler("send_notification", send_service_notification))

        await application.initialize()
        await application.start()
        runner: web.AppRunner | None = None

        async def telegram_webhook_handler(request):
            update = await request.json()
            update = Update.de_json(update, application.bot)
            async with semaphore:
                task = asyncio.create_task(application.process_update(update))
                task.add_done_callback(
                    lambda task: task.exception()
                    and logger.error(
                        "Error in update processing",
                        exc_info=task.exception(),
                    )
                )
            return web.Response(text="OK")

        async def health_handler(request):
            return web.Response(
                text=f"OK v{version} DefaultModel - {settings.openai.default_model}"
            )

        app = web.Application()
        app.router.add_post("/telegram-webhook", telegram_webhook_handler)
        app.router.add_get("/health", health_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, settings.server.host, settings.server.port)
        await site.start()

        await set_telegram_webhook(application)

        logger.info(
            "Bot v%s is running. DefaultModel - %s",
            version,
            settings.openai.default_model,
        )
        try:
            await asyncio.Event().wait()
        finally:
            if runner is not None:
                await runner.cleanup()
            await application.stop()
            await application.shutdown()
            logger.info("Bot has stopped.")

if __name__ == '__main__':
    asyncio.run(main())
