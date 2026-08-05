import asyncio
import base64
import logging
import mimetypes
import os
import tempfile

from aiohttp import web
from PIL import Image
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

from core.openai_api import get_model_answer, transcribe_audio
from core.state_and_commands import (
    TELEGRAM_BOT_TOKEN,
    OpenAI_Models,
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
    set_user_image_edit_session,
    start,
)
from core.tool_helpers import generated_image_from_session_dict
from utils.elastic import get_all_user_notes
from utils.openrouter_client import OpenRouterService
from utils.sql import get_admins, in_user_list, init_db
from utils.yandex_maps import get_address

version="28.0"


# URL вебхука
WEBHOOK_URL = "https://telegram-bot-xmj4.onrender.com/telegram-webhook"

logger = logging.getLogger(__name__)

def get_system_message():
    # Сообщение системы для пользователя
    local_time = get_local_time()
    system_message = {
        "role": "system",
        "content":
f"""
Вы — личный помощник, который СЖАТО И КРАТКО отвечает на вопросы пользователя. Время по Москве — {local_time}.
1. Если по доступному в диалоге контексту видно, что пользователь просит сгенерировать изображение только по текстовому описанию (с нуля), используй функцию generate_image.
2. Если по доступному в диалоге контексту видно, что у пользователя есть текущая картинка для редактирования и он просит изменить, стилизовать, перерисовать, улучшить или сделать вариацию, используй функцию generate_image_from_image.
3. Если пользователь просит сгенерировать видео, анимацию, оживить картинку или сделать видео на основе изображения — используй функцию generate_video.
4. Промпты для генерации видео и изображений создавай на английском языке, если результат генерации требует наличие текста, то этот текст не обязательно должен быть на английском.
5. Если функция недоступна для текущего состояния сессии, не придумывай обходной путь и следуй ограничениям, которые вернет приложение.
""",
    }
    return system_message

max_history_length = 15  # Максимальное количество сообщений в истории

user_histories=get_all_histories()

administrators_ids = get_admins()

semaphore = asyncio.Semaphore(10)

async def get_bot_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_message,
    openrouter_service: OpenRouterService,
)->tuple[str|None, str|None]:
    try:
        user=update.effective_user
        if user is None:
            return None,None
        
        # Проверяем наличие активной visual session
        session = await get_user_image_edit_session(user.id)
        
        # Получение истории сообщений пользователя
        if session is not None and session.get("current_image") is not None:
            try:
                # Берем current_image из visual session
                image_dict = session["current_image"]
                image = generated_image_from_session_dict(image_dict)
                history = await user_histories.get(user.id, [])
                history.append({
                    "role": "user",
                    "content": [
                                {"type": "input_text", "text": user_message},
                                {"type": "input_image", "image": image},
                    ],
                })
            except Exception as e:
                logger.error(
                    "Ошибка при обработке запроса с изображением: type=%s",
                    type(e).__name__,
                )
                return "Извините, произошла ошибка при обработке вашего запроса c изображением.", None
        else:
            # Получаем или создаем историю сообщений для пользователя
            history = await get_history(user.id, user_message)
    
   
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
        )
        
        
        # Добавляем дополнительную информацию в историю
        if m_a.additional_system_messages is not None:
            for message in m_a.additional_system_messages:
                history.append(message)
                logger.info("В историю добавлена системная информация: type=%s", type(message).__name__)
        # Добавляем ответ бота в историю
        if m_a.bot_reply is not None:
            history.append({"role": "assistant", "content": m_a.bot_reply})
        
        # Обновляем историю пользователя
        await user_histories.set(user.id, history)
        logger.info("История пользователя обновлена: items=%s", len(history))
        
        return m_a.bot_reply, f"Токенов: использовано - {m_a.ctx_token}, сгенерировано - {m_a.completion_token}."

    except Exception as e:
        logger.error("Ошибка при обращении к OpenAI API: type=%s", type(e).__name__)
        return "Извините, произошла ошибка при обработке вашего запроса.", None

async def get_history(user_id, user_message):
    history = await user_histories.get(user_id, [])
    # Добавляем новое сообщение пользователя в историю
    history.append({"role": "user", "content": user_message})

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
    if len(other_messages) > allowed_other_count:
        other_messages = other_messages[-allowed_other_count:]

    # Снова объединяем историю, сохранив системные сообщения на месте
    history = system_messages + other_messages
    return history

async def send_big_text(update: Update, text_to_send):
    if update.effective_user is None:
        return
    user = update.effective_user
    if len(text_to_send) > 4096:
        messages = [text_to_send[i:i+4096] for i in range(0, len(text_to_send), 4096)]
        for msg in messages:
            await reply_text(update,msg)
    else:
        await reply_text(update,text_to_send)
    history = await user_histories.get(user.id, [])
    if len(history)==8 or len(history)==14:
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
        
        # Проверяем наличие активной visual session
        existing_session = await get_user_image_edit_session(update.effective_user.id)
        if existing_session is not None:
            await reply_service_text(update,"У вас уже есть активная сессия редактирования изображения. Чтобы начать работу с новой картинкой, выполните /reset.")
            return
        
        # Получаем файл изображения
        photo_file = await update.message.photo[-1].get_file()
        photo_path = f'user_{update.effective_user.id}_image.jpg'
        await photo_file.download_to_drive(photo_path)

        # Определяем MIME-тип
        img_type, _ = mimetypes.guess_type(photo_path)
        if img_type is None:
            img_type = "application/octet-stream"  # На случай, если тип определить не удалось
        with Image.open(photo_path) as img:
            width, height = img.size
        # Кодируем изображение в base64 для OpenAI
        with open(photo_path, 'rb') as image_file:
            image_content = image_file.read()
            # Преобразование в Base64
            img_b64_bytes = base64.b64encode(image_content)
            # Преобразование в строку
            img_b64_str = img_b64_bytes.decode("utf-8")
            image_dict = {"image_type": img_type, 
                          "image": img_b64_str, 
                          "width": width,
                          "height": height,
                          "aspect_ratio": width / height,}
            
            # Создаем новую visual session
            from datetime import datetime
            session = {
                "original_image": image_dict,
                "current_image": image_dict,
                "source_kind": "uploaded",
                "history": [
                    {
                        "step": 0,
                        "kind": "upload",
                        "prompt": None,
                        "created_at": datetime.now().isoformat()
                    }
                ]
            }
            await set_user_image_edit_session(update.effective_user.id, session)
            
        await reply_service_text(update,"Изображение загружено и стало текущей основой для редактирования. Можешь задать вопрос по нему, попросить изменить/стилизовать его или сделать видео на основе. Для последовательных правок не нужно повторно отправлять картинку.")
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
    init_db()
    openrouter_service = OpenRouterService.from_env()

    async with openrouter_service:
        # Инициализация приложения с увеличенными таймаутами для загрузки изображений
        request = HTTPXRequest(
            connection_pool_size=8,
            read_timeout=120,
            write_timeout=180,
            connect_timeout=30,
            pool_timeout=30,
        )
        application = (
            ApplicationBuilder()
            .token(TELEGRAM_BOT_TOKEN)
            .request(request)
            .build()
        )
        application.bot_data["openrouter_service"] = openrouter_service

        # Добавление обработчиков
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
        application.add_handler(MessageHandler(filters.LOCATION, location_handler))
        application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
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
                text=f"OK v{version} DefaultModel - {OpenAI_Models.DEFAULT_MODEL.value}"
            )

        app = web.Application()
        app.router.add_post("/telegram-webhook", telegram_webhook_handler)
        app.router.add_get("/health", health_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", int(os.environ.get("PORT", "8443")))
        await site.start()

        await set_telegram_webhook(application)

        logger.info(
            "Bot v%s is running. DefaultModel - %s",
            version,
            OpenAI_Models.DEFAULT_MODEL.value,
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
