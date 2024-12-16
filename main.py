import asyncio
import base64
import mimetypes
import os
import logging
import tempfile

from functools import partial
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from aiohttp import web
from openai import OpenAI

from openai_api import get_model_answer, transcribe_audio
from state_and_commands import  OpenAI_Models, add_location_button, add_user, get_history, get_last_session, get_local_time, get_user_image, info, list_users, remove_user, reply_service_text, reply_text, reset, set_bot_version, set_session_info, set_user_image, start
from sql import get_admins, in_user_list
from yandex_maps import get_address


version="9.8"

# Инициализация OpenAI и Telegram API
opena_ai_api_key=os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=opena_ai_api_key)

telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')



# URL вебхука
WEBHOOK_URL = "https://telegram-bot-xmj4.onrender.com/telegram-webhook"

def get_system_message():
    # Сообщение системы для пользователя
    local_time = get_local_time()
    system_message = {
        "role": "system",
        "content": 
f"""
Вы — помощник, который отвечает на вопросы пользователя. Время по Москве — {local_time}. 
1. Если для ответа на вопрос пользователя требуется прогноз погоды, то: 
    - внимательно изучи историю сообщений и системную информацию. Если в ней уже есть прогноз, используй его.
    - если нет, то вызови функцию get_weather_description.

2. Если для ответа на вопрос пользователя требуется его гелокация, то: 
    - внимательно изучи историю сообщений и системную информацию.Если в ней уже есть геолокация, используй ее.
    - если нет, то запроси геолокацию через функцию request_geolocation.

3. Если для ответа на вопрос пользователя требуется сгенерировать картинку, то вызови функцию generate_image.
""",
    }
    return system_message

max_history_length = 20  # Максимальное количество сообщений в истории

user_histories=get_history()

administrators_ids = get_admins()

async def get_bot_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, user_message):
    try:
        imgage_dict = await get_user_image(update.effective_user.id)
        if imgage_dict is not None:
            try:
                # Итоговая строка для использования
                img_type=imgage_dict["image_type"]
                img_b64_str = imgage_dict["image"]
                history = await user_histories.get(update.effective_user.id, [])
                history.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
                        },
                    ],
                })
            except Exception as e:
                logging.error(f"Ошибка при при обработке вашего запроса c изображением: {e}")
                return "Извините, произошла ошибка при обработке вашего запроса c изображением."
            finally:
                await set_user_image(update.effective_user.id, None)
        else:
            # Получаем или создаем историю сообщений для пользователя
            history = await get_history(update.effective_user.id, user_message)
    
   
        system_message= get_system_message()
        logging.info([system_message] + history)
        
        
        bot_reply, additional_system_messages, service_after_message = await get_model_answer(openai_client, update, context, [system_message] + history)
        
        # Добавляем дополнительную информацию в историю
        if additional_system_messages is not None:
            for message in additional_system_messages:
                history.append(message)
                logging.info(f"В историю добавлена новая системная информация: {message}")
        # Добавляем ответ бота в историю
        if bot_reply is not None:
            history.append({"role": "assistant", "content": bot_reply})
        
        # Обновляем историю пользователя
        await user_histories.set(update.effective_user.id, history)
        logging.info(f"История пользователя обновлена: {history}")
        
        return bot_reply
    except Exception as e:
        logging.error(f"Ошибка при обращении к OpenAI API: {e}")
        return "Извините, произошла ошибка при обработке вашего запроса."

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
    if len(text_to_send) > 4096:
        messages = [text_to_send[i:i+4096] for i in range(0, len(text_to_send), 4096)]
        for msg in messages:
            await reply_text(update,msg)
    else:
        await reply_text(update,text_to_send)
    history = await user_histories.get(update.effective_user.id, [])
    if len(history)==8 or len(history)==14:
        await reply_service_text(update, 
f"""Не забывайте сбрасывать контекст (историю) беседы с помощью команды /reset или командой из меню. 
Бот в своих ответах учитывает предыдущие {max_history_length} сообщений.
Это было {len(history)} сообщение. 
"""
)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_message = update.message.text
    if not await in_user_list(user):
        await not_authorized_message(update, user)
        return
    return await handle_message_inner(update, context, user_message)

async def handle_message_inner(update: Update, context: ContextTypes.DEFAULT_TYPE, user_message):
    bot_reply = await get_bot_reply(update, context, user_message)
    if bot_reply is None or len(bot_reply) == 0:
        return
    await send_big_text(update, bot_reply)
    await set_session_info(update.effective_user)

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка голосовых сообщений и распознавание текста через OpenAI Whisper API."""
    user = update.effective_user
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
        logging.info(f"Временный файл загружен: {temp_file.name}")

        try:
            # Распознавание речи с использованием OpenAI
            recognized_text=transcribe_audio(openai_client,temp_file.name)
            
            if recognized_text=="":
                 await reply_service_text(update,"Произошла ошибка при распознавании вашего сообщения.")
                 return
            await send_big_text(update, f"Распознаный текст: \n {recognized_text}")
            await handle_message_inner(update, context, recognized_text) 
            logging.info(f"Распознанный текст от пользователя {user.id}: {recognized_text}")
        except Exception as e:
            logging.error(f"Ошибка при распознавании текста через OpenAI: {e}")
            await reply_service_text(update,"Произошла ошибка при распознавании вашего сообщения.")

async def not_authorized_message(update, user):
    await reply_service_text(update,f"Извините, у вас нет доступа к этому боту. Пользователь {user}")
    logging.error(f"Нет доступа: {user}. Допустимые пользователи: {administrators_ids}")

async def set_telegram_webhook(application):
    await application.bot.set_webhook(WEBHOOK_URL)

# Обработка геолокации
async def location_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
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
        logging.error(f"Ошибка в обработчике геолокации: {e}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        # Получаем файл изображения
        photo_file = await update.message.photo[-1].get_file()
        photo_path = f'user_{update.effective_user.id}_image.jpg'
        await photo_file.download_to_drive(photo_path)

        # Определяем MIME-тип
        img_type, _ = mimetypes.guess_type(photo_path)
        if img_type is None:
            img_type = "application/octet-stream"  # На случай, если тип определить не удалось

        # Кодируем изображение в base64 для OpenAI
        with open(photo_path, 'rb') as image_file:
            image_content = image_file.read()
            # Преобразование в Base64
            img_b64_bytes = base64.b64encode(image_content)
            # Преобразование в строку
            img_b64_str = img_b64_bytes.decode("utf-8")
            await set_user_image(update.effective_user.id, {"image_type": img_type, "image":img_b64_str})

        await reply_service_text(update,"Изображение загружено, задайте вопрос по нему")
    except Exception as e:
        await reply_service_text(update,"Ошибка при загрузке изображения")
        logging.error(f"Ошибка в обработчике изображений: {e}")



async def main():
    set_bot_version(version)
    # Инициализация приложения
    application = ApplicationBuilder().token(telegram_token).build()

    # Добавление обработчиков
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.add_handler(MessageHandler(filters.LOCATION, location_handler))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    # Добавление обработчиков команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("list", list_users))
    application.add_handler(CommandHandler("add", add_user))
    application.add_handler(CommandHandler("remove", remove_user))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("last_session", get_last_session))
    application.add_handler(CommandHandler("info", info))
    application.add_handler(CommandHandler("location", add_location_button))
    
    
    

    # Инициализация и запуск приложения
    await application.initialize()
    await application.start()

    # Настройка маршрута вебхука
    async def telegram_webhook_handler(request):
        update = await request.json()
        update = Update.de_json(update, application.bot)
        await application.process_update(update)
        return web.Response(text="OK")

    # Создание веб-приложения aiohttp
    app = web.Application()
    app.router.add_post('/telegram-webhook', telegram_webhook_handler)

    # Запуск вебхука
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', int(os.environ.get('PORT', '8443')))
    await site.start()

    # Установка вебхука в Telegram
    await set_telegram_webhook(application)

    # Запуск бота
    logging.info(f"Bot v{version} is running. DefaultModel - {OpenAI_Models.DEFAULT_MODEL.value}")
    try:
        await asyncio.Event().wait()
    finally:
        # Корректная остановка приложения
        await application.stop()
        await application.shutdown()
        logging.info("Bot has stopped.")

if __name__ == '__main__':
    asyncio.run(main())
