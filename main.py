import asyncio
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

from openai_api import get_model_answer
from state_and_commands import add_location_button, add_user, get_history, get_last_session, get_local_time, info, list_users, remove_user, reply_service_text, reply_text, reset, set_geolocation, set_info, set_session_info, start
from common_types import SafeDict
from sql import get_admins, in_user_list


version="4.2"

# Инициализация OpenAI и Telegram API
opena_ai_api_key=os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=opena_ai_api_key)

telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')

model_name="gpt-4o"
voice_recognition_model_name="whisper-1"


# URL вебхука
WEBHOOK_URL = "https://telegram-bot-xmj4.onrender.com"

def get_system_message():
    # Сообщение системы для пользователя
    local_time = get_local_time()
    system_message = {
        "role": "system",
        "content": 
f"""
Вы — помощник, который отвечает на вопросы пользователей. Время по Москве — {local_time}. 
Если в истории нет геолокации, то нужно запросить соответсвующую функцию.
""",
    }
    return system_message

max_history_length = 20  # Максимальное количество сообщений в истории

user_histories=get_history()

administrators_ids = get_admins()

async def get_bot_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, user_message):

    # Получаем или создаем историю сообщений для пользователя
    history = await get_history(update.effective_user.id, user_message)
    
    try:
        system_message= get_system_message()
        logging.info([system_message] + history)
        
        bot_reply, additional_system_message = await get_model_answer(openai_client, update, context, model_name, [system_message] + history)
        
        if bot_reply is None:
            return None

        # Добавляем дополнительную информацию в историю
        if additional_system_message is not None:
            history.append(additional_system_message)
        # Добавляем ответ бота в историю
        history.append({"role": "assistant", "content": bot_reply})
        
        # Обновляем историю пользователя
        await user_histories.set(update.effective_user.id, history)
        
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
    if len(history)==5 or len(history)==10:
        await reply_service_text(update, 
"""
Не забывайте сбрасывать контекст (историю) беседы с помощью команды /reset или командой из меню. 
Бот в своих ответах учитывает предыдущие сообщения.
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
            transcription = openai_client.audio.transcriptions.create(
                            model=voice_recognition_model_name,
                            file=open(temp_file.name, 'rb')
                            )
            recognized_text=transcription.text
            
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

async def set_webhook(application):
    await application.bot.set_webhook(WEBHOOK_URL)

# Обработка геолокации
async def location_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message.location:
        history = await user_histories.get(update.effective_user.id, [])
        latitude = update.message.location.latitude
        longitude = update.message.location.longitude
        location_message = f"Твои координаты:\nШирота: {latitude}\nДолгота: {longitude}"
        history.append({"role": "system", "content": location_message})
        await set_geolocation(update.effective_user.id, latitude, longitude)
        await reply_service_text(update,location_message)
        await handle_message_inner(update, context, "Геолокация задана.")
        


async def main():
    set_info(model_name, voice_recognition_model_name, version)
    # Инициализация приложения
    application = ApplicationBuilder().token(telegram_token).build()

    # Добавление обработчиков
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.add_handler(MessageHandler(filters.LOCATION, location_handler))
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
    async def webhook_handler(request):
        update = await request.json()
        update = Update.de_json(update, application.bot)
        await application.process_update(update)
        return web.Response(text="OK")

    # Создание веб-приложения aiohttp
    app = web.Application()
    app.router.add_post('/', webhook_handler)

    # Запуск вебхука
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', int(os.environ.get('PORT', '8443')))
    await site.start()

    # Установка вебхука в Telegram
    await set_webhook(application)

    # Запуск бота
    logging.info(f"Bot v{version} is running. Model - {model_name}")
    try:
        await asyncio.Event().wait()
    finally:
        # Корректная остановка приложения
        await application.stop()
        await application.shutdown()
        logging.info("Bot has stopped.")

if __name__ == '__main__':
    asyncio.run(main())
