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

from commands import add_user, get_history, get_last_session, info, list_users, remove_user, reset, set_info, set_session_info, start
from users import get_admins, in_user_list


version="2.0"

# Инициализация OpenAI и Telegram API
opena_ai_api_key=os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=opena_ai_api_key)

telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')

model_name="chatgpt-4o-latest"
voice_recognition_model_name="whisper-1"


# URL вебхука
WEBHOOK_URL = "https://telegram-bot-xmj4.onrender.com"

system_message = {
    "role": "system",
    "content": "Вы — помощник, который отвечает на вопросы пользователей."
}

max_history_length = 20  # Максимальное количество сообщений в истории

user_histories=get_history()

administrators_ids = get_admins()

async def get_bot_reply(user_id, user_message):
    # Получаем или создаем историю сообщений для пользователя
    history = await user_histories.get(user_id, [])
    
    # Добавляем новое сообщение пользователя в историю
    history.append({"role": "user", "content": user_message})
    
    # Ограничиваем историю, чтобы не превышать лимиты по токенам
    
    if len(history) > max_history_length:
        history = history[-max_history_length:]
    
    try:
        logging.info([system_message] + history)
        loop = asyncio.get_event_loop()
        # Вызываем OpenAI API с историей сообщений
        response = await loop.run_in_executor(
            None,
            partial(
                openai_client.chat.completions.create,
                model=model_name,
                messages=[system_message] + history,
                max_tokens=16384,
            )
        )
        bot_reply = response.choices[0].message.content.strip()
        
        # Добавляем ответ бота в историю
        history.append({"role": "assistant", "content": bot_reply})
        
        # Обновляем историю пользователя
        await user_histories.set(user_id, history)
        
        return bot_reply
    except Exception as e:
        logging.error(f"Ошибка при обращении к OpenAI API: {e}")
        return "Извините, произошла ошибка при обработке вашего запроса."

async def send_big_text(update: Update, text_to_send):
    if len(text_to_send) > 4096:
        messages = [text_to_send[i:i+4096] for i in range(0, len(text_to_send), 4096)]
        for msg in messages:
            await update.message.reply_text(msg)
    else:
        await update.message.reply_text(text_to_send)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_message = update.message.text
    if not await in_user_list(user):
        await not_authorized_message(update, user)
        return
    set_session_info(user)
    return await handle_message_inner(update, user, user_message)

async def handle_message_inner(update, user, user_message):
    
    
    bot_reply = await get_bot_reply(user.id, user_message)
    await send_big_text(update, bot_reply)

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка голосовых сообщений и распознавание текста через OpenAI Whisper API."""
    user = update.effective_user
    voice = update.message.voice

    if not voice:
        await update.message.reply_text("Что-то пошло не так. Голосовое сообщение не найдено.")
        return
    
    if not await in_user_list(user):
        await not_authorized_message(update, user)
        return
    set_session_info(user)
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
                 await update.message.reply_text("Произошла ошибка при распознавании вашего сообщения.")
                 return
            await send_big_text(update, f"Распознаный текст: \n {recognized_text}")
            await handle_message_inner(update, user, recognized_text) 
            logging.info(f"Распознанный текст от пользователя {user.id}: {recognized_text}")
        except Exception as e:
            logging.error(f"Ошибка при распознавании текста через OpenAI: {e}")
            await update.message.reply_text("Произошла ошибка при распознавании вашего сообщения.")

async def not_authorized_message(update, user):
    await update.message.reply_text(f"Извините, у вас нет доступа к этому боту. Пользователь {user}")
    logging.error(f"Нет доступа: {user}. Допустимые пользователи: {administrators_ids}")

async def set_webhook(application):
    await application.bot.set_webhook(WEBHOOK_URL)

async def main():
    set_info(model_name, voice_recognition_model_name, version)
    # Инициализация приложения
    application = ApplicationBuilder().token(telegram_token).build()

    # Добавление обработчиков
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    # Добавление обработчиков команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("list", list_users))
    application.add_handler(CommandHandler("add", add_user))
    application.add_handler(CommandHandler("remove", remove_user))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("last_session", get_last_session))
    application.add_handler(CommandHandler("info", info))
    

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
