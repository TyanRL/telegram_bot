import asyncio
import os
import logging
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

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация OpenAI и Telegram API
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')

model_name="chatgpt-4o-latest"

# URL вебхука
WEBHOOK_URL = "https://telegram-bot-xmj4.onrender.com"

alowed_user_ids = []
allowed_user_names = []


def get_users_allowed_from_os():
    global allowed_user_ids, allowed_user_names
    allowed_users_str = os.getenv('ALLOWED_USERS', '')
    allowed_user_ids = [
        int(uid.strip()) for uid in allowed_users_str.split(',') if uid.strip().isdigit()
    ]
    allowed_user_names_str = os.getenv('ALLOWED_USER_NAMES', '')
    allowed_user_names  = [
        name.strip() for name in allowed_user_names_str.split(',') if len(name) > 0
     ]
    logging.info(f"Users uploaded from environment. Ids - {allowed_user_ids} \n Names - {allowed_user_names}")



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    get_users_allowed_from_os()

    if not in_white_list(user):
        await update.message.reply_text(f"Извините, у вас нет доступа к этому боту. Пользователь {user}")
        logging.error(f"Нет доступа: {user}. Допустимые пользователи: {alowed_user_ids}, {allowed_user_names}")

        return
    await update.message.reply_text('Привет! Я бот, интегрированный с ChatGPT. Задайте мне вопрос.')

def in_white_list(user):
    return user.id in alowed_user_ids or user.username in allowed_user_names

async def get_bot_reply(user_message):
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            partial(
                openai_client.chat.completions.create,
                model=model_name,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                max_tokens=12000,
            )
        )
        bot_reply = response.choices[0].message.content.strip()
        return bot_reply
    except Exception as e:
        logging.error(f"Ошибка при обращении к OpenAI API: {e}")
        return "Извините, произошла ошибка при обработке вашего запроса."

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not in_white_list(user):
        await update.message.reply_text(f"Извините, у вас нет доступа к этому боту. Пользователь {user}")
        logging.error(f"Нет доступа: {user}. Допустимые пользователи: {alowed_user_ids}, {allowed_user_names}")
        return
    user_message = update.message.text
    bot_reply = await get_bot_reply(user_message)
    await update.message.reply_text(bot_reply)

async def set_webhook(application):
    await application.bot.set_webhook(WEBHOOK_URL)

async def main():
    # Получение списка пользователей из переменной окружения
    get_users_allowed_from_os() 
    # Инициализация приложения
    application = ApplicationBuilder().token(telegram_token).build()

    # Добавление обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

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
    logging.info(f"Bot is running. Model - {model_name}")
    try:
        await asyncio.Event().wait()
    finally:
        # Корректная остановка приложения
        await application.stop()
        await application.shutdown()
        logging.info("Bot has stopped.")

if __name__ == '__main__':
    asyncio.run(main())
