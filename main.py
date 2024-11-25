import asyncio
import os
import openai
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from aiohttp import web

# Initialize OpenAI and Telegram API
openai.api_key = os.getenv('OPENAI_API_KEY')
telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')

# Get the webhook URL from an environment variable
WEBHOOK_URL = "https://telegram-bot-xmj4.onrender.com"  # Set this in Render

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Привет! Я бот, интегрированный с ChatGPT. Задайте мне вопрос.')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text

    try:
        # Use OpenAI's ChatCompletion API asynchronously
        response = await openai.ChatCompletion.acreate(
            model="chatgpt-4o-latest",
            messages=[
                {"role": "user", "content": user_message}
            ],
            max_tokens=12000,
        )
        bot_reply = response.choices[0].message.content.strip()
    except Exception as e:
        bot_reply = "Извините, произошла ошибка при обработке вашего запроса."
        # Optionally log the exception
        print(f"Error: {e}")
    await update.message.reply_text(bot_reply)

async def set_webhook(application):
    await application.bot.set_webhook(WEBHOOK_URL)

async def main():
    # Initialize the Application
    application = ApplicationBuilder().token(telegram_token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Initialize and start the application
    await application.initialize()
    await application.start()

    # Set up the webhook route
    async def webhook_handler(request):
        # Extract the JSON payload from the request
        update = await request.json()
        # Process the update using the application
        update = Update.de_json(update, application.bot)
        await application.process_update(update)
        return web.Response(text="OK")

    # Create an aiohttp web app
    app = web.Application()
    app.router.add_post('/', webhook_handler)

    # Start the webhook
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', int(os.environ.get('PORT', '8443')))
    await site.start()

    # Set the webhook with Telegram
    await set_webhook(application)

    # Run the bot until Ctrl+C is pressed
    print("Bot is running...")
    try:
        await asyncio.Event().wait()
    finally:
        # Stop and shutdown the application gracefully
        await application.stop()
        await application.shutdown()

if __name__ == '__main__':
    asyncio.run(main())
