import asyncio
import os
import openai
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Initialize OpenAI and Telegram API
openai.api_key = os.getenv('OPENAI_API_KEY')
telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Привет! Я бот, интегрированный с ChatGPT. Задайте мне вопрос.')

def handle_message(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    response = openai.Completion.create(
        engine="chatgpt-4o-latest",
        prompt=user_message,
        max_tokens=32000
    )
    bot_reply = response.choices[0].text.strip()
    update.message.reply_text(bot_reply)

async def main():
    # Initialize the Application
    application = ApplicationBuilder().token(telegram_token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot without closing the event loop
    await application.run_polling(close_loop=False)

if __name__ == '__main__':
    asyncio.run(main())
