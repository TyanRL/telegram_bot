import asyncio
import os
import openai
from telegram import Update, Bot
from telegram.ext import Updater, CommandHandler, MessageHandler, filters, CallbackContext

# Инициализация OpenAI и Telegram API
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

def main():
    bot = Bot(token=telegram_token)
    update_queue = asyncio.Queue()


    updater = Updater(bot=bot, update_queue=update_queue)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
