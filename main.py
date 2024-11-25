import os
import openai
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Инициализация OpenAI и Telegram API
openai.api_key = os.getenv('OPENAI_API_KEY')
telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Привет! Я бот, интегрированный с ChatGPT. Задайте мне вопрос.')

def handle_message(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=user_message,
        max_tokens=150
    )
    bot_reply = response.choices[0].text.strip()
    update.message.reply_text(bot_reply)

def main():
    updater = Updater(telegram_token)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
