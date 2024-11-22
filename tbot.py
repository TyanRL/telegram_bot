import os
import openai
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from dotenv import load_dotenv

load_dotenv()

# Инициализация OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Функция для обработки сообщений
def handle_message(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=user_message,
        max_tokens=150
    )
    bot_reply = response.choices[0].text.strip()
    update.message.reply_text(bot_reply)

def main() -> None:
    # Инициализация бота
    updater = Updater(os.getenv('TELEGRAM_BOT_TOKEN'))
    dispatcher = updater.dispatcher

    # Обработчик текстовых сообщений
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    # Запуск бота
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
