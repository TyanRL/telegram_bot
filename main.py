import os
import openai
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# Инициализация OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Функция для обработки сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Убедитесь, что эта модель доступна
            messages=[
                {"role": "system", "content": "Вы — полезный ассистент."},
                {"role": "user", "content": user_message},
            ],
            max_tokens=150
        )
        bot_reply = response['choices'][0]['message']['content'].strip()
        await update.message.reply_text(bot_reply)
    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка: {str(e)}")

async def main():
    # Инициализация бота
    application = ApplicationBuilder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()

    # Обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск бота
    await application.run_polling()

if __name__ == '__main__':
    import asyncio

    # Создание нового цикла событий в окружении Render
    try:
        loop = asyncio.new_event_loop()  # Создаем новый событийный цикл
        asyncio.set_event_loop(loop)    # Устанавливаем его как текущий
        loop.run_until_complete(main())  # Запускаем бота
    except Exception as e:
        print(f"Ошибка при запуске: {str(e)}")
    finally:
        loop.close()  # Закрываем цикл в случае завершения
