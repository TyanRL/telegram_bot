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
    # Инициализация приложения Telegram
    application = ApplicationBuilder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()

    # Добавляем обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск polling
    await application.run_polling()

if __name__ == '__main__':
    import asyncio

    try:
        # Получаем текущий цикл событий
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Если цикла нет, создаем новый
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        # Запускаем main внутри активного цикла
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("Бот остановлен пользователем")
    finally:
        if not loop.is_closed():
            loop.close()
