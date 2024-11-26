import asyncio
import os
import logging
import mysql.connector

from telegram import Bot
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


version="1.5"

# Получение параметров подключения из переменных окружения
MYSQL_HOST = os.getenv('MYSQL_ADDON_HOST')
MYSQL_DB = os.getenv('MYSQL_ADDON_DB')
MYSQL_USER = os.getenv('MYSQL_ADDON_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_ADDON_PASSWORD')
MYSQL_PORT = os.getenv('MYSQL_ADDON_PORT', '3306')

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация OpenAI и Telegram API
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')

model_name="chatgpt-4o-latest"

# URL вебхука
WEBHOOK_URL = "https://telegram-bot-xmj4.onrender.com"

system_message = {
    "role": "system",
    "content": "Вы — помощник, который отвечает на вопросы пользователей."
}

max_history_length = 20  # Максимальное количество сообщений в истории

# администраторы
administrators_ids = []

class SafeDict:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.data = {}

    async def set(self, key, value):
        async with self.lock:
            self.data[key] = value

    async def get(self, key):
        async with self.lock:
            return self.data.get(key)

class SafeList:
    def __init__(self, l: list):
        self.lock = asyncio.Lock()
        self.data = l

    async def append(self, value):
        async with self.lock:
            self.data.append(value)

    async def get(self, index):
        async with self.lock:
            return self.data[index]
        
    async def remove(self, value):
        async with self.lock:
            if value in self.data:
                self.data.remove(value)
    
    async def get_all(self):
        async with self.lock:
            return list(self.data)

# пользователи
user_ids= SafeList([])
user_histories = SafeDict()


#----------------------------------MySQL------------------------------------------
if not all([MYSQL_HOST, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD]):
    raise EnvironmentError("Не установлены все необходимые переменные окружения для подключения к MySQL.")

def connect_to_db():
    try:
        return mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            port=MYSQL_PORT
        )
    except mysql.connector.Error as err:
        logging.error(f"Ошибка подключения к MySQL: {err}")
        raise

def create_table():
    """Создает таблицу для хранения идентификаторов пользователей."""
    connection = connect_to_db()
    try:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_ids (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id BIGINT NOT NULL
            )
        """)
        connection.commit()
    except mysql.connector.Error as err:
        logging.error(f"Ошибка создания таблицы в MySQL: {err}")
        raise
    finally:
        cursor.close()
        connection.close()

def save_user_id(user_id):
    """Сохраняет идентификатор пользователя в базу данных."""
    connection = connect_to_db()
    try:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO user_ids (user_id) VALUES (%s)", (user_id,))
        connection.commit()
    except mysql.connector.Error as err:
        logging.error(f"Ошибка сохранения пользователя в MySQL: {err}")
        raise
    finally:
        cursor.close()
        connection.close()

def get_user_ids():
    """Получает все идентификаторы пользователей из базы данных."""
    connection = connect_to_db()
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT user_id FROM user_ids")
        result = cursor.fetchall()
        return [row[0] for row in result]
    except mysql.connector.Error as err:
        logging.error(f"Ошибка получения пользователей из MySQL: {err}")
        raise
    finally:
        cursor.close()
        connection.close()

def remove_user_id(user_id):
    """Удаляет идентификатор пользователя из базы данных."""
    connection = connect_to_db()
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM user_ids WHERE user_id = %s", (user_id,))
        connection.commit()
    except mysql.connector.Error as err:
        logging.error(f"Ошибка удаления пользователя из MySQL: {err}")
        raise
    finally:
        cursor.close()
        connection.close()

#-------------------------------------end MySQL-------------------------------------------------------



def get_admins_from_os():
    global administrators_ids
    allowed_users_str = os.getenv('ALLOWED_USER_IDS', '')
    administrators_ids = [
        int(uid.strip()) for uid in allowed_users_str.split(',') if uid.strip().isdigit()
    ]
    logging.info(f"Users uploaded from environment. Ids - {administrators_ids}")
#-----------------------------------------------COMMANDS-----------------------------------------------------------------------

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await user_histories.set(user.id, [])
    await update.message.reply_text("Контекст беседы был сброшен. Начинаем новую беседу.")
    logging.info(f"Context for user {user.id} is reset")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    get_admins_from_os()

    if not await in_user_list(user):
        await update.message.reply_text(f"Извините, у вас нет доступа к этому боту. Пользователь {user}")
        logging.error(f"Нет доступа: {user}. Допустимые пользователи: {administrators_ids}")

        return
    await update.message.reply_text('Привет! Я бот, интегрированный с ChatGPT. Задайте мне вопрос.')

async def add_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global user_ids
    user = update.effective_user
    if in_admin_list(user):
        if len(context.args) == 0:
            await update.message.reply_text("Вы не указали идентификатор пользователя. Команда должна выглядеть так: /add <идентификатор пользователя>")

            return
        elif not context.args[0].isdigit():
            await update.message.reply_text("Идентификатор должен быть числом.")
            return

        new_user_id = int(context.args[0])
        temp_user_ids = await user_ids.get_all()

        if new_user_id not in temp_user_ids:
            save_user_id(new_user_id)
            await user_ids.append(new_user_id)
            await update.message.reply_text("Пользователь добавлен в список допустимых.")
        else:
            await update.message.reply_text("Пользователь уже в списке допустимых.")
    else:
        await update.message.reply_text("У вас нет прав на эту команду.")

async def remove_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global user_ids
    user = update.effective_user
    if in_admin_list(user):
        if len(context.args) == 0:
            await update.message.reply_text("Вы не указали идентификатор пользователя. Команда должна выглядеть так: /remove <идентификатор пользователя>")

            return
        elif not context.args[0].isdigit():
            await update.message.reply_text("Идентификатор должен быть числом.")
            return
        
        new_user_id = int(context.args[0])
        temp_user_ids = await user_ids.get_all()
        if new_user_id in temp_user_ids:
            await user_ids.remove(new_user_id)
            remove_user_id(new_user_id)
            await update.message.reply_text("Пользователь удален из списка допустимых.")
        else:
            await update.message.reply_text("Пользователь не найден в списке допустимых.")
    else:
        await update.message.reply_text("У вас нет прав на эту команду.")

async def list_users(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if in_admin_list(user):
        temp_user_ids = await user_ids.get_all()
        await update.message.reply_text("Список допустимых пользователей: " + str(temp_user_ids))
    else:
        await update.message.reply_text("У вас нет прав на эту команду.")


def in_admin_list(user):
    return user.id in administrators_ids
async def in_user_list(user):
    temp_user_ids = await user_ids.get_all()
    return user.id in temp_user_ids or in_admin_list(user)

#------------------------------------------------end of COMMANDS-----------------------------------------------------------------------


async def get_bot_reply(user_id, user_message):
    # Получаем или создаем историю сообщений для пользователя
    history = await user_histories.get(user_id) or []
    
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
                max_tokens=12000,
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

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    get_admins_from_os()
    user = update.effective_user
    if not await in_user_list(user):
        await update.message.reply_text(f"Извините, у вас нет доступа к этому боту. Пользователь {user}")
        logging.error(f"Нет доступа: {user}. Допустимые пользователи: {administrators_ids}")
        return
    user_message = update.message.text
    bot_reply = await get_bot_reply(user.id, user_message)
    await update.message.reply_text(bot_reply)

async def set_webhook(application):
    await application.bot.set_webhook(WEBHOOK_URL)

async def main():
    global user_ids
    # Получение списка администраторов из переменной окружения
    get_admins_from_os() 
    # Получение списка пользователей из переменной окружения
    connect_to_db()
    create_table()
    user_ids=SafeList(get_user_ids())
    # Инициализация приложения
    application = ApplicationBuilder().token(telegram_token).build()

    # Добавление обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CommandHandler("list", list_users))
    application.add_handler(CommandHandler("add", add_user))
    application.add_handler(CommandHandler("remove", remove_user))
    application.add_handler(CommandHandler("reset", reset))

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
