
import datetime
import logging

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from common_types import SafeDict
from users import get_admins, get_all, in_admin_list, in_user_list, remove_user_id, save_user_id


user_histories = SafeDict()
last_session = SafeDict() 


def get_history():
    return user_histories

async def set_session_info(user) -> None:
    last_session.set("username", user.username)
    last_session.set("userid", user.id)
    last_session.set("time", datetime.datetime.now())

model_name=""
voice_recognition_model_name=""
version=""

def set_info(openai_model_name: str, vr_model_name: str, bot_version: str) -> None:
    global model_name
    global voice_recognition_model_name
    global version
    version = bot_version
    model_name = openai_model_name
    voice_recognition_model_name = vr_model_name



#-----------------------------------------------COMMANDS-----------------------------------------------------------------------

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await user_histories.set(user.id, [])
    await update.message.reply_text("Контекст беседы был сброшен. Начинаем новую беседу.")
    logging.info(f"Context for user {user.id} is reset")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user

    if not await in_user_list(user):
        await update.message.reply_text(f"Извините, у вас нет доступа к этому боту. Пользователь {user}")
        logging.error(f"Нет доступа: {user}. Допустимые пользователи: {get_admins()}")

        return
    await update.message.reply_text('Привет! Я бот, интегрированный с ChatGPT. Задайте мне вопрос.')

async def add_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if in_admin_list(user):
        if len(context.args) == 0:
            await update.message.reply_text("Вы не указали идентификатор пользователя. Команда должна выглядеть так: /add <идентификатор пользователя>")

            return
        elif not context.args[0].isdigit():
            await update.message.reply_text("Идентификатор должен быть числом.")
            return

        new_user_id = int(context.args[0])
        temp_user_ids = await get_all()

        if new_user_id not in temp_user_ids:
            await save_user_id(new_user_id)
            await update.message.reply_text("Пользователь добавлен в список допустимых.")
        else:
            await update.message.reply_text("Пользователь уже в списке допустимых.")
    else:
        await update.message.reply_text("У вас нет прав на эту команду.")

async def remove_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if in_admin_list(user):
        if len(context.args) == 0:
            await update.message.reply_text("Вы не указали идентификатор пользователя. Команда должна выглядеть так: /remove <идентификатор пользователя>")

            return
        elif not context.args[0].isdigit():
            await update.message.reply_text("Идентификатор должен быть числом.")
            return
        
        new_user_id = int(context.args[0])
        temp_user_ids = await get_all()
        if new_user_id in temp_user_ids:
            
            await remove_user_id(new_user_id)
            await update.message.reply_text("Пользователь удален из списка допустимых.")
        else:
            await update.message.reply_text("Пользователь не найден в списке допустимых.")
    else:
        await update.message.reply_text("У вас нет прав на эту команду.")

async def list_users(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if in_admin_list(user):
        temp_user_ids = await get_all()
        await update.message.reply_text("Список допустимых пользователей: " + str(temp_user_ids))
    else:
        await update.message.reply_text("У вас нет прав на эту команду.")

async def get_last_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if in_admin_list(user):
        username=last_session.get("username",None)
        userid=last_session.get("userid",None)
        last_session_time= last_session.get("time",None)
        if username is None or last_session_time is None:
            await update.message.reply_text("Нет данных о последней сессии")
        else:
            await update.message.reply_text(f"Последняя сессия была с {username} ID: {userid} в {last_session_time}")
    else:
        await update.message.reply_text("У вас нет прав на эту команду.")

async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if in_user_list(user):
    
        if version == "":
            await update.message.reply_text("Нет данных.")
        else:
            await update.message.reply_text(f"Версия бота: {version}, модель: {model_name}, модель для распознавания голоса: {voice_recognition_model_name}")
    else:
        await update.message.reply_text("У вас нет прав на эту команду.")

#------------------------------------------------end of COMMANDS-----------------------------------------------------------------------