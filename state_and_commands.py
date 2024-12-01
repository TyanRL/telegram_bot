
import datetime
import logging
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from common_types import SafeDict
from sql import get_admins, get_all, get_all_session, in_admin_list, in_user_list, remove_user_id, save_last_session, save_user_id


user_histories = SafeDict()
translate_mode=SafeDict()


def get_history():
    return user_histories

async def set_session_info(user) -> None:
    # Получение текущего времени в формате UTC
    utc_time = datetime.datetime.now(ZoneInfo("UTC"))    

    # Преобразование времени из UTC в локальное время
    local_timezone = ZoneInfo('Europe/Moscow')  # замените на вашу временную зону
    local_time = utc_time.astimezone(local_timezone)
    await save_last_session(user.id, user.username, local_time)
    
async def set_translate_mode(user) -> None:
    translate_mode


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
        last_sessions= get_all_session()
        last_sesssions_str="Список последних сессий пользователей: \n"
        for last_session in last_sessions:
            username=await last_session.get("username",None)
            userid=await last_session.get("userid",None)
            last_session_time=await last_session.get("time",None)
            if username is not None and last_session_time is not None:
                last_sesssions_str+=f"Пользователь {username} ID: {userid} в {last_session_time}\n"
            
            await update.message.reply_text(last_sesssions_str)
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