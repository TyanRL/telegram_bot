
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
from telegram.helpers import escape_markdown


user_histories = SafeDict()
translate_mode=SafeDict()


def get_history():
    return user_histories

async def set_session_info(user) -> None:
    # Получение текущего времени в формате UTC
    local_time = get_local_time()
    await save_last_session(user.id, user.username, local_time)

def get_local_time():
    utc_time = datetime.datetime.now(ZoneInfo("UTC"))    

    # Преобразование времени из UTC в локальное время
    local_timezone = ZoneInfo('Europe/Moscow')  # замените на вашу временную зону
    local_time = utc_time.astimezone(local_timezone)
    return local_time
    
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

async def reply_text(update: Update, message:str):
    escaped_text = escape_markdown(message, version=2)
    await update.message.reply_text(escaped_text, parse_mode="MarkdownV2")

async def reply_service_text(update: Update, message:str):
    escaped_text = escape_markdown(message, version=2)
    await update.message.reply_text(f"_{escaped_text}_", parse_mode="MarkdownV2")



#-----------------------------------------------COMMANDS-----------------------------------------------------------------------

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await user_histories.set(user.id, [])
    await reply_service_text(update,"Контекст беседы был сброшен. Начинаем новую беседу.")
    logging.info(f"Context for user {user.id} is reset")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user

    if not await in_user_list(user):
        await reply_service_text(update,f"Извините, у вас нет доступа к этому боту. Пользователь {user}")
        logging.error(f"Нет доступа: {user}. Допустимые пользователи: {get_admins()}")

        return
    await reply_text(update, 'Привет! Я бот, интегрированный с ChatGPT. Задайте мне вопрос.')

async def add_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if in_admin_list(user):
        if len(context.args) == 0:
            await reply_service_text(update,"Вы не указали идентификатор пользователя. Команда должна выглядеть так: /add <идентификатор пользователя>")

            return
        elif not context.args[0].isdigit():
            await reply_service_text(update,"Идентификатор должен быть числом.")
            return

        new_user_id = int(context.args[0])
        temp_user_ids = await get_all()

        if new_user_id not in temp_user_ids:
            await save_user_id(new_user_id)
            await reply_service_text(update,"Пользователь добавлен в список допустимых.")
        else:
            await reply_service_text(update,"Пользователь уже в списке допустимых.")
    else:
        await reply_service_text(update,"У вас нет прав на эту команду.")

async def remove_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if in_admin_list(user):
        if len(context.args) == 0:
            await reply_service_text(update,"Вы не указали идентификатор пользователя. Команда должна выглядеть так: /remove <идентификатор пользователя>")

            return
        elif not context.args[0].isdigit():
            await reply_service_text(update,"Идентификатор должен быть числом.")
            return
        
        new_user_id = int(context.args[0])
        temp_user_ids = await get_all()
        if new_user_id in temp_user_ids:
            
            await remove_user_id(new_user_id)
            await reply_service_text(update,"Пользователь удален из списка допустимых.")
        else:
            await reply_service_text(update,"Пользователь не найден в списке допустимых.")
    else:
        await reply_service_text(update,"У вас нет прав на эту команду.")

async def list_users(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if in_admin_list(user):
        temp_user_ids = await get_all()
        await reply_service_text(update,"Список допустимых пользователей: " + str(temp_user_ids))
    else:
        await reply_service_text(update,"У вас нет прав на эту команду.")

async def get_last_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if in_admin_list(user):
        last_sessions= get_all_session()
        if len(last_sessions)==0:
            await reply_service_text(update,"Список последних сессий пользователей пуст.")
            return

        last_sesssions_str="Список последних сессий пользователей: \n"
        for last_session in last_sessions:
            username=last_session.get("username",None)
            userid=last_session.get("userid",None)
            last_session_time=last_session.get("last_session_time",None)
            if username is not None and last_session_time is not None:
                last_sesssions_str+=f"Пользователь {username} ID: {userid} в {last_session_time}\n"
            
        await reply_service_text(update,last_sesssions_str)
    else:
        await reply_service_text(update,"У вас нет прав на эту команду.")

async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if in_user_list(user):
    
        if version == "":
            await reply_service_text(update,"Нет данных.")
        else:
            await reply_service_text(update,f"Версия бота: {version}, модель: {model_name}, модель для распознавания голоса: {voice_recognition_model_name}")
    else:
        await reply_service_text(update,"У вас нет прав на эту команду.")

#------------------------------------------------end of COMMANDS-----------------------------------------------------------------------