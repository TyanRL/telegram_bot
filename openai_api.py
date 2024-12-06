
import asyncio
from functools import partial
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from state_and_commands import add_location_button

# Запрос геолокации у пользователя:
async def request_geolocation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await add_location_button(update, context)

async def get_model_answer(openai_client: OpenAI,update: Update, context: ContextTypes.DEFAULT_TYPE, model_name: str, messages):
    loop = asyncio.get_event_loop()

    # Описываем доступные функции для модели:
    functions = [
        {
            "name": "request_geolocation",
            "description": "Получить геолокацию пользователя",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    ]
    
    # Вызываем OpenAI API с историей сообщений и объявленной функцией
    response = await loop.run_in_executor(
        None,
        partial(
            openai_client.chat.completions.create,
            model=model_name,
            messages=messages,
            functions=functions,
            # Можно оставить модель самой решать, вызывать ли функцию или нет:
            function_call="auto",
            # Или требовать явный вызов:
            # function_call={"name": "request_geolocation"},
            max_tokens=16384
        )
    )
    
    # Проверяем, вернулась ли функция для вызова
    if (response.choices and 
        len(response.choices) > 0 and
        response.choices[0].message.get("function_call")):
        
        function_call = response.choices[0].message["function_call"]
        
        if function_call["name"] == "request_geolocation":
            # Вызываем функцию запроса геолокации
            await request_geolocation(update, context)
            

    # Если функция не вызвалась, возвращаем обычный текстовый ответ:
    bot_reply = response.choices[0].message.content.strip()
    return bot_reply


