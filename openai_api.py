import asyncio
import logging
import openai
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

async def get_model_answer(openai_client, update: Update, context: ContextTypes.DEFAULT_TYPE, model_name: str, messages):
    try:
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

        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            functions=functions,
            function_call="auto",  # Let the model decide whether to call the function
            max_tokens=16384
        )

        # Check if a function call was returned
        if (response.choices and 
            len(response.choices) > 0 and
            hasattr(response.choices[0].message, "function_call")):
            
            function_call = response.choices[0].message.function_call
            
            if function_call.name == "request_geolocation":
                # Вызываем функцию запроса геолокации
                await request_geolocation(update, context)
                return None

        # Если функция не вызвалась, возвращаем обычный текстовый ответ:
        bot_reply = response.choices[0].message.content.strip()
        return bot_reply

    except Exception as e:
        # Логируем ошибки
        logging.error(f"Ошибка при обращении к OpenAI API: {e}")
        return "Произошла ошибка при обработке запроса."
