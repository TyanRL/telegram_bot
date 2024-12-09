import asyncio
from functools import partial
import json
import logging
import os
import openai
import requests
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from state_and_commands import add_location_button, get_geolocation
from weather import get_weather_description





# Описываем доступные функции для модели:
functions = [
    {
        "name": "request_geolocation",
        "description": "Получить геолокацию пользователя",
        "parameters": {
            "type": "object",
            "properties": {}
        },

        "name": "get_weather_description",
        "description": "Получить описание погоды по геолокации пользователя.",
        "parameters": {
            "type": "object",
            "properties": {}
            },

        "name": "generate_image",
        "description": "Сгенерировать изображение по запросу пользователя.",
        "parameters": {
            "type": "object",
            "properties": {
                 "prompt": {
                    "type": "string",
                    "description": "Запрос пользователя по которому сгенерируется картинка"
                    },
                    "style": {
                    "type": "string",
                    "enum": ["vivid", "natural"],
                    "description": "Стиль изображения 'vivid' или 'natural'"
                    },  

                }
            },
    }
]



# Запрос геолокации у пользователя:
async def request_geolocation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await add_location_button(update, context)

# 
def generate_image(prompt:str, style:str):
    response = None
    try:
        if prompt is None or prompt == "":
            logging.info("Пустой запрос на генерацию изображения")
            return
    
        if style  != 'vivid' or style!= 'natural':
            style='vivid'

        response = openai.Image.create(
            model='dall-e-3',
            prompt=prompt,
            n=1,
            size='1024x1024',
            quality='hd',  # Опционально: 'standard' или 'hd'
            style=style  # Опционально: 'vivid' или 'natural'
            )
        image_url = response['data'][0]['url']
    except Exception as e:
        logging.info(f"Response: {str(response)}")
        logging.error("Ошибка при генерации изображения: " + str(e))
        return
    # Отправка изображения пользователю
    return image_url

    


async def get_model_answer(openai_client, update: Update, context: ContextTypes.DEFAULT_TYPE, model_name: str, messages, recursion_depth=0):
    try:
        
        if recursion_depth > 10:
            logging.error("Recursion depth exceeded")
            return None, None

        additional_system_messages=[]
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            partial(
                openai_client.chat.completions.create,
                model=model_name,
                messages=messages,
                functions=functions,
                function_call="auto",  
                max_tokens=16384
            )
        )
        
        if (response.choices and 
            len(response.choices) > 0 and
            hasattr(response.choices[0].message, "function_call")):
    
            function_call = response.choices[0].message.function_call
            
            if function_call and function_call.name == "request_geolocation":
                # Вызываем функцию запроса геолокации
                logging.info("Вызываем функцию запроса геолокации")
                await request_geolocation(update, context)
                return None, None
            if function_call and function_call.name == "get_weather_description":
                logging.info("Вызываем функцию запроса погоды")
                # смотрим есть ли геолокация в для этого пользователя
                geolocation =  await get_geolocation(update.effective_user.id)
                if geolocation is None:
                    # Если геолокации нет, то вызываем функцию запроса геолокации
                    logging.info("Геолокации нет. Вызываем функцию запроса геолокации")
                    await request_geolocation(update, context)
                    return None, None
                else:
                    # Если геолокация есть, то вызываем функцию получения погоды
                    (attitude,longtitude)= geolocation
                    result = get_weather_description(attitude, longtitude)
                    new_system_message={"role": "system", "content": result}
                    additional_system_messages.append(new_system_message)
                    messages.append(new_system_message)
                    (answer, additional_system_messages2) = await get_model_answer(openai_client, update, context, model_name, messages, recursion_depth+1)
                    return answer, additional_system_messages+additional_system_messages2
            if function_call and function_call.name == "generate_image":
                function_args = response.choices[0].message.function_call.arguments
                logging.info(f"Вызываем функцию генерации изображения. Аргументы: {function_args}, Тип: {type(function_args)}")
                function_args_dict = json.loads(function_args)
                image_url = generate_image(function_args_dict["prompt"], function_args_dict["style"])
                if image_url is None:
                    bot_reply = "Не удалось сгенерировать изображение. Попробуйте другой prompt или style."
                    return bot_reply, additional_system_messages
                else:
                    # Если генерация прошла успешно, то отправляем пользователю картинку
                    await update.message.reply_photo(photo=image_url)
                    bot_reply = "Я сделал :)"
                    return bot_reply, additional_system_messages

   
        # Если функция не вызвалась, возвращаем обычный текстовый ответ:
        bot_reply = response.choices[0].message.content.strip()
        return bot_reply, additional_system_messages

    except Exception as e:
        # Логируем ошибки
        logging.error(f"Ошибка при обращении к OpenAI API: {e}")
        return "Произошла ошибка при обработке запроса."
