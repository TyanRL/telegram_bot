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
from state_and_commands import OpenAI_Models, add_location_button, get_OpenAI_Models, get_user_model, get_voice_recognition_model, reply_service_text, set_user_model
from weather import  get_weather_description2, get_weekly_forecast
from yandex_maps import get_location_by_address


MAXIMUM_RECURSION_ANSWER_DEPTH = 10



# Описываем доступные функции для модели:
functions=[
    {
        "name": "request_geolocation",
        "description": "Получить геолокацию пользователя",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
       "name": "get_weather_description",
        "description": "Получить текущую погоду.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Широта."
                },
                "longitude": {
                    "type": "number",
                    "description": "Долгота."
                }
            },
            "required": ["latitude", "longitude"]
        }
    },
    {
        "name": "get_weekly_forecast",
        "description": "Получить прогноз погоды на неделю вперед.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Широта."
                },
                "longitude": {
                    "type": "number",
                    "description": "Долгота."
                }
            },
            "required": ["latitude", "longitude"]
        }
    },
    {
        "name": "get_location_by_address",
        "description": "Получить широту и долготу по адресу или названию местности.",

       "parameters": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "Адрес иил название местности по которому нужно получить геолокацию."
                },
            }
        }
    },

    {
        "name": "generate_image",
        "description": "Сгенерировать изображение по запросу пользователя.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Запрос пользователя, по которому сгенерируется картинка"
                },
                "style": {
                    "type": "string",
                    "enum": ["vivid", "natural"],
                    "description": "Стиль изображения: 'vivid' или 'natural'"
                }
            }
        }
    },
    {
        "name": "change_model",
        "description": "Сменить модель генерации текста",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": [f"{OpenAI_Models.DEFAULT_MODEL.value}", f"{OpenAI_Models.O1_MINI.value}"],
                    "description": f"'{OpenAI_Models.DEFAULT_MODEL.value}' - основная используемая модель широкого назначения. '{OpenAI_Models.O1_MINI.value}' - модель с рассуждениями, подходящая для решения логических задач, написания кода и научных целей. Не имеет function calling, не может работать с изображениями. Если пользователя не устраивают текущие результаты, то можно сменить модель."
                 },
                 "style": {
                     "type": "string",
                     "enum": ["vivid", "natural"],
                     "description": "Стиль изображения: 'vivid' или 'natural'"

                }
            }
        }
    }
]




# Запрос геолокации у пользователя:
async def request_geolocation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await add_location_button(update, context)

# 
def generate_image(openai_client, prompt:str, style:str):
    response = None
    try:
        if prompt is None or prompt == "":
            logging.info("Пустой запрос на генерацию изображения")
            return
    
        if style  != 'vivid' or style!= 'natural':
            style='vivid'

        response = openai_client.images.generate(
            model='dall-e-3',
            prompt=prompt,
            n=1,
            size='1024x1024',
            #quality='hd',  # Опционально: 'standard' или 'hd'
            style=style  # Опционально: 'vivid' или 'natural'
            )
        # Получаем первый объект изображения из списка data
        image = response.data[0]

        # Извлекаем URL изображения
        image_url = image.url
    except Exception as e:
        logging.info(f"Response: {str(response)}")
        logging.error("Ошибка при генерации изображения: " + str(e))
        return
    # Отправка изображения пользователю
    return image_url

def transcribe_audio(openai_client, audio_filename):
    try:
         # Распознавание речи с использованием OpenAI
        transcription = openai_client.audio.transcriptions.create(
                            model=get_voice_recognition_model(),
                            file=open(audio_filename, 'rb')
                            )
        recognized_text=transcription.text
    except Exception as e:
        logging.error("Ошибка при распознавании речи: " + str(e))
        return
    # Отправка текста пользователю
    return recognized_text





async def get_model_answer(openai_client, update: Update, context: ContextTypes.DEFAULT_TYPE, messages, recursion_depth=0):
    try:
        logging.info(f"Запрос к модели: {str(messages[-1])}, глубина рекурсии {recursion_depth}")   

        if recursion_depth > MAXIMUM_RECURSION_ANSWER_DEPTH:
            logging.error("Recursion depth exceeded")
            return None, None

        additional_system_messages=[]
        model_name=await get_user_model(update.effective_user.id)
        functions_to_call= functions
        function_call_parameter="auto"

        # так как модели o1 не поддерживают сиcтемные сообщения то удалим их
        if model_name == OpenAI_Models.O1_MINI.value:
             filtered_messages = [message for message in messages if message["role"] != "system"]
             messages=filtered_messages
             function_call_parameter="",  



        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            partial(
                openai_client.chat.completions.create,
                model=model_name,
                messages=messages,
                functions=functions_to_call,
                function_call=function_call_parameter,  
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
            if function_call and (function_call.name == "get_weather_description" or function_call.name == "get_weekly_forecast"):
                function_args = response.choices[0].message.function_call.arguments
                logging.info(f"Вызываем функцию запроса погоды. Аргументы: {function_args}, Тип: {type(function_args)}")
                function_args_dict = json.loads(function_args)
                latitude=function_args_dict["latitude"]
                longitude=function_args_dict["longitude"]
                # Если геолокация есть, то вызываем функцию получения погоды
                if function_call.name == "get_weather_description":
                    result = get_weather_description2(latitude, longitude)
                elif function_call.name == "get_weekly_forecast":
                    result = get_weekly_forecast(latitude, longitude)
                    
                new_system_message={"role": "system", "content": result}
                additional_system_messages.append(new_system_message)
                messages.append(new_system_message)
                (answer, additional_system_messages2) = await get_model_answer(openai_client, update, context, messages, recursion_depth+1)
                return answer, additional_system_messages+additional_system_messages2


            if function_call and function_call.name == "generate_image":
                function_args = response.choices[0].message.function_call.arguments
                logging.info(f"Вызываем функцию генерации изображения. Аргументы: {function_args}, Тип: {type(function_args)}")
                function_args_dict = json.loads(function_args)
                image_url = generate_image(openai_client, function_args_dict["prompt"], function_args_dict["style"])
                if image_url is None:
                    bot_reply = "Не удалось сгенерировать изображение. Попробуйте другой prompt или style."
                    return bot_reply, additional_system_messages
                else:
                    # Если генерация прошла успешно, то отправляем пользователю картинку
                    await update.message.reply_photo(photo=image_url)
                    bot_reply = "Я сделал :)"
                    return bot_reply, additional_system_messages
            
            if function_call and function_call.name == "change_model":
                function_args = response.choices[0].message.function_call.arguments
                logging.info(f"Вызываем функцию смены модели. Аргументы: {function_args}, Тип: {type(function_args)}")
                function_args_dict = json.loads(function_args)
                new_model_name_str = function_args_dict["model"]
                new_model_name = get_OpenAI_Models(new_model_name_str)
                if new_model_name_str!=model_name:
                    await set_user_model(update.effective_user.id,new_model_name)
                    await reply_service_text(update, f"Модель успешно изменена на {new_model_name_str}. Модель не поддерживает работу с инструментами (погода, геолокация, картинки и т.д.). Для возврата на стандартную модель сбросьте контекст (/reset)")
                    new_system_message={"role": "system", "content": f"Модель успешно изменена на {new_model_name_str}. Модель не поддерживает работу с инструментами (погода, геолокация, картинки и т.д.)."}
                    additional_system_messages.append(new_system_message)
                    messages.append(new_system_message)
                (answer, additional_system_messages2) = await get_model_answer(openai_client, update, context, messages, recursion_depth+1)
                return answer, additional_system_messages+additional_system_messages2
            
            if function_call and function_call.name == "get_location_by_address":
                function_args = response.choices[0].message.function_call.arguments
                logging.info(f"Вызываем функцию получения геолокации по адресу. Аргументы: {function_args}, Тип: {type(function_args)}")
                function_args_dict = json.loads(function_args)
                address=function_args_dict["address"]
                geoloc = get_location_by_address(address)
                if geoloc is None:
                    bot_reply = "Не удалось получить геолокацию."
                    return bot_reply, additional_system_messages
                else:
                    (latitude, longitude) = geoloc
                    result = f"Геолокация {address} установлена. Широта: {latitude}, Долгота: {longitude}"
                    logging.info(result)
                    new_system_message={"role": "system", "content": result}
                    additional_system_messages.append(new_system_message)
                    messages.append(new_system_message)
                    (answer, additional_system_messages2) = await get_model_answer(openai_client, update, context, messages, recursion_depth+1)
                    return answer, additional_system_messages+additional_system_messages2


   
        # Если функция не вызвалась, возвращаем обычный текстовый ответ:
        bot_reply = response.choices[0].message.content.strip()
        return bot_reply, additional_system_messages

    except Exception as e:
        # Логируем ошибки
        logging.error(f"Ошибка при обращении к OpenAI API: {e}")
        return "Произошла ошибка при обработке запроса."
