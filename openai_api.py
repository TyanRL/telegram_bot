import asyncio
from functools import partial
import json
import logging
import os
from typing import Tuple
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
from openai import OpenAI
from common_types import dict_to_markdown
from elastic import add_note, get_all_user_notes, get_notes_by_query, remove_notes
from state_and_commands import OpenAI_Models, add_location_button, get_OpenAI_Models, get_notes_text, get_user_model, get_voice_recognition_model, reply_service_text, set_user_model
from weather import  get_weather_description2, get_weekly_forecast
from yandex_maps import get_location_by_address



# Инициализация OpenAI
opena_ai_api_key=os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=opena_ai_api_key)

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
            },
             "required": [
                 "prompt",
             ]
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
            },
            "required": [
                 "model",
             ]
        }
    },
    {
        "name": "add_note",
        "description": "Добавить заметку по запросу пользователя",

       "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Название заметки"
                },
                "body": {
                    "type": "string",
                    "description": "Тело заметки"
                },
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Массив тегов, связанных с заметкой"
                },
            },
            "required": [
                "title",
                "body",
            ]

        }
    },
    {
        "name": "get_notes_by_query",
        "description": "Найти существующую заметку по запросу пользователя",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "Ключевые слова для поиска заметок в Elasticsearch"
                },
                "start_created_date": {
                    "type": "string",
                    "description": "Начальная дата диапазона для поиска заметок в Elasticsearch по дате СОЗДАНИЯ заметки. Это не дата события внутри заметки."
                },
                "end_created_date": {
                    "type": "string",
                    "description": "Конечная дата диапазона для поиска заметок в Elasticsearch по дате СОЗДАНИЯ заметкию Это не дата события внутри заметки."
                },
            },
            "required": [
                 "search_query",
             ]
        }
    },
    {
        "name": "get_all_user_notes",
        "description": "Получить все заметки пользователя",

         "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "remove_notes",
        "description": "Удалить заметку с идентификаторами note_ids",
        "parameters": {
            "type": "object",
            "properties": {
                 "note_ids": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "Идентификаторы удаляемых заметок"
                },
            },
            "required": [
                "note_ids",
             ]

        }
    },
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

def transcribe_audio(audio_filename):
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





async def get_model_answer(update: Update, context: ContextTypes.DEFAULT_TYPE, messages, recursion_depth=0)->Tuple[str, list[dict], str]:
    try:
        logging.info(f"Запрос к модели: {str(messages[-1])}, глубина рекурсии {recursion_depth}")   

        if recursion_depth > MAXIMUM_RECURSION_ANSWER_DEPTH:
            logging.error("Recursion depth exceeded")
            return None, None, None


        additional_system_messages=[]
        model_name=await get_user_model(update.effective_user.id)
        
        partial_param = partial(
                openai_client.chat.completions.create,
                model=model_name,
                messages=messages,
                functions=functions,
                function_call="auto",  
                max_tokens=16384
            )

        # так как модели o1 не поддерживают сиcтемные сообщения то удалим их
        if model_name == OpenAI_Models.O1_MINI.value:
            filtered_messages = [message for message in messages if message["role"] != "system"]
            partial_param = partial(
                openai_client.chat.completions.create,
                model=model_name,
                messages=filtered_messages,
                max_completion_tokens=32768
            )
             

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            partial_param
        )
        
        if (response.choices and 
            len(response.choices) > 0 and
            hasattr(response.choices[0].message, "function_call")):
    
            function_call = response.choices[0].message.function_call
            
            if function_call and function_call.name == "request_geolocation":
                # Вызываем функцию запроса геолокации
                logging.info("Вызываем функцию запроса геолокации")
                await request_geolocation(update, context)
                return None, None, None
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
                (answer, additional_system_messages2, service_after_message) = await get_model_answer(update, context, messages, recursion_depth+1)
                return answer, additional_system_messages+additional_system_messages2, service_after_message


            if function_call and function_call.name == "generate_image":
                function_args = response.choices[0].message.function_call.arguments
                logging.info(f"Вызываем функцию генерации изображения. Аргументы: {function_args}, Тип: {type(function_args)}")
                function_args_dict = json.loads(function_args)
                image_url = generate_image(openai_client, function_args_dict["prompt"], function_args_dict["style"])
                if image_url is None:
                    bot_reply = "Не удалось сгенерировать изображение. Попробуйте другой prompt или style."
                    return bot_reply, additional_system_messages, None
                else:
                    # Если генерация прошла успешно, то отправляем пользователю картинку
                    await update.message.reply_photo(photo=image_url)
                    bot_reply = "Я сделал :)"
                    return bot_reply, additional_system_messages, None
            
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
                    return None,None, None
            
            if function_call and function_call.name == "get_location_by_address":
                function_args = response.choices[0].message.function_call.arguments
                logging.info(f"Вызываем функцию получения геолокации по адресу. Аргументы: {function_args}, Тип: {type(function_args)}")
                function_args_dict = json.loads(function_args)
                address=function_args_dict["address"]
                geoloc = get_location_by_address(address)
                if geoloc is None:
                    bot_reply = "Не удалось получить геолокацию."
                    return bot_reply, additional_system_messages, None
                else:
                    (latitude, longitude) = geoloc
                    result = f"Геолокация {address} установлена. Широта: {latitude}, Долгота: {longitude}"
                    logging.info(result)
                    new_system_message={"role": "system", "content": result}
                    additional_system_messages.append(new_system_message)
                    messages.append(new_system_message)
                    (answer, additional_system_messages2, service_after_message) = await get_model_answer(update, context, messages, recursion_depth+1)
                    return answer, additional_system_messages+additional_system_messages2, service_after_message

            if function_call and (function_call.name == "add_note"):
                logging.info(f"Function call arguments 1: {response.choices[0]}")
                logging.info(f"Function call arguments 2: {response.choices[0].message}")
                logging.info(f"Function call arguments 3: {response.choices[0].message.function_call}")
                logging.info(f"Function call arguments 4: {response.choices[0].message.function_call.arguments}")

                function_args = response.choices[0].message.function_call.arguments
                logging.info(f"Вызываем функцию добавления заметки. Аргументы: {function_args}, Тип: {type(function_args)}")
                # Если function_args это строка, парсим её
                if isinstance(function_args, str):
                    function_args_dict = json.loads(function_args)
                else:
                    function_args_dict = function_args  # если это уже словарь
                
                title=function_args_dict["title"]
                body=function_args_dict["body"]
                tags=function_args_dict["tags"]
                
                add_note(update.effective_user.id,title, body, tags)
                await reply_service_text(update,f"Заметка '{title}' добавлена.")
                bot_reply = "Я сделал :)"
                return bot_reply, additional_system_messages, None
                

            if function_call and (function_call.name == "get_all_user_notes"):
                logging.info(f"Вызываем функцию получения всех заметок.")
                documents = get_all_user_notes(update.effective_user.id)
                
                if len(documents) == 0:
                    await reply_service_text(update,"Заметки не найдены.")
                    return None, None, None
                
                answer, system_message_body = get_notes_text(documents)

                new_system_message={"role": "system", "content": system_message_body}
                additional_system_messages.append(new_system_message)
                messages.append(new_system_message)
                
                await reply_service_text(update,f"Найдено {len(documents)} заметки(-ок).")
                return answer, additional_system_messages, None
            
            if function_call and (function_call.name == "get_notes_by_query"):
                function_args = response.choices[0].message.function_call.arguments
                logging.info(f"Вызываем функцию поиска заметки. Аргументы: {function_args}, Тип: {type(function_args)}")
                function_args_dict = json.loads(function_args)
                search_query=function_args_dict["search_query"]
                start_date=function_args_dict.get("start_created_date",None)
                end_date=function_args_dict.get("end_created_date",None)

                
                documents = get_notes_by_query(update.effective_user.id, search_query, start_date, end_date)
                if len(documents) == 0:
                    await reply_service_text(update,"Заметки не найдены.")
                    return None, None, None
                
                answer, system_message_body = get_notes_text(documents)
                new_system_message={"role": "system", "content": system_message_body}
                additional_system_messages.append(new_system_message)
                messages.append(new_system_message)
                (answer, additional_system_messages2, service_after_message) = await get_model_answer(update, context, messages, recursion_depth+1)
                return answer, additional_system_messages+additional_system_messages2, service_after_message

            if function_call and (function_call.name == "remove_notes"):
                function_args = response.choices[0].message.function_call.arguments
                logging.info(f"Вызываем функцию удаления заметок. Аргументы: {function_args}, Тип: {type(function_args)}")
                # Если function_args это строка, парсим её
                if isinstance(function_args, str):
                    function_args_dict = json.loads(function_args)
                else:
                    function_args_dict = function_args  # если это уже словарь
                
                note_ids=[int(x) for x in function_args_dict["note_ids"]]
                await remove_notes(note_ids)
                bot_reply = "Заметки удалены"
                return bot_reply, additional_system_messages, None  


   
        # Если функция не вызвалась, возвращаем обычный текстовый ответ:
        bot_reply = response.choices[0].message.content.strip()
        return bot_reply, additional_system_messages, None

    except Exception as e:
        # Логируем ошибки
        logging.error(f"Ошибка при обращении к OpenAI API: {e}", exc_info=True)
        return "Произошла ошибка при обработке запроса.", None, None


