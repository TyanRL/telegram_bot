import asyncio
from functools import partial
import json
import logging
import os
from typing import Tuple
import openai
import requests
from telegram import Update
from utils.openrouter_images import generate_image_openrouter
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from openai import OpenAI
from openai.types.responses import Response
from common_types import dict_to_markdown
from utils.elastic import add_note, get_all_user_notes, get_notes_by_query, remove_notes
from utils.google_search import get_search_results
from state_and_commands import OpenAI_Models, add_location_button, get_OpenAI_Models, get_notes_text, get_user_model, get_voice_recognition_model, reply_service_text, set_user_model
from utils.weather import  get_weather_description2, get_weekly_forecast
from utils.yandex_maps import get_location_by_address


logger = logging.getLogger(__name__)


def _normalize_function_args(function_args):
    if isinstance(function_args, str):
        try:
            return json.loads(function_args)
        except Exception as e:
            logger.warning(
                "Не удалось распарсить аргументы tool call как JSON: %s. Значение: %r",
                e,
                function_args,
                exc_info=True,
            )
            return {}
    if isinstance(function_args, dict):
        return function_args
    if function_args is None:
        return {}

    logger.warning(
        "Неожиданный тип аргументов tool call: %s. Значение: %r",
        type(function_args),
        function_args,
    )
    return {}


def _extract_function_call(response: Response):
    output_items = list(response.output or [])
    logger.info(
        "Responses API output: output_text=%r, items=%s",
        getattr(response, "output_text", None),
        [getattr(item, "type", type(item).__name__) for item in output_items],
    )

    for item in output_items:
        item_type = getattr(item, "type", None)

        if item_type in ("function_call", "custom_tool_call"):
            function_call_name = getattr(item, "name", None)
            function_args = getattr(item, "arguments", None)
            logger.info(
                "Найден function tool call: name=%s, args_type=%s",
                function_call_name,
                type(function_args),
            )
            return function_call_name, _normalize_function_args(function_args)

        if item_type == "tool":
            tool = getattr(item, "tool", None)
            function_call_name = getattr(tool, "name", None) if tool else None
            function_args = getattr(tool, "arguments", None) if tool else None
            logger.info(
                "Найден legacy tool call: name=%s, args_type=%s",
                function_call_name,
                type(function_args),
            )
            return function_call_name, _normalize_function_args(function_args)

    logger.warning(
        "В ответе Responses API не найден tool call. output_text=%r, raw_output=%r",
        getattr(response, "output_text", None),
        output_items,
    )
    return None, {}

# Инициализация OpenAI
opena_ai_api_key=os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=opena_ai_api_key)

MAXIMUM_RECURSION_ANSWER_DEPTH = 10

class ModelAnswer():
    bot_reply: str|None
    additional_system_messages: list[dict]
    ctx_token: int
    completion_token: int

    def __init__(self, bot_reply: str|None, additional_system_messages: list[dict]=[], ctx_token: int=0, completion_token: int=0):
        self.bot_reply = bot_reply
        self.additional_system_messages = additional_system_messages
        self.ctx_token = ctx_token
        self.completion_token = completion_token

# Описываем доступные функции для модели:
functions=[
    {
        "type": "function",
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
            "required": ["prompt", "style"]
        }
    },
    { "type": "web_search" },
]




# Запрос геолокации у пользователя:
async def request_geolocation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await add_location_button(update, context)

# 
def generate_image(prompt: str | None, style: str | None):
    try:
        if prompt is None or prompt == "":
            logger.info("Пустой запрос на генерацию изображения")
            return None
        image_urls = generate_image_openrouter(prompt=prompt)
        if not image_urls:
            logger.error("OpenRouter не вернул изображений")
            return None
        return image_urls[0]
    except Exception as e:
        logger.error("Ошибка при генерации изображения через OpenRouter: " + str(e), exc_info=True)
        return None

def transcribe_audio(audio_filename):
    try:
         # Распознавание речи с использованием OpenAI
        transcription = openai_client.audio.transcriptions.create(
                            model=get_voice_recognition_model(),
                            file=open(audio_filename, 'rb')
                            )
        recognized_text=transcription.text
    except Exception as e:
        logger.error("Ошибка при распознавании речи: " + str(e))
        return
    # Отправка текста пользователю
    return recognized_text





async def get_model_answer(update: Update, context: ContextTypes.DEFAULT_TYPE, messages: list[dict], recursion_depth=0)->ModelAnswer:
    try:
        logger.info(f"Запрос к модели: {str(messages[-1])}, глубина рекурсии {recursion_depth}")   

        if recursion_depth > MAXIMUM_RECURSION_ANSWER_DEPTH:
            logger.error("Recursion depth exceeded")
            return ModelAnswer(None)

        if update.effective_user is None:
            logger.error("User is None")
            return ModelAnswer(None)

        context_tokens=0
        completion_tokens=0
        additional_system_messages=[]
        model_name=await get_user_model(update.effective_user.id)
        response = await get_simple_answer(messages, model_name)
        
        if isinstance(response, Response) and response.usage is not None:
            if hasattr(response.usage, "input_tokens"):
                context_tokens += response.usage.input_tokens  # type: ignore
            if hasattr(response.usage, "output_tokens"):
                completion_tokens += response.usage.output_tokens  # type: ignore


        # Обработка tool-calls Responses API
        if isinstance(response, Response):
            function_call_name, function_args_dict = _extract_function_call(response)
            logger.info(
                "Результат разбора tool call: name=%s, args=%r",
                function_call_name,
                function_args_dict,
            )

            if function_call_name == "request_geolocation":
                logger.info("Вызываем функцию запроса геолокации")
                await request_geolocation(update, context)
                return ModelAnswer(None, [], context_tokens, completion_tokens)

            if function_call_name in ("get_weather_description", "get_weekly_forecast"):
                try:
                    latitude = function_args_dict["latitude"]
                    longitude = function_args_dict["longitude"]

                    logger.info(f"Вызываем {function_call_name} для координат: {latitude}, {longitude}")

                    if function_call_name == "get_weather_description":
                        result = get_weather_description2(latitude, longitude)
                    else:
                        result = get_weekly_forecast(latitude, longitude)

                    if not result or result.startswith("Ошибка"):
                        error_msg = f"Не удалось получить данные о погоде для координат {latitude}, {longitude}"
                        logger.error(error_msg)
                        await reply_service_text(update, error_msg)
                        return ModelAnswer(error_msg, additional_system_messages, context_tokens, completion_tokens)

                    logger.info(f"Данные о погоде получены: {result[:100]}...")
                    new_system_message = {"role": "system", "content": result}
                    additional_system_messages.append(new_system_message)
                    messages.append(new_system_message)

                    inner_answer = await get_model_answer(update, context, messages, recursion_depth + 1)
                    context_tokens += inner_answer.ctx_token
                    completion_tokens += inner_answer.completion_token
                    return ModelAnswer(
                        inner_answer.bot_reply,
                        additional_system_messages + inner_answer.additional_system_messages,
                        context_tokens,
                        completion_tokens
                    )
                except Exception as e:
                    error_msg = f"Ошибка при получении данных о погоде: {e}"
                    logger.error(error_msg, exc_info=True)
                    await reply_service_text(update, "Произошла ошибка при получении данных о погоде. Попробуйте позже.")
                    return ModelAnswer("Произошла ошибка при обработке запроса.", additional_system_messages, context_tokens, completion_tokens)

            if function_call_name == "generate_image":
                image_url = generate_image(
                    function_args_dict.get("prompt"),
                    function_args_dict.get("style")
                )
                if image_url is None:
                    logger.warning(
                        "Генерация изображения не удалась. prompt=%r, style=%r",
                        function_args_dict.get("prompt"),
                        function_args_dict.get("style"),
                    )
                    bot_reply = "Не удалось сгенерировать изображение. Попробуйте другой prompt или style."
                    return ModelAnswer(bot_reply, additional_system_messages, context_tokens, completion_tokens)
                await update.message.reply_photo(photo=image_url)  # type: ignore
                bot_reply = "Я сделал :)"
                return ModelAnswer(bot_reply, additional_system_messages, context_tokens, completion_tokens)

            if function_call_name == "change_model":
                new_model_name_str = function_args_dict["model"]
                new_model_name = get_OpenAI_Models(new_model_name_str)
                if new_model_name_str != model_name:
                    await set_user_model(update.effective_user.id, new_model_name)
                    await reply_service_text(update, f"Модель успешно изменена на {new_model_name_str}. Для возврата на стандартную модель сбросьте контекст (/reset)")
                    return ModelAnswer(None, [], context_tokens, completion_tokens)

            if function_call_name == "get_location_by_address":
                address = function_args_dict["address"]
                logger.info(f"Вызываем get_location_by_address для адреса: {address}")

                try:
                    geoloc = get_location_by_address(address)
                    if geoloc is None:
                        error_msg = f"Не удалось получить геолокацию для адреса '{address}'. Проверьте правильность написания адреса и доступность сервиса геокодирования."
                        logger.error(error_msg)
                        await reply_service_text(update, error_msg)
                        return ModelAnswer(error_msg, additional_system_messages, context_tokens, completion_tokens)

                    (latitude, longitude) = geoloc
                    result = f"Геолокация для '{address}' установлена. Широта: {latitude}, Долгота: {longitude}"
                    logger.info(result)

                    new_system_message = {"role": "system", "content": result}
                    additional_system_messages.append(new_system_message)
                    messages.append(new_system_message)

                    inner_answer = await get_model_answer(update, context, messages, recursion_depth + 1)
                    context_tokens += inner_answer.ctx_token
                    completion_tokens += inner_answer.completion_token
                    return ModelAnswer(
                        inner_answer.bot_reply,
                        additional_system_messages + inner_answer.additional_system_messages,
                        context_tokens,
                        completion_tokens
                    )
                except Exception as e:
                    error_msg = f"Ошибка при получении геолокации для адреса '{address}': {e}"
                    logger.error(error_msg, exc_info=True)
                    await reply_service_text(update, f"Произошла ошибка при получении геолокации. Попробуйте позже.")
                    return ModelAnswer("Произошла ошибка при обработке запроса.", additional_system_messages, context_tokens, completion_tokens)

            if function_call_name == "add_note":
                title = function_args_dict["title"]
                body = function_args_dict["body"]
                tags = function_args_dict["tags"]
                add_note(update.effective_user.id, title, body, tags)
                await reply_service_text(update, f"Заметка '{title}' добавлена.")
                bot_reply = "Я сделал :)"
                return ModelAnswer(bot_reply, additional_system_messages, context_tokens, completion_tokens)

            if function_call_name == "get_all_user_notes":
                logger.info("Вызываем функцию получения всех заметок.")
                documents = get_all_user_notes(update.effective_user.id)
                if len(documents) == 0:
                    await reply_service_text(update, "Заметки не найдены.")
                    return ModelAnswer(None, [], context_tokens, completion_tokens)

                answer, system_message_body = get_notes_text(documents)
                new_system_message = {"role": "system", "content": system_message_body}
                additional_system_messages.append(new_system_message)
                messages.append(new_system_message)

                await reply_service_text(update, f"Найдено {len(documents)} заметки(-ок).")
                return ModelAnswer(answer, additional_system_messages, context_tokens, completion_tokens)

            if function_call_name == "get_notes_by_query":
                search_query = function_args_dict["search_query"]
                start_date = function_args_dict.get("start_created_date") or ""
                end_date = function_args_dict.get("end_created_date") or ""

                documents = get_notes_by_query(update.effective_user.id, search_query, start_date, end_date)
                if len(documents) == 0:
                    await reply_service_text(update, "Заметки не найдены.")
                    return ModelAnswer(None, [], context_tokens, completion_tokens)

                answer, system_message_body = get_notes_text(documents)
                new_system_message = {"role": "system", "content": system_message_body}
                additional_system_messages.append(new_system_message)
                messages.append(new_system_message)
                inner_answer = await get_model_answer(update, context, messages, recursion_depth + 1)
                context_tokens += inner_answer.ctx_token
                completion_tokens += inner_answer.completion_token
                return ModelAnswer(
                    inner_answer.bot_reply,
                    additional_system_messages + inner_answer.additional_system_messages,
                    context_tokens,
                    completion_tokens
                )

            if function_call_name == "remove_notes":
                note_ids = [int(x) for x in function_args_dict["note_ids"]]
                await remove_notes(note_ids)
                bot_reply = "Заметки удалены"
                return ModelAnswer(bot_reply, additional_system_messages, context_tokens, completion_tokens)

   
        # Если функция не вызвалась, возвращаем обычный текстовый ответ:
        if isinstance(response, Response):
            bot_reply = (getattr(response, "output_text", None) or "").strip()
            if bot_reply == "":
                logger.warning(
                    "Пустой output_text без обработанного tool call. raw_output=%r",
                    list(response.output or []),
                )
                bot_reply = "Произошла ошибка при обработке запроса."
            else:
                logger.info("Текстовый ответ модели успешно извлечён: %r", bot_reply[:500])
        else:
            logger.error("Неожиданный тип ответа от get_simple_answer: %s, значение=%r", type(response), response)
            bot_reply = "Произошла ошибка при обработке запроса."
        
        return ModelAnswer(bot_reply,
                                additional_system_messages,
                                context_tokens,completion_tokens)

    except Exception as e:
        # Логируем ошибки
        logger.error(f"Ошибка при обращении к OpenAI API: {e}", exc_info=True)
        return ModelAnswer("Произошла ошибка при обработке запроса.")

async def get_simple_answer(messages, model_name) -> Response:
        partial_param = partial(
                openai_client.responses.create,
                model=model_name,
                input=messages,
                max_output_tokens=16384,
                tools=functions,
                text={"verbosity": "low"},
                reasoning={"effort": "medium"}
            )  # type: ignore

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, partial_param)

        return response



