import asyncio
from functools import partial
import json
import logging
import os
from telegram import Update
from utils.openrouter_images import generate_image_openrouter, OpenRouterImageError
from utils.openrouter_videos import (
    download_video,
    poll_video_generation,
    submit_video_generation,
    OpenRouterVideoError,
)
from core.generation_error_mapper import map_generation_error
from telegram.ext import (
    ContextTypes,
)
from openai import OpenAI
from openai.types.responses import Response
from utils.elastic import add_note, get_all_user_notes, get_notes_by_query, remove_notes
from core.state_and_commands import add_location_button, animate_service_message, get_OpenAI_Models, get_notes_text, get_user_generation_source_image, get_user_model, get_voice_recognition_model, reply_service_message, reply_service_text, set_user_generation_source_image, set_user_model
from utils.weather import  get_weather_description2, get_weekly_forecast
from utils.yandex_maps import get_location_by_address
import base64
from io import BytesIO
from telegram import InputFile
from PIL import Image


logger = logging.getLogger(__name__)


def _normalize_function_args(function_args):
    if isinstance(function_args, str):
        try:
            return json.loads(function_args)
        except Exception as e:
            logger.warning(
                f"Не удалось распарсить аргументы tool call как JSON: {e}. Значение: {function_args!r}",
                exc_info=True,
            )
            return {}
    if isinstance(function_args, dict):
        return function_args
    if function_args is None:
        return {}

    logger.warning(
        f"Неожиданный тип аргументов tool call: {type(function_args)}. Значение: {function_args!r}",
    )
    return {}


def _extract_function_call(response: Response):
    output_items = list(response.output or [])
    logger.info(
        f"Responses API output: output_text={getattr(response, 'output_text', None)!r}, items={[getattr(item, 'type', type(item).__name__) for item in output_items]}",
    )

    for item in output_items:
        item_type = getattr(item, "type", None)

        if item_type in ("function_call", "custom_tool_call"):
            function_call_name = getattr(item, "name", None)
            function_args = getattr(item, "arguments", None)
            logger.info(
                f"Найден function tool call: name={function_call_name}, args_type={type(function_args)}",
            )
            return function_call_name, _normalize_function_args(function_args)

        if item_type == "tool":
            tool = getattr(item, "tool", None)
            function_call_name = getattr(tool, "name", None) if tool else None
            function_args = getattr(tool, "arguments", None) if tool else None
            logger.info(
                f"Найден legacy tool call: name={function_call_name}, args_type={type(function_args)}",
            )
            return function_call_name, _normalize_function_args(function_args)

    logger.warning(
        f"В ответе Responses API не найден tool call. output_text={getattr(response, 'output_text', None)!r}, raw_output={output_items!r}",
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
        "description": "Сгенерировать изображение только по текстовому описанию пользователя. Используй, когда пользователь просит нарисовать что-то с нуля без опоры на ранее присланное изображение.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Запрос пользователя, по которому сгенерируется картинка"
                },
            },
            "required": ["prompt"]
        }
    },
    {
        "type": "function",
        "name": "generate_image_from_image",
        "description": "Сгенерировать или преобразовать изображение на основе последней картинки, отправленной пользователем, с учетом текстовой инструкции. Используй, когда пользователь просит изменить, стилизовать, перерисовать, улучшить или сделать вариацию на основе ранее присланного изображения.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Инструкция, как преобразовать изображение пользователя"
                },
            },
            "required": ["prompt"]
        }
    },
    {
        "type": "function",
        "name": "generate_video",
        "description": "Сгенерировать видео или анимацию по текстовому описанию пользователя. Если у пользователя есть сохранённое изображение, используй его как основу для видео. Используй, когда пользователь просит видео, анимацию, оживить картинку или сделать видео на основе изображения.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Запрос пользователя, по которому сгенерируется видео"
                },
            },
            "required": ["prompt"]
        }
    },
    { "type": "web_search" },
]




# Запрос геолокации у пользователя:
async def request_geolocation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await add_location_button(update, context)

#
def _prepare_image_for_telegram(b64_data: str, max_size: int = 1280, quality: int = 88) -> BytesIO:
    """Декодирует base64, ресайзит и конвертирует изображение в JPEG для отправки в Telegram."""
    image_bytes = base64.b64decode(b64_data)
    img = Image.open(BytesIO(image_bytes))
    # Конвертируем в RGB, чтобы избежать проблем с альфа-каналом при сохранении в JPEG
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    # Уменьшаем размер, сохраняя пропорции
    img.thumbnail((max_size, max_size))
    bio = BytesIO()
    img.save(bio, format="JPEG", quality=quality, optimize=True)
    bio.seek(0)
    bio.name = "generated.jpg"
    return bio

async def generate_image(prompt: str | None):
    if prompt is None or prompt == "":
        logger.info("Пустой запрос на генерацию изображения")
        return None
    # Пробрасываем OpenRouterImageError для корректной обработки в вызывающем коде
    image_urls = await generate_image_openrouter(prompt=prompt)
    if not image_urls:
        logger.error("OpenRouter не вернул изображений")
        return None
    return image_urls[0]

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
                f"Результат разбора tool call: name={function_call_name}, args={function_args_dict!r}",
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
                try:
                    image_data_url = await generate_image(
                    function_args_dict.get("prompt"),
                    )
                except OpenRouterImageError as e:
                    mapped = map_generation_error(e, context="image")
                    await reply_service_text(update, mapped.user_message)
                    return ModelAnswer(None, additional_system_messages, context_tokens, completion_tokens)
                
                if image_data_url is None:
                    bot_reply = "Не удалось сгенерировать изображение. Внутренняя ошибка сервера"
                    return ModelAnswer(bot_reply, additional_system_messages, context_tokens, completion_tokens)

                try:
                    if image_data_url.startswith("data:"):
                        header, b64_data = image_data_url.split(",", 1)
                        bio = _prepare_image_for_telegram(b64_data)
                        await update.message.reply_photo(photo=InputFile(bio))  # type: ignore
                    else:
                        await update.message.reply_photo(photo=image_data_url)  # type: ignore
                except Exception as e:
                    logger.error(f"Ошибка при отправке картинки в Telegram: {e}", exc_info=True)
                    return ModelAnswer("Картинка сгенерировалась, но не удалось отправить её в Telegram.")

                bot_reply = "Я сделал :)"
                return ModelAnswer(bot_reply, additional_system_messages, context_tokens, completion_tokens)

            if function_call_name == "generate_image_from_image":
                user_id = update.effective_user.id
                source_image_dict = await get_user_generation_source_image(user_id)
                if source_image_dict is None:
                    logger.warning(
                        f"generate_image_from_image вызван, но у пользователя {user_id} нет сохраненного изображения",
                    )
                    bot_reply = "Сначала отправьте изображение, которое будет использоваться как основа для генерации."
                    return ModelAnswer(bot_reply, additional_system_messages, context_tokens, completion_tokens)

                try:
                    img_type = source_image_dict["image_type"]
                    img_b64_str = source_image_dict["image"]
                    data_url = f"data:{img_type};base64,{img_b64_str}"
                except Exception as e:
                    logger.error(f"Ошибка при сборке data URL из сохраненного изображения: {e}", exc_info=True)
                    bot_reply = "Не удалось подготовить исходное изображение для генерации."
                    return ModelAnswer(bot_reply, additional_system_messages, context_tokens, completion_tokens)

                prompt = function_args_dict.get("prompt")
                if not prompt:
                    logger.warning("Пустой prompt для generate_image_from_image")
                    bot_reply = "Не удалось сгенерировать изображение: запрос пустой."
                    return ModelAnswer(bot_reply, additional_system_messages, context_tokens, completion_tokens)
                try:
                    image_urls = await generate_image_openrouter(prompt=prompt, input_images=[data_url])
                    if not image_urls:
                        logger.error("OpenRouter не вернул изображений для img2img")
                        bot_reply = "Не удалось сгенерировать изображение. Внутренняя ошибка сервера"
                        return ModelAnswer(bot_reply, additional_system_messages, context_tokens, completion_tokens)
                    image_data_url = image_urls[0]
                except OpenRouterImageError as e:
                    mapped = map_generation_error(e, context="image")
                    await reply_service_text(update, mapped.user_message)
                    return ModelAnswer(None, additional_system_messages, context_tokens, completion_tokens)
                except Exception as e:
                    logger.error(f"Ошибка при генерации изображения через OpenRouter (img2img): {e}", exc_info=True)
                    bot_reply = "Не удалось сгенерировать изображение. Внутренняя ошибка сервера"
                    return ModelAnswer(bot_reply, additional_system_messages, context_tokens, completion_tokens)

                try:
                    if image_data_url.startswith("data:"):
                        header, b64_data = image_data_url.split(",", 1)
                        bio = _prepare_image_for_telegram(b64_data)
                        await update.message.reply_photo(photo=InputFile(bio))  # type: ignore
                    else:
                        await update.message.reply_photo(photo=image_data_url)  # type: ignore
                except Exception as e:
                    logger.error(f"Ошибка при отправке картинки в Telegram: {e}", exc_info=True)
                    return ModelAnswer("Картинка сгенерировалась, но не удалось отправить её в Telegram.")

                # Очищаем generation source после успешной генерации, чтобы не переиспользовать случайно
                await set_user_generation_source_image(user_id, None)
                bot_reply = "Я сделал :)"
                return ModelAnswer(bot_reply, additional_system_messages, context_tokens, completion_tokens)

            if function_call_name == "generate_video":
                user_id = update.effective_user.id
                prompt = function_args_dict.get("prompt")
                if not prompt:
                    logger.warning("Пустой prompt для generate_video")
                    bot_reply = "Не удалось сгенерировать видео: запрос пустой."
                    return ModelAnswer(bot_reply, additional_system_messages, context_tokens, completion_tokens)

                # Проверяем наличие сохранённого изображения для использования как основы
                source_image_dict = await get_user_generation_source_image(user_id)
                input_references = None
                if source_image_dict is not None:
                    try:
                        img_type = source_image_dict["image_type"]
                        img_b64_str = source_image_dict["image"]
                        data_url = f"data:{img_type};base64,{img_b64_str}"
                        input_references = [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            }
                        ]
                    except Exception as e:
                        logger.error(f"Ошибка при сборке data URL для видео: {e}", exc_info=True)

                try:
                    job_id, polling_url = await submit_video_generation(prompt, input_references)
                    logger.info(f"Video job submitted: {job_id}")
                except OpenRouterVideoError as e:
                    mapped = map_generation_error(e, context="video")
                    await reply_service_text(update, mapped.user_message)
                    return ModelAnswer(None, additional_system_messages, context_tokens, completion_tokens)
                except Exception as e:
                    logger.error(f"Ошибка при отправке запроса на генерацию видео: {e}", exc_info=True)
                    bot_reply = "Не удалось начать генерацию видео. Попробуйте позже."
                    return ModelAnswer(bot_reply, additional_system_messages, context_tokens, completion_tokens)

                status_message = await reply_service_message(update, "Видео генерируется, подождите")
                animation_task = asyncio.create_task(
                    animate_service_message(status_message, "Видео генерируется, подождите", interval=1.5)
                )

                try:
                    unsigned_urls = await poll_video_generation(polling_url)
                    video_url = unsigned_urls[0]
                    video_path = await download_video(video_url)
                except OpenRouterVideoError as e:
                    logger.error(f"Ошибка при генерации видео: {e}", exc_info=True)
                    try:
                        await status_message.edit_text("_Не удалось сгенерировать видео._", parse_mode="MarkdownV2")
                    except Exception:
                        pass
                    return ModelAnswer(None, additional_system_messages, context_tokens, completion_tokens)
                except Exception as e:
                    logger.error(f"Неожиданная ошибка при генерации видео: {e}", exc_info=True)
                    try:
                        await status_message.edit_text("_Произошла ошибка при генерации видео._", parse_mode="MarkdownV2")
                    except Exception:
                        pass
                    return ModelAnswer(None, additional_system_messages, context_tokens, completion_tokens)
                finally:
                    animation_task.cancel()
                    try:
                        await animation_task
                    except asyncio.CancelledError:
                        pass

                try:
                    await status_message.edit_text("_Видео готово, отправляю..._", parse_mode="MarkdownV2")
                except Exception:
                    pass

                try:
                    with open(video_path, "rb") as video_file:
                        await update.message.reply_video(video=InputFile(video_file))  # type: ignore
                except Exception as e:
                    logger.error(f"Ошибка при отправке видео как video: {e}", exc_info=True)
                    try:
                        with open(video_path, "rb") as video_file:
                            await update.message.reply_document(document=InputFile(video_file))  # type: ignore
                    except Exception as e2:
                        logger.error(f"Ошибка при отправке видео как document: {e2}", exc_info=True)
                        try:
                            await status_message.edit_text("_Видео сгенерировано, но не удалось отправить его в Telegram._", parse_mode="MarkdownV2")
                        except Exception:
                            pass
                        return ModelAnswer(None, additional_system_messages, context_tokens, completion_tokens)
                finally:
                    try:
                        os.remove(video_path)
                    except Exception:
                        pass

                # Очищаем generation source после успешной генерации
                await set_user_generation_source_image(user_id, None)
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
                    await reply_service_text(update, "Произошла ошибка при получении геолокации. Попробуйте позже.")
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
                    f"Пустой output_text без обработанного tool call. raw_output={list(response.output or [])!r}",
                )
                bot_reply = "Произошла ошибка при обработке запроса."
            else:
                logger.info(f"Текстовый ответ модели успешно извлечён: {bot_reply[:500]!r}")
        else:
            logger.error(f"Неожиданный тип ответа от get_simple_answer: {type(response)}, значение={response!r}")
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



