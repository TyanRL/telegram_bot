import asyncio
import base64
from functools import partial
import json
import logging
import os


from openai import OpenAI
from openai.types.responses import Response
from telegram import Update
from telegram.ext import ContextTypes

from core.common_types import GeneratedImage, ModelAnswer
from core.state_and_commands import (
    get_user_model,
    get_voice_recognition_model,
)
from core.tool_registry import TOOLS_SCHEMA, get_tool_registry, ToolExecutionContext
from utils.openrouter_client import OpenRouterService

# Импорт handler-модулей для регистрации в реестре
from core.tool_handlers import weather, media, model, notes  # noqa: F401

logger = logging.getLogger(__name__)

# Инициализация OpenAI
opena_ai_api_key = os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=opena_ai_api_key)

MAXIMUM_RECURSION_ANSWER_DEPTH = 10


def _normalize_function_args(function_args):
    if isinstance(function_args, str):
        try:
            return json.loads(function_args)
        except Exception as e:
            logger.warning(
                "Не удалось распарсить аргументы tool call как JSON: type=%s",
                type(e).__name__,
            )
            return {}
    if isinstance(function_args, dict):
        return function_args
    if function_args is None:
        return {}

    logger.warning(
        "Неожиданный тип аргументов tool call: %s",
        type(function_args).__name__,
    )
    return {}


def _extract_function_call(response: Response):
    output_items = list(response.output or [])
    logger.info(
        "Responses API output: items=%s",
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
                type(function_args).__name__,
            )
            return function_call_name, _normalize_function_args(function_args)

        if item_type == "tool":
            tool = getattr(item, "tool", None)
            function_call_name = getattr(tool, "name", None) if tool else None
            function_args = getattr(tool, "arguments", None) if tool else None
            logger.info(
                "Найден tool call: name=%s, args_type=%s",
                function_call_name,
                type(function_args).__name__,
            )
            return function_call_name, _normalize_function_args(function_args)

    logger.warning(
        "В ответе Responses API не найден tool call: item_count=%s",
        len(output_items),
    )
    return None, {}


async def get_model_answer(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    messages: list[dict],
    recursion_depth: int = 0,
    openrouter_service: OpenRouterService | None = None,
) -> ModelAnswer:
    try:
        logger.info(
            "Запрос к модели: message_count=%s, глубина рекурсии %s",
            len(messages),
            recursion_depth,
        )

        if recursion_depth > MAXIMUM_RECURSION_ANSWER_DEPTH:
            logger.error("Recursion depth exceeded")
            return ModelAnswer(None)

        if update.effective_user is None:
            logger.error("User is None")
            return ModelAnswer(None)

        if openrouter_service is None:
            logger.error("OpenRouter service не передан в get_model_answer")
            return ModelAnswer("Произошла ошибка конфигурации media-сервиса.")

        context_tokens = 0
        completion_tokens = 0
        additional_system_messages: list[dict] = []
        model_name = await get_user_model(update.effective_user.id)
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
                "Результат разбора tool call: name=%s, argument_keys=%s",
                function_call_name,
                sorted(function_args_dict),
            )

            registry = get_tool_registry()
            if function_call_name and registry.has(function_call_name):
                tool_ctx = ToolExecutionContext(
                    update=update,
                    context=context,
                    messages=messages,
                    model_name=model_name,
                    recursion_depth=recursion_depth,
                    openrouter_service=openrouter_service,
                    context_tokens=context_tokens,
                    completion_tokens=completion_tokens,
                    additional_system_messages=additional_system_messages,
                )
                handler = registry.get(function_call_name)
                assert handler is not None
                try:
                    result = await handler(tool_ctx, function_args_dict)
                except Exception as e:
                    logger.error(
                        "Ошибка в handler '%s': type=%s",
                        function_call_name,
                        type(e).__name__,
                    )
                    return ModelAnswer("Произошла ошибка при обработке запроса.")

                context_tokens = result.ctx_token
                completion_tokens = result.completion_token
                additional_system_messages = list(result.additional_system_messages)

                if result.recurse:
                    inner_answer = await get_model_answer(
                        update,
                        context,
                        messages,
                        recursion_depth + 1,
                        openrouter_service,
                    )
                    context_tokens += inner_answer.ctx_token
                    completion_tokens += inner_answer.completion_token
                    return ModelAnswer(
                        inner_answer.bot_reply,
                        additional_system_messages + inner_answer.additional_system_messages,
                        context_tokens,
                        completion_tokens,
                    )
                return ModelAnswer(
                    result.bot_reply,
                    additional_system_messages,
                    context_tokens,
                    completion_tokens,
                )

            if function_call_name:
                logger.warning(f"Неизвестный tool call: {function_call_name}")

        # Если функция не вызвалась, возвращаем обычный текстовый ответ:
        if isinstance(response, Response):
            bot_reply = (getattr(response, "output_text", None) or "").strip()
            if bot_reply == "":
                logger.warning(
                    "Пустой output_text без обработанного tool call: item_count=%s",
                    len(list(response.output or [])),
                )
                bot_reply = "Произошла ошибка при обработке запроса."
            else:
                logger.info("Текстовый ответ модели успешно извлечён: length=%s", len(bot_reply))
        else:
            logger.error(
                "Неожиданный тип ответа от get_simple_answer: %s",
                type(response).__name__,
            )
            bot_reply = "Произошла ошибка при обработке запроса."

        return ModelAnswer(bot_reply, additional_system_messages, context_tokens, completion_tokens)

    except Exception as e:
        logger.error("Ошибка при обращении к OpenAI API: type=%s", type(e).__name__)
        return ModelAnswer("Произошла ошибка при обработке запроса.")


def _prepare_openai_value(value):
    """Сериализует бинарные изображения только на границе OpenAI SDK."""

    if isinstance(value, GeneratedImage):
        encoded = base64.b64encode(value.content).decode("ascii")
        return f"data:{value.mime_type};base64,{encoded}"
    if isinstance(value, dict):
        if value.get("type") == "input_image" and isinstance(value.get("image"), GeneratedImage):
            image = value["image"]
            encoded = base64.b64encode(image.content).decode("ascii")
            prepared = {key: item for key, item in value.items() if key != "image"}
            prepared["image_url"] = f"data:{image.mime_type};base64,{encoded}"
            return prepared
        return {key: _prepare_openai_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_prepare_openai_value(item) for item in value]
    return value


async def get_simple_answer(messages, model_name) -> Response:
    prepared_messages = _prepare_openai_value(messages)
    partial_param = partial(
        openai_client.responses.create,
        model=model_name,
        input=prepared_messages,
        max_output_tokens=16384,
        tools=TOOLS_SCHEMA,
        text={"verbosity": "low"},
        reasoning={"effort": "medium"},
    )  # type: ignore

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, partial_param)
    return response


def transcribe_audio(audio_filename):
    try:
        transcription = openai_client.audio.transcriptions.create(
            model=get_voice_recognition_model(),
            file=open(audio_filename, 'rb')
        )
        recognized_text = transcription.text
    except Exception as e:
        logger.error("Ошибка при распознавании речи: " + str(e))
        return
    return recognized_text
