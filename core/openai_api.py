import asyncio
from functools import partial
import json
import logging
import os


from openai import OpenAI
from openai.types.responses import Response
from telegram import Update
from telegram.ext import ContextTypes

from core.common_types import ModelAnswer
from core.state_and_commands import (
    get_user_model,
    get_voice_recognition_model,
)
from core.tool_registry import TOOLS_SCHEMA, get_tool_registry, ToolExecutionContext

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


async def get_model_answer(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    messages: list[dict],
    recursion_depth: int = 0,
) -> ModelAnswer:
    try:
        logger.info(f"Запрос к модели: {str(messages[-1])}, глубина рекурсии {recursion_depth}")

        if recursion_depth > MAXIMUM_RECURSION_ANSWER_DEPTH:
            logger.error("Recursion depth exceeded")
            return ModelAnswer(None)

        if update.effective_user is None:
            logger.error("User is None")
            return ModelAnswer(None)

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
                f"Результат разбора tool call: name={function_call_name}, args={function_args_dict!r}",
            )

            registry = get_tool_registry()
            if function_call_name and registry.has(function_call_name):
                tool_ctx = ToolExecutionContext(
                    update=update,
                    context=context,
                    messages=messages,
                    model_name=model_name,
                    recursion_depth=recursion_depth,
                    context_tokens=context_tokens,
                    completion_tokens=completion_tokens,
                    additional_system_messages=additional_system_messages,
                )
                handler = registry.get(function_call_name)
                assert handler is not None
                try:
                    result = await handler(tool_ctx, function_args_dict)
                except Exception as e:
                    logger.error(f"Ошибка в handler '{function_call_name}': {e}", exc_info=True)
                    return ModelAnswer("Произошла ошибка при обработке запроса.")

                context_tokens = result.ctx_token
                completion_tokens = result.completion_token
                additional_system_messages = list(result.additional_system_messages)

                if result.recurse:
                    inner_answer = await get_model_answer(
                        update, context, messages, recursion_depth + 1
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
                    f"Пустой output_text без обработанного tool call. raw_output={list(response.output or [])!r}",
                )
                bot_reply = "Произошла ошибка при обработке запроса."
            else:
                logger.info(f"Текстовый ответ модели успешно извлечён: {bot_reply[:500]!r}")
        else:
            logger.error(f"Неожиданный тип ответа от get_simple_answer: {type(response)}, значение={response!r}")
            bot_reply = "Произошла ошибка при обработке запроса."

        return ModelAnswer(bot_reply, additional_system_messages, context_tokens, completion_tokens)

    except Exception as e:
        logger.error(f"Ошибка при обращении к OpenAI API: {e}", exc_info=True)
        return ModelAnswer("Произошла ошибка при обработке запроса.")


async def get_simple_answer(messages, model_name) -> Response:
    partial_param = partial(
        openai_client.responses.create,
        model=model_name,
        input=messages,
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
