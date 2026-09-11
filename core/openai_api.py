import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any

from openai import OpenAI
from openai.types.responses import Response
from telegram import Update
from telegram.ext import ContextTypes

from core.common_types import GeneratedImage, ModelAnswer, ToolResult
from core.config import settings
from core.state_and_commands import (
    get_user_model,
)

# Импорт handler-модулей для регистрации в реестре
from core.tool_handlers import media, model, notes, weather  # noqa: F401
from core.tool_registry import TOOLS_SCHEMA, ToolExecutionContext, get_tool_registry
from utils.openrouter_client import OpenRouterService

logger = logging.getLogger(__name__)

# Инициализация OpenAI
opena_ai_api_key = settings.secrets.openai_api_key
openai_client = OpenAI(api_key=opena_ai_api_key)

MAXIMUM_TOOL_ROUNDS = settings.application.max_tool_rounds
# Сохраняем старое имя для внешнего кода, который мог импортировать константу.
MAXIMUM_RECURSION_ANSWER_DEPTH = MAXIMUM_TOOL_ROUNDS


@dataclass(frozen=True, slots=True)
class _FunctionCall:
    """Минимальные данные function call, необходимые для continuation API."""

    call_id: str
    name: str
    arguments: dict[str, Any]


def _normalize_function_args(function_args):
    if isinstance(function_args, str):
        try:
            parsed_args = json.loads(function_args)
            if isinstance(parsed_args, dict):
                return parsed_args
            logger.warning(
                "Аргументы tool call после JSON-декодирования имеют тип %s, ожидается object",
                type(parsed_args).__name__,
            )
            return {}
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


def _extract_function_calls(response: Response) -> list[_FunctionCall]:
    output_items = list(response.output or [])
    logger.info(
        "Responses API output: items=%s",
        [getattr(item, "type", type(item).__name__) for item in output_items],
    )

    function_calls: list[_FunctionCall] = []
    for item in output_items:
        item_type = getattr(item, "type", None)

        if item_type in ("function_call", "custom_tool_call"):
            function_call_name = getattr(item, "name", None)
            function_args = getattr(item, "arguments", None)
            if function_args is None:
                function_args = getattr(item, "input", None)
            call_id = getattr(item, "call_id", None)
            logger.info(
                "Найден function tool call: name=%s, call_id=%s, args_type=%s",
                function_call_name,
                call_id,
                type(function_args).__name__,
            )

            if not isinstance(function_call_name, str) or not function_call_name:
                logger.warning("Пропущен function call без имени")
                continue
            if not isinstance(call_id, str) or not call_id:
                logger.error(
                    "Пропущен function call '%s' без call_id; продолжение невозможно",
                    function_call_name,
                )
                continue

            function_calls.append(
                _FunctionCall(
                    call_id=call_id,
                    name=function_call_name,
                    arguments=_normalize_function_args(function_args),
                )
            )

    if not function_calls:
        logger.info(
            "В ответе Responses API нет function call: item_count=%s",
            len(output_items),
        )
    return function_calls


def _response_usage(response: Response) -> tuple[int, int]:
    """Безопасно извлекает usage из ответа SDK."""

    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    return int(input_tokens), int(output_tokens)


def _serialize_tool_output(output: Any) -> str:
    """Преобразует доменный результат в строку для function_call_output."""

    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        logger.warning(
            "Не удалось сериализовать результат tool; используется строковое представление"
        )
        return str(output)


def _failed_tool_result(message: str) -> ToolResult:
    """Создаёт безопасный результат ошибки без внутренних деталей приложения."""

    return ToolResult({"ok": False, "error": message})


async def _execute_tool(
    registry,
    function_call: _FunctionCall,
    tool_ctx: ToolExecutionContext,
) -> ToolResult:
    """Выполняет один function call и всегда возвращает модельный результат."""

    handler = registry.get(function_call.name)
    if handler is None:
        logger.warning("Неизвестный tool call: %s", function_call.name)
        return _failed_tool_result(
            f"Инструмент '{function_call.name}' недоступен в текущей конфигурации."
        )

    try:
        result = await handler(tool_ctx, function_call.arguments)
    except Exception as error:
        logger.error(
            "Ошибка в handler '%s': type=%s",
            function_call.name,
            type(error).__name__,
            exc_info=True,
        )
        return _failed_tool_result(
            f"Не удалось выполнить инструмент '{function_call.name}'."
        )

    if not isinstance(result, ToolResult):
        logger.error(
            "Handler '%s' вернул неподдерживаемый тип результата: %s",
            function_call.name,
            type(result).__name__,
        )
        return _failed_tool_result(
            f"Инструмент '{function_call.name}' вернул некорректный результат."
        )

    return result


async def get_model_answer(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    messages: list[dict],
    recursion_depth: int = 0,
    openrouter_service: OpenRouterService | None = None,
) -> ModelAnswer:
    try:
        logger.info(
            "Запрос к модели: message_count=%s, начальная глубина tool cycle %s",
            len(messages),
            recursion_depth,
        )

        if recursion_depth > MAXIMUM_TOOL_ROUNDS:
            logger.error("Tool cycle limit exceeded before request")
            return ModelAnswer("Не удалось завершить обработку запроса.")

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
        registry = get_tool_registry()

        for tool_round in range(MAXIMUM_TOOL_ROUNDS + 1):
            if not isinstance(response, Response):
                logger.error(
                    "Неожиданный тип ответа от get_simple_answer: %s",
                    type(response).__name__,
                )
                return ModelAnswer(
                    "Произошла ошибка при обработке запроса.",
                    additional_system_messages,
                    context_tokens,
                    completion_tokens,
                )

            input_tokens, output_tokens = _response_usage(response)
            context_tokens += input_tokens
            completion_tokens += output_tokens

            function_calls = _extract_function_calls(response)
            if not function_calls:
                bot_reply = (getattr(response, "output_text", None) or "").strip()
                if bot_reply == "":
                    logger.warning(
                        "Пустой output_text без function call: item_count=%s",
                        len(list(response.output or [])),
                    )
                    bot_reply = "Произошла ошибка при обработке запроса."
                else:
                    logger.info(
                        "Текстовый ответ модели успешно извлечён: length=%s",
                        len(bot_reply),
                    )
                return ModelAnswer(
                    bot_reply,
                    additional_system_messages,
                    context_tokens,
                    completion_tokens,
                )

            if tool_round >= MAXIMUM_TOOL_ROUNDS:
                logger.error("Tool cycle limit exceeded: rounds=%s", MAXIMUM_TOOL_ROUNDS)
                return ModelAnswer(
                    "Не удалось завершить обработку запроса: превышен лимит вызовов инструментов.",
                    additional_system_messages,
                    context_tokens,
                    completion_tokens,
                )

            tool_outputs: list[dict[str, str]] = []
            for function_call in function_calls:
                logger.info(
                    "Обработка tool call: name=%s, call_id=%s, argument_keys=%s",
                    function_call.name,
                    function_call.call_id,
                    sorted(function_call.arguments),
                )
                tool_ctx = ToolExecutionContext(
                    update=update,
                    context=context,
                    model_name=model_name,
                    recursion_depth=recursion_depth + tool_round,
                    openrouter_service=openrouter_service,
                    context_tokens=context_tokens,
                    completion_tokens=completion_tokens,
                )
                result = await _execute_tool(registry, function_call, tool_ctx)
                additional_system_messages.extend(result.additional_system_messages)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": function_call.call_id,
                        "output": _serialize_tool_output(result.output),
                    }
                )

            response_id = getattr(response, "id", None)
            if not isinstance(response_id, str) or not response_id:
                logger.error("Ответ с function call не содержит response.id")
                return ModelAnswer(
                    "Произошла ошибка при продолжении обработки запроса.",
                    additional_system_messages,
                    context_tokens,
                    completion_tokens,
                )

            # previous_response_id сохраняет reasoning и исходные function call
            # на стороне Responses API; в input передаются только их результаты.
            response = await get_simple_answer(
                tool_outputs,
                model_name,
                previous_response_id=response_id,
            )

        return ModelAnswer(
            "Не удалось завершить обработку запроса.",
            additional_system_messages,
            context_tokens,
            completion_tokens,
        )

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


async def get_simple_answer(
    messages,
    model_name,
    *,
    previous_response_id: str | None = None,
) -> Response:
    prepared_messages = _prepare_openai_value(messages)
    request_kwargs: dict[str, Any] = {
        "model": model_name,
        "input": prepared_messages,
        "max_output_tokens": settings.openai.max_output_tokens,
        "tools": TOOLS_SCHEMA,
        "text": {"verbosity": settings.openai.verbosity},
        "reasoning": {"effort": settings.openai.reasoning_effort},
        # Без сохранённого response нельзя надёжно продолжить reasoning через
        # previous_response_id после выполнения function call.
        "store": settings.openai.store_responses,
    }
    if previous_response_id is not None:
        request_kwargs["previous_response_id"] = previous_response_id

    partial_param = partial(openai_client.responses.create, **request_kwargs)  # type: ignore

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, partial_param)
    return response
