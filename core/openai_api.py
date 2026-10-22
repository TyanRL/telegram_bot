import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, cast

from openai import APIError, NotFoundError, OpenAI
from openai.types.responses import Response
from telegram import Update
from telegram.ext import ContextTypes

from core.common_types import (
    GeneratedImage,
    ModelAnswer,
    PendingQuestionRequest,
    PendingToolCall,
    QuestionAnswer,
    QuestionAnswerKind,
    QuestionAnswerMode,
    ToolQuestionCheck,
    ToolQuestionPause,
    ToolResult,
)
from core.config import settings
from core.question_state import QuestionContinuationContext
from core.state_and_commands import (
    get_user_model,
)

# Импорт handler-модулей для регистрации в реестре
from core.tool_handlers import media, model, notes, questions, weather  # noqa: F401
from core.tool_handlers.media import run_video_request
from core.tool_registry import (
    TOOLS_SCHEMA,
    ImageToolTurnState,
    ToolExecutionContext,
    VideoToolTurnState,
    get_tool_registry,
)
from core.video_requests import VideoRequestDraft, parse_duration_answer, validate_duration
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


_IMAGE_TOOL_NAMES = {"generate_image", "generate_image_from_image"}
_VIDEO_TOOL_NAMES = {"generate_video"}
_RETRY_FORBIDDEN_INSTRUCTION = (
    "Не вызывай платные инструменты изображений повторно в этом пользовательском ходе; "
    "сообщи пользователю о результате и дождись нового запроса."
)
_VIDEO_RETRY_FORBIDDEN_INSTRUCTION = (
    "Не вызывай платный инструмент видео повторно в этом пользовательском ходе; "
    "сообщи пользователю о результате и дождись нового запроса."
)

# Безопасные сообщения при недоступном продолжении работы.
_CONTINUATION_UNAVAILABLE_TEXT = (
    "Не удалось продолжить задачу после вашего ответа. "
    "Платный запрос не выполнялся и не будет повторён."
)
_ANSWER_STORAGE_UNAVAILABLE_TEXT = (
    "Не удалось задать уточняющий вопрос: продолжение после вашего ответа "
    "недоступно. Платный запрос не выполнялся."
)


def _failed_tool_result(
    message: str,
    *,
    retry_forbidden: bool = False,
    blocked: bool = False,
) -> ToolResult:
    """Создаёт безопасный результат ошибки без внутренних деталей приложения."""

    output: dict[str, Any] = {"ok": False, "error": message}
    meta: dict[str, Any] = {}
    if retry_forbidden:
        output["retry_forbidden"] = True
        output["instruction"] = _RETRY_FORBIDDEN_INSTRUCTION
        meta["retry_forbidden"] = True
    if blocked:
        output["blocked_after_image_failure"] = True
        meta["blocked_after_image_failure"] = True
    return ToolResult(output, meta=meta)


def _mark_image_retry_forbidden(
    image_state: ImageToolTurnState,
    tool_round: int,
) -> None:
    image_state.retry_forbidden = True
    if image_state.failure_round is None:
        image_state.failure_round = tool_round


_ERROR_MESSAGE_LOG_LIMIT = 2000


def _describe_api_error(error: Exception, model: str | None) -> str:
    """Собирает диагностическую строку об ошибке для логов.

    Достаёт из исключений OpenAI SDK статус, код и параметр, чтобы
    причина сбоя (например, 404 model_not_found) была видна в логах.
    В SDK v2 атрибуты code/param у исключений не заполняются, поэтому
    значения дополнительно извлекаются из body ответа.
    """

    parts = [f"type={type(error).__name__}", f"model={model or 'unknown'}"]

    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        parts.append(f"status_code={status_code}")

    error_code = getattr(error, "code", None)
    error_param = getattr(error, "param", None)
    error_body = getattr(error, "body", None)
    body_error = error_body.get("error") if isinstance(error_body, dict) else None
    if isinstance(body_error, dict):
        error_code = error_code or body_error.get("code")
        error_param = error_param or body_error.get("param")

    if error_code:
        parts.append(f"code={error_code}")
    if error_param:
        parts.append(f"param={error_param}")

    message = str(error).strip() or "<без описания>"
    if len(message) > _ERROR_MESSAGE_LOG_LIMIT:
        message = message[:_ERROR_MESSAGE_LOG_LIMIT] + "…<обрезано>"
    parts.append(f"message={message}")

    return ", ".join(parts)


async def _execute_tool(
    registry,
    function_call: _FunctionCall,
    tool_ctx: ToolExecutionContext,
) -> ToolResult:
    """Выполняет один function call и всегда возвращает модельный результат."""

    is_image_tool = function_call.name in _IMAGE_TOOL_NAMES
    image_state = tool_ctx.image_tool_state
    if is_image_tool:
        call_id = function_call.call_id
        tool_round = tool_ctx.recursion_depth
        duplicate_call = call_id in image_state.attempted_image_call_ids
        failed_in_prior_round = image_state.retry_forbidden and (
            image_state.failure_round is None
            or tool_round > image_state.failure_round
        )
        if duplicate_call or failed_in_prior_round:
            if duplicate_call:
                _mark_image_retry_forbidden(image_state, tool_round)
            return _failed_tool_result(
                "Платная операция с изображением уже выполнялась или завершилась ошибкой "
                "в этом ходе. Повторный вызов заблокирован; передайте пользователю "
                "понятное сообщение.",
                retry_forbidden=True,
                blocked=True,
            )
        image_state.attempted_image_call_ids.add(call_id)
        image_state.image_call_attempted = False

    is_video_tool = function_call.name in _VIDEO_TOOL_NAMES
    video_state = tool_ctx.video_tool_state
    if is_video_tool:
        duplicate_call = function_call.call_id in video_state.attempted_video_call_ids
        repeated_operation = video_state.is_known_operation(
            getattr(tool_ctx, "video_operation_id", None)
        )
        if duplicate_call or repeated_operation or video_state.retry_forbidden:
            logger.warning(
                "Повторный платный вызов видео заблокирован: call_id=%s, "
                "retry_forbidden=%s, repeated_operation=%s",
                function_call.call_id,
                video_state.retry_forbidden,
                repeated_operation,
            )
            output: dict[str, Any] = {
                "ok": False,
                "error": (
                    "Платная генерация видео уже выполнялась или была отклонена "
                    "в этом ходе. Повторный вызов заблокирован: дождитесь нового "
                    "запроса пользователя."
                ),
                "retry_forbidden": True,
                "instruction": _VIDEO_RETRY_FORBIDDEN_INSTRUCTION,
            }
            video_state.retry_forbidden = True
            return ToolResult(output, meta={"retry_forbidden": True})
        video_state.attempted_video_call_ids.add(function_call.call_id)

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
        failed_result = _failed_tool_result(
            f"Не удалось выполнить инструмент '{function_call.name}'.",
            retry_forbidden=is_image_tool,
        )
        if is_image_tool:
            _mark_image_retry_forbidden(image_state, tool_ctx.recursion_depth)
        return failed_result

    if not isinstance(result, ToolResult):
        logger.error(
            "Handler '%s' вернул неподдерживаемый тип результата: %s",
            function_call.name,
            type(result).__name__,
        )
        failed_result = _failed_tool_result(
            f"Инструмент '{function_call.name}' вернул некорректный результат.",
            retry_forbidden=is_image_tool,
        )
        if is_image_tool:
            _mark_image_retry_forbidden(image_state, tool_ctx.recursion_depth)
        return failed_result

    if is_image_tool:
        retry_forbidden = result.meta.get("retry_forbidden") is True or (
            isinstance(result.output, dict)
            and result.output.get("retry_forbidden") is True
        )
        returned_error = (
            isinstance(result.output, dict) and result.output.get("ok") is False
        )
        if retry_forbidden or (image_state.image_call_attempted and returned_error):
            _mark_image_retry_forbidden(image_state, tool_ctx.recursion_depth)
        image_state.image_call_attempted = False

    if is_video_tool:
        # Отклонённая заявка (например, превышение предела референсов) блокирует
        # повторный платный запуск в этом же ходе: модель не должна повторить
        # генерацию с незаметно урезанным набором референсов.
        video_retry_forbidden = result.meta.get("retry_forbidden") is True or (
            isinstance(result.output, dict)
            and result.output.get("retry_forbidden") is True
        )
        if video_retry_forbidden:
            video_state.retry_forbidden = True
        operation_id = result.meta.get("operation_id")
        delivered = isinstance(result.output, dict) and result.output.get("ok") is True
        if delivered:
            video_state.mark_completed(
                operation_id if isinstance(operation_id, str) else None
            )
        else:
            video_state.mark_failed(
                operation_id if isinstance(operation_id, str) else None
            )

    return result


@dataclass
class _ToolCycleState:
    """Общее состояние цикла инструментов.

    Состояние переживает приостановку до ответа пользователя: исходный
    пользовательский текст, лимит раундов, токены и защитные состояния
    инструментов сохраняются между обновлениями.
    """

    update: Update
    context: ContextTypes.DEFAULT_TYPE
    openrouter_service: OpenRouterService
    model_name: str
    original_user_text: str | None = None
    recursion_depth: int = 0
    context_tokens: int = 0
    completion_tokens: int = 0
    tool_round: int = 0
    image_tool_state: ImageToolTurnState = field(default_factory=ImageToolTurnState)
    video_tool_state: VideoToolTurnState = field(default_factory=VideoToolTurnState)
    video_request: VideoRequestDraft | None = None
    video_operation_id: str | None = None
    video_answer: QuestionAnswer | None = None
    additional_system_messages: list[dict] = field(default_factory=list)


def _build_tool_context(state: _ToolCycleState) -> ToolExecutionContext:
    """Собирает контекст выполнения инструмента из состояния цикла."""

    return ToolExecutionContext(
        update=state.update,
        context=state.context,
        model_name=state.model_name,
        recursion_depth=state.recursion_depth + state.tool_round,
        openrouter_service=state.openrouter_service,
        context_tokens=state.context_tokens,
        completion_tokens=state.completion_tokens,
        user_text=state.original_user_text,
        image_tool_state=state.image_tool_state,
        video_tool_state=state.video_tool_state,
        video_request=state.video_request,
        video_operation_id=state.video_operation_id,
        video_answer=state.video_answer,
    )


def _pending_tool_calls(function_calls: list[_FunctionCall]) -> tuple[PendingToolCall, ...]:
    """Преобразует вызовы ответа в сохраняемые вызовы контекста продолжения."""

    return tuple(
        PendingToolCall(
            call_id=call.call_id,
            name=call.name,
            arguments=dict(call.arguments),
        )
        for call in function_calls
    )


def _continuation_is_available(response_id: str | None) -> bool:
    """Проверяет, доступно ли интерактивное продолжение ответа.

    Продолжение через `previous_response_id` возможно только при сохранённых
    ответах OpenAI и при наличии идентификатора ответа.
    """

    if not settings.openai.store_responses:
        return False
    return isinstance(response_id, str) and bool(response_id)


def _build_continuation_context(
    state: _ToolCycleState,
    *,
    response_id: str,
    function_calls: list[_FunctionCall],
    completed_outputs: list[dict[str, str]],
    current_call_index: int,
    question: PendingQuestionRequest,
    video_request: VideoRequestDraft | None = None,
    video_operation_id: str | None = None,
) -> QuestionContinuationContext:
    """Собирает серверный контекст продолжения прерванного цикла."""

    return QuestionContinuationContext(
        response_id=response_id,
        tool_calls=_pending_tool_calls(function_calls),
        completed_outputs=tuple(completed_outputs),
        current_call_index=current_call_index,
        model_name=state.model_name,
        original_user_text=state.original_user_text,
        context_tokens=state.context_tokens,
        completion_tokens=state.completion_tokens,
        tool_round=state.tool_round,
        image_tool_state=state.image_tool_state,
        video_tool_state=state.video_tool_state,
        video_request=video_request,
        video_operation_id=video_operation_id,
        answer_mode=question.answer_mode,
        validator=question.validator,
    )


async def _find_pending_question(
    state: _ToolCycleState,
    function_calls: list[_FunctionCall],
    registry: Any,
) -> tuple[ToolQuestionCheck, int] | None:
    """Ищет в пакете вызовов уточнение, требующее ответа пользователя.

    Порядок вызовов в ответе модели не важен: уточнение находится до
    выполнения любого вызова, поэтому зависимые платные операции не
    выполняются раньше вопроса.
    """

    get_question_check = getattr(registry, "get_question_check", None)
    if not callable(get_question_check):
        return None
    lookup = cast(Callable[[str], Any], get_question_check)

    for index, function_call in enumerate(function_calls):
        check_handler = lookup(function_call.name)
        if not callable(check_handler):
            continue
        try:
            check = await cast(Any, check_handler)(
                _build_tool_context(state),
                function_call.arguments,
            )
        except Exception as error:
            logger.warning(
                "Предварительная проверка инструмента '%s' не удалась: type=%s",
                function_call.name,
                type(error).__name__,
            )
            continue
        if check is not None:
            return check, index
    return None


async def _execute_tool_batch(
    state: _ToolCycleState,
    registry: Any,
    function_calls: list[_FunctionCall],
    start_index: int = 0,
    end_index: int | None = None,
) -> tuple[list[dict[str, str]], ToolQuestionPause | None, int]:
    """Выполняет пакет вызовов инструментов.

    Возвращает пару (накопленные результаты, приостановка, позиция
    приостановки). Приостановка означает, что вызов на указанной позиции
    требует ответа пользователя: уже выполненные вызовы не повторяются, а
    их результаты сохраняются. Вызовы начиная с `end_index` не выполняются
    вовсе: они зависят от ответа пользователя.
    """

    tool_outputs: list[dict[str, str]] = []
    last_index = len(function_calls) if end_index is None else min(end_index, len(function_calls))
    for index in range(start_index, last_index):
        function_call = function_calls[index]
        logger.info(
            "Обработка tool call: name=%s, call_id=%s, argument_keys=%s",
            function_call.name,
            function_call.call_id,
            sorted(function_call.arguments),
        )
        result = await _execute_tool(registry, function_call, _build_tool_context(state))
        state.additional_system_messages.extend(result.additional_system_messages)
        if result.pause is not None:
            return tool_outputs, result.pause, index
        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": function_call.call_id,
                "output": _serialize_tool_output(result.output),
            }
        )
    return tool_outputs, None, last_index


async def _run_tool_cycle(
    state: _ToolCycleState,
    response: Response,
) -> ModelAnswer:
    """Выполняет цикл инструментов, поддерживая приостановку до ответа.

    Возвращает обычный ответ модели либо `ModelAnswer` с явным ожиданием
    пользователя: в этом случае очередной запрос к модели не выполняется.
    """

    registry = get_tool_registry()

    for tool_round in range(MAXIMUM_TOOL_ROUNDS + 1):
        if not isinstance(response, Response):
            logger.error(
                "Неожиданный тип ответа от get_simple_answer: %s",
                type(response).__name__,
            )
            return ModelAnswer(
                "Произошла ошибка при обработке запроса.",
                state.additional_system_messages,
                state.context_tokens,
                state.completion_tokens,
            )

        input_tokens, output_tokens = _response_usage(response)
        state.context_tokens += input_tokens
        state.completion_tokens += output_tokens
        state.tool_round = tool_round

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
                state.additional_system_messages,
                state.context_tokens,
                state.completion_tokens,
            )

        if tool_round >= MAXIMUM_TOOL_ROUNDS:
            logger.error("Tool cycle limit exceeded: rounds=%s", MAXIMUM_TOOL_ROUNDS)
            return ModelAnswer(
                "Не удалось завершить обработку запроса: превышен лимит вызовов инструментов.",
                state.additional_system_messages,
                state.context_tokens,
                state.completion_tokens,
            )

        response_id = getattr(response, "id", None)
        if not isinstance(response_id, str) or not response_id:
            logger.error("Ответ с function call не содержит response.id")
            return ModelAnswer(
                "Произошла ошибка при продолжении обработки запроса.",
                state.additional_system_messages,
                state.context_tokens,
                state.completion_tokens,
            )

        # Уточнение ищется до выполнения пакета: зависимые платные операции
        # не выполняются раньше вопроса пользователю.
        pending = await _find_pending_question(state, function_calls, registry)
        if pending is not None:
            check, index = pending
            if not _continuation_is_available(response_id):
                logger.error(
                    "Продолжение недоступно: store_responses=%s, response_id=%s",
                    settings.openai.store_responses,
                    response_id,
                )
                return ModelAnswer(
                    _ANSWER_STORAGE_UNAVAILABLE_TEXT,
                    state.additional_system_messages,
                    state.context_tokens,
                    state.completion_tokens,
                )
            # Пакет переупорядочивается: вопрос становится первым вызовом,
            # поэтому зависимые платные операции ждут ответа пользователя
            # независимо от порядка вызовов в ответе модели.
            ordered_calls = [function_calls[index]] + [
                call for position, call in enumerate(function_calls) if position != index
            ]
            continuation = _build_continuation_context(
                state,
                response_id=response_id,
                function_calls=ordered_calls,
                completed_outputs=[],
                current_call_index=0,
                question=check.request,
                video_request=check.video_request,
                video_operation_id=check.video_operation_id,
            )
            logger.info(
                "Приостановка цикла инструментов: вопрос=%s, вариантов=%s",
                function_calls[index].name,
                len(check.request.options),
            )
            return ModelAnswer(
                None,
                state.additional_system_messages,
                state.context_tokens,
                state.completion_tokens,
                pending_question=check.request,
                pending_continuation=continuation,
            )

        tool_outputs, pause, index = await _execute_tool_batch(
            state, registry, function_calls, 0
        )
        if pause is not None:
            # Приостановка может прийти из самого инструмента: проверка выше
            # не нашла вопроса, но обработчик запросил его явно.
            if not _continuation_is_available(response_id):
                logger.error(
                    "Продолжение недоступно после приостановки инструмента: "
                    "store_responses=%s",
                    settings.openai.store_responses,
                )
                return ModelAnswer(
                    _ANSWER_STORAGE_UNAVAILABLE_TEXT,
                    state.additional_system_messages,
                    state.context_tokens,
                    state.completion_tokens,
                )
            continuation = _build_continuation_context(
                state,
                response_id=response_id,
                function_calls=function_calls,
                completed_outputs=tool_outputs,
                current_call_index=index,
                question=pause.request,
                video_request=pause.video_request,
            )
            return ModelAnswer(
                None,
                state.additional_system_messages,
                state.context_tokens,
                state.completion_tokens,
                pending_question=pause.request,
                pending_continuation=continuation,
            )

        # previous_response_id сохраняет reasoning и исходные function call
        # на стороне Responses API; в input передаются только их результаты.
        response = await get_simple_answer(
            tool_outputs,
            state.model_name,
            previous_response_id=response_id,
        )

    return ModelAnswer(
        "Не удалось завершить обработку запроса.",
        state.additional_system_messages,
        state.context_tokens,
        state.completion_tokens,
    )


async def get_model_answer(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    messages: list[dict],
    recursion_depth: int = 0,
    openrouter_service: OpenRouterService | None = None,
    user_text: str | None = None,
) -> ModelAnswer:
    try:
        model_name: str | None = None

        if recursion_depth > MAXIMUM_TOOL_ROUNDS:
            logger.error("Tool cycle limit exceeded before request")
            return ModelAnswer("Не удалось завершить обработку запроса.")

        if update.effective_user is None:
            logger.error("User is None")
            return ModelAnswer(None)

        if openrouter_service is None:
            logger.error("OpenRouter service не передан в get_model_answer")
            return ModelAnswer("Произошла ошибка конфигурации media-сервиса.")

        model_name = await get_user_model(update.effective_user.id)
        logger.info(
            "Запрос к модели: model=%s, message_count=%s, начальная глубина tool cycle %s",
            model_name,
            len(messages),
            recursion_depth,
        )
        response = await get_simple_answer(messages, model_name)

        state = _ToolCycleState(
            update=update,
            context=context,
            openrouter_service=openrouter_service,
            model_name=model_name,
            original_user_text=user_text,
            recursion_depth=recursion_depth,
        )
        return await _run_tool_cycle(state, response)

    except Exception as e:
        logger.error(
            "Ошибка при обращении к OpenAI API: %s",
            _describe_api_error(e, model_name),
            exc_info=not isinstance(e, APIError),
        )
        return ModelAnswer("Произошла ошибка при обработке запроса.")


async def verify_default_model_available() -> bool:
    """Проверяет доступность модели по умолчанию при старте бота.

    Возвращает False только при явной недоступности модели (404), когда
    каждый запрос гарантированно завершится ошибкой. Временные сбои
    сети не блокируют запуск.
    """

    model_name = settings.openai.default_model
    try:
        await asyncio.to_thread(openai_client.models.retrieve, model_name)
    except NotFoundError:
        logger.error(
            "Модель по умолчанию недоступна для OpenAI API: model=%s. "
            "Проверьте openai.default_model в config.yaml: модель не существует "
            "или недоступна для текущего API-ключа.",
            model_name,
        )
        return False
    except APIError as error:
        logger.warning(
            "Не удалось проверить модель по умолчанию, запуск продолжается: %s",
            _describe_api_error(error, model_name),
        )
        return True
    except Exception as error:
        logger.warning(
            "Не удалось проверить модель по умолчанию, запуск продолжается: "
            "type=%s, message=%s",
            type(error).__name__,
            str(error)[:500],
        )
        return True

    logger.info("Модель по умолчанию доступна: model=%s", model_name)
    return True


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
    logger.info(
        "Responses API запрос: model=%s, input_type=%s, input_items=%s, previous_response_id=%s",
        model_name,
        type(prepared_messages).__name__,
        len(prepared_messages) if isinstance(prepared_messages, list) else "-",
        previous_response_id,
    )
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


def _plain_answer_payload(
    question: PendingQuestionRequest,
    answer: QuestionAnswer,
) -> dict[str, Any]:
    """Строит окончательный результат обычного вопроса для модели.

    Результат структурированный: модель получает выбранный вариант с его
    идентификатором и подписью либо собственный текст пользователя.
    """

    if answer.kind is QuestionAnswerKind.OPTION and answer.option_id is not None:
        label = answer.text
        for option in question.options:
            if option.option_id == answer.option_id:
                label = option.text
                break
        return {
            "ok": True,
            "kind": "option",
            "option_id": answer.option_id,
            "label": label,
            "answer": answer.text,
        }
    return {
        "ok": True,
        "kind": "custom",
        "option_id": None,
        "label": None,
        "answer": answer.text,
    }


async def _run_video_continuation(
    state: _ToolCycleState,
    continuation: QuestionContinuationContext,
    answer: QuestionAnswer,
) -> ToolResult:
    """Выполняет сохранённую видеозаявку с проверенной длительностью.

    Заявка выполняется ровно один раз: промпт, референсы и снимок профиля
    остаются прежними, меняется только длительность.
    """

    draft = continuation.video_request
    if draft is None:
        return ToolResult(
            output={
                "ok": False,
                "error": "Сохранённая видеозаявка не найдена.",
            },
        )

    profile = draft.profile
    duration: int | None = None
    if isinstance(answer.payload, int) and not isinstance(answer.payload, bool):
        # Готовый вариант кнопки хранит проверенное значение на сервере.
        try:
            duration = validate_duration(answer.payload, profile)
        except ValueError:
            duration = None
    if duration is None:
        duration = parse_duration_answer(answer.text, profile)
    if duration is None:
        allowed = ", ".join(str(value) for value in profile.allowed_durations)
        return ToolResult(
            output={
                "ok": False,
                "error": f"Длительность {answer.text!r} не поддерживается. Допустимые значения: {allowed}.",
            },
        )

    tool_ctx = _build_tool_context(state)
    tool_ctx.video_request = draft
    tool_ctx.video_operation_id = draft.operation_id
    tool_ctx.video_answer = answer
    return await run_video_request(tool_ctx, draft.with_duration(duration))


def _mark_video_call_attempted(
    state: _ToolCycleState,
    call_id: str,
) -> None:
    """Отмечает платный вызов видео как совершённый.

    Защита от повторного платного запуска сохраняется между обновлениями:
    ответ пользователя не обнуляет её.
    """

    state.video_tool_state.attempted_video_call_ids.add(call_id)


async def resume_model_answer(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    continuation: QuestionContinuationContext,
    answer: QuestionAnswer,
    openrouter_service: OpenRouterService | None = None,
    question: PendingQuestionRequest | None = None,
) -> ModelAnswer:
    """Продолжает прерванный цикл инструментов после ответа пользователя.

    Возобновляется тот же ответ OpenAI через `previous_response_id`. Каждый
    исходный вызов получает ровно один окончательный `function_call_output`:
    маркер ожидания не подменяет результат человеческого ответа.
    """

    try:
        if not _continuation_is_available(continuation.response_id):
            logger.error(
                "Продолжение недоступно: store_responses=%s, response_id=%s",
                settings.openai.store_responses,
                continuation.response_id,
            )
            return ModelAnswer(_CONTINUATION_UNAVAILABLE_TEXT)

        if openrouter_service is None:
            logger.error("OpenRouter service не передан в resume_model_answer")
            return ModelAnswer("Произошла ошибка конфигурации media-сервиса.")

        model_name = continuation.model_name or settings.openai.default_model
        state = _ToolCycleState(
            update=update,
            context=context,
            openrouter_service=openrouter_service,
            model_name=model_name,
            original_user_text=continuation.original_user_text,
            recursion_depth=0,
            context_tokens=continuation.context_tokens,
            completion_tokens=continuation.completion_tokens,
            tool_round=continuation.tool_round,
            image_tool_state=continuation.image_tool_state,
            video_tool_state=continuation.video_tool_state,
            video_request=continuation.video_request,
            video_operation_id=continuation.video_operation_id,
            video_answer=answer,
        )

        registry = get_tool_registry()
        pending_calls = list(continuation.tool_calls)
        index = continuation.current_call_index
        if index < 0 or index >= len(pending_calls):
            logger.error(
                "Некорректная позиция продолжения: index=%s, calls=%s",
                index,
                len(pending_calls),
            )
            return ModelAnswer(_CONTINUATION_UNAVAILABLE_TEXT)

        current_call = pending_calls[index]
        tool_ctx = _build_tool_context(state)

        if continuation.answer_mode is QuestionAnswerMode.VIDEO_REQUEST:
            _mark_video_call_attempted(state, current_call.call_id)
            result = await _run_video_continuation(state, continuation, answer)
        else:
            request = question
            if request is None:
                request = PendingQuestionRequest(
                    text="Уточняющий вопрос",
                    options=(),
                    answer_mode=QuestionAnswerMode.PLAIN,
                )
            result = ToolResult(
                output=_plain_answer_payload(request, answer),
            )

        logger.info(
            "Продолжение вызова '%s' (call_id=%s): результат получен",
            current_call.name,
            current_call.call_id,
        )

        tool_outputs: list[dict[str, str]] = [
            {
                "type": "function_call_output",
                "call_id": item["call_id"],
                "output": item["output"],
            }
            for item in continuation.completed_outputs
        ]
        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": current_call.call_id,
                "output": _serialize_tool_output(result.output),
            }
        )

        # Оставшиеся вызовы пакета выполняются по порядку.
        remaining_calls = [
            _FunctionCall(
                call_id=call.call_id,
                name=call.name,
                arguments=dict(call.arguments),
            )
            for call in pending_calls[index + 1:]
        ]
        if remaining_calls:
            # Уточнение среди оставшихся вызовов проверяется до их
            # выполнения: зависимые платные операции снова ждут ответа.
            next_pending = await _find_pending_question(state, remaining_calls, registry)
            if next_pending is not None:
                next_check, next_index = next_pending
                response_id = continuation.response_id
                if not _continuation_is_available(response_id):
                    return ModelAnswer(
                        _ANSWER_STORAGE_UNAVAILABLE_TEXT,
                        state.additional_system_messages,
                        state.context_tokens,
                        state.completion_tokens,
                    )
                reordered = [remaining_calls[next_index]] + [
                    call for position, call in enumerate(remaining_calls) if position != next_index
                ]
                merged_calls = pending_calls[: index + 1] + [
                    PendingToolCall(
                        call_id=call.call_id,
                        name=call.name,
                        arguments=dict(call.arguments),
                    )
                    for call in reordered
                ]
                next_continuation = _build_continuation_context(
                    state,
                    response_id=str(response_id),
                    function_calls=[
                        _FunctionCall(
                            call_id=call.call_id,
                            name=call.name,
                            arguments=dict(call.arguments),
                        )
                        for call in merged_calls
                    ],
                    completed_outputs=tool_outputs,
                    current_call_index=index + 1,
                    question=next_check.request,
                    video_request=next_check.video_request,
                    video_operation_id=next_check.video_operation_id,
                )
                return ModelAnswer(
                    None,
                    state.additional_system_messages,
                    state.context_tokens,
                    state.completion_tokens,
                    pending_question=next_check.request,
                    pending_continuation=next_continuation,
                )

            extra_outputs, pause, pause_index = await _execute_tool_batch(
                state,
                registry,
                remaining_calls,
                0,
            )
            tool_outputs.extend(extra_outputs)
            if pause is not None:
                response_id = continuation.response_id
                if not _continuation_is_available(response_id):
                    return ModelAnswer(
                        _ANSWER_STORAGE_UNAVAILABLE_TEXT,
                        state.additional_system_messages,
                        state.context_tokens,
                        state.completion_tokens,
                    )
                merged_calls = pending_calls[: index + 1] + [
                    PendingToolCall(
                        call_id=call.call_id,
                        name=call.name,
                        arguments=dict(call.arguments),
                    )
                    for call in pending_calls[index + 1:]
                ]
                next_continuation = _build_continuation_context(
                    state,
                    response_id=str(response_id),
                    function_calls=[
                        _FunctionCall(
                            call_id=call.call_id,
                            name=call.name,
                            arguments=dict(call.arguments),
                        )
                        for call in merged_calls
                    ],
                    completed_outputs=tool_outputs,
                    current_call_index=index + 1 + pause_index,
                    question=pause.request,
                    video_request=pause.video_request,
                )
                return ModelAnswer(
                    None,
                    state.additional_system_messages,
                    state.context_tokens,
                    state.completion_tokens,
                    pending_question=pause.request,
                    pending_continuation=next_continuation,
                )

        # previous_response_id сохраняет reasoning и исходные function call
        # на стороне Responses API; в input передаются только их результаты.
        response = await get_simple_answer(
            tool_outputs,
            state.model_name,
            previous_response_id=continuation.response_id,
        )
        return await _run_tool_cycle(state, response)

    except Exception as error:
        logger.error(
            "Ошибка при продолжении работы после ответа: %s",
            _describe_api_error(error, continuation.model_name),
            exc_info=not isinstance(error, APIError),
        )
        return ModelAnswer(_CONTINUATION_UNAVAILABLE_TEXT)


def transcribe_audio(audio_filename):
    try:
        transcription = openai_client.audio.transcriptions.create(
            model=settings.openai.speech_model,
            file=open(audio_filename, 'rb')
        )
        recognized_text = transcription.text
    except Exception as e:
        logger.error("Ошибка при распознавании речи: " + str(e))
        return
    return recognized_text
