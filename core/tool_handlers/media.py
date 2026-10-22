import asyncio
import logging
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from uuid import uuid4

from PIL import Image

from core.common_types import (
    GeneratedImage,
    GeneratedVideo,
    ImageEditOptions,
    ImageGenerationOptions,
    PendingQuestionRequest,
    QuestionAnswerMode,
    ToolQuestionCheck,
    ToolResult,
    VideoGenerationOptions,
)
from core.config import settings
from core.generation_error_mapper import (
    GenerationFailureReason,
    map_generation_error,
    user_message_for_reason,
)
from core.image_observability import (
    log_image_generation_attempt,
    log_image_operation_summary,
)
from core.image_routing import (
    ImageRetryContext,
    ImageRoutingDecision,
    resolve_image_retry_context,
    resolve_image_route,
)
from core.state_and_commands import (
    add_user_image_session_history,
    append_user_image_to_session,
    animate_service_message,
    get_user_image_edit_session,
    reply_service_message,
    reply_service_text,
)
from core.tool_helpers import (
    build_image_dict_from_image,
    get_session_image_entries,
    resolve_selected_images,
    send_image_to_telegram,
    send_video_to_telegram,
)
from core.tool_registry import (
    ToolExecutionContext,
    register_tool,
    register_tool_question_check,
)
from core.user_questions import build_duration_question_options
from core.video_requests import (
    AmbiguousDurationError,
    AmbiguousImageOrderError,
    ReferenceLimitExceededError,
    UnsupportedDurationError,
    VideoModelProfile,
    VideoRequestDraft,
    build_duration_validator,
    build_video_request_draft,
    parse_explicit_duration,
    parse_user_image_indices,
    resolve_video_references,
)
from utils.openrouter_client import OpenRouterMediaError

logger = logging.getLogger(__name__)

_IMAGE_GENERATION_STATUS_TEXT = "Изображение генерируется, подождите"
_RETRY_FORBIDDEN_TEXT = (
    "Не вызывай платные инструменты изображений повторно в этом пользовательском ходе; "
    "сообщи пользователю о результате и дождись нового запроса."
)
_VIDEO_RETRY_FORBIDDEN_TEXT = (
    "Не вызывай платный инструмент видео повторно в этом пользовательском ходе: "
    "набор референсов нужно явно сократить по запросу пользователя. "
    "Сообщи пользователю о результате и дождись нового запроса."
)
_SAFE_IMAGE_ERROR_CODES = frozenset(
    {
        "ambiguous_provider_failure",
        "authentication",
        "cancelled",
        "configuration",
        "connection_error",
        "content_policy",
        "empty_prompt",
        "generation_failed",
        "insufficient_credits",
        "invalid_response",
        "invalid_source_image",
        "rate_limit",
        "sdk_unavailable",
        "spending_limit",
        "temporary_provider_error",
        "timeout",
        "unsupported_model",
        "unsupported_parameter",
        "unknown",
    }
)


def _sdk_aspect_ratio(value: object) -> str | None:
    """Приводит сохранённое числовое соотношение к enum SDK."""

    supported = {
        "21:9": 21 / 9,
        "2:1": 2 / 1,
        "16:9": 16 / 9,
        "3:2": 3 / 2,
        "7:5": 7 / 5,
        "4:3": 4 / 3,
        "5:4": 5 / 4,
        "1:1": 1.0,
        "4:5": 4 / 5,
        "3:4": 3 / 4,
        "5:7": 5 / 7,
        "2:3": 2 / 3,
        "9:16": 9 / 16,
        "1:2": 1 / 2,
        "9:21": 9 / 21,
    }

    if isinstance(value, str):
        return value if value in supported or value == "auto" else None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        return None

    return min(supported, key=lambda name: abs(supported[name] - float(value)))


def _answer(
    ctx: ToolExecutionContext,
    text: str | None,
    *,
    ok: bool | None = None,
    details: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    retry_forbidden: bool = False,
    retry_forbidden_text: str | None = None,
) -> ToolResult:
    output: dict[str, Any] = {
        "ok": text is not None if ok is None else ok,
        "message": text or "Операция не выполнена.",
    }
    if details:
        output.update(details)
    result_meta = dict(meta or {})
    if retry_forbidden:
        output["retry_forbidden"] = True
        output["instruction"] = retry_forbidden_text or _RETRY_FORBIDDEN_TEXT
        result_meta["retry_forbidden"] = True
    return ToolResult(
        output,
        ctx_token=getattr(ctx, "context_tokens", 0),
        completion_token=getattr(ctx, "completion_tokens", 0),
        meta=result_meta,
    )


async def _start_image_generation_status(
    update: Any,
) -> tuple[Any | None, asyncio.Task[None] | None]:
    try:
        status_message = await reply_service_message(
            update,
            _IMAGE_GENERATION_STATUS_TEXT,
        )
    except Exception:
        # Некоторые тестовые Update не содержат message.reply_text.
        logger.debug("Не удалось отправить статусное сообщение генерации изображения")
        return None, None

    animation_task = asyncio.create_task(
        animate_service_message(
            status_message,
            _IMAGE_GENERATION_STATUS_TEXT,
            interval=settings.telegram.service_animation_interval_seconds,
        )
    )
    return status_message, animation_task


async def _cancel_image_generation_animation(
    animation_task: asyncio.Task[None] | None,
) -> None:
    if animation_task is None:
        return
    animation_task.cancel()
    try:
        await animation_task
    except asyncio.CancelledError:
        pass
    except Exception as error:
        logger.debug("Анимация статуса завершилась с ошибкой: type=%s", type(error).__name__)


async def _delete_image_generation_status(status_message: Any | None) -> None:
    if status_message is None:
        return
    try:
        await status_message.delete()
    except Exception:
        pass


def _safe_error_code(error: BaseException) -> str:
    reason = getattr(error, "reason", None)
    if isinstance(reason, str) and reason in _SAFE_IMAGE_ERROR_CODES:
        return reason
    return type(error).__name__[:80]


def _known_cost(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


def _actual_model_quality(
    operation: str,
    actual_model: str | None,
    decision: ImageRoutingDecision,
    fallback_used: bool,
) -> str:
    if fallback_used:
        # OpenRouterService разрешает резерв только с Klein на старшую модель.
        return "pro"
    if operation == "generate":
        standard_model = settings.media.image_generation_fast_model
        pro_model = settings.media.image_generation_model
    else:
        standard_model = settings.media.image_edit_fast_model
        pro_model = settings.media.image_edit_model
    if (
        actual_model == standard_model
        and actual_model == pro_model
        and isinstance(actual_model, str)
        and "klein" in actual_model.casefold()
    ):
        return "standard"
    if actual_model == standard_model and standard_model != pro_model:
        return "standard"
    if actual_model == pro_model and pro_model != standard_model:
        return "pro"
    return decision.quality


def _image_dimensions(image: GeneratedImage) -> tuple[int | None, int | None]:
    try:
        with Image.open(BytesIO(image.content)) as decoded:
            return decoded.width, decoded.height
    except Exception:
        return None, None


def _attempt_rows(value: Any) -> list[dict[str, Any]]:
    attempts = getattr(value, "attempts", None)
    if not isinstance(attempts, list):
        metadata = getattr(value, "metadata", None)
        attempts = metadata.get("attempts") if isinstance(metadata, Mapping) else None
    if not isinstance(attempts, list):
        return []
    return [dict(item) for item in attempts if isinstance(item, Mapping)]


def _log_attempt_events(
    *,
    value: Any,
    error: BaseException | None,
    operation_id: str,
    operation: str,
    requested_model: str,
    route_reason: str,
    resolution: str | None,
    reference_count: int,
    elapsed_ms: float,
    cancelled: bool = False,
) -> list[dict[str, Any]]:
    attempts = _attempt_rows(value) if value is not None else []
    if not attempts and error is not None:
        attempts.extend(_attempt_rows(error))

    if not attempts:
        if value is not None:
            metadata = getattr(value, "metadata", {})
            if not isinstance(metadata, Mapping):
                metadata = {}
            attempts.append(
                {
                    "model": metadata.get("model") or requested_model,
                    "status_code": 200,
                    "duration_ms": elapsed_ms,
                    "cost": metadata.get("cost"),
                    "billing_unknown": metadata.get("cost") is None,
                }
            )
        elif error is not None and (
            cancelled
            or isinstance(error, OpenRouterMediaError)
            or not isinstance(error, OpenRouterMediaError)
            or getattr(error, "status_code", None) is not None
            or getattr(error, "billing_unknown", False)
            or getattr(error, "cost", None) is not None
        ):
            attempts.append(
                {
                    "model": getattr(error, "model", None) or requested_model,
                    "status_code": getattr(error, "status_code", None),
                    "duration_ms": elapsed_ms,
                    "cost": getattr(error, "cost", None),
                    "billing_unknown": getattr(error, "billing_unknown", True),
                    "error_reason": "cancelled" if cancelled else _safe_error_code(error),
                }
            )

    normalized: list[dict[str, Any]] = []
    for number, attempt in enumerate(attempts, start=1):
        actual_model = attempt.get("model")
        if not isinstance(actual_model, str) or not actual_model:
            actual_model = requested_model
        status_code = attempt.get("status_code", attempt.get("status"))
        if isinstance(status_code, bool) or not isinstance(status_code, int):
            status_code = None
        cost_value = attempt.get("cost")
        if cost_value is None and "cost" not in attempt:
            cost_value = attempt.get("known_cost")
        cost = _known_cost(cost_value)
        error_code = attempt.get("error_reason", attempt.get("reason"))
        if not isinstance(error_code, str) or error_code not in _SAFE_IMAGE_ERROR_CODES:
            error_code = None
        attempt_duration = _known_cost(attempt.get("duration_ms"))
        if attempt_duration is None:
            attempt_duration = max(0.0, elapsed_ms)
        row = {
            "operation_id": operation_id,
            "attempt": number,
            "operation": operation,
            "requested_model": requested_model,
            "actual_model": actual_model,
            "route_reason": route_reason,
            "resolution": resolution,
            "reference_count": reference_count,
            "duration_ms": round(attempt_duration, 3),
            "status": (
                "success" if status_code == 200 and error_code is None else "error"
            ),
            "error_code": error_code,
            "status_code": status_code,
            "cost": cost,
            "cost_unknown": bool(attempt.get("billing_unknown", False)) or cost is None,
        }
        log_image_generation_attempt(row)
        normalized.append(
            {
                **row,
                "known_cost": cost,
                "billing_unknown": row["cost_unknown"],
            }
        )
    return normalized


def _accounting(
    value: Any,
    attempts: list[dict[str, Any]],
) -> tuple[float, bool, float | None]:
    metadata = getattr(value, "metadata", None)
    if not isinstance(metadata, Mapping):
        metadata = {}
    total_known_cost = _known_cost(metadata.get("total_known_cost"))
    if total_known_cost is None:
        total_known_cost = math.fsum(
            row["known_cost"] for row in attempts if row.get("known_cost") is not None
        )
    billing_unknown = bool(metadata.get("billing_unknown", False)) or any(
        row.get("billing_unknown", False) for row in attempts
    )
    result_cost = _known_cost(metadata.get("cost"))
    return total_known_cost, billing_unknown, result_cost


def _log_operation_summary(
    *,
    operation_id: str,
    task_id: str,
    operation: str,
    requested_model: str,
    requested_quality: str,
    actual_model: str | None,
    quality: str,
    route_reason: str,
    delivered: bool,
    attempts: list[dict[str, Any]],
    fallback_used: bool,
    is_complaint_retry: bool,
    elapsed_ms: float,
    image: GeneratedImage | None,
    fallback_width: int | None = None,
    fallback_height: int | None = None,
) -> tuple[float, bool, float | None]:
    total_known_cost, billing_unknown, result_cost = _accounting(image, attempts)
    width: int | None = None
    height: int | None = None
    byte_count: int | None = None
    if image is not None:
        width, height = _image_dimensions(image)
        width = width or fallback_width
        height = height or fallback_height
        byte_count = len(image.content)
    log_image_operation_summary(
        {
            "operation_id": operation_id,
            "task_id": task_id,
            "operation": operation,
            "requested_model": requested_model,
            "requested_quality": requested_quality,
            "actual_model": actual_model,
            "quality": quality,
            "route_reason": route_reason,
            "delivered": delivered,
            "attempt_count": len(attempts),
            "fallback_used": fallback_used,
            "is_complaint_retry": is_complaint_retry,
            "total_duration_ms": round(max(0.0, elapsed_ms), 3),
            "width": width,
            "height": height,
            "bytes": byte_count,
            "result_cost": result_cost,
            "total_known_cost": total_known_cost,
            "billing_unknown": billing_unknown,
        }
    )
    return total_known_cost, billing_unknown, result_cost


async def _run_image_operation(
    ctx: ToolExecutionContext,
    *,
    user_id: int,
    operation: str,
    prompt: str,
    source_images: list[GeneratedImage],
    source_image_indices: list[int],
    aspect_ratio: str | None,
    decision: ImageRoutingDecision,
    retry_context: ImageRetryContext,
    references_truncated: bool = False,
) -> ToolResult:
    """Единый lifecycle доставки, статуса, метаданных и telemetry для image tools."""

    operation_id = uuid4().hex
    target_task_id = retry_context.target_task_id
    target_operation_id = retry_context.target_operation_id
    inherited_task_id = target_task_id or target_operation_id
    task_id = (
        inherited_task_id
        if retry_context.is_retry
        and isinstance(inherited_task_id, str)
        and re.fullmatch(r"[0-9a-f]{32}", inherited_task_id)
        else operation_id
    )
    reference_count = len(source_images)
    started_at = time.perf_counter()
    status_message: Any | None = None
    animation_task: asyncio.Task[None] | None = None
    image: GeneratedImage | None = None
    attempts: list[dict[str, Any]] = []
    attempts_logged = False
    summary_logged = False
    api_call_started = False
    delivered = False
    generated_width: int | None = None
    generated_height: int | None = None
    actual_model = decision.model
    requested_model = decision.model
    actual_quality = decision.quality
    fallback_used = False

    def elapsed_ms() -> float:
        return (time.perf_counter() - started_at) * 1000

    def update_actual_result(value: Any) -> None:
        nonlocal actual_model, actual_quality, fallback_used
        metadata = getattr(value, "metadata", None)
        if not isinstance(metadata, Mapping):
            metadata = {}
        model_value = metadata.get("model")
        if not isinstance(model_value, str) or not model_value:
            model_value = getattr(value, "model", None)
        if isinstance(model_value, str) and model_value:
            actual_model = model_value
        fallback_used = metadata.get("fallback_used") is True or getattr(
            value, "fallback_used", False
        ) is True
        actual_quality = _actual_model_quality(
            operation,
            actual_model,
            decision,
            fallback_used,
        )

    def summarize() -> tuple[float, bool, float | None]:
        nonlocal summary_logged
        summary_logged = True
        return _log_operation_summary(
            operation_id=operation_id,
            task_id=task_id,
            operation=operation,
            requested_model=requested_model,
            requested_quality=decision.quality,
            actual_model=actual_model,
            quality=actual_quality,
            route_reason=decision.reason,
            delivered=delivered,
            attempts=attempts,
            fallback_used=fallback_used,
            is_complaint_retry=retry_context.is_retry,
            elapsed_ms=elapsed_ms(),
            image=image,
            fallback_width=generated_width,
            fallback_height=generated_height,
        )

    try:
        status_message, animation_task = await _start_image_generation_status(ctx.update)
        try:
            api_call_started = True
            image_tool_state = getattr(ctx, "image_tool_state", None)
            if image_tool_state is not None:
                image_tool_state.image_call_attempted = True
            if operation == "generate":
                image = await ctx.openrouter_service.generate(
                    prompt,
                    ImageGenerationOptions(
                        aspect_ratio=aspect_ratio,
                        resolution=decision.resolution,
                        model=decision.model,
                    ),
                )
            else:
                image = await ctx.openrouter_service.edit(
                    source_images,
                    prompt,
                    ImageEditOptions(
                        aspect_ratio=aspect_ratio,
                        resolution=decision.resolution,
                        model=decision.model,
                    ),
                )
        except asyncio.CancelledError as error:
            attempts = _log_attempt_events(
                value=None,
                error=error,
                operation_id=operation_id,
                operation=operation,
                requested_model=requested_model,
                route_reason=decision.reason,
                resolution=decision.resolution,
                reference_count=reference_count,
                elapsed_ms=elapsed_ms(),
                cancelled=True,
            )
            attempts_logged = True
            update_actual_result(error)
            summarize()
            raise
        except OpenRouterMediaError as error:
            attempts = _log_attempt_events(
                value=None,
                error=error,
                operation_id=operation_id,
                operation=operation,
                requested_model=requested_model,
                route_reason=decision.reason,
                resolution=decision.resolution,
                reference_count=reference_count,
                elapsed_ms=elapsed_ms(),
            )
            attempts_logged = True
            update_actual_result(error)
            summarize()
            mapped = map_generation_error(error, context="image")
            await reply_service_text(ctx.update, mapped.user_message)
            return _answer(
                ctx,
                mapped.user_message,
                ok=False,
                details={
                    "operation_id": operation_id,
                    "operation": operation,
                    "requested_model": requested_model,
                    "actual_model": actual_model,
                    "quality": actual_quality,
                    "fallback_used": fallback_used,
                },
                meta={
                    "operation_id": operation_id,
                    "operation": operation,
                    "error_code": _safe_error_code(error),
                },
                retry_forbidden=True,
            )
        except Exception as error:
            logger.error("Ошибка image-операции: type=%s", type(error).__name__)
            attempts = _log_attempt_events(
                value=None,
                error=error,
                operation_id=operation_id,
                operation=operation,
                requested_model=requested_model,
                route_reason=decision.reason,
                resolution=decision.resolution,
                reference_count=reference_count,
                elapsed_ms=elapsed_ms(),
            )
            attempts_logged = True
            update_actual_result(error)
            summarize()
            return _answer(
                ctx,
                "Не удалось выполнить операцию с изображением. Внутренняя ошибка сервера.",
                ok=False,
                details={
                    "operation_id": operation_id,
                    "operation": operation,
                    "requested_model": requested_model,
                    "actual_model": actual_model,
                    "quality": actual_quality,
                    "fallback_used": fallback_used,
                },
                meta={
                    "operation_id": operation_id,
                    "operation": operation,
                    "error_code": _safe_error_code(error),
                },
                retry_forbidden=True,
            )

        update_actual_result(image)
        attempts = _log_attempt_events(
            value=image,
            error=None,
            operation_id=operation_id,
            operation=operation,
            requested_model=requested_model,
            route_reason=decision.reason,
            resolution=decision.resolution,
            reference_count=reference_count,
            elapsed_ms=elapsed_ms(),
        )
        attempts_logged = True

        try:
            normalized_b64, generated_width, generated_height, generated_aspect_ratio = (
                await send_image_to_telegram(ctx.update, image)
            )
        except asyncio.CancelledError:
            summarize()
            raise
        except Exception as error:
            logger.error(
                "Ошибка доставки image-результата в Telegram: type=%s",
                type(error).__name__,
            )
            summarize()
            return _answer(
                ctx,
                "Картинка сгенерировалась, но не удалось отправить её в Telegram.",
                ok=False,
                details={
                    "operation_id": operation_id,
                    "operation": operation,
                    "requested_model": requested_model,
                    "actual_model": actual_model,
                    "quality": actual_quality,
                    "fallback_used": fallback_used,
                },
                meta={"operation_id": operation_id, "operation": operation},
                retry_forbidden=True,
            )

        delivered = True
        total_known_cost, billing_unknown, result_cost = summarize()
        generation_metadata = {
            "operation_id": operation_id,
            "task_id": task_id,
            "operation": operation,
            "model": actual_model,
            "quality": actual_quality,
            "requested_model": requested_model,
            "fallback_used": fallback_used,
            "prompt": prompt,
            "source_image_indices": source_image_indices,
            "aspect_ratio": aspect_ratio,
            "resolution": decision.resolution,
            "route_reason": decision.reason,
            "total_known_cost": total_known_cost,
            "billing_unknown": billing_unknown,
        }

        try:
            new_image_dict = build_image_dict_from_image(
                image,
                normalized_b64,
                generated_width,
                generated_height,
                generated_aspect_ratio,
                generation=generation_metadata,
            )
            image_number = await append_user_image_to_session(
                user_id,
                new_image_dict,
                "text2image" if operation == "generate" else "edit",
                prompt,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error(
                "Ошибка сохранения image-результата в сессию: type=%s",
                type(error).__name__,
            )
            return _answer(
                ctx,
                "Изображение отправлено, но не удалось сохранить его в сессии.",
                ok=True,
                details={
                    "operation_id": operation_id,
                    "operation": operation,
                    "requested_model": requested_model,
                    "actual_model": actual_model,
                    "quality": actual_quality,
                    "fallback_used": fallback_used,
                    "image_number": None,
                    "result_cost": result_cost,
                    "total_known_cost": total_known_cost,
                    "billing_unknown": billing_unknown,
                    "delivered": True,
                },
                meta={"operation_id": operation_id, "operation": operation},
                retry_forbidden=True,
            )

        limit_notice = (
            "В Images API можно передать не более 10 референсов; использованы последние 10. "
            if references_truncated
            else ""
        )
        if operation == "generate":
            success_message = (
                f"Я сделал :) Изображение №{image_number} добавлено в сессию и доступно "
                "как референс для следующих операций."
            )
        else:
            success_message = (
                f"Результат добавлен в сессию как изображение №{image_number}. "
                "Можете выбрать его или другие изображения по номерам для следующих операций."
            )
        return _answer(
            ctx,
            limit_notice + success_message,
            ok=True,
            details={
                "operation_id": operation_id,
                "operation": operation,
                "requested_model": requested_model,
                "actual_model": actual_model,
                "quality": actual_quality,
                "fallback_used": fallback_used,
                "image_number": image_number,
                "result_cost": result_cost,
                "total_known_cost": total_known_cost,
                "billing_unknown": billing_unknown,
                "delivered": True,
            },
            meta={
                "operation_id": operation_id,
                "operation": operation,
                "actual_model": actual_model,
                "quality": actual_quality,
                "image_number": image_number,
                "fallback_used": fallback_used,
            },
        )
    except asyncio.CancelledError as error:
        if api_call_started and not attempts_logged:
            attempts = _log_attempt_events(
                value=image,
                error=error,
                operation_id=operation_id,
                operation=operation,
                requested_model=requested_model,
                route_reason=decision.reason,
                resolution=decision.resolution,
                reference_count=reference_count,
                elapsed_ms=elapsed_ms(),
                cancelled=True,
            )
            attempts_logged = True
        if not summary_logged:
            summarize()
        raise
    finally:
        await _cancel_image_generation_animation(animation_task)
        await _delete_image_generation_status(status_message)


def _image_indices(args: dict[str, Any]) -> list[int] | None:
    indices = args.get("image_indices")
    if indices is None:
        return None
    if not isinstance(indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in indices
    ):
        raise ValueError("image_indices должен быть списком целых номеров")
    return indices


_USER_IMAGE_INDEX_RE = re.compile(
    r"(?<!\w)(?:изображени\w*|картин\w*|фотографи\w*|фото|images?|pictures?|photos?)"
    r"\s*(?:(?:№|#|номер|number|no\.?\s*)\s*)?(\d+)(?!\w)|(?<!\w)(?:№|#)\s*(\d+)(?!\w)",
    re.IGNORECASE,
)
_RESULT_REFERENCE_RE = re.compile(
    r"(?<!\w)(?:"
    r"последн\w*\s+(?:результат\w*|изображени\w*|картинк\w*)|"
    r"(?:этот|эту|этого|полученн\w*|созданн\w*)\s+"
    r"(?:результат\w*|изображени\w*|картинк\w*)|"
    r"this\s+(?:result|image|picture)|"
    r"(?:last|latest)\s+(?:result|image|picture)|"
    r"image\s+(?:you|we)\s+(?:just\s+)?(?:made|created|generated)"
    r")(?!\w)",
    re.IGNORECASE,
)


def _image_indices_from_user_text(user_text: Any) -> list[int]:
    if not isinstance(user_text, str):
        return []
    indices: list[int] = []
    for match in _USER_IMAGE_INDEX_RE.finditer(user_text):
        raw_index = match.group(1) or match.group(2)
        if raw_index is None:
            continue
        index = int(raw_index)
        if index > 0 and index not in indices:
            indices.append(index)
    return indices


@register_tool("generate_image")
async def handle_generate_image(
    ctx: ToolExecutionContext, args: dict[str, Any]
) -> ToolResult:
    user_id = ctx.update.effective_user.id  # type: ignore
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _answer(
            ctx,
            "Не удалось сгенерировать изображение: запрос пустой.",
            ok=False,
        )
    prompt = prompt.strip()
    user_text = getattr(ctx, "user_text", None)
    session = await get_user_image_edit_session(user_id)
    entries = get_session_image_entries(session) if isinstance(session, dict) else []
    retry_context = resolve_image_retry_context(
        user_text,
        entries,
        "generate",
        current_prompt=prompt,
    )

    if retry_context.is_retry and retry_context.target_missing:
        return _answer(
            ctx,
            "Указанный результат для повтора недоступен. Проверьте номер изображения.",
            ok=False,
        )

    # Если модель ошибочно вызвала text-to-image для повтора неудачного edit,
    # повторяем исходную edit-операцию с её исходниками, а не создаём другой тип результата.
    if retry_context.is_retry and retry_context.target_operation == "edit":
        edit_retry = resolve_image_retry_context(
            user_text,
            entries,
            "edit",
            current_prompt=prompt,
        )
        if edit_retry.source_indices_available is not True:
            missing = ", ".join(
                f"№{index}" for index in edit_retry.missing_source_indices
            )
            detail = f" Недоступны источники: {missing}." if missing else ""
            return _answer(
                ctx,
                "Не удалось восстановить исходные изображения неудачного редактирования."
                + detail,
                ok=False,
            )
        redirected_args = dict(args)
        redirected_args["prompt"] = prompt
        redirected_args["image_indices"] = list(edit_retry.reference_image_indices)
        return await handle_generate_image_from_image(ctx, redirected_args)

    resolved_prompt = retry_context.prompt or prompt
    decision = resolve_image_route(
        settings,
        "generate",
        quality=args.get("quality"),
        prompt=resolved_prompt,
        user_text=user_text,
        reference_count=0,
        retry_target=retry_context if retry_context.is_retry else None,
    )
    return await _run_image_operation(
        ctx,
        user_id=user_id,
        operation="generate",
        prompt=resolved_prompt,
        source_images=[],
        source_image_indices=[],
        aspect_ratio=_sdk_aspect_ratio(args.get("aspect_ratio")),
        decision=decision,
        retry_context=retry_context,
    )


@register_tool("generate_image_from_image")
async def handle_generate_image_from_image(
    ctx: ToolExecutionContext, args: dict[str, Any]
) -> ToolResult:
    user_id = ctx.update.effective_user.id  # type: ignore
    session = await get_user_image_edit_session(user_id)
    if session is None:
        return _answer(
            ctx,
            "Сначала отправьте изображение, которое будет использоваться как основа для генерации.",
            ok=False,
        )
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _answer(
            ctx,
            "Не удалось сгенерировать изображение: запрос пустой.",
            ok=False,
        )
    prompt = prompt.strip()
    user_text = getattr(ctx, "user_text", None)
    entries = get_session_image_entries(session)
    retry_context = resolve_image_retry_context(
        user_text,
        entries,
        "edit",
        current_prompt=prompt,
    )

    if retry_context.is_retry and retry_context.target_missing:
        return _answer(
            ctx,
            "Указанный результат для повтора недоступен. Проверьте номер изображения.",
            ok=False,
        )

    if retry_context.is_retry and retry_context.target_operation == "generate":
        # Повтор text-to-image не должен превращаться в редактирование плохого результата.
        redirected_args = dict(args)
        redirected_args["prompt"] = prompt
        return await handle_generate_image(ctx, redirected_args)

    if retry_context.is_retry:
        if retry_context.target_operation == "edit":
            if retry_context.source_indices_available is not True:
                missing = ", ".join(
                    f"№{index}" for index in retry_context.missing_source_indices
                )
                if missing:
                    explanation = f"Недоступны источники: {missing}."
                else:
                    explanation = "В метаданных результата нет данных об исходниках."
                return _answer(
                    ctx,
                    "Не удалось восстановить исходные изображения неудачного редактирования. "
                    + explanation,
                    ok=False,
                )
            image_indices = list(retry_context.reference_image_indices)
        elif retry_context.target_operation is None:
            try:
                tool_selected_indices = _image_indices(args)
            except ValueError as error:
                return _answer(ctx, str(error), ok=False)
            user_selected_indices = _image_indices_from_user_text(user_text)
            if user_selected_indices:
                image_indices = user_selected_indices
            elif tool_selected_indices:
                # Старые записи можно использовать только после явного выбора
                # номеров; неизвестное происхождение не даёт права брать всё.
                image_indices = tool_selected_indices
            else:
                return _answer(
                    ctx,
                    "У этого старого результата нет данных об исходниках. Укажите изображения, "
                    "которые нужно использовать, или попросите создать результат с нуля.",
                    ok=False,
                )
        else:
            return _answer(ctx, "Не удалось определить тип операции для повтора.", ok=False)
    else:
        try:
            image_indices = _image_indices(args)
        except ValueError as error:
            return _answer(ctx, str(error), ok=False)
        # Пользовательский выбор номеров приоритетнее автоматической подсказки
        # языковой модели; без такого выбора обычная правка использует все картинки.
        explicit_user_indices = _image_indices_from_user_text(user_text)
        if explicit_user_indices:
            image_indices = explicit_user_indices
        elif (
            retry_context.is_improvement
            and retry_context.use_result_as_reference
            and retry_context.target_index is not None
            and isinstance(user_text, str)
            and _RESULT_REFERENCE_RE.search(user_text)
        ):
            image_indices = [retry_context.target_index]

    max_refs = min(10, settings.media.max_session_images)
    try:
        selected_entries = get_session_image_entries(session, image_indices)
        if not selected_entries:
            return _answer(ctx, "В сессии нет доступных изображений для редактирования.", ok=False)
        references_truncated = len(selected_entries) > max_refs
        source_images, source_image_indices = resolve_selected_images(
            session,
            image_indices,
            max_images=max_refs,
        )
    except Exception as error:
        logger.info("Не удалось разрешить image-источники: type=%s", type(error).__name__)
        return _answer(
            ctx,
            "Не удалось подготовить выбранные исходные изображения. Проверьте их номера.",
            ok=False,
        )

    resolved_prompt = retry_context.prompt if retry_context.is_retry else prompt
    if not isinstance(resolved_prompt, str) or not resolved_prompt.strip():
        resolved_prompt = prompt
    explicit_aspect_ratio = _sdk_aspect_ratio(args.get("aspect_ratio"))
    if explicit_aspect_ratio is not None:
        aspect_ratio = explicit_aspect_ratio
    else:
        aspect_ratio = _sdk_aspect_ratio(source_images[0].metadata.get("aspect_ratio"))
        if len(source_images) == 1:
            primary_image = source_images[0]
            source_width = primary_image.metadata.get("width")
            source_height = primary_image.metadata.get("height")
            if (
                isinstance(source_width, int)
                and not isinstance(source_width, bool)
                and isinstance(source_height, int)
                and not isinstance(source_height, bool)
                and source_width > 0
                and source_height > 0
            ):
                resolved_prompt = (
                    f"{resolved_prompt}\n\n"
                    "Important: preserve the original image aspect ratio exactly. "
                    f"The source image is {source_width}x{source_height}. Do not make it wider or taller. "
                    "Keep the same framing and canvas proportions."
                )

    # Решение принимается только после дедупликации, лимита и получения реальных номеров.
    decision = resolve_image_route(
        settings,
        "edit",
        quality=args.get("quality"),
        prompt=resolved_prompt,
        user_text=user_text,
        reference_count=len(source_images),
        retry_target=retry_context if retry_context.is_retry else None,
        generated_result_improvement=(
            retry_context.is_improvement
            and retry_context.use_result_as_reference
            and retry_context.target_index in source_image_indices
            and isinstance(user_text, str)
            and _RESULT_REFERENCE_RE.search(user_text) is not None
        ),
    )
    return await _run_image_operation(
        ctx,
        user_id=user_id,
        operation="edit",
        prompt=resolved_prompt,
        source_images=source_images,
        source_image_indices=source_image_indices,
        aspect_ratio=aspect_ratio,
        decision=decision,
        retry_context=retry_context,
        references_truncated=references_truncated,
    )


_DURATION_QUESTION_TEXT = "Сколько секунд должно длиться видео?"
_DURATION_QUESTION_INPUT_HINT = (
    "Напишите длительность в секундах (например, 7 или «7 секунд»)."
)


def _invalid_duration_message(profile: VideoModelProfile) -> str:
    """Возвращает понятное сообщение о неверной длительности видео."""

    allowed = ", ".join(str(value) for value in profile.allowed_durations)
    return (
        f"{user_message_for_reason(GenerationFailureReason.INVALID_DURATION)} "
        f"Допустимые значения: {allowed}."
    )


def build_duration_question_request(profile: VideoModelProfile) -> PendingQuestionRequest:
    """Строит обязательный вопрос о длительности видео.

    Варианты кнопок берутся из активного видеопрофиля, а собственный ответ
    проверяется тем же валидатором: длительность подтверждает пользователь.
    """

    presets = profile.question_duration_presets() or profile.allowed_durations
    return PendingQuestionRequest(
        text=_DURATION_QUESTION_TEXT,
        options=build_duration_question_options(presets),
        input_hint=_DURATION_QUESTION_INPUT_HINT,
        answer_mode=QuestionAnswerMode.VIDEO_REQUEST,
        validator=build_duration_validator(profile),
    )


def _confirmed_duration(ctx: ToolExecutionContext, profile: VideoModelProfile) -> int | None:
    """Возвращает длительность, явно названную пользователем.

    Значение, переданное моделью в аргументах инструмента, согласием
    пользователя не является и подтверждённой длительностью не считается.

    Противоречивые, приблизительные и неоднозначные значения не угадываются:
    в этом случае возвращается ``None`` и задаётся вопрос о длительности.
    Явно запрошенная неподдерживаемая длительность — ошибка, а не вопрос.
    """

    try:
        return parse_explicit_duration(getattr(ctx, "user_text", None), profile)
    except AmbiguousDurationError:
        logger.info("Длительность в исходном запросе неоднозначна: задаётся вопрос")
        return None
    except UnsupportedDurationError:
        raise


def _user_image_indices(ctx: ToolExecutionContext) -> list[int]:
    """Возвращает номера референсов, явно названные пользователем.

    Явный порядок пользователя в тексте запроса имеет приоритет над
    ``image_indices``, переданными моделью.
    """

    return parse_user_image_indices(getattr(ctx, "user_text", None))


@dataclass(frozen=True, slots=True)
class _VideoDraftPreparation:
    """Результат подготовки снимка видеозаявки.

    ``draft`` заполнен только при успешной подготовке. ``error`` содержит
    понятный текст ошибки, а ``blocks_retry`` означает, что платный запуск в
    этом ходе запрещён: модель не должна повторить генерацию с незаметно
    урезанным или подменённым набором референсов. В этом случае вопрос о
    длительности не задаётся — сначала нужно разобраться с самим запросом.
    """

    draft: VideoRequestDraft | None = None
    error: str | None = None
    blocks_retry: bool = False


async def _prepare_video_draft(
    ctx: ToolExecutionContext,
    args: dict[str, Any],
) -> _VideoDraftPreparation:
    """Готовит снимок видеозаявки.

    Снимок фиксируется до вопроса о длительности: промпт, упорядоченные
    реальные номера референсов, байты выбранных изображений и снимок
    активного видеопрофиля. Новые загрузки в уже созданную заявку не
    попадают.
    """

    user_id = ctx.update.effective_user.id  # type: ignore
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        logger.warning("Пустой prompt для generate_video")
        return _VideoDraftPreparation(
            error="Не удалось сгенерировать видео: запрос пустой."
        )

    profile = settings.video_profile

    try:
        user_indices = _user_image_indices(ctx)
    except AmbiguousImageOrderError as error:
        logger.warning("Неоднозначный порядок референсов в запросе: %s", error)
        return _VideoDraftPreparation(
            error=f"Не удалось подготовить видео: {error}",
            blocks_retry=True,
        )

    try:
        tool_indices = _image_indices(args)
    except ValueError as error:
        logger.warning("Некорректные image_indices: %s", error)
        return _VideoDraftPreparation(
            error=f"Не удалось подготовить видео: {error}",
            blocks_retry=True,
        )

    # Явный выбор пользователя приоритетнее автоматически восстановленного.
    image_indices = user_indices or tool_indices

    session = await get_user_image_edit_session(user_id)
    try:
        selection = resolve_video_references(
            session if isinstance(session, dict) else None,
            image_indices,
            profile=profile,
        )
    except ReferenceLimitExceededError as error:
        logger.warning("Превышен предел референсов видеопрофиля: %s", error)
        return _VideoDraftPreparation(
            error=f"Не удалось подготовить видео: {error}",
            blocks_retry=True,
        )
    except ValueError as error:
        logger.warning("Некорректный выбор референсов: %s", error)
        return _VideoDraftPreparation(
            error=f"Не удалось подготовить видео: {error}",
            blocks_retry=True,
        )

    try:
        draft = build_video_request_draft(
            prompt,
            profile,
            reference_images=selection.images,
            reference_indices=selection.indices,
        )
    except ValueError as error:
        logger.warning("Заявка на видео отклонена: %s", error)
        return _VideoDraftPreparation(
            error=f"Не удалось подготовить видео: {error}"
        )

    return _VideoDraftPreparation(draft=draft)


@register_tool_question_check("generate_video")
async def check_generate_video(
    ctx: ToolExecutionContext,
    args: dict[str, Any]
) -> ToolQuestionCheck | None:
    """Проверяет, нужно ли уточнить длительность видео.

    Проверка выполняется до пакета вызовов: обязательный вопрос о длительности
    всегда предшествует платным операциям, даже если модель поставила вызов
    видео после них.
    """

    preparation = await _prepare_video_draft(ctx, args)
    draft = preparation.draft
    if draft is None:
        if preparation.blocks_retry:
            logger.info("Заявка на видео отклонена: платный повтор заблокирован")
        return None
    try:
        duration = _confirmed_duration(ctx, draft.profile)
    except UnsupportedDurationError as error:
        logger.warning("Недопустимая длительность в исходном запросе: %s", error)
        return None
    if duration is not None:
        return None
    return ToolQuestionCheck(
        request=build_duration_question_request(draft.profile),
        video_request=draft,
        video_operation_id=draft.operation_id,
    )


def _mark_video_operation_failed(ctx: ToolExecutionContext, operation_id: str) -> None:
    """Запрещает автоматический платный повтор видео после ошибки."""

    state = getattr(ctx, "video_tool_state", None)
    if state is None:
        return
    state.mark_failed(operation_id)


def _mark_video_operation_completed(ctx: ToolExecutionContext, operation_id: str) -> None:
    """Отмечает видеооперацию как выполненную."""

    state = getattr(ctx, "video_tool_state", None)
    if state is None:
        return
    state.mark_completed(operation_id)


def _video_result_payload(
    draft: VideoRequestDraft,
    video: GeneratedVideo,
) -> dict[str, Any]:
    """Собирает структурированный результат генерации видео.

    Возвращаются фактические модель и разрешение активного профиля,
    длительность, пропорция, номера референсов в том же порядке,
    идентификатор задачи провайдера и известная стоимость. Бинарное
    содержимое картинок и data URL в результат не попадают.
    """

    profile = draft.profile
    cost = getattr(video, "cost", None)
    raw_metadata = getattr(video, "metadata", None)
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    job_id = metadata.get("job_id")
    return {
        "model": profile.model,
        "resolution": profile.resolution,
        "duration": draft.duration,
        "aspect_ratio": draft.aspect_ratio or settings.media.video_aspect_ratio,
        "reference_indices": list(draft.reference_indices),
        "reference_count": len(draft.reference_images),
        "job_id": job_id if isinstance(job_id, str) and job_id else None,
        "cost": cost,
        "billing_unknown": cost is None,
    }


async def run_video_request(
    ctx: ToolExecutionContext,
    draft: VideoRequestDraft,
) -> ToolResult:
    """Выполняет подготовленную видеозаявку.

    Используется и обычным обработчиком инструмента, и продолжением после
    ответа пользователя: платный запуск выполняется ровно один раз.
    """

    profile = draft.profile
    if draft.duration is None:
        return _answer(
            ctx,
            _invalid_duration_message(profile),
            ok=False,
        )

    options = VideoGenerationOptions(
        duration=draft.duration,
        model=profile.model,
        resolution=profile.resolution,
        aspect_ratio=settings.media.video_aspect_ratio,
    )
    reference_images = list(draft.reference_images)

    logger.info(
        "Видео: operation_id=%s, model=%s, resolution=%s, duration=%s, "
        "references=%s",
        draft.operation_id,
        profile.model,
        profile.resolution,
        draft.duration,
        list(draft.reference_indices),
    )

    status_message = await reply_service_message(ctx.update, "Видео генерируется, подождите")
    animation_task = asyncio.create_task(
        animate_service_message(
            status_message,
            "Видео генерируется, подождите",
            interval=settings.telegram.service_animation_interval_seconds,
        )
    )

    try:
        video = await ctx.openrouter_service.generate_video(
            draft.prompt,
            options,
            reference_images=reference_images,
        )
    except OpenRouterMediaError as error:
        mapped = map_generation_error(error, context="video")
        logger.warning(
            "Видео: operation_id=%s, этап=create/poll, результат=ошибка, "
            "job_id=%s, reason=%s, cost=%s",
            draft.operation_id,
            getattr(error, "job_id", None),
            mapped.reason.value,
            getattr(error, "cost", None),
        )
        # После ошибки автоматический платный повтор модели запрещён.
        _mark_video_operation_failed(ctx, draft.operation_id)
        try:
            await status_message.edit_text(mapped.user_message)
        except Exception:
            pass
        await reply_service_text(ctx.update, mapped.user_message)
        return _answer(ctx, None, meta={"operation_id": draft.operation_id})
    except Exception as error:
        logger.error(
            "Видео: operation_id=%s, результат=ошибка, type=%s",
            draft.operation_id,
            type(error).__name__,
        )
        _mark_video_operation_failed(ctx, draft.operation_id)
        try:
            await status_message.edit_text(
                "_Произошла ошибка при генерации видео._",
                parse_mode=settings.telegram.parse_mode,
            )
        except Exception:
            pass
        return _answer(ctx, None, meta={"operation_id": draft.operation_id})
    finally:
        animation_task.cancel()
        try:
            await animation_task
        except asyncio.CancelledError:
            pass

    try:
        await status_message.edit_text(
            "_Видео готово, отправляю..._",
            parse_mode=settings.telegram.parse_mode,
        )
    except Exception:
        pass

    payload = _video_result_payload(draft, video)
    try:
        success = await send_video_to_telegram(ctx.update, video, status_message)
    finally:
        video.cleanup()
    if success:
        _mark_video_operation_completed(ctx, draft.operation_id)
    else:
        # Сбой доставки не приводит к повторной генерации видео.
        _mark_video_operation_failed(ctx, draft.operation_id)
    logger.info(
        "Видео: operation_id=%s, этап=delivery, job_id=%s, результат=%s, cost=%s",
        draft.operation_id,
        payload["job_id"],
        "отправлено" if success else "ошибка доставки",
        payload["cost"] if payload["cost"] is not None else "unknown",
    )
    if not success:
        return _answer(
            ctx,
            None,
            details={"delivery": "failed"},
            meta={"operation_id": draft.operation_id},
        )

    try:
        await add_user_image_session_history(
            ctx.update.effective_user.id,  # type: ignore
            "video",
            draft.prompt,
        )
    except Exception as error:
        logger.error(
            "Ошибка при обновлении history visual session после видео: %s",
            error,
            exc_info=True,
        )

    return _answer(
        ctx,
        (
            "Я сделал :) Видео создано на основе выбранных изображений. "
            "Все картинки сессии сохранены для дальнейших операций."
            if reference_images
            else "Я сделал :) Видео создано по текстовому описанию."
        ),
        details=payload,
        meta={"operation_id": draft.operation_id},
    )


@register_tool("generate_video")
async def handle_generate_video(
    ctx: ToolExecutionContext, args: dict[str, Any]
) -> ToolResult:
    preparation = await _prepare_video_draft(ctx, args)
    draft = preparation.draft
    if draft is None:
        return _answer(
            ctx,
            preparation.error or "Не удалось подготовить видео.",
            ok=False,
            retry_forbidden=preparation.blocks_retry,
            retry_forbidden_text=_VIDEO_RETRY_FORBIDDEN_TEXT,
        )

    profile = draft.profile
    try:
        duration = _confirmed_duration(ctx, profile)
    except UnsupportedDurationError as error:
        logger.warning("Недопустимая длительность в исходном запросе: %s", error)
        return _answer(ctx, f"Не удалось подготовить видео: {error}", ok=False)

    if duration is None:
        # Значение duration, придуманное моделью, согласием пользователя не
        # является: без явного указания пользователя задаётся вопрос.
        return _answer(
            ctx,
            _invalid_duration_message(profile),
            ok=False,
            details={
                "requires_duration": True,
                "duration_question": build_duration_question_request(profile).text,
            },
        )

    return await run_video_request(ctx, draft.with_duration(duration))
