"""Единый lifecycle и адаптер ошибок OpenRouter Media API."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Sequence, TypeVar

from openrouter import OpenRouter
from openrouter import errors as sdk_errors

from core.config import Settings, settings
from core.common_types import (
    GeneratedImage,
    GeneratedVideo,
    ImageEditOptions,
    ImageGenerationOptions,
    VideoGenerationOptions,
)

T = TypeVar("T")


def _known_cost(value: Any) -> float | None:
    """Преобразует только конечные неотрицательные суммы в известную стоимость."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


def _monotonic() -> float:
    """Часы операции, вынесенные отдельно для детерминированного тестирования deadline."""

    return time.monotonic()


def _image_attempt_record(
    *,
    model: str,
    duration_ms: float,
    status_code: int | None,
    cost: Any,
    billing_unknown: bool,
    error_reason: str | None = None,
) -> dict[str, Any]:
    known_cost = _known_cost(cost)
    if status_code is None and error_reason == "invalid_response":
        # Изображение и usage пришли от API, но не прошли локальную проверку.
        status_code = 200
    return {
        "model": model,
        "status": status_code,
        "status_code": status_code,
        "duration_ms": max(0.0, round(duration_ms, 3)),
        "cost": known_cost,
        "known_cost": known_cost,
        "billing_unknown": billing_unknown or known_cost is None,
        "error_reason": error_reason,
        "reason": error_reason,
    }


def _attach_image_attempts(
    error: OpenRouterMediaError | asyncio.CancelledError,
    *,
    attempts: Sequence[dict[str, Any]],
    requested_model: str,
    attempted_model: str,
    fallback_used: bool,
    fallback_reason: str | None,
) -> None:
    copied_attempts = [dict(attempt) for attempt in attempts]
    total_known_cost = math.fsum(
        attempt["known_cost"]
        for attempt in copied_attempts
        if attempt.get("known_cost") is not None
    )
    billing_unknown = any(attempt.get("billing_unknown", False) for attempt in copied_attempts)
    for name, value in (
        ("attempts", copied_attempts),
        ("total_known_cost", total_known_cost),
        ("billing_unknown", billing_unknown),
        ("requested_model", requested_model),
        ("model", attempted_model),
        ("fallback_used", fallback_used),
        ("fallback_reason", fallback_reason),
    ):
        setattr(error, name, value)


def _safe_provider_message(value: object | None) -> str | None:
    """Возвращает короткое сообщение провайдера без полного тела ответа."""

    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("message") or value.get("error") or value.get("code")
    message = " ".join(str(value).split())
    if not message:
        return None
    lowered = message.lower()
    if "data:image/" in lowered or "base64" in lowered:
        return "Провайдер вернул сообщение, содержащее бинарные данные"
    return message[:300]


class OpenRouterMediaError(Exception):
    """Базовая ошибка media-операций OpenRouter."""

    default_reason = "unknown"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        reason: str | None = None,
        provider_message: object | None = None,
        operation: str | None = None,
        model: str | None = None,
        requested_model: str | None = None,
        cost: Any = None,
        attempt_metadata: dict[str, Any] | None = None,
        fallback_eligible: bool = False,
        billing_unknown: bool = False,
        attempts: Sequence[dict[str, Any]] | None = None,
        total_known_cost: float = 0.0,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason or self.default_reason
        self.operation = operation
        self.model = model
        self.requested_model = requested_model or model
        self.cost = cost
        self.attempt_metadata = dict(attempt_metadata or {})
        self.fallback_eligible = fallback_eligible
        self.billing_unknown = billing_unknown
        self.attempts = [dict(attempt) for attempt in attempts or ()]
        self.total_known_cost = total_known_cost
        self.fallback_used = fallback_used
        self.fallback_reason = fallback_reason
        self.provider_message = _safe_provider_message(provider_message)
        self.safe_provider_message = self.provider_message


class OpenRouterConfigurationError(OpenRouterMediaError):
    default_reason = "configuration"


class OpenRouterAuthenticationError(OpenRouterMediaError):
    default_reason = "authentication"


class OpenRouterBillingError(OpenRouterMediaError):
    default_reason = "insufficient_credits"


class OpenRouterUnsupportedError(OpenRouterMediaError):
    default_reason = "unsupported"


class OpenRouterUnsupportedModelError(OpenRouterUnsupportedError):
    default_reason = "unsupported_model"


class OpenRouterUnsupportedParameterError(OpenRouterUnsupportedError):
    default_reason = "unsupported_parameter"


class OpenRouterRateLimitError(OpenRouterMediaError):
    default_reason = "rate_limit"


class OpenRouterTimeoutError(OpenRouterMediaError):
    default_reason = "timeout"


class OpenRouterConnectionError(OpenRouterMediaError):
    default_reason = "connection_error"


class OpenRouterResponseError(OpenRouterMediaError):
    default_reason = "invalid_response"


class OpenRouterTemporaryError(OpenRouterMediaError):
    default_reason = "temporary_provider_error"


class OpenRouterSdkUnavailableError(OpenRouterMediaError):
    default_reason = "sdk_unavailable"


class OpenRouterVideoFailedError(OpenRouterMediaError):
    default_reason = "video_failed"


class OpenRouterVideoCancelledError(OpenRouterMediaError):
    default_reason = "video_cancelled"


class OpenRouterVideoExpiredError(OpenRouterMediaError):
    default_reason = "video_expired"


@dataclass(frozen=True, slots=True)
class OpenRouterConfig:
    """Конфигурация SDK без отдельных URL для media API."""

    api_key: str
    http_referer: str | None = None
    x_open_router_title: str | None = None
    x_open_router_categories: str | None = None
    timeout_ms: int = settings.openrouter.timeout_ms
    video_timeout_seconds: float = settings.media.video_timeout_seconds
    video_polling_interval_seconds: float = settings.media.video_polling_interval_seconds

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise OpenRouterConfigurationError(
                "Не задан секрет OPENROUTER_API_KEY"
            )
        if self.timeout_ms <= 0:
            raise OpenRouterConfigurationError("Таймаут OpenRouter должен быть положительным")
        if self.video_timeout_seconds <= 0 or self.video_polling_interval_seconds <= 0:
            raise OpenRouterConfigurationError(
                "Параметры ожидания видео должны быть положительными"
            )

    @classmethod
    def from_settings(cls, app_settings: Settings = settings) -> "OpenRouterConfig":
        """Создаёт конфигурацию SDK из единого объекта настроек."""

        return cls(
            api_key=app_settings.secrets.openrouter_api_key,
            http_referer=app_settings.openrouter.http_referer,
            x_open_router_title=app_settings.openrouter.title,
            x_open_router_categories=app_settings.openrouter.categories,
            timeout_ms=app_settings.openrouter.timeout_ms,
            video_timeout_seconds=app_settings.media.video_timeout_seconds,
            video_polling_interval_seconds=app_settings.media.video_polling_interval_seconds,
        )

    @classmethod
    def from_env(cls) -> "OpenRouterConfig":
        """Совместимый алиас: источник настроек теперь находится в ``core.config``."""

        return cls.from_settings()


class OpenRouterService:
    """Долгоживущий async SDK-клиент OpenRouter.

    Экземпляр создаётся на время жизни приложения и не создаёт SDK-клиент при
    импорте модуля.
    """

    def __init__(
        self,
        config: OpenRouterConfig | None = None,
        *,
        client_factory: Callable[..., OpenRouter] = OpenRouter,
    ) -> None:
        self.config = config or OpenRouterConfig.from_settings()
        self._client_factory = client_factory
        self._sdk_client: OpenRouter | None = None
        self._client: OpenRouter | None = None

    @classmethod
    def from_settings(
        cls,
        app_settings: Settings = settings,
        *,
        client_factory: Callable[..., OpenRouter] = OpenRouter,
    ) -> "OpenRouterService":
        return cls(
            OpenRouterConfig.from_settings(app_settings),
            client_factory=client_factory,
        )

    @classmethod
    def from_env(
        cls, *, client_factory: Callable[..., OpenRouter] = OpenRouter
    ) -> "OpenRouterService":
        """Совместимый алиас для старых точек запуска."""

        return cls.from_settings(client_factory=client_factory)

    @property
    def started(self) -> bool:
        return self._client is not None

    async def __aenter__(self) -> "OpenRouterService":
        if self._client is not None:
            return self

        try:
            sdk_client = self._client_factory(
                api_key=self.config.api_key,
                http_referer=self.config.http_referer,
                x_open_router_title=self.config.x_open_router_title,
                x_open_router_categories=self.config.x_open_router_categories,
                timeout_ms=self.config.timeout_ms,
            )
            self._sdk_client = sdk_client
            self._client = await sdk_client.__aenter__()
        except Exception as exc:
            sdk_client = self._sdk_client
            if sdk_client is not None:
                try:
                    await sdk_client.__aexit__(type(exc), exc, exc.__traceback__)
                except Exception:
                    pass
            self._sdk_client = None
            raise self._translate_exception(exc, "lifecycle") from exc
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        sdk_client = self._sdk_client
        self._client = None
        self._sdk_client = None
        if sdk_client is not None:
            await sdk_client.__aexit__(exc_type, exc_value, traceback)

    async def close(self) -> None:
        await self.__aexit__(None, None, None)

    def _require_client(self) -> OpenRouter:
        if self._client is None:
            raise OpenRouterSdkUnavailableError(
                "OpenRouter SDK-клиент не запущен", operation="lifecycle"
            )
        return self._client

    async def _run(
        self, operation: str, call: Callable[[], Awaitable[T]]
    ) -> T:
        self._require_client()
        try:
            return await call()
        except OpenRouterMediaError:
            raise
        except Exception as exc:
            raise self._translate_exception(exc, operation) from exc

    async def _run_image_operation(
        self,
        *,
        operation: str,
        requested_model: str,
        fast_model: str,
        pro_model: str,
        call_for_model: Callable[[str, int], Awaitable[GeneratedImage]],
    ) -> GeneratedImage:
        """Выполняет image-вызов с максимум одним подтверждённым резервом."""

        deadline = _monotonic() + self.config.timeout_ms / 1000
        attempts: list[dict[str, Any]] = []
        fallback_used = False
        fallback_reason: str | None = None
        attempted_model = requested_model

        async def invoke(model: str, *, reason: str | None = None) -> GeneratedImage:
            nonlocal attempted_model, fallback_used, fallback_reason
            remaining = deadline - _monotonic()
            if remaining <= 0:
                raise OpenRouterTimeoutError(
                    "Общий лимит времени генерации исчерпан",
                    operation=operation,
                    model=attempted_model,
                    requested_model=requested_model,
                    billing_unknown=bool(attempts),
                )

            if model != requested_model:
                attempted_model = model
                fallback_used = True
                fallback_reason = reason
            timeout_ms = max(1, math.floor(remaining * 1000))
            started_at = _monotonic()
            try:
                image = await asyncio.wait_for(
                    call_for_model(model, timeout_ms),
                    timeout=remaining,
                )
            except asyncio.CancelledError:
                attempts.append(
                    _image_attempt_record(
                        model=model,
                        duration_ms=(_monotonic() - started_at) * 1000,
                        status_code=None,
                        cost=None,
                        billing_unknown=True,
                        error_reason="cancelled",
                    )
                )
                raise
            except Exception as raw_error:
                error = self._translate_exception(raw_error, operation)
                if (
                    error.status_code is not None
                    or error.attempt_metadata
                    or error.billing_unknown
                    or error.cost is not None
                    or isinstance(raw_error, OpenRouterMediaError)
                ):
                    attempts.append(
                        _image_attempt_record(
                            model=model,
                            duration_ms=(_monotonic() - started_at) * 1000,
                            status_code=error.status_code,
                            cost=error.cost,
                            billing_unknown=error.billing_unknown,
                            error_reason=error.reason,
                        )
                    )
                if error is raw_error:
                    raise
                raise error from raw_error

            attempts.append(
                _image_attempt_record(
                    model=model,
                    duration_ms=(_monotonic() - started_at) * 1000,
                    status_code=200,
                    cost=image.metadata.get("cost"),
                    billing_unknown=False,
                )
            )
            return image

        fallback_allowed_for_route = (
            settings.media.image_fallback_enabled is True
            and requested_model == fast_model
            and fast_model != pro_model
        )

        try:
            image = await invoke(requested_model)
        except asyncio.CancelledError as exc:
            _attach_image_attempts(
                exc,
                attempts=attempts,
                requested_model=requested_model,
                attempted_model=requested_model,
                fallback_used=False,
                fallback_reason=None,
            )
            raise
        except OpenRouterMediaError as error:
            if not (fallback_allowed_for_route and error.fallback_eligible is True):
                _attach_image_attempts(
                    error,
                    attempts=attempts,
                    requested_model=requested_model,
                    attempted_model=requested_model,
                    fallback_used=False,
                    fallback_reason=None,
                )
                raise

            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                cancelled = asyncio.CancelledError()
                _attach_image_attempts(
                    cancelled,
                    attempts=attempts,
                    requested_model=requested_model,
                    attempted_model=requested_model,
                    fallback_used=False,
                    fallback_reason=None,
                )
                raise cancelled from error

            remaining = deadline - _monotonic()
            if remaining <= 0:
                _attach_image_attempts(
                    error,
                    attempts=attempts,
                    requested_model=requested_model,
                    attempted_model=requested_model,
                    fallback_used=False,
                    fallback_reason=None,
                )
                raise

            try:
                image = await invoke(pro_model, reason=error.reason)
            except asyncio.CancelledError as cancelled:
                _attach_image_attempts(
                    cancelled,
                    attempts=attempts,
                    requested_model=requested_model,
                    attempted_model=attempted_model,
                    fallback_used=fallback_used,
                    fallback_reason=fallback_reason,
                )
                raise
            except OpenRouterMediaError as fallback_error:
                _attach_image_attempts(
                    fallback_error,
                    attempts=attempts,
                    requested_model=requested_model,
                    attempted_model=attempted_model,
                    fallback_used=fallback_used,
                    fallback_reason=fallback_reason,
                )
                raise
        except Exception as raw_error:
            error = self._translate_exception(raw_error, operation)
            _attach_image_attempts(
                error,
                attempts=attempts,
                requested_model=requested_model,
                attempted_model=requested_model,
                fallback_used=False,
                fallback_reason=None,
            )
            raise error from raw_error

        metadata = dict(image.metadata)
        metadata.update(
            {
                "model": attempted_model,
                "requested_model": requested_model,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "attempts": [dict(attempt) for attempt in attempts],
                "total_known_cost": math.fsum(
                    attempt["known_cost"]
                    for attempt in attempts
                    if attempt.get("known_cost") is not None
                ),
                "billing_unknown": any(
                    attempt.get("billing_unknown", False) for attempt in attempts
                ),
            }
        )
        return GeneratedImage(content=image.content, mime_type=image.mime_type, metadata=metadata)

    async def generate_image(
        self,
        prompt: str,
        options: ImageGenerationOptions | None = None,
    ) -> GeneratedImage:
        if not prompt or not prompt.strip():
            raise OpenRouterUnsupportedParameterError(
                "Запрос на генерацию изображения пуст", reason="empty_prompt", operation="image"
            )
        from utils.openrouter_images import generate

        client = self._require_client()
        selected_options = options if options is not None else ImageGenerationOptions()
        if selected_options.resolution is None and settings.media.image_default_resolution is not None:
            selected_options = replace(
                selected_options,
                resolution=settings.media.image_default_resolution,
            )
        default_model = settings.media.image_generation_model
        if selected_options.model is not None and not isinstance(selected_options.model, str):
            raise OpenRouterUnsupportedParameterError(
                "Параметр model должен быть строкой",
                reason="unsupported_parameter",
                operation="image",
            )
        requested_model = (
            selected_options.model.strip()
            if isinstance(selected_options.model, str) and selected_options.model.strip()
            else default_model.strip()
        )
        return await self._run_image_operation(
            operation="image",
            requested_model=requested_model,
            fast_model=settings.media.image_generation_fast_model.strip(),
            pro_model=settings.media.image_generation_model.strip(),
            call_for_model=lambda model, timeout_ms: generate(
                prompt,
                replace(selected_options, model=model),
                client=client,
                timeout_ms=timeout_ms,
            ),
        )

    async def generate(
        self,
        prompt: str,
        options: ImageGenerationOptions | None = None,
    ) -> GeneratedImage:
        """Каноническая операция генерации изображения."""

        return await self.generate_image(prompt, options)

    async def edit_image(
        self,
        source_image: GeneratedImage | Sequence[GeneratedImage],
        instruction: str,
        options: ImageEditOptions | None = None,
    ) -> GeneratedImage:
        if not instruction or not instruction.strip():
            raise OpenRouterUnsupportedParameterError(
                "Инструкция редактирования пуста", reason="empty_prompt", operation="image_edit"
            )
        source_images = (
            [source_image]
            if isinstance(source_image, GeneratedImage)
            else list(source_image)
        )
        if not source_images or any(not image.content for image in source_images):
            raise OpenRouterUnsupportedParameterError(
                "Список исходных изображений пуст или содержит пустое изображение",
                reason="invalid_source_image",
                operation="image_edit",
            )
        from utils.openrouter_images import edit

        client = self._require_client()
        selected_options = options if options is not None else ImageEditOptions()
        if selected_options.resolution is None and settings.media.image_default_resolution is not None:
            selected_options = replace(
                selected_options,
                resolution=settings.media.image_default_resolution,
            )
        default_model = settings.media.image_edit_model
        if selected_options.model is not None and not isinstance(selected_options.model, str):
            raise OpenRouterUnsupportedParameterError(
                "Параметр model должен быть строкой",
                reason="unsupported_parameter",
                operation="image_edit",
            )
        requested_model = (
            selected_options.model.strip()
            if isinstance(selected_options.model, str) and selected_options.model.strip()
            else default_model.strip()
        )
        source_snapshot = list(source_images)
        return await self._run_image_operation(
            operation="image_edit",
            requested_model=requested_model,
            fast_model=settings.media.image_edit_fast_model.strip(),
            pro_model=settings.media.image_edit_model.strip(),
            call_for_model=lambda model, timeout_ms: edit(
                source_snapshot,
                instruction,
                replace(selected_options, model=model),
                client=client,
                timeout_ms=timeout_ms,
            ),
        )

    async def edit(
        self,
        source_image: GeneratedImage | Sequence[GeneratedImage],
        instruction: str,
        options: ImageEditOptions | None = None,
    ) -> GeneratedImage:
        """Каноническая операция редактирования изображения."""

        return await self.edit_image(source_image, instruction, options)

    async def generate_video(
        self,
        prompt: str,
        reference_image: GeneratedImage | Sequence[GeneratedImage] | None = None,
        options: VideoGenerationOptions | None = None,
    ) -> GeneratedVideo:
        if not prompt or not prompt.strip():
            raise OpenRouterUnsupportedParameterError(
                "Запрос на генерацию видео пуст", reason="empty_prompt", operation="video"
            )
        if reference_image is not None:
            reference_images = (
                [reference_image]
                if isinstance(reference_image, GeneratedImage)
                else list(reference_image)
            )
            if not reference_images or any(not image.content for image in reference_images):
                raise OpenRouterUnsupportedParameterError(
                    "Список исходных изображений пуст или содержит пустое изображение",
                    reason="invalid_source_image",
                    operation="video",
                )
            reference_image = reference_images
        from utils.openrouter_videos import generate

        client = self._require_client()
        selected_options = options or VideoGenerationOptions(
            timeout_seconds=self.config.video_timeout_seconds,
            polling_interval_seconds=self.config.video_polling_interval_seconds,
        )
        return await self._run(
            "video",
            lambda: generate(prompt, reference_image, selected_options, client=client),
        )

    @staticmethod
    def _translate_exception(exc: Exception, operation: str) -> OpenRouterMediaError:
        if isinstance(exc, OpenRouterMediaError):
            return exc

        status_code = getattr(exc, "status_code", None)
        if not isinstance(status_code, int):
            raw_response = getattr(exc, "raw_response", None)
            candidate_status = getattr(raw_response, "status_code", None)
            status_code = candidate_status if isinstance(candidate_status, int) else None
        provider_message = getattr(exc, "body", None)
        if provider_message is None:
            provider_message = getattr(exc, "message", None)

        name = type(exc).__name__.lower()
        message_text = getattr(exc, "message", None)
        provider_text = " ".join(
            str(value)
            for value in (provider_message, message_text, exc)
            if value is not None
        ).lower()

        content_policy = any(
            marker in provider_text
            for marker in (
                "content policy",
                "content_policy",
                "policy violation",
                "moderation",
                "safety",
                "unsafe",
                "blocked",
                "prohibited content",
                "policy violation",
            )
        )
        billing_markers = (
            "billing",
            "insufficient credit",
            "insufficient fund",
            "insufficient balance",
            "not enough funds",
            "not enough credit",
            "out of credit",
            "spending limit",
            "billing limit",
            "budget limit",
            "payment required",
            "credit balance",
        )
        billing_message = any(marker in provider_text for marker in billing_markers)
        if content_policy:
            return OpenRouterUnsupportedError(
                "Запрос отклонён фильтрами безопасности",
                status_code=status_code,
                reason="content_policy",
                provider_message=provider_message,
                operation=operation,
                billing_unknown=True,
            )

        if status_code == 402 or billing_message:
            spending_limit = any(
                marker in provider_text
                for marker in (
                    "limit",
                    "budget",
                    "credit balance",
                )
            )
            return OpenRouterBillingError(
                "OpenRouter сообщил об ограничении оплаты или расходов",
                status_code=status_code or 402,
                reason="spending_limit" if spending_limit else "insufficient_credits",
                provider_message=provider_message,
                operation=operation,
                billing_unknown=False,
            )

        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timeout" in name:
            return OpenRouterTimeoutError(
                "OpenRouter не ответил вовремя",
                status_code=status_code,
                provider_message=provider_message,
                operation=operation,
                billing_unknown=True,
            )

        if status_code in {401, 403} or "unauthorized" in name or "forbidden" in name:
            return OpenRouterAuthenticationError(
                "OpenRouter отклонил авторизацию",
                status_code=status_code,
                provider_message=provider_message,
                operation=operation,
                billing_unknown=True,
            )

        if status_code == 429 or "toomanyrequests" in name or "ratelimit" in name:
            return OpenRouterRateLimitError(
                "OpenRouter ограничил частоту запросов",
                status_code=status_code,
                provider_message=provider_message,
                operation=operation,
                billing_unknown=True,
            )

        if status_code == 404:
            model_unavailable = any(
                marker in provider_text
                for marker in (
                    "model",
                    "provider",
                    "no available",
                    "no endpoints",
                    "not found for",
                )
            )
            if model_unavailable:
                image_operation = operation in {"image", "image_edit"}
                return OpenRouterUnsupportedModelError(
                    "Выбранная модель или подходящий провайдер недоступны",
                    status_code=404,
                    reason="unsupported_model",
                    provider_message=provider_message,
                    operation=operation,
                    cost=0.0 if image_operation else None,
                    fallback_eligible=image_operation,
                    billing_unknown=not image_operation,
                )
            return OpenRouterUnsupportedError(
                "OpenRouter не нашёл запрошенный ресурс",
                status_code=status_code,
                reason="unsupported_model",
                provider_message=provider_message,
                operation=operation,
                billing_unknown=True,
            )

        if status_code in {400, 413, 422}:
            return OpenRouterUnsupportedParameterError(
                "OpenRouter отклонил параметры media-операции",
                status_code=status_code,
                reason="unsupported_parameter",
                provider_message=provider_message,
                operation=operation,
                billing_unknown=True,
            )

        if status_code == 502:
            if operation not in {"image", "image_edit"}:
                return OpenRouterTemporaryError(
                    "OpenRouter временно недоступен",
                    status_code=502,
                    provider_message=provider_message,
                    operation=operation,
                    billing_unknown=True,
                )
            if any(marker in provider_text for marker in ("timeout", "timed out", "connection reset")):
                return OpenRouterTemporaryError(
                    "OpenRouter не подтвердил завершение генерации",
                    status_code=502,
                    reason="ambiguous_provider_failure",
                    provider_message=provider_message,
                    operation=operation,
                    billing_unknown=True,
                )
            return OpenRouterTemporaryError(
                "OpenRouter сообщил о неуспешной генерации",
                status_code=502,
                reason="generation_failed",
                provider_message=provider_message,
                operation=operation,
                cost=0.0,
                fallback_eligible=True,
                billing_unknown=False,
            )

        if status_code is not None and status_code >= 500:
            return OpenRouterTemporaryError(
                "OpenRouter временно недоступен",
                status_code=status_code,
                provider_message=provider_message,
                operation=operation,
                billing_unknown=True,
            )

        if isinstance(exc, ValueError):
            attempt_metadata = getattr(exc, "attempt_metadata", None)
            model = getattr(exc, "model", None)
            if not isinstance(attempt_metadata, dict) or not model:
                return OpenRouterUnsupportedParameterError(
                    "Параметры media-запроса некорректны или не поддерживаются",
                    status_code=status_code,
                    reason="unsupported_parameter",
                    provider_message=provider_message,
                    operation=operation,
                    billing_unknown=False,
                )
            cost = getattr(exc, "cost", None)
            return OpenRouterResponseError(
                "OpenRouter вернул некорректный media-результат",
                status_code=status_code,
                provider_message=provider_message,
                operation=operation,
                model=model,
                requested_model=getattr(exc, "requested_model", None),
                cost=cost,
                attempt_metadata=attempt_metadata,
                billing_unknown=cost is None,
            )

        if any(
            marker in name
            for marker in (
                "connectionerror",
                "connecterror",
                "readerror",
                "writeerror",
                "remoteprotocolerror",
                "pooltimeout",
            )
        ) or isinstance(exc, (ConnectionError, OSError)):
            return OpenRouterConnectionError(
                "Соединение с OpenRouter прервалось до подтверждения результата",
                status_code=status_code,
                provider_message=provider_message,
                operation=operation,
                billing_unknown=True,
            )

        if isinstance(exc, sdk_errors.OpenRouterError):
            return OpenRouterTemporaryError(
                "OpenRouter SDK временно недоступен",
                status_code=status_code,
                provider_message=provider_message,
                operation=operation,
                billing_unknown=True,
            )

        return OpenRouterSdkUnavailableError(
            "OpenRouter SDK временно недоступен",
            status_code=status_code,
            provider_message=provider_message,
            operation=operation,
            billing_unknown=True,
        )
