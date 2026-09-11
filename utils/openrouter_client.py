"""Единый lifecycle и адаптер ошибок OpenRouter Media API."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

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
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason or self.default_reason
        self.operation = operation
        self.provider_message = _safe_provider_message(provider_message)
        self.safe_provider_message = self.provider_message


class OpenRouterConfigurationError(OpenRouterMediaError):
    default_reason = "configuration"


class OpenRouterAuthenticationError(OpenRouterMediaError):
    default_reason = "authentication"


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
    speech_model: str = settings.openrouter.speech_model
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
            speech_model=app_settings.openrouter.speech_model,
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

    async def transcribe_audio(self, audio_filename: str) -> str:
        """Распознаёт Telegram OGG/Opus через Qwen3 ASR в OpenRouter."""
        client = self._require_client()

        async def transcribe() -> str:
            # STT принимает исходные байты в base64, без префикса data URI.
            # Чтение и кодирование файла выносим из цикла обработки Telegram.
            def encode_audio() -> str:
                content = Path(audio_filename).read_bytes()
                if not content:
                    raise ValueError("Аудиофайл пуст")
                return base64.b64encode(content).decode("ascii")

            audio_data = await asyncio.to_thread(encode_audio)
            response = await client.stt.create_transcription_async(
                model=self.config.speech_model,
                input_audio={"data": audio_data, "format_": "ogg"},
                response_format="json",
            )
            # Язык определяется автоматически; пустой текст обрабатывает handler.
            if not isinstance(response.text, str):
                raise ValueError("Ответ STT не содержит текст")
            return response.text.strip()

        # Сохраняем единое преобразование таймаутов и ошибок провайдера.
        return await self._run("transcription", transcribe)

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
        return await self._run(
            "image",
            lambda: generate(prompt, options, client=client),
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
        source_image: GeneratedImage,
        instruction: str,
        options: ImageEditOptions | None = None,
    ) -> GeneratedImage:
        if not instruction or not instruction.strip():
            raise OpenRouterUnsupportedParameterError(
                "Инструкция редактирования пуста", reason="empty_prompt", operation="image_edit"
            )
        if not source_image.content:
            raise OpenRouterUnsupportedParameterError(
                "Исходное изображение пусто",
                reason="invalid_source_image",
                operation="image_edit",
            )
        from utils.openrouter_images import edit

        client = self._require_client()
        return await self._run(
            "image_edit",
            lambda: edit(source_image, instruction, options, client=client),
        )

    async def edit(
        self,
        source_image: GeneratedImage,
        instruction: str,
        options: ImageEditOptions | None = None,
    ) -> GeneratedImage:
        """Каноническая операция редактирования изображения."""

        return await self.edit_image(source_image, instruction, options)

    async def generate_video(
        self,
        prompt: str,
        reference_image: GeneratedImage | None = None,
        options: VideoGenerationOptions | None = None,
    ) -> GeneratedVideo:
        if not prompt or not prompt.strip():
            raise OpenRouterUnsupportedParameterError(
                "Запрос на генерацию видео пуст", reason="empty_prompt", operation="video"
            )
        if reference_image is not None and not reference_image.content:
            raise OpenRouterUnsupportedParameterError(
                "Исходное изображение пусто",
                reason="invalid_source_image",
                operation="video",
            )
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
            status_code = None
        provider_message = getattr(exc, "body", None)
        if provider_message is None:
            provider_message = getattr(exc, "message", None)

        name = type(exc).__name__.lower()
        provider_text = str(provider_message or "").lower()

        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timeout" in name:
            return OpenRouterTimeoutError(
                "OpenRouter не ответил вовремя",
                status_code=status_code,
                provider_message=provider_message,
                operation=operation,
            )

        if status_code in {401, 403} or "unauthorized" in name or "forbidden" in name:
            return OpenRouterAuthenticationError(
                "OpenRouter отклонил авторизацию",
                status_code=status_code,
                provider_message=provider_message,
                operation=operation,
            )

        if status_code == 429 or "toomanyrequests" in name or "ratelimit" in name:
            return OpenRouterRateLimitError(
                "OpenRouter ограничил частоту запросов",
                status_code=status_code,
                provider_message=provider_message,
                operation=operation,
            )

        if status_code in {400, 404, 413, 422}:
            if any(
                keyword in provider_text
                for keyword in ("policy", "moderation", "safety", "blocked", "rejected")
            ):
                reason = "content_policy"
            elif status_code == 404 or "model" in provider_text:
                reason = "unsupported_model"
            else:
                reason = "unsupported_parameter"
            return OpenRouterUnsupportedError(
                "OpenRouter отклонил параметры media-операции",
                status_code=status_code,
                reason=reason,
                provider_message=provider_message,
                operation=operation,
            )

        if status_code is not None and status_code >= 500:
            return OpenRouterTemporaryError(
                "OpenRouter временно недоступен",
                status_code=status_code,
                provider_message=provider_message,
                operation=operation,
            )

        if isinstance(exc, ValueError):
            return OpenRouterResponseError(
                "OpenRouter вернул некорректный media-результат",
                status_code=status_code,
                provider_message=provider_message,
                operation=operation,
            )

        if isinstance(exc, sdk_errors.OpenRouterError):
            return OpenRouterTemporaryError(
                "OpenRouter SDK временно недоступен",
                status_code=status_code,
                provider_message=provider_message,
                operation=operation,
            )

        return OpenRouterSdkUnavailableError(
            "OpenRouter SDK временно недоступен",
            status_code=status_code,
            provider_message=provider_message,
            operation=operation,
        )
