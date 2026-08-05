import logging
from dataclasses import dataclass
from enum import Enum

from utils.openrouter_client import (
    OpenRouterAuthenticationError,
    OpenRouterConfigurationError,
    OpenRouterMediaError,
    OpenRouterRateLimitError,
    OpenRouterSdkUnavailableError,
    OpenRouterTemporaryError,
    OpenRouterTimeoutError,
    OpenRouterUnsupportedError,
    OpenRouterVideoCancelledError,
    OpenRouterVideoExpiredError,
    OpenRouterVideoFailedError,
)

logger = logging.getLogger(__name__)


class GenerationFailureReason(str, Enum):
    CONTENT_POLICY = "content_policy"
    EMPTY_PROMPT = "empty_prompt"
    INVALID_SOURCE_IMAGE = "invalid_source_image"
    PROMPT_TOO_COMPLEX = "prompt_too_complex"
    GENERATION_TIMEOUT = "generation_timeout"
    RATE_LIMITED = "rate_limited"
    TEMPORARY_PROVIDER_ERROR = "temporary_provider_error"
    DELIVERY_ERROR = "delivery_error"
    INCORRECT_RESOLUTION = "resolution_error"
    CONFIGURATION_ERROR = "configuration_error"
    AUTHENTICATION_ERROR = "authentication_error"
    UNSUPPORTED_MODEL = "unsupported_model"
    UNSUPPORTED_PARAMETER = "unsupported_parameter"
    VIDEO_FAILED = "video_failed"
    VIDEO_CANCELLED = "video_cancelled"
    VIDEO_EXPIRED = "video_expired"
    SDK_UNAVAILABLE = "sdk_unavailable"
    INTERNAL_ERROR = "internal_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class UserFacingGenerationError:
    reason: GenerationFailureReason
    user_message: str
    is_user_actionable: bool = True


def _reason_from_openrouter_error(
    error: OpenRouterMediaError,
) -> GenerationFailureReason:
    reason = error.reason
    explicit_reasons = {
        "empty_prompt": GenerationFailureReason.EMPTY_PROMPT,
        "invalid_source_image": GenerationFailureReason.INVALID_SOURCE_IMAGE,
        "content_policy": GenerationFailureReason.CONTENT_POLICY,
        "unsupported_model": GenerationFailureReason.UNSUPPORTED_MODEL,
        "unsupported_parameter": GenerationFailureReason.UNSUPPORTED_PARAMETER,
    }
    if reason in explicit_reasons:
        return explicit_reasons[reason]
    if isinstance(error, OpenRouterConfigurationError):
        return GenerationFailureReason.CONFIGURATION_ERROR
    if isinstance(error, OpenRouterAuthenticationError):
        return GenerationFailureReason.AUTHENTICATION_ERROR
    if isinstance(error, OpenRouterRateLimitError):
        return GenerationFailureReason.RATE_LIMITED
    if isinstance(error, OpenRouterTimeoutError):
        return GenerationFailureReason.GENERATION_TIMEOUT
    if isinstance(error, OpenRouterVideoFailedError):
        return GenerationFailureReason.VIDEO_FAILED
    if isinstance(error, OpenRouterVideoCancelledError):
        return GenerationFailureReason.VIDEO_CANCELLED
    if isinstance(error, OpenRouterVideoExpiredError):
        return GenerationFailureReason.VIDEO_EXPIRED
    if isinstance(error, OpenRouterSdkUnavailableError):
        return GenerationFailureReason.SDK_UNAVAILABLE
    if isinstance(error, OpenRouterTemporaryError):
        return GenerationFailureReason.TEMPORARY_PROVIDER_ERROR
    if isinstance(error, OpenRouterUnsupportedError):
        return GenerationFailureReason.UNSUPPORTED_PARAMETER
    if error.status_code == 429:
        return GenerationFailureReason.RATE_LIMITED
    if error.status_code in {401, 403}:
        return GenerationFailureReason.AUTHENTICATION_ERROR
    if error.status_code and 500 <= error.status_code < 600:
        return GenerationFailureReason.TEMPORARY_PROVIDER_ERROR
    return GenerationFailureReason.UNKNOWN_ERROR


def _reason_from_legacy_shape(exc: Exception) -> GenerationFailureReason:
    """Безопасный fallback для ошибок вне адаптера SDK."""

    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return GenerationFailureReason.RATE_LIMITED
    if status_code in {401, 403}:
        return GenerationFailureReason.AUTHENTICATION_ERROR
    if isinstance(status_code, int) and 500 <= status_code < 600:
        return GenerationFailureReason.TEMPORARY_PROVIDER_ERROR

    text = str(exc).lower()
    if "timeout" in text or "timed out" in text:
        return GenerationFailureReason.GENERATION_TIMEOUT
    if any(keyword in text for keyword in ("policy", "moderation", "safety", "blocked")):
        return GenerationFailureReason.CONTENT_POLICY
    if "resolution" in text:
        return GenerationFailureReason.INCORRECT_RESOLUTION
    if "unsupported" in text or "invalid parameter" in text:
        return GenerationFailureReason.UNSUPPORTED_PARAMETER
    return GenerationFailureReason.UNKNOWN_ERROR


def map_generation_error(
    exc: Exception,
    context: str = "image",
) -> UserFacingGenerationError:
    """Преобразует техническую ошибку media SDK в безопасный текст."""

    reason = (
        _reason_from_openrouter_error(exc)
        if isinstance(exc, OpenRouterMediaError)
        else _reason_from_legacy_shape(exc)
    )
    logger.info(
        "Mapping %s generation error: type=%s reason=%s status=%s",
        context,
        type(exc).__name__,
        reason.value,
        getattr(exc, "status_code", None),
    )

    messages = {
        GenerationFailureReason.CONTENT_POLICY: (
            "Не удалось сгенерировать результат: запрос отклонён фильтрами безопасности. "
            "Попробуйте переформулировать его."
        ),
        GenerationFailureReason.EMPTY_PROMPT: "Не удалось выполнить запрос: описание не задано.",
        GenerationFailureReason.INVALID_SOURCE_IMAGE: (
            "Не удалось использовать исходное изображение. Отправьте изображение ещё раз."
        ),
        GenerationFailureReason.CONFIGURATION_ERROR: (
            "Media-сервис не настроен. Обратитесь к администратору."
        ),
        GenerationFailureReason.AUTHENTICATION_ERROR: (
            "Media-сервис не прошёл авторизацию. Обратитесь к администратору."
        ),
        GenerationFailureReason.UNSUPPORTED_MODEL: (
            "Выбранная модель не поддерживает эту операцию. Попробуйте позже."
        ),
        GenerationFailureReason.UNSUPPORTED_PARAMETER: (
            "Параметры запроса не поддерживаются выбранным media-сервисом."
        ),
        GenerationFailureReason.RATE_LIMITED: (
            "Сервис генерации сейчас перегружен. Попробуйте повторить запрос чуть позже."
        ),
        GenerationFailureReason.GENERATION_TIMEOUT: (
            "Генерация заняла слишком много времени. Попробуйте повторить позже."
        ),
        GenerationFailureReason.VIDEO_FAILED: "Сервис не смог сгенерировать видео. Попробуйте изменить запрос.",
        GenerationFailureReason.VIDEO_CANCELLED: "Генерация видео была отменена провайдером.",
        GenerationFailureReason.VIDEO_EXPIRED: "Задача генерации видео истекла. Попробуйте ещё раз.",
        GenerationFailureReason.SDK_UNAVAILABLE: "Сервис генерации временно недоступен. Попробуйте позже.",
        GenerationFailureReason.TEMPORARY_PROVIDER_ERROR: "Сервис генерации временно недоступен. Попробуйте позже.",
        GenerationFailureReason.PROMPT_TOO_COMPLEX: "Попробуйте сделать описание короче и проще.",
        GenerationFailureReason.INCORRECT_RESOLUTION: "Указано неподдерживаемое разрешение изображения.",
        GenerationFailureReason.DELIVERY_ERROR: "Результат создан, но не удалось отправить его в Telegram.",
        GenerationFailureReason.INTERNAL_ERROR: "Произошла внутренняя ошибка при генерации.",
        GenerationFailureReason.UNKNOWN_ERROR: "Произошла внутренняя ошибка при генерации. Попробуйте позже.",
    }
    return UserFacingGenerationError(
        reason=reason,
        user_message=messages[reason],
        is_user_actionable=reason
        not in {
            GenerationFailureReason.CONFIGURATION_ERROR,
            GenerationFailureReason.AUTHENTICATION_ERROR,
            GenerationFailureReason.TEMPORARY_PROVIDER_ERROR,
            GenerationFailureReason.SDK_UNAVAILABLE,
        },
    )
