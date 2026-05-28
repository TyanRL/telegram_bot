from dataclasses import dataclass
from enum import Enum
import logging

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
    INTERNAL_ERROR = "internal_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class UserFacingGenerationError:
    reason: GenerationFailureReason
    user_message: str
    is_user_actionable: bool = True

def _get_reason_from_text(text: str, status_code: int | None = None) -> GenerationFailureReason:
    text_lower = text.lower()
    
    if status_code == 429:
        return GenerationFailureReason.RATE_LIMITED
    
    # Расширенный список ключевых слов для content policy
    content_policy_keywords = [
        "policy", "safety", "moderation", "request moderated", "content violation",
        "censored", "nsfw", "rejected", "blocked", "provider returned error",
        "denied", "flagged", "sexual", "explicit", "inappropriate", "offensive",
        "adult content", "nudity", "violence"
    ]
    if any(keyword in text_lower for keyword in content_policy_keywords):
        return GenerationFailureReason.CONTENT_POLICY
    
    if any(keyword in text_lower for keyword in ["timeout", "timed out"]):
        return GenerationFailureReason.GENERATION_TIMEOUT
        
    if any(keyword in text_lower for keyword in ["too complex", "invalid prompt", "unsupported"]):
        return GenerationFailureReason.PROMPT_TOO_COMPLEX
    
    if any(keyword in text_lower for keyword in ["resolution"]):
            return GenerationFailureReason.INCORRECT_RESOLUTION
        
    if status_code and 500 <= status_code < 600:
        return GenerationFailureReason.TEMPORARY_PROVIDER_ERROR
    
    # Если status_code 400 и есть provider_message, скорее всего это content policy
    if status_code == 400:
        return GenerationFailureReason.CONTENT_POLICY
        
    return GenerationFailureReason.UNKNOWN_ERROR

def map_generation_error(exc: Exception, context: str = "image") -> UserFacingGenerationError:
    logger.info(f"Mapping {context} error: {exc}")
    
    # Defaults
    reason = GenerationFailureReason.UNKNOWN_ERROR
    user_message = "Произошла внутренняя ошибка при генерации. Попробуйте еще раз позже."
    
    # Try to extract info if possible (e.g. from custom exceptions)
    status_code = getattr(exc, "status_code", None)
    error_text = str(exc)
    
    reason = _get_reason_from_text(error_text, status_code)
    
    if reason == GenerationFailureReason.CONTENT_POLICY:
        user_message = "Не удалось сгенерировать результат: запрос отклонён фильтрами безопасности. Попробуйте переформулировать его, избегая запрещённого, откровенного или чрезмерно жестокого контента."
    elif reason == GenerationFailureReason.RATE_LIMITED:
        user_message = "Сервис генерации сейчас перегружен. Попробуйте повторить запрос чуть позже."
    elif reason == GenerationFailureReason.GENERATION_TIMEOUT:
        user_message = "Генерация заняла слишком много времени. Попробуйте упростить запрос или повторить позже."
    elif reason == GenerationFailureReason.PROMPT_TOO_COMPLEX:
        user_message = "Не удалось сгенерировать результат по текущему описанию. Попробуйте сделать запрос короче и проще."
    elif reason == GenerationFailureReason.TEMPORARY_PROVIDER_ERROR:
        user_message = "Сервис генерации временно недоступен. Попробуйте позже."
    elif reason == GenerationFailureReason.INCORRECT_RESOLUTION:
        user_message = "Произошла внутренняя ошибка при генерации. Неверно задано разрешение. Обратитесь к администратору."
        
    return UserFacingGenerationError(reason=reason, user_message=user_message)
