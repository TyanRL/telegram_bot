import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

image_model_name = "black-forest-labs/flux.2-max"
openrouter_base_url = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterImageError(Exception):
    def __init__(self, message: str, status_code: int | None = None, provider_message: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.provider_message = provider_message


class OpenRouterConfigError(OpenRouterImageError):
    pass


def _get_headers() -> Dict[str, str]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise OpenRouterConfigError("Переменная окружения OPENROUTER_API_KEY не задана или пуста")

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    site_url = os.getenv("OPENROUTER_SITE_URL")
    site_name = os.getenv("OPENROUTER_SITE_NAME")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name

    return headers


def _validate_data_image_url(url: str) -> None:
    if not isinstance(url, str) or not url:
        raise OpenRouterImageError("Изображение должно быть непустой строкой")
    if not url.startswith("data:image/") or ";base64," not in url:
        raise OpenRouterImageError(
            "Поддерживаются только data:image/...;base64,... URL"
        )


async def generate_image_openrouter(
    prompt: str,
    input_images: Optional[List[str]] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> List[str]:
    if not prompt:
        logger.info("Пустой prompt для генерации изображения через OpenRouter")
        return []

    headers = _get_headers()
    model = model or image_model_name

    messages = kwargs.pop("messages", None)
    if messages is None:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_data_url in input_images or []:
            _validate_data_image_url(image_data_url)
            content.append(
                {"type": "image_url", "image_url": {"url": image_data_url}}
            )
        messages = [{"role": "user", "content": content}]

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "modalities": ["image"],   # для Flux это правильно
    }
    payload.update(kwargs)

    logger.info(f"Image model: {model}")
    logger.info(f"POST {openrouter_base_url}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                openrouter_base_url,
                headers=headers,
                json=payload,
                timeout=120,
            )
    except httpx.RequestError as e:
        logger.error(f"Ошибка HTTP при обращении к OpenRouter: {e}", exc_info=True)
        raise OpenRouterImageError("Ошибка HTTP при обращении к OpenRouter") from e

    content_type = response.headers.get("Content-Type", "")
    response_preview = response.text[:1000]

    logger.info(
        f"OpenRouter response: status={response.status_code} content_type={content_type} body_preview={response_preview!r}",
    )

    if not response.is_success:
        logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
        raise OpenRouterImageError(
            f"OpenRouter вернул HTTP {response.status_code}: {response.text}",
            status_code=response.status_code,
            provider_message=response.text,
        )

    try:
        data = response.json()
    except ValueError as e:
        raise OpenRouterImageError(
            f"Некорректный JSON-ответ от OpenRouter "
            f"(status={response.status_code}, content_type={content_type}, body={response_preview!r})"
        ) from e

    # Проверяем наличие ошибки в JSON даже при HTTP 200
    error_data = data.get("error")
    if error_data:
        error_message = error_data.get("message", "Неизвестная ошибка провайдера")
        error_code = error_data.get("code")
        logger.error(f"OpenRouter вернул ошибку в JSON: {error_data}")
        raise OpenRouterImageError(
            f"OpenRouter вернул ошибку: {error_message}",
            status_code=error_code if isinstance(error_code, int) else None,
            provider_message=error_message,
        )

    choices = data.get("choices")
    if not choices:
        raise OpenRouterImageError(f"Ответ OpenRouter не содержит choices: {data!r}")

    message = choices[0].get("message", {})
    images = message.get("images")
    if not images:
        raise OpenRouterImageError(f"Ответ OpenRouter не содержит images: {data!r}")

    data_urls: List[str] = []
    for img in images:
        image_url = img.get("image_url", {})
        url_value = image_url.get("url")
        if isinstance(url_value, str) and url_value:
            data_urls.append(url_value)

    if not data_urls:
        raise OpenRouterImageError(f"Не удалось извлечь image_url.url: {data!r}")

    return data_urls
