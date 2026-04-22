import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

image_model_name = "black-forest-labs/flux.2-max"
openrouter_base_url = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterImageError(Exception):
    pass


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


def generate_image_openrouter(
    prompt: str,
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
        messages = [{"role": "user", "content": prompt}]

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "modalities": ["image"],   # для Flux это правильно
    }
    payload.update(kwargs)

    logger.info("Image model: %s", model)
    logger.info("POST %s", openrouter_base_url)

    try:
        response = requests.post(
            openrouter_base_url,
            headers=headers,
            json=payload,
            timeout=120,
        )
    except requests.RequestException as e:
        logger.error("Ошибка HTTP при обращении к OpenRouter: %s", e, exc_info=True)
        raise OpenRouterImageError("Ошибка HTTP при обращении к OpenRouter") from e

    content_type = response.headers.get("Content-Type", "")
    response_preview = response.text[:1000]

    logger.info(
        "OpenRouter response: status=%s content_type=%s body_preview=%r",
        response.status_code,
        content_type,
        response_preview,
    )

    if not response.ok:
        raise OpenRouterImageError(
            f"OpenRouter вернул HTTP {response.status_code}: {response_preview}"
        )

    try:
        data = response.json()
    except ValueError as e:
        raise OpenRouterImageError(
            f"Некорректный JSON-ответ от OpenRouter "
            f"(status={response.status_code}, content_type={content_type}, body={response_preview!r})"
        ) from e

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