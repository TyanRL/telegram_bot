import logging
import os
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)

image_model_name="black-forest-labs/flux.2-max"
openrouter_base_url="https://openrouter.ai/api/v1"



class OpenRouterImageError(Exception):
    """Базовый класс ошибок генерации изображений через OpenRouter."""


class OpenRouterConfigError(OpenRouterImageError):
    """Ошибка конфигурации OpenRouter (переменные окружения, модель и т.п.)."""






def _get_headers() -> Dict[str, str]:
    """Сформировать заголовки для запроса к OpenRouter."""

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise OpenRouterConfigError("Переменная окружения OPENROUTER_API_KEY не задана или пуста")

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Необязательные рекомендованные заголовки OpenRouter
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
    """Центральная точка входа для генерации изображений через OpenRouter.

    Выполняет HTTP-запрос к эндпоинту /chat/completions с параметрами:
      - model
      - messages: [{"role": "user", "content": prompt}]
      - modalities: ["image", "text"]

    Возвращает список data-URL строк с изображениями
    (например, "data:image/png;base64,....").

    В случае ошибок выбрасывает OpenRouterImageError / OpenRouterConfigError.
    """


    if not prompt:
        logger.info("Пустой prompt для генерации изображения через OpenRouter")
        return []

    try:
        url = openrouter_base_url
        headers = _get_headers()
    except OpenRouterConfigError as e:
        logger.error(f"Ошибка конфигурации OpenRouter: {e}")
        raise

    if model is None:
        try:
            model = image_model_name
        except Exception as e:  # подстраховка, чтобы не падать без логгирования
            logger.error(
                "Не удалось определить модель OpenRouter для генерации изображений: %s",
                e,
                exc_info=True,
            )
            raise OpenRouterConfigError("Не указана модель OpenRouter для генерации изображений") from e

    messages = kwargs.pop("messages", None)
    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "modalities": ["image"],
    }
    payload.update(kwargs)

    logger.info(f"Image model: {model}")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        logger.error("Ошибка HTTP при обращении к OpenRouter: %s", e, exc_info=True)
        raise OpenRouterImageError("Ошибка HTTP при обращении к OpenRouter") from e

    if response.status_code < 200 or response.status_code >= 300:
        logger.error(
            "Неверный HTTP-статус от OpenRouter: %s - %s",
            response.status_code,
            response.text,
        )
        raise OpenRouterImageError(f"Неверный HTTP-статус от OpenRouter: {response.status_code}")

    try:
        data = response.json()
    except ValueError as e:
        logger.error("Не удалось распарсить JSON-ответ OpenRouter: %s", e, exc_info=True)
        raise OpenRouterImageError("Некорректный JSON-ответ от OpenRouter") from e

    try:
        choices = data["choices"]
    except (KeyError, TypeError):
        logger.error("Ответ OpenRouter не содержит поля 'choices'")
        raise OpenRouterImageError("Ответ OpenRouter не содержит ожидаемого поля 'choices'")

    if not choices:
        logger.error("Ответ OpenRouter не содержит элементов в 'choices'")
        raise OpenRouterImageError("Ответ OpenRouter не содержит изображений")

    first_choice = choices[0] if isinstance(choices, list) else None
    if not isinstance(first_choice, dict):
        logger.error("Неожиданная структура первого элемента 'choices' в ответе OpenRouter")
        raise OpenRouterImageError("Неожиданная структура ответа OpenRouter (choices[0])")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        logger.error("Ответ OpenRouter не содержит корректного поля 'message' в первом choice")
        raise OpenRouterImageError("Ответ OpenRouter не содержит данных сообщения с изображением")

    images = message.get("images")
    if not images:
        logger.error("Ответ OpenRouter не содержит поля 'images'")
        raise OpenRouterImageError("Ответ OpenRouter не содержит изображения")

    data_urls: List[str] = []
    for img in images:
        if not isinstance(img, dict):
            continue
        img_url = img.get("image_url")
        if isinstance(img_url, dict):
            url_value = img_url.get("url")
            if isinstance(url_value, str) and url_value:
                data_urls.append(url_value)

    if not data_urls:
        logger.error("Ответ OpenRouter не содержит валидных ссылок на изображения")
        raise OpenRouterImageError("Ответ OpenRouter не содержит валидных изображений")

    return data_urls


def generate_image_openrouter_base64(
    prompt: str,
    model: Optional[str] = None,
    **kwargs: Any,
) -> List[str]:
    """Сгенерировать изображения через OpenRouter и вернуть чистые base64-строки.

    Обёртка над generate_image_openrouter, которая из data-URL вида
    "data:image/png;base64,..." извлекает только base64-часть.
    """

    data_urls = generate_image_openrouter(prompt, model=model, **kwargs)

    result: List[str] = []
    for url in data_urls:
        if not isinstance(url, str):
            continue
        if url.startswith("data:"):
            _, _, b64 = url.partition(",")
            if b64:
                result.append(b64)
        else:
            # Если по какой-то причине пришла не data-URL, считаем, что это уже base64
            result.append(url)

    return result
