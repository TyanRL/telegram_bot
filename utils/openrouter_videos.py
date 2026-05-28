import asyncio
import logging
import tempfile
from typing import Any, Dict, List, Optional

import httpx

from utils.openrouter_images import _get_headers

logger = logging.getLogger(__name__)

DEFAULT_VIDEO_MODEL = "google/veo-3.1-fast"
OPENROUTER_VIDEOS_URL = "https://openrouter.ai/api/v1/videos"


class OpenRouterVideoError(Exception):
    def __init__(self, message: str, status_code: int | None = None, provider_message: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.provider_message = provider_message


class OpenRouterVideoConfigError(OpenRouterVideoError):
    pass


async def submit_video_generation(
    prompt: str,
    input_references: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
    resolution: str = "480p",
) -> tuple[str, str]:
    """Отправляет запрос на генерацию видео в OpenRouter.

    Returns:
        (job_id, polling_url)
    """
    if not prompt:
        raise OpenRouterVideoError("Пустой prompt для генерации видео")

    headers = _get_headers()
    model = model or DEFAULT_VIDEO_MODEL

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "resolution": resolution,
    }
    if input_references:
        payload["input_references"] = input_references

    logger.info(f"Video model: {model}, resolution: {resolution}")
    logger.info(f"POST {OPENROUTER_VIDEOS_URL}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                OPENROUTER_VIDEOS_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
    except httpx.RequestError as e:
        logger.error(f"Ошибка HTTP при отправке запроса на генерацию видео: {e}", exc_info=True)
        raise OpenRouterVideoError("Ошибка HTTP при обращении к OpenRouter для генерации видео") from e

    response_preview = response.text[:1000]
    logger.info(
        f"OpenRouter video submit response: status={response.status_code} body_preview={response_preview!r}",
    )

    if not response.is_success:
        raise OpenRouterVideoError(
            f"OpenRouter вернул HTTP {response.status_code}: {response_preview}",
            status_code=response.status_code,
            provider_message=response_preview,
        )

    try:
        data = response.json()
    except ValueError as e:
        raise OpenRouterVideoError(
            f"Некорректный JSON-ответ от OpenRouter при создании видео "
            f"(status={response.status_code}, body={response_preview!r})"
        ) from e

    job_id = data.get("id")
    polling_url = data.get("polling_url")
    if not job_id or not polling_url:
        raise OpenRouterVideoError(f"Ответ OpenRouter не содержит id или polling_url: {data!r}")

    logger.info(f"Video job submitted: {job_id}, polling_url={polling_url}")
    return str(job_id), str(polling_url)


async def poll_video_generation(
    polling_url: str,
    timeout: int = 180,
    interval: int = 5,
) -> List[str]:
    """Опрашивает статус генерации видео.

    Returns:
        Список URL готовых видео (unsigned_urls).

    Raises:
        OpenRouterVideoError: при таймауте или ошибке генерации.
    """
    headers = _get_headers()
    start_time = asyncio.get_event_loop().time()

    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            raise OpenRouterVideoError(f"Таймаут ожидания генерации видео ({timeout} сек)")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    polling_url,
                    headers=headers,
                    timeout=30,
                )
        except httpx.RequestError as e:
            logger.error(f"Ошибка HTTP при опросе статуса видео: {e}", exc_info=True)
            raise OpenRouterVideoError("Ошибка HTTP при опросе статуса видео") from e

        if not response.is_success:
            response_preview = response.text[:1000]
            raise OpenRouterVideoError(
                f"OpenRouter вернул HTTP {response.status_code} при опросе: {response_preview}",
                status_code=response.status_code,
                provider_message=response_preview,
            )

        try:
            data = response.json()
        except ValueError as e:
            response_preview = response.text[:1000]
            raise OpenRouterVideoError(
                f"Некорректный JSON при опросе статуса видео: {response_preview!r}"
            ) from e

        status = data.get("status")
        logger.info(f"Video poll status: {status}")

        if status == "completed":
            unsigned_urls = data.get("unsigned_urls", [])
            if not unsigned_urls:
                raise OpenRouterVideoError(f"Видео готово, но URL отсутствуют: {data!r}")
            logger.info(f"Video generation completed, urls={unsigned_urls}")
            return unsigned_urls

        if status == "failed":
            error_msg = data.get("error", "Unknown error")
            raise OpenRouterVideoError(f"Генерация видео завершилась с ошибкой: {error_msg}")

        await asyncio.sleep(interval)


async def download_video(url: str) -> str:
    """Скачивает видео по URL во временный файл.

    Returns:
        Путь к сохранённому временному файлу.
    """
    headers = _get_headers()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=120)
            response.raise_for_status()
    except httpx.RequestError as e:
        logger.error(f"Ошибка при скачивании видео: {e}", exc_info=True)
        raise OpenRouterVideoError("Ошибка при скачивании видео") from e

    suffix = ".mp4"
    content_type = response.headers.get("Content-Type", "")
    if "webm" in content_type:
        suffix = ".webm"
    elif "mov" in content_type:
        suffix = ".mov"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        async for chunk in response.aiter_bytes(chunk_size=8192):
            if chunk:
                tmp_file.write(chunk)
        tmp_path = tmp_file.name

    logger.info(f"Video downloaded to {tmp_path}")
    return tmp_path
