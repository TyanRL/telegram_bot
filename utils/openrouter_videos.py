"""OpenRouter Video Generation API adapter."""

from __future__ import annotations

import asyncio
import base64
import tempfile
from pathlib import Path
from typing import Any, cast

from openrouter import OpenRouter
from openrouter.components.contentpartimage import ContentPartImageTypedDict
from openrouter.components.videogenerationrequest import (
    VideoGenerationRequestAspectRatio,
    VideoGenerationRequestResolution,
)

from core.common_types import GeneratedImage, GeneratedVideo, VideoGenerationOptions
from utils.openrouter_client import (
    OpenRouterResponseError,
    OpenRouterTimeoutError,
    OpenRouterVideoCancelledError,
    OpenRouterVideoExpiredError,
    OpenRouterVideoFailedError,
)
from core.config import settings

DEFAULT_VIDEO_MODEL = settings.media.video_model
TERMINAL_VIDEO_STATUSES = {"failed", "cancelled", "expired"}


def _status_value(status: object) -> str:
    return str(getattr(status, "value", status))


def _image_to_data_url(image: GeneratedImage) -> str:
    encoded = base64.b64encode(image.content).decode("ascii")
    return f"data:{image.mime_type};base64,{encoded}"


def _video_suffix(response: Any) -> str:
    headers = getattr(response, "headers", {}) or {}
    content_type = str(headers.get("content-type", headers.get("Content-Type", ""))).lower()
    if "webm" in content_type:
        return ".webm"
    if "quicktime" in content_type or "mov" in content_type:
        return ".mov"
    return ".mp4"


def _write_video_file(content: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
        temporary_file.write(content)
        return Path(temporary_file.name)


async def generate(
    prompt: str,
    reference_image: GeneratedImage | None = None,
    options: VideoGenerationOptions | None = None,
    *,
    client: OpenRouter,
) -> GeneratedVideo:
    """Генерирует видео, ожидает готовность и возвращает временный файл."""

    selected_options = options or VideoGenerationOptions()
    input_references: list[ContentPartImageTypedDict] | None = None
    if reference_image is not None:
        input_references = [
            cast(
                ContentPartImageTypedDict,
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_url(reference_image)},
                },
            )
        ]

    job = await client.video_generation.generate_async(
        model=selected_options.model or settings.media.video_model,
        prompt=prompt,
        resolution=cast(
            VideoGenerationRequestResolution | None,
            selected_options.resolution,
        ),
        aspect_ratio=cast(
            VideoGenerationRequestAspectRatio | None,
            selected_options.aspect_ratio,
        ),
        duration=selected_options.duration,
        generate_audio=selected_options.generate_audio,
        seed=selected_options.seed,
        size=selected_options.size,
        input_references=input_references,
    )
    job_id = getattr(job, "id", None)
    if not job_id:
        raise OpenRouterResponseError(
            "OpenRouter не вернул идентификатор видео-задачи", operation="video"
        )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + selected_options.timeout_seconds
    while True:
        if loop.time() >= deadline:
            raise OpenRouterTimeoutError(
                "Генерация видео не завершилась вовремя", operation="video"
            )

        result = await client.video_generation.get_generation_async(job_id=str(job_id))
        status = _status_value(getattr(result, "status", None))

        if status == "completed":
            break

        if status == "failed":
            raise OpenRouterVideoFailedError(
                "Генерация видео завершилась с ошибкой",
                provider_message=getattr(result, "error", None),
                operation="video",
            )
        if status == "cancelled":
            raise OpenRouterVideoCancelledError(
                "Генерация видео отменена провайдером", operation="video"
            )
        if status == "expired":
            raise OpenRouterVideoExpiredError(
                "Срок действия задачи генерации видео истёк", operation="video"
            )
        if status not in TERMINAL_VIDEO_STATUSES:
            await asyncio.sleep(selected_options.polling_interval_seconds)

    response = await client.video_generation.get_video_content_async(
        job_id=str(job_id),
        index=selected_options.index,
    )
    try:
        video_content = await response.aread()
    finally:
        await response.aclose()

    if not video_content:
        raise OpenRouterResponseError(
            "OpenRouter вернул пустой видеофайл", operation="video"
        )

    path = await asyncio.to_thread(_write_video_file, video_content, _video_suffix(response))
    return GeneratedVideo(path=path)
