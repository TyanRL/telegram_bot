"""OpenRouter Images API adapter.

Этот модуль работает только с SDK-клиентом, переданным сервисом. Data URL
создаётся здесь и используется исключительно в `input_references`.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any, cast

from openrouter import OpenRouter
from openrouter.components.contentpartimage import ContentPartImageTypedDict
from openrouter.components.imagegenerationrequest import (
    ImageGenerationRequestAspectRatio,
    ImageGenerationRequestBackground,
    ImageGenerationRequestOutputFormat,
    ImageGenerationRequestQuality,
    ImageGenerationRequestResolution,
)

from core.common_types import GeneratedImage, ImageEditOptions, ImageGenerationOptions


GENERATION_IMAGE_MODEL = "qwen/qwen-image-3-pro"
EDIT_IMAGE_MODEL = "google/gemini-3.1-flash-image"


def _image_to_data_url(image: GeneratedImage) -> str:
    encoded = base64.b64encode(image.content).decode("ascii")
    return f"data:{image.mime_type};base64,{encoded}"


def _sdk_image_options(
    options: ImageGenerationOptions | ImageEditOptions,
) -> dict[str, Any]:
    """Приводит доменные строки к типам enum-like, объявленным SDK."""

    return {
        "aspect_ratio": cast(ImageGenerationRequestAspectRatio | None, options.aspect_ratio),
        "resolution": cast(ImageGenerationRequestResolution | None, options.resolution),
        "output_format": cast(ImageGenerationRequestOutputFormat | None, options.output_format),
        "output_compression": options.output_compression,
        "quality": cast(ImageGenerationRequestQuality | None, options.quality),
        "background": cast(ImageGenerationRequestBackground | None, options.background),
        "size": options.size,
        "seed": options.seed,
    }


def _decode_image_response(response: Any, *, model: str, operation: str) -> GeneratedImage:
    data = getattr(response, "data", None)
    if not data:
        raise ValueError("OpenRouter не вернул изображение")

    image = data[0]
    encoded = getattr(image, "b64_json", None)
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("В ответе OpenRouter отсутствует b64_json")

    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("OpenRouter вернул некорректное содержимое изображения") from exc

    if not content:
        raise ValueError("OpenRouter вернул пустое изображение")

    mime_type = getattr(image, "media_type", None) or "image/png"
    metadata: dict[str, Any] = {"model": model, "operation": operation}
    created = getattr(response, "created", None)
    if created is not None:
        metadata["created"] = created
    usage = getattr(response, "usage", None)
    cost = getattr(usage, "cost", None) if usage is not None else None
    if cost is not None:
        metadata["cost"] = cost

    return GeneratedImage(content=content, mime_type=str(mime_type), metadata=metadata)


async def generate(
    prompt: str,
    options: ImageGenerationOptions | None = None,
    *,
    client: OpenRouter,
) -> GeneratedImage:
    """Генерирует изображение через фиксированную Qwen-модель."""

    selected_options = options or ImageGenerationOptions()
    response = await client.images.generate_async(
        model=GENERATION_IMAGE_MODEL,
        prompt=prompt,
        n=selected_options.n,
        stream=False,
        **_sdk_image_options(selected_options),
    )
    return _decode_image_response(
        response,
        model=GENERATION_IMAGE_MODEL,
        operation="generate",
    )


async def edit(
    source_image: GeneratedImage,
    instruction: str,
    options: ImageEditOptions | None = None,
    *,
    client: OpenRouter,
) -> GeneratedImage:
    """Редактирует изображение через фиксированную Gemini-модель."""

    selected_options = options or ImageEditOptions()
    input_references: list[ContentPartImageTypedDict] = [
        cast(
            ContentPartImageTypedDict,
            {
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(source_image)},
            },
        )
    ]
    response = await client.images.generate_async(
        model=EDIT_IMAGE_MODEL,
        prompt=instruction,
        input_references=input_references,
        n=selected_options.n,
        stream=False,
        **_sdk_image_options(selected_options),
    )
    return _decode_image_response(
        response,
        model=EDIT_IMAGE_MODEL,
        operation="edit",
    )
