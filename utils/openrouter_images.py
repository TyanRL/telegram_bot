"""OpenRouter Images API adapter.

Этот модуль работает только с SDK-клиентом, переданным сервисом. Data URL
создаётся здесь и используется исключительно в `input_references`.

Для обеих моделей FLUX 3 Image Images API документирует `aspect_ratio`,
`resolution`, `n` и `input_references` для image-to-image. `seed` в контракте
Images API отсутствует и явно отклоняется до вызова SDK.
"""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
from typing import Any, Sequence, cast

from openrouter import OpenRouter
from openrouter.components.contentpartimage import ContentPartImageTypedDict
from PIL import Image

from core.common_types import GeneratedImage, ImageEditOptions, ImageGenerationOptions
from core.config import settings

# Совместимые имена для кода, который импортировал старые константы.
GENERATION_IMAGE_MODEL = settings.media.image_generation_model
EDIT_IMAGE_MODEL = settings.media.image_edit_model


def _image_to_data_url(image: GeneratedImage) -> str:
    encoded = base64.b64encode(image.content).decode("ascii")
    return f"data:{image.mime_type};base64,{encoded}"


def _sdk_image_options(
    options: ImageGenerationOptions | ImageEditOptions,
) -> dict[str, Any]:
    """Формирует только заданные параметры из контракта FLUX Images API."""

    if options.seed is not None:
        raise ValueError("Параметр seed не поддерживается FLUX Images API")

    return {
        name: value
        for name, value in (
            ("aspect_ratio", options.aspect_ratio),
            ("resolution", options.resolution),
        )
        if value is not None
    }


class InvalidImageResponseError(ValueError):
    """Ошибка ответа с безопасными данными попытки и известным биллингом."""

    def __init__(
        self,
        message: str,
        *,
        model: str,
        operation: str,
        cost: Any = None,
        created: Any = None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.requested_model = model
        self.operation = operation
        self.cost = cost
        self.created = created
        self.attempt_metadata: dict[str, Any] = {
            "model": model,
            "requested_model": model,
            "operation": operation,
        }
        if cost is not None:
            self.attempt_metadata["cost"] = cost
        if created is not None:
            self.attempt_metadata["created"] = created


def _decode_image_response(response: Any, *, model: str, operation: str) -> GeneratedImage:
    # Usage must be read before inspecting image data: a malformed image can
    # still represent a billed generation.
    usage = getattr(response, "usage", None)
    cost = getattr(usage, "cost", None) if usage is not None else None
    created = getattr(response, "created", None)

    def invalid(message: str) -> InvalidImageResponseError:
        return InvalidImageResponseError(
            message,
            model=model,
            operation=operation,
            cost=cost,
            created=created,
        )

    data = getattr(response, "data", None)
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)) or not data:
        raise invalid("OpenRouter не вернул изображение")

    image = data[0]
    encoded = getattr(image, "b64_json", None)
    if not isinstance(encoded, str) or not encoded:
        raise invalid("В ответе OpenRouter отсутствует b64_json")

    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise invalid("OpenRouter вернул некорректное содержимое изображения") from exc

    if not content:
        raise invalid("OpenRouter вернул пустое изображение")

    try:
        with Image.open(BytesIO(content)) as decoded_image:
            decoded_image.verify()
    except Exception as exc:
        raise invalid("OpenRouter вернул повреждённые байты изображения") from exc

    mime_type = getattr(image, "media_type", None) or "image/png"
    if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
        raise invalid("OpenRouter вернул некорректный media_type")

    metadata: dict[str, Any] = {
        "model": model,
        "requested_model": model,
        "operation": operation,
    }
    if created is not None:
        metadata["created"] = created
    if cost is not None:
        metadata["cost"] = cost

    return GeneratedImage(content=content, mime_type=str(mime_type), metadata=metadata)


def _requested_model(options: ImageGenerationOptions | ImageEditOptions, default: str) -> str:
    model = options.model
    if model is None:
        return default
    if not isinstance(model, str):
        raise ValueError("Параметр model должен быть строкой")
    return model.strip() if model.strip() else default


def _validate_single_image(options: ImageGenerationOptions | ImageEditOptions) -> None:
    if type(options.n) is not int or options.n != 1:
        raise ValueError("FLUX Images API поддерживает только n == 1")


def _image_request_controls(timeout_ms: int | None) -> dict[str, Any]:
    """Отключает скрытые повторы SDK только для запросов генерации изображений."""

    controls: dict[str, Any] = {"retries": None}
    if timeout_ms is not None:
        if type(timeout_ms) is not int or timeout_ms <= 0:
            raise ValueError("Таймаут image-запроса должен быть положительным целым числом")
        controls["timeout_ms"] = timeout_ms
    return controls


async def generate(
    prompt: str,
    options: ImageGenerationOptions | None = None,
    *,
    client: OpenRouter,
    timeout_ms: int | None = None,
) -> GeneratedImage:
    """Генерирует одно изображение через выбранную или старшую модель."""

    selected_options = options if options is not None else ImageGenerationOptions()
    _validate_single_image(selected_options)
    sdk_options = _sdk_image_options(selected_options)
    if selected_options.resolution is None and settings.media.image_default_resolution is not None:
        sdk_options["resolution"] = settings.media.image_default_resolution
    model = _requested_model(selected_options, settings.media.image_generation_model)
    request_controls = _image_request_controls(timeout_ms)
    response = await client.images.generate_async(
        model=model,
        prompt=prompt,
        n=selected_options.n,
        **sdk_options,
        **request_controls,
    )
    return _decode_image_response(
        response,
        model=model,
        operation="generate",
    )


async def edit(
    source_image: GeneratedImage | Sequence[GeneratedImage],
    instruction: str,
    options: ImageEditOptions | None = None,
    *,
    client: OpenRouter,
    timeout_ms: int | None = None,
) -> GeneratedImage:
    """Редактирует изображение через выбранную или старшую модель.

    Редактирование выполняется как image-to-image: исходное изображение
    передаётся data URL'ом в `input_references`, а инструкция — в `prompt`.
    """

    selected_options = options if options is not None else ImageEditOptions()
    _validate_single_image(selected_options)
    sdk_options = _sdk_image_options(selected_options)
    if selected_options.resolution is None and settings.media.image_default_resolution is not None:
        sdk_options["resolution"] = settings.media.image_default_resolution
    model = _requested_model(selected_options, settings.media.image_edit_model)
    selected_images = (
        [source_image]
        if isinstance(source_image, GeneratedImage)
        else list(source_image)
    )
    if not selected_images:
        raise ValueError("Для редактирования нужно хотя бы одно исходное изображение")
    if len(selected_images) > 10:
        raise ValueError("Images API поддерживает не более 10 input_references")

    input_references: list[ContentPartImageTypedDict] = [
        cast(
            ContentPartImageTypedDict,
            {
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(image)},
            },
        )
        for image in selected_images
    ]
    request_controls = _image_request_controls(timeout_ms)
    response = await client.images.generate_async(
        model=model,
        prompt=instruction,
        input_references=input_references,
        n=selected_options.n,
        **sdk_options,
        **request_controls,
    )
    return _decode_image_response(
        response,
        model=model,
        operation="edit",
    )
