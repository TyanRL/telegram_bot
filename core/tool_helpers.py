import base64
import binascii
import logging
from io import BytesIO
from typing import Any
from uuid import uuid4

from PIL import Image
from telegram import InputFile, Update

from core.common_types import GeneratedImage, GeneratedVideo
from core.config import settings

logger = logging.getLogger(__name__)

_MAX_IMAGE_REFERENCES = 10
_GENERATION_QUALITY_VALUES = ("standard", "pro")


def _pad_image_to_aspect_ratio(img: Image.Image, target_ratio: float) -> Image.Image:
    current_ratio = img.width / img.height

    if abs(current_ratio - target_ratio) < 0.01:
        return img

    if current_ratio > target_ratio:
        new_w = img.width
        new_h = round(new_w / target_ratio)
    else:
        new_h = img.height
        new_w = round(new_h * target_ratio)

    canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    x = (new_w - img.width) // 2
    y = (new_h - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def normalize_image_for_session(
    image_content: bytes,
    max_size: int = settings.media.telegram_image_max_size,
    quality: int = settings.media.telegram_image_jpeg_quality,
) -> dict[str, Any]:
    """Нормализует входное изображение перед сохранением в visual session."""
    with Image.open(BytesIO(image_content)) as source_image:
        img = source_image.convert("RGB")

    img.thumbnail((max_size, max_size))
    final_width, final_height = img.size
    final_aspect_ratio = final_width / final_height if final_height > 0 else 1.0

    normalized_image = BytesIO()
    img.save(normalized_image, format="JPEG", quality=quality, optimize=True)

    return {
        "image_type": "image/jpeg",
        "image": base64.b64encode(normalized_image.getvalue()).decode("utf-8"),
        "width": final_width,
        "height": final_height,
        "aspect_ratio": final_aspect_ratio,
    }


def prepare_image_for_telegram(
    image: GeneratedImage,
    max_size: int = settings.media.telegram_image_max_size,
    quality: int = settings.media.telegram_image_jpeg_quality,
    target_aspect_ratio: float | None = None,
) -> tuple[BytesIO, str, int, int, float]:
    """Подготавливает бинарное изображение для Telegram.

    При необходимости приводит его к нужному aspect ratio,
    ресайзит и конвертирует изображение в JPEG для Telegram.
    Возвращает BytesIO, обновлённый base64, ширину, высоту и aspect ratio для сохранения в visual session.
    """
    img = Image.open(BytesIO(image.content))

    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    if target_aspect_ratio is not None and target_aspect_ratio > 0:
        img = _pad_image_to_aspect_ratio(img, target_aspect_ratio)

    img.thumbnail((max_size, max_size))

    final_width, final_height = img.size
    final_aspect_ratio = final_width / final_height if final_height > 0 else 1.0

    bio = BytesIO()
    img.save(bio, format="JPEG", quality=quality, optimize=True)

    normalized_b64 = base64.b64encode(bio.getvalue()).decode("utf-8")

    bio.seek(0)
    bio.name = "generated.jpg"

    return bio, normalized_b64, final_width, final_height, final_aspect_ratio


async def send_image_to_telegram(
    update: Update,
    image: GeneratedImage,
    target_aspect_ratio: float | None = None,
) -> tuple[str | None, int | None, int | None, float | None]:
    """Отправляет бинарный результат в Telegram.

    Возвращает base64 только для последующего сохранения в visual session.
    """
    bio, normalized_b64_data, gen_width, gen_height, gen_aspect_ratio = (
        prepare_image_for_telegram(image, target_aspect_ratio=target_aspect_ratio)
    )
    target = get_delivery_target(update)
    if target is None:
        raise ValueError("Нет сообщения-адресата для отправки изображения")
    await target.reply_photo(photo=InputFile(bio))
    return normalized_b64_data, gen_width, gen_height, gen_aspect_ratio


def get_delivery_target(update: Update) -> Any:
    """Возвращает сообщение-адресат для отправки файла.

    Обычное сообщение пользователя адресуется напрямую. Обновление от
    inline-кнопки не содержит `update.message`: адресатом становится
    сообщение бота, к которому привязана кнопка (`update.effective_message`).
    Ответ так уходит в исходный чат и тему, а не в чат бота по умолчанию.
    """

    from core.state_and_commands import get_delivery_message

    return get_delivery_message(update)


async def send_video_to_telegram(
    update: Update, video: GeneratedVideo, status_message: Any
) -> bool:
    """Отправляет видео в Telegram, пробуя video, затем document. Возвращает True при успехе.

    Сбой доставки не приводит к повторной генерации: файл освобождается, а
    вызывающий код сообщает пользователю о неудаче.
    """
    target = get_delivery_target(update)
    if target is None:
        logger.error("Нет сообщения-адресата для отправки видео")
        video.cleanup()
        return False
    try:
        with video.path.open("rb") as video_file:
            await target.reply_video(
                video=InputFile(video_file),
                write_timeout=settings.telegram.video_write_timeout,
                read_timeout=settings.telegram.video_read_timeout,
                connect_timeout=settings.telegram.video_connect_timeout,
                pool_timeout=settings.telegram.video_pool_timeout,
            )
        return True
    except Exception as e:
        logger.error(f"Ошибка при отправке видео как video: {e}", exc_info=True)
        try:
            with video.path.open("rb") as video_file:
                await target.reply_document(
                    document=InputFile(video_file),
                    write_timeout=settings.telegram.video_write_timeout,
                    read_timeout=settings.telegram.video_read_timeout,
                    connect_timeout=settings.telegram.video_connect_timeout,
                    pool_timeout=settings.telegram.video_pool_timeout,
                )
            return True
        except Exception as e2:
            logger.error(f"Ошибка при отправке видео как document: {e2}", exc_info=True)
            try:
                await status_message.edit_text(
                    "_Видео сгенерировано, но не удалось отправить его в Telegram._",
                    parse_mode=settings.telegram.parse_mode,
                )
            except Exception:
                pass
            return False
    finally:
        video.cleanup()


def build_image_dict_from_image(
    image: GeneratedImage,
    normalized_b64: str | None,
    width: int | None,
    height: int | None,
    aspect_ratio: float | None,
    *,
    generation: dict[str, Any] | None = None,
) -> dict:
    """Создаёт данные visual session из бинарного изображения.

    `normalized_b64` появляется только на границе хранения и уже содержит
    JPEG-версию, отправленную в Telegram. Параметры операции хранятся отдельно
    от бинарного результата; в `metadata.cost` остаётся стоимость именно этого
    результата, а сводка стоимости операции лежит в `generation`.
    """
    if normalized_b64 is not None and width is not None and height is not None and aspect_ratio is not None:
        image_metadata = image.metadata
        image_generation = image_metadata.get("generation")
        if not isinstance(image_generation, dict):
            image_generation = {}
        generation_values = dict(generation or {})

        def value(name: str) -> Any:
            if name in generation_values and generation_values[name] is not None:
                return generation_values[name]
            if name in image_generation and image_generation[name] is not None:
                return image_generation[name]
            return image_metadata.get(name)

        model = _non_empty_string(value("model"))
        operation = value("operation")
        if operation not in ("generate", "edit"):
            operation = None

        quality = value("quality")
        if quality not in _GENERATION_QUALITY_VALUES:
            quality = _quality_for_model(model, operation)

        requested_model = _non_empty_string(value("requested_model")) or model
        operation_id = _non_empty_string(value("operation_id")) or uuid4().hex
        task_id = _non_empty_string(value("task_id")) or operation_id
        prompt = value("prompt")
        if not isinstance(prompt, str):
            prompt = None

        fallback_used = value("fallback_used")
        if not isinstance(fallback_used, bool):
            fallback_used = False

        source_image_indices = _normalized_source_indices(
            value("source_image_indices")
        )
        requested_aspect_ratio = value("aspect_ratio")
        if not (
            isinstance(requested_aspect_ratio, str)
            or (
                isinstance(requested_aspect_ratio, (int, float))
                and not isinstance(requested_aspect_ratio, bool)
            )
        ):
            requested_aspect_ratio = None
        resolution = _non_empty_string(value("resolution"))
        route_reason = _non_empty_string(value("route_reason"))

        cost = _cost_value(image_metadata.get("cost"))
        total_known_cost = _cost_value(value("total_known_cost"))
        if total_known_cost is None:
            total_known_cost = cost
        billing_unknown = value("billing_unknown")
        if not isinstance(billing_unknown, bool):
            billing_unknown = value("cost_unknown")
        if not isinstance(billing_unknown, bool):
            billing_unknown = not _has_cost(cost)

        stored_metadata: dict[str, Any] = {}
        if cost is not None:
            stored_metadata["cost"] = cost

        return {
            "image_type": "image/jpeg",
            "image": normalized_b64,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "source_mime_type": image.mime_type,
            "metadata": stored_metadata,
            "generation": {
                "operation_id": operation_id,
                "task_id": task_id,
                "model": model,
                "quality": quality,
                "requested_model": requested_model,
                "fallback_used": fallback_used,
                "prompt": prompt,
                "operation": operation,
                "source_image_indices": source_image_indices,
                "aspect_ratio": requested_aspect_ratio,
                "resolution": resolution,
                "route_reason": route_reason,
                "total_known_cost": total_known_cost,
                "billing_unknown": billing_unknown,
            },
        }
    return {}


def _non_empty_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _cost_value(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _has_cost(value: Any) -> bool:
    return _cost_value(value) is not None


def _quality_for_model(model: str | None, operation: str | None) -> str | None:
    """Определяет известный уровень только по фактическому ID текущего вызова."""

    if model is None:
        return None

    if operation == "edit":
        standard_model = settings.media.image_edit_fast_model
        pro_model = settings.media.image_edit_model
    else:
        standard_model = settings.media.image_generation_fast_model
        pro_model = settings.media.image_generation_model

    if model == standard_model and model != pro_model:
        return "standard"
    if model == pro_model and model != standard_model:
        return "pro"
    return None


def _normalized_source_indices(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []

    indices: list[int] = []
    seen: set[int] = set()
    for index in value:
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            continue
        if index not in seen:
            indices.append(index)
            seen.add(index)

    if len(indices) > _MAX_IMAGE_REFERENCES:
        indices = indices[-_MAX_IMAGE_REFERENCES:]
    return indices


def _read_generation_block(value: Any) -> dict[str, Any] | None:
    """Читает только известные поля generation, не интерпретируя старые записи."""

    if not isinstance(value, dict):
        return None

    generation: dict[str, Any] = {}
    for key in (
        "operation_id",
        "task_id",
        "model",
        "requested_model",
        "prompt",
        "resolution",
        "route_reason",
    ):
        string_value = _non_empty_string(value.get(key))
        if string_value is not None:
            generation[key] = string_value

    quality = value.get("quality")
    if quality in _GENERATION_QUALITY_VALUES:
        generation["quality"] = quality

    operation = value.get("operation")
    if operation in ("generate", "edit"):
        generation["operation"] = operation

    fallback_used = value.get("fallback_used")
    if isinstance(fallback_used, bool):
        generation["fallback_used"] = fallback_used

    if "source_image_indices" in value:
        generation["source_image_indices"] = _normalized_source_indices(
            value.get("source_image_indices")
        )

    requested_aspect_ratio = value.get("aspect_ratio")
    if isinstance(requested_aspect_ratio, str) or (
        isinstance(requested_aspect_ratio, (int, float))
        and not isinstance(requested_aspect_ratio, bool)
    ):
        generation["aspect_ratio"] = requested_aspect_ratio

    total_known_cost = _cost_value(value.get("total_known_cost"))
    if total_known_cost is not None:
        generation["total_known_cost"] = total_known_cost

    for key in ("billing_unknown", "cost_unknown"):
        flag = value.get(key)
        if isinstance(flag, bool):
            generation[key] = flag

    return generation


def generated_image_from_session_dict(image_dict: dict[str, Any]) -> GeneratedImage:
    """Восстанавливает бинарное изображение на границе visual session."""
    image_type = image_dict.get("image_type")
    encoded = image_dict.get("image")
    if not isinstance(image_type, str) or not image_type:
        raise ValueError("В visual session отсутствует MIME-тип изображения")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("В visual session отсутствует изображение")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("В visual session содержится некорректное изображение") from exc
    if not content:
        raise ValueError("В visual session содержится пустое изображение")

    metadata: dict[str, Any] = {
        "width": image_dict.get("width"),
        "height": image_dict.get("height"),
        "aspect_ratio": image_dict.get("aspect_ratio"),
    }
    stored_metadata = image_dict.get("metadata")
    if isinstance(stored_metadata, dict):
        cost = _cost_value(stored_metadata.get("cost"))
        if cost is not None:
            metadata["cost"] = cost

    generation = _read_generation_block(image_dict.get("generation"))
    if generation is not None:
        metadata["generation"] = generation
        for key in (
            "model",
            "task_id",
            "quality",
            "requested_model",
            "fallback_used",
            "operation",
            "source_image_indices",
            "resolution",
            "route_reason",
            "total_known_cost",
            "billing_unknown",
            "cost_unknown",
        ):
            if key in generation:
                metadata[key] = generation[key]

    return GeneratedImage(
        content=content,
        mime_type=image_type,
        metadata=metadata,
    )


def get_session_image_entries(
    session: dict[str, Any], indices: list[int] | None = None
) -> list[tuple[int, dict[str, Any]]]:
    """Возвращает пары (реальный 1-based номер, запись изображения) сессии.

    Visual session имеет единственный актуальный формат: список изображений
    ``images``. Старый формат с единственным текущим изображением не
    поддерживается и не преобразуется.

    Пустой список индексов, как и отсутствие индексов, означает «все картинки».
    Порядок сохраняется, а повторный номер используется один раз по первому
    вхождению.
    """

    images = session.get("images")
    available = (
        [image for image in images if isinstance(image, dict)]
        if isinstance(images, list)
        else []
    )

    if not indices:
        return list(enumerate(available, start=1))

    selected: list[tuple[int, dict[str, Any]]] = []
    seen: set[int] = set()
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Номера изображений должны быть целыми числами")
        if index < 1 or index > len(available):
            raise ValueError(
                f"Номер изображения {index} вне диапазона 1..{len(available)}"
            )
        if index not in seen:
            selected.append((index, available[index - 1]))
            seen.add(index)
    return selected


def get_session_images(
    session: dict[str, Any], indices: list[int] | None = None
) -> list[dict[str, Any]]:
    """Возвращает записи изображений сессии в выбранном порядке."""

    return [image for _, image in get_session_image_entries(session, indices)]


def resolve_selected_images(
    session: dict[str, Any],
    indices: list[int] | None = None,
    *,
    max_images: int | None = None,
) -> tuple[list[GeneratedImage], list[int]]:
    """Разрешает выбранные изображения и возвращает их реальные номера.

    Дедупликация выполняется до ограничения количества. При превышении лимита
    сохраняются последние изображения в выбранном порядке, как и в media tools.
    """

    entries = get_session_image_entries(session, indices)
    if max_images is not None:
        if isinstance(max_images, bool) or not isinstance(max_images, int) or max_images < 1:
            raise ValueError("max_images должен быть положительным целым числом")
        entries = entries[-max_images:]

    return (
        [generated_image_from_session_dict(image_dict) for _, image_dict in entries],
        [index for index, _ in entries],
    )


def describe_image_provenance(
    images: list[dict[str, Any]],
    *,
    start_index: int = 1,
    all_images: list[dict[str, Any]] | None = None,
) -> str:
    """Формирует безопасную краткую сводку происхождения для внутреннего контекста."""

    complete_list = all_images if all_images is not None else images
    latest_generated_index: int | None = None
    for index in range(len(complete_list), 0, -1):
        if _is_generated_image(complete_list[index - 1]):
            latest_generated_index = index
            break

    lines: list[str] = []
    if latest_generated_index is not None:
        lines.append(
            f"Последний успешно созданный результат: изображение №{latest_generated_index}."
        )

    for index, image_dict in enumerate(images, start=start_index):
        source_kind = image_dict.get("source_kind")
        generation = _read_generation_block(image_dict.get("generation")) or {}
        if source_kind == "uploaded":
            description = "загружено пользователем"
        elif _is_generated_image(image_dict):
            operation = generation.get("operation")
            if operation == "edit" or source_kind == "edit":
                description = "отредактировано"
            else:
                description = "сгенерировано"
            if index == latest_generated_index:
                description += ", последний успешный результат"

            quality = generation.get("quality")
            model = generation.get("model")
            if quality in _GENERATION_QUALITY_VALUES:
                description += f", уровень {quality}"
            if isinstance(model, str):
                description += f", модель {model[:100]}"

            source_indices = generation.get("source_image_indices")
            if isinstance(source_indices, list) and source_indices:
                description += ", источники " + ", ".join(
                    f"№{source_index}" for source_index in source_indices
                )

            prompt = generation.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                compact_prompt = " ".join(prompt.split())
                if len(compact_prompt) > 240:
                    compact_prompt = compact_prompt[:237] + "..."
                description += f"; исходная инструкция: «{compact_prompt}»"
        else:
            description = "происхождение неизвестно"

        lines.append(f"Изображение №{index}: {description}.")

    if not lines:
        return ""
    return "Сведения о происхождении изображений:\n" + "\n".join(lines)


def _is_generated_image(image_dict: dict[str, Any]) -> bool:
    source_kind = image_dict.get("source_kind")
    return (
        isinstance(image_dict.get("generation"), dict)
        or (
            isinstance(source_kind, str)
            and source_kind in {"generated", "text2image", "edit"}
        )
    )


def append_image_to_session(
    session: dict[str, Any],
    image_dict: dict[str, Any],
    kind: str,
    prompt: str | None = None,
) -> int:
    """Добавляет картинку в visual session и возвращает её 1-based номер.

    Сессия хранит только список изображений: дублирующие переходные поля
    верхнего уровня не сохраняются.
    """

    images = get_session_images(session)
    stored_image = dict(image_dict)
    for metadata_key in ("metadata", "generation"):
        metadata_value = stored_image.get(metadata_key)
        if isinstance(metadata_value, dict):
            metadata_copy = dict(metadata_value)
            if metadata_key == "generation" and isinstance(
                metadata_copy.get("source_image_indices"), list
            ):
                metadata_copy["source_image_indices"] = list(
                    metadata_copy["source_image_indices"]
                )
            stored_image[metadata_key] = metadata_copy
    stored_image["source_kind"] = kind
    images.append(stored_image)

    session["images"] = images
    add_history_entry(session, kind, prompt)
    return len(images)


def add_history_entry(session: dict, kind: str, prompt: str | None = None) -> None:
    """Добавляет запись в history visual session."""
    from datetime import datetime

    history = session.get("history", [])
    history.append({
        "step": len(history),
        "kind": kind,
        "prompt": prompt,
        "created_at": datetime.now().isoformat(),
    })
    session["history"] = history
