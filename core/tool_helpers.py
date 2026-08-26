import base64
import binascii
import logging
from io import BytesIO
from typing import Any

from PIL import Image
from telegram import InputFile, Update

from core.common_types import GeneratedImage, GeneratedVideo
from core.config import settings

logger = logging.getLogger(__name__)


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
    await update.message.reply_photo(photo=InputFile(bio))  # type: ignore
    return normalized_b64_data, gen_width, gen_height, gen_aspect_ratio


async def send_video_to_telegram(
    update: Update, video: GeneratedVideo, status_message: Any
) -> bool:
    """Отправляет видео в Telegram, пробуя video, затем document. Возвращает True при успехе."""
    try:
        with video.path.open("rb") as video_file:
            await update.message.reply_video(
                video=InputFile(video_file),
                write_timeout=settings.telegram.video_write_timeout,
                read_timeout=settings.telegram.video_read_timeout,
                connect_timeout=settings.telegram.video_connect_timeout,
                pool_timeout=settings.telegram.video_pool_timeout,
            )  # type: ignore
        return True
    except Exception as e:
        logger.error(f"Ошибка при отправке видео как video: {e}", exc_info=True)
        try:
            with video.path.open("rb") as video_file:
                await update.message.reply_document(
                    document=InputFile(video_file),
                    write_timeout=settings.telegram.video_write_timeout,
                    read_timeout=settings.telegram.video_read_timeout,
                    connect_timeout=settings.telegram.video_connect_timeout,
                    pool_timeout=settings.telegram.video_pool_timeout,
                )  # type: ignore
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
) -> dict:
    """Создаёт данные visual session из бинарного изображения.

    `normalized_b64` появляется только на границе хранения и уже содержит
    JPEG-версию, отправленную в Telegram.
    """
    if normalized_b64 is not None and width is not None and height is not None and aspect_ratio is not None:
        return {
            "image_type": "image/jpeg",
            "image": normalized_b64,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "source_mime_type": image.mime_type,
        }
    return {}


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
    return GeneratedImage(
        content=content,
        mime_type=image_type,
        metadata={
            "width": image_dict.get("width"),
            "height": image_dict.get("height"),
            "aspect_ratio": image_dict.get("aspect_ratio"),
        },
    )


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
