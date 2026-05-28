import base64
import logging
import os
from io import BytesIO
from typing import Any

from PIL import Image
from telegram import InputFile, Update

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
    b64_data: str,
    max_size: int = 1280,
    quality: int = 88,
    target_aspect_ratio: float | None = None,
) -> tuple[BytesIO, str, int, int, float]:
    """Декодирует base64, при необходимости приводит к нужному aspect ratio,
    ресайзит и конвертирует изображение в JPEG для Telegram.
    Возвращает BytesIO, обновлённый base64, ширину, высоту и aspect ratio для сохранения в visual session.
    """
    image_bytes = base64.b64decode(b64_data)
    img = Image.open(BytesIO(image_bytes))

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
    image_data_url: str,
    target_aspect_ratio: float | None = None,
) -> tuple[str | None, int | None, int | None, float | None]:
    """Отправляет изображение в Telegram. Возвращает normalized_b64, width, height, aspect_ratio."""
    if image_data_url.startswith("data:"):
        header, b64_data = image_data_url.split(",", 1)
        bio, normalized_b64_data, gen_width, gen_height, gen_aspect_ratio = prepare_image_for_telegram(
            b64_data, target_aspect_ratio=target_aspect_ratio
        )
        await update.message.reply_photo(photo=InputFile(bio))  # type: ignore
        return normalized_b64_data, gen_width, gen_height, gen_aspect_ratio
    else:
        await update.message.reply_photo(photo=image_data_url)  # type: ignore
        return None, None, None, None


async def send_video_to_telegram(update: Update, video_path: str, status_message: Any) -> bool:
    """Отправляет видео в Telegram, пробуя video, затем document. Возвращает True при успехе."""
    try:
        with open(video_path, "rb") as video_file:
            await update.message.reply_video(video=InputFile(video_file), write_timeout=180, read_timeout=120, connect_timeout=30,pool_timeout=30)  # type: ignore
        return True
    except Exception as e:
        logger.error(f"Ошибка при отправке видео как video: {e}", exc_info=True)
        try:
            with open(video_path, "rb") as video_file:
                await update.message.reply_document(document=InputFile(video_file), write_timeout=180, read_timeout=120, connect_timeout=30,pool_timeout=30)  # type: ignore
            return True
        except Exception as e2:
            logger.error(f"Ошибка при отправке видео как document: {e2}", exc_info=True)
            try:
                await status_message.edit_text(
                    "_Видео сгенерировано, но не удалось отправить его в Telegram._",
                    parse_mode="MarkdownV2",
                )
            except Exception:
                pass
            return False
    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass


def build_image_dict_from_data_url(image_data_url: str, normalized_b64: str | None, width: int | None, height: int | None, aspect_ratio: float | None) -> dict:
    """Создаёт словарь с данными изображения из data URL."""
    if image_data_url.startswith("data:") and normalized_b64 is not None and width is not None and height is not None and aspect_ratio is not None:
        header, _ = image_data_url.split(",", 1)
        img_type = header.split(";")[0].replace("data:", "")
        return {
            "image_type": img_type,
            "image": normalized_b64,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
        }
    return {}


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
