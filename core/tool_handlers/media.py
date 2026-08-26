import asyncio
import logging
from typing import Any

from core.common_types import (
    GeneratedImage,
    ImageEditOptions,
    ImageGenerationOptions,
    ToolResult,
)
from core.config import settings
from core.generation_error_mapper import map_generation_error
from core.state_and_commands import (
    animate_service_message,
    get_user_image_edit_session,
    reply_service_message,
    reply_service_text,
    set_user_image_edit_session,
)
from core.tool_helpers import (
    add_history_entry,
    build_image_dict_from_image,
    generated_image_from_session_dict,
    send_image_to_telegram,
    send_video_to_telegram,
)
from core.tool_registry import ToolExecutionContext, register_tool
from utils.openrouter_client import OpenRouterMediaError

logger = logging.getLogger(__name__)


def _sdk_aspect_ratio(value: object) -> str | None:
    """Приводит сохранённое числовое соотношение к enum SDK."""

    if isinstance(value, str):
        return value
    if not isinstance(value, (int, float)) or value <= 0:
        return None

    supported = {
        "1:1": 1.0,
        "4:3": 4 / 3,
        "3:4": 3 / 4,
        "3:2": 3 / 2,
        "2:3": 2 / 3,
        "16:9": 16 / 9,
        "9:16": 9 / 16,
        "21:9": 21 / 9,
        "9:21": 9 / 21,
    }
    return min(supported, key=lambda name: abs(supported[name] - float(value)))


def _answer(ctx: ToolExecutionContext, text: str | None) -> ToolResult:
    return ToolResult(
        {"ok": text is not None, "message": text or "Операция не выполнена."},
        ctx_token=ctx.context_tokens,
        completion_token=ctx.completion_tokens,
    )


async def _report_media_error(
    ctx: ToolExecutionContext,
    error: OpenRouterMediaError,
    *,
    context: str,
) -> ToolResult:
    mapped = map_generation_error(error, context=context)
    await reply_service_text(ctx.update, mapped.user_message)
    return _answer(ctx, None)


def _source_image_from_session(session: dict[str, Any]) -> GeneratedImage:
    current_image = session.get("current_image")
    if not isinstance(current_image, dict):
        raise ValueError("В visual session отсутствует current_image")
    return generated_image_from_session_dict(current_image)


@register_tool("generate_image")
async def handle_generate_image(
    ctx: ToolExecutionContext, args: dict[str, Any]
) -> ToolResult:
    user_id = ctx.update.effective_user.id  # type: ignore

    session = await get_user_image_edit_session(user_id)
    if session is not None:
        return _answer(
            ctx,
            "У вас уже есть активная сессия редактирования изображения. "
            "Чтобы сгенерировать новое изображение с нуля, сначала выполните /reset.",
        )

    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        logger.info("Пустой запрос на генерацию изображения")
        return _answer(ctx, "Не удалось сгенерировать изображение: запрос пустой.")

    try:
        image = await ctx.openrouter_service.generate(
            prompt,
            ImageGenerationOptions(),
        )
    except OpenRouterMediaError as error:
        return await _report_media_error(ctx, error, context="image")
    except Exception as error:
        logger.error("Ошибка при генерации изображения: %s", error, exc_info=True)
        return _answer(ctx, "Не удалось сгенерировать изображение. Внутренняя ошибка сервера")

    try:
        normalized_b64, gen_width, gen_height, gen_aspect_ratio = (
            await send_image_to_telegram(ctx.update, image)
        )
    except Exception as error:
        logger.error("Ошибка при отправке картинки в Telegram: %s", error, exc_info=True)
        return _answer(ctx, "Картинка сгенерировалась, но не удалось отправить её в Telegram.")

    try:
        new_image_dict = build_image_dict_from_image(
            image,
            normalized_b64,
            gen_width,
            gen_height,
            gen_aspect_ratio,
        )
        new_session = {
            "original_image": new_image_dict,
            "current_image": new_image_dict,
            "source_kind": "generated",
            "history": [],
        }
        add_history_entry(new_session, "text2image", prompt)
        await set_user_image_edit_session(user_id, new_session)
    except Exception as error:
        logger.error(
            "Ошибка при создании visual session после generate_image: %s",
            error,
            exc_info=True,
        )

    return _answer(
        ctx,
        "Я сделал :) Изображение сгенерировано и стало текущей основой для редактирования.",
    )


@register_tool("generate_image_from_image")
async def handle_generate_image_from_image(
    ctx: ToolExecutionContext, args: dict[str, Any]
) -> ToolResult:
    user_id = ctx.update.effective_user.id  # type: ignore

    session = await get_user_image_edit_session(user_id)
    if session is None:
        logger.warning(
            "generate_image_from_image вызван, но у пользователя %s нет visual session",
            user_id,
        )
        return _answer(
            ctx,
            "Сначала отправьте изображение, которое будет использоваться как основа для генерации.",
        )

    try:
        source_image = _source_image_from_session(session)
    except Exception as error:
        logger.error("Ошибка при чтении исходного изображения: %s", error, exc_info=True)
        return _answer(ctx, "Не удалось подготовить исходное изображение для генерации.")

    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        logger.warning("Пустой prompt для generate_image_from_image")
        return _answer(ctx, "Не удалось сгенерировать изображение: запрос пустой.")

    source_width = source_image.metadata.get("width")
    source_height = source_image.metadata.get("height")
    if not source_width or not source_height:
        current_image = session.get("current_image", {})
        source_width = current_image.get("width")
        source_height = current_image.get("height")
    if source_width and source_height:
        prompt = (
            f"{prompt}\n\n"
            "Important: preserve the original image aspect ratio exactly. "
            f"The source image is {source_width}x{source_height}. Do not make it wider or taller. "
            "Keep the same framing and canvas proportions."
        )

    current_image = session.get("current_image", {})
    aspect_ratio = current_image.get("aspect_ratio")

    try:
        image = await ctx.openrouter_service.edit(
            source_image,
            prompt,
            ImageEditOptions(aspect_ratio=_sdk_aspect_ratio(aspect_ratio)),
        )
    except OpenRouterMediaError as error:
        return await _report_media_error(ctx, error, context="image")
    except Exception as error:
        logger.error(
            "Ошибка при редактировании изображения через OpenRouter: %s",
            error,
            exc_info=True,
        )
        return _answer(ctx, "Не удалось сгенерировать изображение. Внутренняя ошибка сервера")

    try:
        normalized_b64, gen_width, gen_height, gen_aspect_ratio = (
            await send_image_to_telegram(
                ctx.update,
                image,
                target_aspect_ratio=aspect_ratio,
            )
        )
    except Exception as error:
        logger.error("Ошибка при отправке картинки в Telegram: %s", error, exc_info=True)
        return _answer(ctx, "Картинка сгенерировалась, но не удалось отправить её в Telegram.")

    try:
        new_image_dict = build_image_dict_from_image(
            image,
            normalized_b64,
            gen_width,
            gen_height,
            gen_aspect_ratio,
        )
        session["current_image"] = new_image_dict
        add_history_entry(session, "edit", prompt)
        await set_user_image_edit_session(user_id, session)
    except Exception as error:
        logger.error("Ошибка при обновлении visual session: %s", error, exc_info=True)

    return _answer(
        ctx,
        "Я сделал :) Можете отправлять следующую инструкцию для редактирования без повторной загрузки картинки.",
    )


@register_tool("generate_video")
async def handle_generate_video(
    ctx: ToolExecutionContext, args: dict[str, Any]
) -> ToolResult:
    user_id = ctx.update.effective_user.id  # type: ignore
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        logger.warning("Пустой prompt для generate_video")
        return _answer(ctx, "Не удалось сгенерировать видео: запрос пустой.")

    reference_image: GeneratedImage | None = None
    session = await get_user_image_edit_session(user_id)
    if session is not None and session.get("current_image") is not None:
        try:
            reference_image = _source_image_from_session(session)
        except Exception as error:
            logger.error("Ошибка при чтении изображения для видео: %s", error, exc_info=True)

    status_message = await reply_service_message(ctx.update, "Видео генерируется, подождите")
    animation_task = asyncio.create_task(
        animate_service_message(
            status_message,
            "Видео генерируется, подождите",
            interval=settings.telegram.service_animation_interval_seconds,
        )
    )

    try:
        video = await ctx.openrouter_service.generate_video(prompt, reference_image)
    except OpenRouterMediaError as error:
        mapped = map_generation_error(error, context="video")
        try:
            await status_message.edit_text(mapped.user_message)
        except Exception:
            pass
        await reply_service_text(ctx.update, mapped.user_message)
        return _answer(ctx, None)
    except Exception as error:
        logger.error("Ошибка при генерации видео: %s", error, exc_info=True)
        try:
            await status_message.edit_text(
                "_Произошла ошибка при генерации видео._",
                parse_mode=settings.telegram.parse_mode,
            )
        except Exception:
            pass
        return _answer(ctx, None)
    finally:
        animation_task.cancel()
        try:
            await animation_task
        except asyncio.CancelledError:
            pass

    try:
        await status_message.edit_text(
            "_Видео готово, отправляю..._",
            parse_mode=settings.telegram.parse_mode,
        )
    except Exception:
        pass

    try:
        success = await send_video_to_telegram(ctx.update, video, status_message)
    finally:
        video.cleanup()
    if not success:
        return _answer(ctx, None)

    try:
        session = await get_user_image_edit_session(user_id)
        if session is not None:
            add_history_entry(session, "video", prompt)
            await set_user_image_edit_session(user_id, session)
    except Exception as error:
        logger.error(
            "Ошибка при обновлении history visual session после видео: %s",
            error,
            exc_info=True,
        )

    return _answer(
        ctx,
        "Я сделал :) Видео создано на основе текущей версии изображения. "
        "Текущая картинка для дальнейших правок сохранена.",
    )
