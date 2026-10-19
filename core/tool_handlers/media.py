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
    add_user_image_session_history,
    append_user_image_to_session,
    animate_service_message,
    get_user_image_edit_session,
    reply_service_message,
    reply_service_text,
)
from core.tool_helpers import (
    build_image_dict_from_image,
    generated_image_from_session_dict,
    get_session_images,
    send_image_to_telegram,
    send_video_to_telegram,
)
from core.tool_registry import ToolExecutionContext, register_tool
from utils.openrouter_client import OpenRouterMediaError

logger = logging.getLogger(__name__)

_IMAGE_GENERATION_STATUS_TEXT = "Изображение генерируется, подождите"


def _sdk_aspect_ratio(value: object) -> str | None:
    """Приводит сохранённое числовое соотношение к enum SDK."""

    supported = {
        "21:9": 21 / 9,
        "2:1": 2 / 1,
        "16:9": 16 / 9,
        "3:2": 3 / 2,
        "7:5": 7 / 5,
        "4:3": 4 / 3,
        "5:4": 5 / 4,
        "1:1": 1.0,
        "4:5": 4 / 5,
        "3:4": 3 / 4,
        "5:7": 5 / 7,
        "2:3": 2 / 3,
        "9:16": 9 / 16,
        "1:2": 1 / 2,
        "9:21": 9 / 21,
    }

    if isinstance(value, str):
        return value if value in supported or value == "auto" else None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        return None

    return min(supported, key=lambda name: abs(supported[name] - float(value)))


def _answer(ctx: ToolExecutionContext, text: str | None) -> ToolResult:
    return ToolResult(
        {"ok": text is not None, "message": text or "Операция не выполнена."},
        ctx_token=ctx.context_tokens,
        completion_token=ctx.completion_tokens,
    )


async def _start_image_generation_status(
    update: Any,
) -> tuple[Any | None, asyncio.Task[None] | None]:
    try:
        status_message = await reply_service_message(
            update,
            _IMAGE_GENERATION_STATUS_TEXT,
        )
    except Exception:
        # Некоторые тестовые Update не содержат message.reply_text.
        logger.debug(
            "Не удалось отправить статусное сообщение генерации изображения",
            exc_info=True,
        )
        return None, None

    animation_task = asyncio.create_task(
        animate_service_message(
            status_message,
            _IMAGE_GENERATION_STATUS_TEXT,
            interval=settings.telegram.service_animation_interval_seconds,
        )
    )
    return status_message, animation_task


async def _cancel_image_generation_animation(
    animation_task: asyncio.Task[None] | None,
) -> None:
    if animation_task is None:
        return
    animation_task.cancel()
    try:
        await animation_task
    except asyncio.CancelledError:
        pass


async def _delete_image_generation_status(status_message: Any | None) -> None:
    if status_message is None:
        return
    try:
        await status_message.delete()
    except Exception:
        pass


async def _report_media_error(
    ctx: ToolExecutionContext,
    error: OpenRouterMediaError,
    *,
    context: str,
) -> ToolResult:
    mapped = map_generation_error(error, context=context)
    await reply_service_text(ctx.update, mapped.user_message)
    return _answer(ctx, None)


def _image_indices(args: dict[str, Any]) -> list[int] | None:
    indices = args.get("image_indices")
    if indices is None:
        return None
    if not isinstance(indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in indices
    ):
        raise ValueError("image_indices должен быть списком целых номеров")
    return indices


@register_tool("generate_image")
async def handle_generate_image(
    ctx: ToolExecutionContext, args: dict[str, Any]
) -> ToolResult:
    user_id = ctx.update.effective_user.id  # type: ignore

    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        logger.info("Пустой запрос на генерацию изображения")
        return _answer(ctx, "Не удалось сгенерировать изображение: запрос пустой.")

    aspect_ratio = _sdk_aspect_ratio(args.get("aspect_ratio"))
    image_number: int | None = None
    status_message, animation_task = await _start_image_generation_status(ctx.update)
    try:
        try:
            image = await ctx.openrouter_service.generate(
                prompt,
                ImageGenerationOptions(aspect_ratio=aspect_ratio),
            )
        except OpenRouterMediaError as error:
            return await _report_media_error(ctx, error, context="image")
        except Exception as error:
            logger.error("Ошибка при генерации изображения: %s", error, exc_info=True)
            return _answer(
                ctx,
                "Не удалось сгенерировать изображение. Внутренняя ошибка сервера",
            )
        finally:
            await _cancel_image_generation_animation(animation_task)

        try:
            normalized_b64, gen_width, gen_height, gen_aspect_ratio = (
                await send_image_to_telegram(ctx.update, image)
            )
        except Exception as error:
            logger.error(
                "Ошибка при отправке картинки в Telegram: %s",
                error,
                exc_info=True,
            )
            return _answer(
                ctx,
                "Картинка сгенерировалась, но не удалось отправить её в Telegram.",
            )

        await _delete_image_generation_status(status_message)
        status_message = None

        try:
            new_image_dict = build_image_dict_from_image(
                image,
                normalized_b64,
                gen_width,
                gen_height,
                gen_aspect_ratio,
            )
            image_number = await append_user_image_to_session(
                user_id,
                new_image_dict,
                "text2image",
                prompt,
            )
        except Exception as error:
            logger.error(
                "Ошибка при создании visual session после generate_image: %s",
                error,
                exc_info=True,
            )

        return _answer(
            ctx,
            (
                f"Я сделал :) Изображение №{image_number} добавлено в сессию и доступно "
                "как референс для следующих операций."
                if image_number is not None
                else "Я сделал :) Изображение сгенерировано, но не удалось сохранить его в сессии."
            ),
        )
    finally:
        await _delete_image_generation_status(status_message)


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
        image_indices = _image_indices(args)
        source_image_dicts = get_session_images(session, image_indices)
        source_images = [
            generated_image_from_session_dict(image_dict)
            for image_dict in source_image_dicts
        ]
        if not source_images:
            raise ValueError("В visual session отсутствуют изображения")
    except Exception as error:
        logger.error("Ошибка при чтении исходного изображения: %s", error, exc_info=True)
        return _answer(ctx, "Не удалось подготовить исходное изображение для генерации.")

    # Images API принимает не более 10 input_references; учитываем также
    # настраиваемый лимит референсов для visual session.
    max_refs = min(10, settings.media.max_session_images)
    references_truncated = len(source_images) > max_refs
    source_images = source_images[-max_refs:]
    source_image_dicts = source_image_dicts[-max_refs:]

    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        logger.warning("Пустой prompt для generate_image_from_image")
        return _answer(ctx, "Не удалось сгенерировать изображение: запрос пустой.")

    explicit_aspect_ratio = _sdk_aspect_ratio(args.get("aspect_ratio"))
    if explicit_aspect_ratio is not None:
        aspect_ratio = explicit_aspect_ratio
    else:
        aspect_ratio = _sdk_aspect_ratio(source_image_dicts[0].get("aspect_ratio"))
        if len(source_images) == 1:
            primary_image = source_images[0]
            source_width = primary_image.metadata.get("width")
            source_height = primary_image.metadata.get("height")
            if not source_width or not source_height:
                source_width = source_image_dicts[0].get("width")
                source_height = source_image_dicts[0].get("height")
            if source_width and source_height:
                prompt = (
                    f"{prompt}\n\n"
                    "Important: preserve the original image aspect ratio exactly. "
                    f"The source image is {source_width}x{source_height}. Do not make it wider or taller. "
                    "Keep the same framing and canvas proportions."
                )

    status_message, animation_task = await _start_image_generation_status(ctx.update)
    try:
        try:
            image = await ctx.openrouter_service.edit(
                source_images,
                prompt,
                ImageEditOptions(aspect_ratio=aspect_ratio),
            )
        except OpenRouterMediaError as error:
            return await _report_media_error(ctx, error, context="image")
        except Exception as error:
            logger.error(
                "Ошибка при редактировании изображения через OpenRouter: %s",
                error,
                exc_info=True,
            )
            return _answer(
                ctx,
                "Не удалось сгенерировать изображение. Внутренняя ошибка сервера",
            )
        finally:
            await _cancel_image_generation_animation(animation_task)

        try:
            normalized_b64, gen_width, gen_height, gen_aspect_ratio = (
                await send_image_to_telegram(ctx.update, image)
            )
        except Exception as error:
            logger.error("Ошибка при отправке картинки в Telegram: %s", error, exc_info=True)
            return _answer(
                ctx,
                "Картинка сгенерировалась, но не удалось отправить её в Telegram.",
            )

        await _delete_image_generation_status(status_message)
        status_message = None

        image_number: int | None = None
        try:
            new_image_dict = build_image_dict_from_image(
                image,
                normalized_b64,
                gen_width,
                gen_height,
                gen_aspect_ratio,
            )
            image_number = await append_user_image_to_session(
                user_id,
                new_image_dict,
                "edit",
                prompt,
            )
        except Exception as error:
            logger.error("Ошибка при обновлении visual session: %s", error, exc_info=True)

        return _answer(
            ctx,
            (
                (
                    "В Images API можно передать не более 10 референсов; использованы последние 10. "
                    if references_truncated
                    else ""
                )
                + f"Я сделал :) Результат добавлен в сессию как изображение №{image_number}. "
                "Можете выбрать его или другие изображения по номерам для следующих операций."
                if image_number is not None
                else "Я сделал :) Изображение создано, но не удалось сохранить его в сессии."
            ),
        )
    finally:
        await _delete_image_generation_status(status_message)


@register_tool("generate_video")
async def handle_generate_video(
    ctx: ToolExecutionContext, args: dict[str, Any]
) -> ToolResult:
    user_id = ctx.update.effective_user.id  # type: ignore
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        logger.warning("Пустой prompt для generate_video")
        return _answer(ctx, "Не удалось сгенерировать видео: запрос пустой.")

    reference_images: list[GeneratedImage] | None = None
    session = await get_user_image_edit_session(user_id)
    try:
        image_indices = _image_indices(args)
        if session is not None:
            selected_image_dicts = get_session_images(session, image_indices)
            if selected_image_dicts:
                reference_images = [
                    generated_image_from_session_dict(image_dict)
                    for image_dict in selected_image_dicts
                ]
        elif image_indices:
            raise ValueError("В visual session отсутствуют изображения")
    except Exception as error:
        logger.error("Ошибка при чтении изображений для видео: %s", error, exc_info=True)
        return _answer(ctx, "Не удалось подготовить выбранные изображения для видео.")

    max_refs = min(10, settings.media.max_session_images)
    references_truncated = False
    if reference_images is not None and len(reference_images) > max_refs:
        references_truncated = True
        reference_images = reference_images[-max_refs:]

    status_message = await reply_service_message(ctx.update, "Видео генерируется, подождите")
    animation_task = asyncio.create_task(
        animate_service_message(
            status_message,
            "Видео генерируется, подождите",
            interval=settings.telegram.service_animation_interval_seconds,
        )
    )

    try:
        video = await ctx.openrouter_service.generate_video(prompt, reference_images)
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
        await add_user_image_session_history(user_id, "video", prompt)
    except Exception as error:
        logger.error(
            "Ошибка при обновлении history visual session после видео: %s",
            error,
            exc_info=True,
        )

    return _answer(
        ctx,
        (
            "В Video API переданы последние 10 референсов. "
            if references_truncated
            else ""
        )
        + (
            "Я сделал :) Видео создано на основе выбранных изображений. "
            "Все картинки сессии сохранены для дальнейших операций."
            if reference_images
            else "Я сделал :) Видео создано по текстовому описанию."
        ),
    )
