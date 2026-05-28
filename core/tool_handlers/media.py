import asyncio
import logging
from typing import Any

from core.common_types import ModelAnswer
from core.generation_error_mapper import map_generation_error
from core.state_and_commands import (
    animate_service_message,
    get_user_image_edit_session,
    reply_service_message,
    reply_service_text,
    set_user_generation_source_image,
    set_user_image_edit_session,
)
from core.tool_helpers import (
    add_history_entry,
    build_image_dict_from_data_url,
    
    send_image_to_telegram,
    send_video_to_telegram,
)
from core.tool_registry import register_tool, ToolExecutionContext
from utils.openrouter_images import generate_image_openrouter, OpenRouterImageError
from utils.openrouter_videos import (
    download_video,
    poll_video_generation,
    submit_video_generation,
    OpenRouterVideoError,
)

logger = logging.getLogger(__name__)


@register_tool("generate_image")
async def handle_generate_image(ctx: ToolExecutionContext, args: dict[str, Any]) -> ModelAnswer:
    user_id = ctx.update.effective_user.id # type: ignore

    session = await get_user_image_edit_session(user_id)
    if session is not None:
        bot_reply = (
            "У вас уже есть активная сессия редактирования изображения. "
            "Чтобы сгенерировать новое изображение с нуля, сначала выполните /reset."
        )
        return ModelAnswer(bot_reply, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)

    prompt = args.get("prompt")
    if not prompt:
        logger.info("Пустой запрос на генерацию изображения")
        return ModelAnswer("Не удалось сгенерировать изображение: запрос пустой.", ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)

    try:
        image_urls = await generate_image_openrouter(prompt=prompt)
        if not image_urls:
            logger.error("OpenRouter не вернул изображений")
            return ModelAnswer("Не удалось сгенерировать изображение. Внутренняя ошибка сервера", ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)
        image_data_url = image_urls[0]
    except OpenRouterImageError as e:
        mapped = map_generation_error(e, context="image")
        await reply_service_text(ctx.update, mapped.user_message)
        return ModelAnswer(None, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)
    except Exception as e:
        logger.error(f"Ошибка при генерации изображения: {e}", exc_info=True)
        return ModelAnswer("Не удалось сгенерировать изображение. Внутренняя ошибка сервера", ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)

    try:
        normalized_b64, gen_width, gen_height, gen_aspect_ratio = await send_image_to_telegram(
            ctx.update, image_data_url
        )
    except Exception as e:
        logger.error(f"Ошибка при отправке картинки в Telegram: {e}", exc_info=True)
        return ModelAnswer("Картинка сгенерировалась, но не удалось отправить её в Telegram.")

    # Создаём новую visual session
    try:
        if image_data_url.startswith("data:") and normalized_b64:
            new_image_dict = build_image_dict_from_data_url(
                image_data_url, normalized_b64, gen_width, gen_height, gen_aspect_ratio
            )
            new_session = {
                "original_image": new_image_dict,
                "current_image": new_image_dict,
                "source_kind": "generated",
                "history": [],
            }
            add_history_entry(new_session, "text2image", prompt)
            await set_user_image_edit_session(user_id, new_session)
            await set_user_generation_source_image(user_id, new_image_dict)
    except Exception as e:
        logger.error(f"Ошибка при создании visual session после generate_image: {e}", exc_info=True)

    bot_reply = "Я сделал :) Изображение сгенерировано и стало текущей основой для редактирования."
    return ModelAnswer(bot_reply, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)


@register_tool("generate_image_from_image")
async def handle_generate_image_from_image(ctx: ToolExecutionContext, args: dict[str, Any]) -> ModelAnswer:
    user_id = ctx.update.effective_user.id # type: ignore

    session = await get_user_image_edit_session(user_id)
    if session is None:
        logger.warning(f"generate_image_from_image вызван, но у пользователя {user_id} нет активной visual session")
        bot_reply = "Сначала отправьте изображение, которое будет использоваться как основа для генерации."
        return ModelAnswer(bot_reply, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)

    source_image_dict = session.get("current_image")
    if source_image_dict is None:
        logger.error(f"Visual session существует, но current_image отсутствует для пользователя {user_id}")
        bot_reply = "Ошибка: в сессии отсутствует текущее изображение."
        return ModelAnswer(bot_reply, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)

    try:
        img_type = source_image_dict["image_type"]
        img_b64_str = source_image_dict["image"]
        data_url = f"data:{img_type};base64,{img_b64_str}"
    except Exception as e:
        logger.error(f"Ошибка при сборке data URL из сохраненного изображения: {e}", exc_info=True)
        bot_reply = "Не удалось подготовить исходное изображение для генерации."
        return ModelAnswer(bot_reply, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)

    prompt = args.get("prompt")
    if not prompt:
        logger.warning("Пустой prompt для generate_image_from_image")
        bot_reply = "Не удалось сгенерировать изображение: запрос пустой."
        return ModelAnswer(bot_reply, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)

    w = source_image_dict.get("width")
    h = source_image_dict.get("height")
    if w and h:
        prompt = (
            f"{prompt}\n\n"
            f"Important: preserve the original image aspect ratio exactly. "
            f"The source image is {w}x{h}. Do not make it wider or taller. "
            f"Keep the same framing and canvas proportions."
        )

    aspect_ratio = source_image_dict.get("aspect_ratio")

    try:
        image_urls = await generate_image_openrouter(prompt=prompt, input_images=[data_url], aspect_ratio=aspect_ratio)
        if not image_urls:
            logger.error("OpenRouter не вернул изображений для img2img")
            bot_reply = "Не удалось сгенерировать изображение. Внутренняя ошибка сервера"
            return ModelAnswer(bot_reply, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)
        image_data_url = image_urls[0]
    except OpenRouterImageError as e:
        mapped = map_generation_error(e, context="image")
        await reply_service_text(ctx.update, mapped.user_message)
        return ModelAnswer(None, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)
    except Exception as e:
        logger.error(f"Ошибка при генерации изображения через OpenRouter (img2img): {e}", exc_info=True)
        bot_reply = "Не удалось сгенерировать изображение. Внутренняя ошибка сервера"
        return ModelAnswer(bot_reply, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)

    try:
        normalized_b64, gen_width, gen_height, gen_aspect_ratio = await send_image_to_telegram(
            ctx.update, image_data_url, target_aspect_ratio=aspect_ratio
        )
    except Exception as e:
        logger.error(f"Ошибка при отправке картинки в Telegram: {e}", exc_info=True)
        return ModelAnswer("Картинка сгенерировалась, но не удалось отправить её в Telegram.")

    # Обновляем session.current_image
    try:
        if image_data_url.startswith("data:") and normalized_b64:
            new_image_dict = build_image_dict_from_data_url(
                image_data_url, normalized_b64, gen_width, gen_height, gen_aspect_ratio
            )
        else:
            new_image_dict = source_image_dict

        session["current_image"] = new_image_dict
        add_history_entry(session, "edit", prompt)
        await set_user_image_edit_session(user_id, session)
        await set_user_generation_source_image(user_id, new_image_dict)
    except Exception as e:
        logger.error(f"Ошибка при обновлении visual session: {e}", exc_info=True)

    bot_reply = "Я сделал :) Можете отправлять следующую инструкцию для редактирования без повторной загрузки картинки."
    return ModelAnswer(bot_reply, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)


@register_tool("generate_video")
async def handle_generate_video(ctx: ToolExecutionContext, args: dict[str, Any]) -> ModelAnswer:
    user_id = ctx.update.effective_user.id # type: ignore
    prompt = args.get("prompt")
    if not prompt:
        logger.warning("Пустой prompt для generate_video")
        bot_reply = "Не удалось сгенерировать видео: запрос пустой."
        return ModelAnswer(bot_reply, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)

    session = await get_user_image_edit_session(user_id)
    input_references = None
    if session is not None:
        source_image_dict = session.get("current_image")
        if source_image_dict is not None:
            try:
                img_type = source_image_dict["image_type"]
                img_b64_str = source_image_dict["image"]
                data_url = f"data:{img_type};base64,{img_b64_str}"
                input_references = [
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            except Exception as e:
                logger.error(f"Ошибка при сборке data URL для видео: {e}", exc_info=True)

    try:
        job_id, polling_url = await submit_video_generation(prompt, input_references)
        logger.info(f"Video job submitted: {job_id}")
    except OpenRouterVideoError as e:
        mapped = map_generation_error(e, context="video")
        await reply_service_text(ctx.update, mapped.user_message)
        return ModelAnswer(None, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)
    except Exception as e:
        logger.error(f"Ошибка при отправке запроса на генерацию видео: {e}", exc_info=True)
        bot_reply = "Не удалось начать генерацию видео. Попробуйте позже."
        return ModelAnswer(bot_reply, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)

    status_message = await reply_service_message(ctx.update, "Видео генерируется, подождите")
    animation_task = asyncio.create_task(
        animate_service_message(status_message, "Видео генерируется, подождите", interval=1.5)
    )

    try:
        unsigned_urls = await poll_video_generation(polling_url)
        video_url = unsigned_urls[0]
        video_path = await download_video(video_url)
    except OpenRouterVideoError as e:
        logger.error(f"Ошибка при генерации видео: {e}", exc_info=True)
        try:
            await status_message.edit_text("_Не удалось сгенерировать видео._", parse_mode="MarkdownV2")
        except Exception:
            pass
        return ModelAnswer(None, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)
    except Exception as e:
        logger.error(f"Неожиданная ошибка при генерации видео: {e}", exc_info=True)
        try:
            await status_message.edit_text("_Произошла ошибка при генерации видео._", parse_mode="MarkdownV2")
        except Exception:
            pass
        return ModelAnswer(None, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)
    finally:
        animation_task.cancel()
        try:
            await animation_task
        except asyncio.CancelledError:
            pass

    try:
        await status_message.edit_text("_Видео готово, отправляю..._", parse_mode="MarkdownV2")
    except Exception:
        pass

    success = await send_video_to_telegram(ctx.update, video_path, status_message)
    if not success:
        return ModelAnswer(None, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)

    # Добавляем запись в history
    try:
        session = await get_user_image_edit_session(user_id)
        if session is not None:
            add_history_entry(session, "video", prompt)
            await set_user_image_edit_session(user_id, session)
    except Exception as e:
        logger.error(f"Ошибка при обновлении history visual session после видео: {e}", exc_info=True)

    bot_reply = "Я сделал :) Видео создано на основе текущей версии изображения. Текущая картинка для дальнейших правок сохранена."
    return ModelAnswer(bot_reply, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)
