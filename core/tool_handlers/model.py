import logging
from typing import Any

from core.common_types import ToolResult
from core.state_and_commands import get_OpenAI_Models, reply_service_text, set_user_model
from core.tool_registry import register_tool, ToolExecutionContext

logger = logging.getLogger(__name__)


@register_tool("change_model")
async def handle_change_model(ctx: ToolExecutionContext, args: dict[str, Any]) -> ToolResult:
    new_model_name_str = args["model"]
    new_model_name = get_OpenAI_Models(new_model_name_str)
    if new_model_name_str != ctx.model_name:
        await set_user_model(ctx.update.effective_user.id, new_model_name) # type: ignore
        await reply_service_text(
            ctx.update,
            f"Модель успешно изменена на {new_model_name_str}. Для возврата на стандартную модель сбросьте контекст (/reset)",
        )
    return ToolResult(
        {"ok": True, "message": f"Модель пользователя установлена: {new_model_name_str}."},
        ctx_token=ctx.context_tokens,
        completion_token=ctx.completion_tokens,
    )


@register_tool("get_location_by_address")
async def handle_get_location_by_address(ctx: ToolExecutionContext, args: dict[str, Any]) -> ToolResult:
    address = args["address"]
    logger.info(f"Вызываем get_location_by_address для адреса: {address}")

    try:
        from utils.yandex_maps import get_location_by_address
        geoloc = get_location_by_address(address)
        if geoloc is None:
            error_msg = f"Не удалось получить геолокацию для адреса '{address}'. Проверьте правильность написания адреса и доступность сервиса геокодирования."
            logger.error(error_msg)
            await reply_service_text(ctx.update, error_msg)
            return ToolResult(
                {"ok": False, "error": error_msg},
                ctx_token=ctx.context_tokens,
                completion_token=ctx.completion_tokens,
            )

        latitude, longitude = geoloc
        result = f"Геолокация для '{address}' установлена. Широта: {latitude}, Долгота: {longitude}"
        logger.info(result)

        new_system_message = {"role": "system", "content": result}
        return ToolResult(
            {"ok": True, "result": result},
            additional_system_messages=[new_system_message],
            ctx_token=ctx.context_tokens,
            completion_token=ctx.completion_tokens,
        )
    except Exception as e:
        error_msg = f"Ошибка при получении геолокации для адреса '{address}': {e}"
        logger.error(error_msg, exc_info=True)
        await reply_service_text(ctx.update, "Произошла ошибка при получении геолокации. Попробуйте позже.")
        return ToolResult(
            {"ok": False, "error": "Произошла ошибка при получении геолокации."},
            ctx_token=ctx.context_tokens,
            completion_token=ctx.completion_tokens,
        )
