import logging
from typing import Any

from core.common_types import ToolResult
from core.state_and_commands import add_location_button, reply_service_text
from core.tool_registry import register_tool, ToolExecutionContext
from utils.weather import get_weather_description2, get_weekly_forecast

logger = logging.getLogger(__name__)


@register_tool("request_geolocation")
async def handle_request_geolocation(
    ctx: ToolExecutionContext, _args: dict[str, Any]
) -> ToolResult:
    logger.info("Вызываем функцию запроса геолокации")
    await add_location_button(ctx.update, ctx.context)
    return ToolResult(
        {"ok": True, "message": "Пользователю показана кнопка отправки геолокации."},
        ctx_token=ctx.context_tokens,
        completion_token=ctx.completion_tokens,
    )


@register_tool("get_weather_description")
async def handle_get_weather_description(
    ctx: ToolExecutionContext, args: dict[str, Any]
) -> ToolResult:
    return await _handle_weather_tool(ctx, args, get_weather_description2, "get_weather_description")


@register_tool("get_weekly_forecast")
async def handle_get_weekly_forecast(
    ctx: ToolExecutionContext, args: dict[str, Any]
) -> ToolResult:
    return await _handle_weather_tool(ctx, args, get_weekly_forecast, "get_weekly_forecast")


async def _handle_weather_tool(
    ctx: ToolExecutionContext,
    args: dict[str, Any],
    weather_func,
    func_name: str,
) -> ToolResult:
    try:
        latitude = args["latitude"]
        longitude = args["longitude"]

        logger.info(f"Вызываем {func_name} для координат: {latitude}, {longitude}")

        result = weather_func(latitude, longitude)

        if not result or result.startswith("Ошибка"):
            error_msg = f"Не удалось получить данные о погоде для координат {latitude}, {longitude}"
            logger.error(error_msg)
            await reply_service_text(ctx.update, error_msg)
            return ToolResult(
                {"ok": False, "error": error_msg},
                ctx_token=ctx.context_tokens,
                completion_token=ctx.completion_tokens,
            )

        logger.info(f"Данные о погоде получены: {result[:100]}...")
        new_system_message = {"role": "system", "content": result}
        return ToolResult(
            {"ok": True, "result": result},
            additional_system_messages=[new_system_message],
            ctx_token=ctx.context_tokens,
            completion_token=ctx.completion_tokens,
        )
    except Exception as e:
        error_msg = f"Ошибка при получении данных о погоде: {e}"
        logger.error(error_msg, exc_info=True)
        await reply_service_text(ctx.update, "Произошла ошибка при получении данных о погоде. Попробуйте позже.")
        return ToolResult(
            {"ok": False, "error": "Произошла ошибка при получении данных о погоде."},
            ctx_token=ctx.context_tokens,
            completion_token=ctx.completion_tokens,
        )
