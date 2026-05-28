import logging
from typing import Any

from core.common_types import ModelAnswer
from core.state_and_commands import add_location_button, reply_service_text
from core.tool_registry import register_tool, ToolExecutionContext
from utils.weather import get_weather_description2, get_weekly_forecast

logger = logging.getLogger(__name__)


@register_tool("request_geolocation")
async def handle_request_geolocation(
    ctx: ToolExecutionContext, _args: dict[str, Any]
) -> ModelAnswer:
    logger.info("Вызываем функцию запроса геолокации")
    await add_location_button(ctx.update, ctx.context)
    return ModelAnswer(None, [], ctx.context_tokens, ctx.completion_tokens)


@register_tool("get_weather_description")
async def handle_get_weather_description(
    ctx: ToolExecutionContext, args: dict[str, Any]
) -> ModelAnswer:
    return await _handle_weather_tool(ctx, args, get_weather_description2, "get_weather_description")


@register_tool("get_weekly_forecast")
async def handle_get_weekly_forecast(
    ctx: ToolExecutionContext, args: dict[str, Any]
) -> ModelAnswer:
    return await _handle_weather_tool(ctx, args, get_weekly_forecast, "get_weekly_forecast")


async def _handle_weather_tool(
    ctx: ToolExecutionContext,
    args: dict[str, Any],
    weather_func,
    func_name: str,
) -> ModelAnswer:
    try:
        latitude = args["latitude"]
        longitude = args["longitude"]

        logger.info(f"Вызываем {func_name} для координат: {latitude}, {longitude}")

        result = weather_func(latitude, longitude)

        if not result or result.startswith("Ошибка"):
            error_msg = f"Не удалось получить данные о погоде для координат {latitude}, {longitude}"
            logger.error(error_msg)
            await reply_service_text(ctx.update, error_msg)
            return ModelAnswer(error_msg, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)

        logger.info(f"Данные о погоде получены: {result[:100]}...")
        new_system_message = {"role": "system", "content": result}
        ctx.additional_system_messages.append(new_system_message)
        ctx.messages.append(new_system_message)

        # Рекурсивный вызов обрабатывается в orchestrator
        return ModelAnswer(None, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens, recurse=True)
    except Exception as e:
        error_msg = f"Ошибка при получении данных о погоде: {e}"
        logger.error(error_msg, exc_info=True)
        await reply_service_text(ctx.update, "Произошла ошибка при получении данных о погоде. Попробуйте позже.")
        return ModelAnswer("Произошла ошибка при обработке запроса.", ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)
