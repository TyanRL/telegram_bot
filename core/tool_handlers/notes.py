import logging
from typing import Any

from core.common_types import ToolResult
from core.state_and_commands import get_notes_text, reply_service_text
from core.tool_registry import register_tool, ToolExecutionContext

logger = logging.getLogger(__name__)


@register_tool("add_note")
async def handle_add_note(ctx: ToolExecutionContext, args: dict[str, Any]) -> ToolResult:
    from utils.elastic import add_note
    title = args["title"]
    body = args["body"]
    tags = args["tags"]
    add_note(ctx.update.effective_user.id,# type: ignore
             title, body, tags) 
    await reply_service_text(ctx.update, f"Заметка '{title}' добавлена.")
    return ToolResult(
        {"ok": True, "message": f"Заметка '{title}' добавлена."},
        ctx_token=ctx.context_tokens,
        completion_token=ctx.completion_tokens,
    )


@register_tool("get_all_user_notes")
async def handle_get_all_user_notes(ctx: ToolExecutionContext, _args: dict[str, Any]) -> ToolResult:
    logger.info("Вызываем функцию получения всех заметок.")
    from utils.elastic import get_all_user_notes
    documents = get_all_user_notes(ctx.update.effective_user.id)# type: ignore
    if len(documents) == 0:
        await reply_service_text(ctx.update, "Заметки не найдены.")
        return ToolResult(
            {"ok": True, "result": "У пользователя нет заметок."},
            ctx_token=ctx.context_tokens,
            completion_token=ctx.completion_tokens,
        )

    answer, system_message_body = get_notes_text(documents)
    new_system_message = {"role": "system", "content": system_message_body}

    await reply_service_text(ctx.update, f"Найдено {len(documents)} заметки(-ок).")
    return ToolResult(
        {"ok": True, "result": answer},
        additional_system_messages=[new_system_message],
        ctx_token=ctx.context_tokens,
        completion_token=ctx.completion_tokens,
    )


@register_tool("get_notes_by_query")
async def handle_get_notes_by_query(ctx: ToolExecutionContext, args: dict[str, Any]) -> ToolResult:
    search_query = args["search_query"]
    start_date = args.get("start_created_date") or ""
    end_date = args.get("end_created_date") or ""

    from utils.elastic import get_notes_by_query
    documents = get_notes_by_query(ctx.update.effective_user.id,# type: ignore
                                   search_query, start_date, end_date)
    if len(documents) == 0:
        await reply_service_text(ctx.update, "Заметки не найдены.")
        return ToolResult(
            {"ok": True, "result": "По заданному запросу заметки не найдены."},
            ctx_token=ctx.context_tokens,
            completion_token=ctx.completion_tokens,
        )

    answer, system_message_body = get_notes_text(documents)
    new_system_message = {"role": "system", "content": system_message_body}
    return ToolResult(
        {"ok": True, "result": answer},
        additional_system_messages=[new_system_message],
        ctx_token=ctx.context_tokens,
        completion_token=ctx.completion_tokens,
    )


@register_tool("remove_notes")
async def handle_remove_notes(ctx: ToolExecutionContext, args: dict[str, Any]) -> ToolResult:
    note_ids = [int(x) for x in args["note_ids"]]
    from utils.elastic import remove_notes
    await remove_notes(note_ids)
    return ToolResult(
        {"ok": True, "message": "Заметки удалены."},
        ctx_token=ctx.context_tokens,
        completion_token=ctx.completion_tokens,
    )
