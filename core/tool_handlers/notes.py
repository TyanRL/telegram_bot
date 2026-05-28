import logging
from typing import Any

from core.common_types import ModelAnswer
from core.state_and_commands import get_notes_text, reply_service_text
from core.tool_registry import register_tool, ToolExecutionContext

logger = logging.getLogger(__name__)


@register_tool("add_note")
async def handle_add_note(ctx: ToolExecutionContext, args: dict[str, Any]) -> ModelAnswer:
    from utils.elastic import add_note
    title = args["title"]
    body = args["body"]
    tags = args["tags"]
    add_note(ctx.update.effective_user.id,# type: ignore
             title, body, tags) 
    await reply_service_text(ctx.update, f"Заметка '{title}' добавлена.")
    bot_reply = "Я сделал :)"
    return ModelAnswer(bot_reply, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)


@register_tool("get_all_user_notes")
async def handle_get_all_user_notes(ctx: ToolExecutionContext, _args: dict[str, Any]) -> ModelAnswer:
    logger.info("Вызываем функцию получения всех заметок.")
    from utils.elastic import get_all_user_notes
    documents = get_all_user_notes(ctx.update.effective_user.id)# type: ignore
    if len(documents) == 0:
        await reply_service_text(ctx.update, "Заметки не найдены.")
        return ModelAnswer(None, [], ctx.context_tokens, ctx.completion_tokens)

    answer, system_message_body = get_notes_text(documents)
    new_system_message = {"role": "system", "content": system_message_body}
    ctx.additional_system_messages.append(new_system_message)

    await reply_service_text(ctx.update, f"Найдено {len(documents)} заметки(-ок).")
    return ModelAnswer(answer, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)


@register_tool("get_notes_by_query")
async def handle_get_notes_by_query(ctx: ToolExecutionContext, args: dict[str, Any]) -> ModelAnswer:
    search_query = args["search_query"]
    start_date = args.get("start_created_date") or ""
    end_date = args.get("end_created_date") or ""

    from utils.elastic import get_notes_by_query
    documents = get_notes_by_query(ctx.update.effective_user.id,# type: ignore
                                   search_query, start_date, end_date)
    if len(documents) == 0:
        await reply_service_text(ctx.update, "Заметки не найдены.")
        return ModelAnswer(None, [], ctx.context_tokens, ctx.completion_tokens)

    answer, system_message_body = get_notes_text(documents)
    new_system_message = {"role": "system", "content": system_message_body}
    ctx.additional_system_messages.append(new_system_message)
    ctx.messages.append(new_system_message)

    return ModelAnswer(None, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens, recurse=True)


@register_tool("remove_notes")
async def handle_remove_notes(ctx: ToolExecutionContext, args: dict[str, Any]) -> ModelAnswer:
    note_ids = [int(x) for x in args["note_ids"]]
    from utils.elastic import remove_notes
    await remove_notes(note_ids)
    bot_reply = "Заметки удалены"
    return ModelAnswer(bot_reply, ctx.additional_system_messages, ctx.context_tokens, ctx.completion_tokens)
