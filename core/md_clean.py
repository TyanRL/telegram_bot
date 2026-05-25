import asyncio
import logging
import re
from telegram.helpers import escape_markdown
import telegramify_markdown

from telegramify_markdown import telegramify


logger = logging.getLogger(__name__)

CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def extract_non_mermaid_codeblocks(text: str):
    codeblocks = []

    def repl(match: re.Match) -> str:
        lang = (match.group(1) or "").strip().lower()
        code = match.group(2)

        if lang == "mermaid":
            return match.group(0)

        placeholder = f"CODEBLOCK{len(codeblocks)}CODEBLOCK"
        codeblocks.append((placeholder, lang, code))
        return placeholder

    new_text = CODE_BLOCK_RE.sub(repl, text)
    return new_text, codeblocks


def restore_codeblocks(text: str, codeblocks: list[tuple[str, str, str]]) -> str:
    for placeholder, lang, code in codeblocks:
        # Внутри ```...``` MarkdownV2 не требует экранирования
        code = code.rstrip("\n")
        lang_part = f"{lang}\n" if lang else "\n"
        block = f"```{lang_part}{code}\n```"
        text = text.replace(placeholder, block)
    return text


async def clean_message(message: str) -> tuple[str, str | None]:
    prepared_text, codeblocks = extract_non_mermaid_codeblocks(message)

    try:
        results = await telegramify(prepared_text)
    except Exception:
        logger.exception("telegramify failed")
        return escape_markdown(message), None

    if not results:
        return escape_markdown(message), None

    text_parts = []

    for item in results:
        logger.info(f"telegramify item type: {type(item)}")

        if isinstance(item, telegramify_markdown.type.Text):
            text_parts.append(item.content)

        elif isinstance(item, telegramify_markdown.type.File):
            logger.warning("Пропускаю File item")
            continue

        elif isinstance(item, telegramify_markdown.type.Photo):
            logger.warning("Пропускаю Photo item")
            continue

    if not text_parts:
        return escape_markdown(message), None

    final_text = "\n".join(text_parts)
    final_text = restore_codeblocks(final_text, codeblocks)

    return final_text, "MarkdownV2"
