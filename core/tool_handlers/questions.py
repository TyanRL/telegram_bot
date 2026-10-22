"""Универсальный инструмент вопросов пользователю.

Инструмент `ask_user_question` позволяет языковой модели уточнить
недостающий параметр задачи. Модель задаёт только текст вопроса и от одного
до трёх готовых вариантов ответа; кнопка «Свой ответ» добавляется сервисом
вопросов автоматически и не может быть отключена.

Владелец вопроса, идентификаторы Telegram, параметры видеопрофиля и любые
исполняемые обработчики продолжения задаются сервером, а не моделью: такие
поля отклоняются на сервере.

Проверка структуры выполняется дважды: до выполнения пакета вызовов
([`check_ask_user_question`](core/tool_handlers/questions.py:118)) и в самом
обработчике ([`handle_ask_user_question`](core/tool_handlers/questions.py:147)).
Первая проверка гарантирует, что уточнение всегда предшествует зависимым
платным операциям того же пакета.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from core.common_types import (
    PendingQuestionRequest,
    QuestionAnswerMode,
    QuestionOption,
    ToolQuestionCheck,
    ToolQuestionPause,
    ToolResult,
)
from core.tool_registry import (
    ToolExecutionContext,
    register_tool,
    register_tool_question_check,
)
from core.user_questions import (
    MAX_UNIVERSAL_OPTIONS,
    validate_universal_option_count,
)

logger = logging.getLogger(__name__)

# Ограничения длины текстов, которые модель может передать в вопрос.
MAX_QUESTION_TEXT_LENGTH = 1000
MAX_OPTION_TEXT_LENGTH = 200
MAX_OPTION_ID_LENGTH = 64
MAX_INPUT_HINT_LENGTH = 200

# Поля, которые модель не задаёт: владелец вопроса, идентификаторы
# Telegram, параметры видеопрофиля и обработчик продолжения. Их присутствие
# отклоняется, а не игнорируется: иначе модель могла бы повлиять на серверную
# маршрутизацию ответа.
FORBIDDEN_ARGUMENT_KEYS = frozenset(
    {
        "user_id",
        "chat_id",
        "message_thread_id",
        "owner",
        "question_id",
        "callback_data",
        "duration",
        "model",
        "resolution",
        "video_profile",
        "continuation",
        "handler",
    }
)

__all__ = [
    "FORBIDDEN_ARGUMENT_KEYS",
    "MAX_OPTION_ID_LENGTH",
    "MAX_OPTION_TEXT_LENGTH",
    "MAX_QUESTION_TEXT_LENGTH",
    "MAX_UNIVERSAL_OPTIONS",
    "QuestionToolError",
    "build_question_options",
    "check_ask_user_question",
    "handle_ask_user_question",
]


class QuestionToolError(ValueError):
    """Некорректные аргументы универсального вопроса.

    Ошибка безопасна для показа модели: она не содержит внутренних деталей
    приложения.
    """


def _read_question_text(args: dict[str, Any]) -> str:
    """Читает и проверяет текст вопроса."""

    question = args.get("question")
    if not isinstance(question, str) or not question.strip():
        raise QuestionToolError("Поле question должно содержать непустой текст вопроса.")
    if len(question) > MAX_QUESTION_TEXT_LENGTH:
        raise QuestionToolError(
            f"Текст вопроса длиннее {MAX_QUESTION_TEXT_LENGTH} символов."
        )
    return question.strip()


def _read_option_id(option: dict[str, Any], position: int) -> str:
    """Читает и проверяет идентификатор варианта."""

    option_id = option.get("id")
    if not isinstance(option_id, str) or not option_id.strip():
        raise QuestionToolError(
            f"Вариант {position} должен иметь непустой строковый идентификатор id."
        )
    cleaned = option_id.strip()
    if len(cleaned) > MAX_OPTION_ID_LENGTH:
        raise QuestionToolError(
            f"Идентификатор варианта {position} длиннее {MAX_OPTION_ID_LENGTH} символов."
        )
    return cleaned


def _read_option_text(option: dict[str, Any], position: int) -> str:
    """Читает и проверяет подпись варианта.

    Подпись принимается в полях `label` или `text`: модели не обязаны
    использовать одно конкретное имя.
    """

    label = option.get("label")
    if not isinstance(label, str) or not label.strip():
        label = option.get("text")
    if not isinstance(label, str) or not label.strip():
        raise QuestionToolError(
            f"Вариант {position} должен иметь непустую подпись label или text."
        )
    if len(label) > MAX_OPTION_TEXT_LENGTH:
        raise QuestionToolError(
            f"Подпись варианта {position} длиннее {MAX_OPTION_TEXT_LENGTH} символов."
        )
    return label.strip()


def build_question_options(raw_options: Any) -> tuple[QuestionOption, ...]:
    """Проверяет список готовых вариантов и строит варианты вопроса.

    Допускается от одного до трёх вариантов с уникальными идентификаторами и
    непустыми подписями. «Свой ответ» добавляется сервисом вопросов и в
    проверку не входит.
    """

    if not isinstance(raw_options, (list, tuple)):
        raise QuestionToolError("Поле options должно быть списком готовых вариантов.")
    try:
        validate_universal_option_count(len(raw_options))
    except (TypeError, ValueError) as error:
        raise QuestionToolError(str(error)) from error

    options: list[QuestionOption] = []
    seen_ids: set[str] = set()
    for position, item in enumerate(raw_options, start=1):
        if not isinstance(item, dict):
            raise QuestionToolError(f"Вариант {position} должен быть объектом с полями id и label.")
        option_id = _read_option_id(item, position)
        if option_id in seen_ids:
            raise QuestionToolError(
                f"Идентификатор варианта '{option_id}' повторяется: идентификаторы должны "
                "быть уникальными внутри вопроса."
            )
        seen_ids.add(option_id)
        options.append(
            QuestionOption(
                option_id=option_id,
                text=_read_option_text(item, position),
            )
        )
    return tuple(options)


def _read_input_hint(args: dict[str, Any]) -> str | None:
    """Читает необязательную подсказку для собственного ответа."""

    hint = args.get("custom_input_hint")
    if hint is None:
        hint = args.get("input_hint")
    if hint is None:
        return None
    if not isinstance(hint, str) or not hint.strip():
        raise QuestionToolError("Подсказка ввода должна быть непустой строкой.")
    if len(hint) > MAX_INPUT_HINT_LENGTH:
        raise QuestionToolError(
            f"Подсказка ввода длиннее {MAX_INPUT_HINT_LENGTH} символов."
        )
    return hint.strip()


def _reject_forbidden_arguments(args: dict[str, Any]) -> None:
    """Отклоняет поля, которые задаёт только сервер."""

    forbidden = sorted(key for key in args if key in FORBIDDEN_ARGUMENT_KEYS)
    if forbidden:
        raise QuestionToolError(
            "Инструмент вопроса не принимает параметры: " + ", ".join(forbidden)
        )


def _build_request(args: dict[str, Any]) -> PendingQuestionRequest:
    """Проверяет аргументы и строит запрос вопроса пользователю."""

    _reject_forbidden_arguments(args)
    text = _read_question_text(args)
    options = build_question_options(args.get("options"))
    return PendingQuestionRequest(
        text=text,
        options=options,
        input_hint=_read_input_hint(args),
        answer_mode=QuestionAnswerMode.PLAIN,
    )


@register_tool_question_check("ask_user_question")
async def check_ask_user_question(
    ctx: ToolExecutionContext,
    args: dict[str, Any],
) -> ToolQuestionCheck:
    """Проверяет инструмент до выполнения пакета вызовов.

    Возвращает запрос вопроса, если аргументы корректны. Некорректные
    аргументы приводят к исключению: вызов получает обычную ошибку, а не
    вопрос, и модель может исправить запрос.
    """

    request = _build_request(args)
    logger.info(
        "Универсальный вопрос: вариантов=%s, длина_текста=%s",
        len(request.options),
        len(request.text),
    )
    return ToolQuestionCheck(request=request)


@register_tool("ask_user_question")
async def handle_ask_user_question(
    ctx: ToolExecutionContext,
    args: dict[str, Any],
) -> ToolResult:
    """Обрабатывает универсальный вопрос языковой модели.

    Возвращает типизированную приостановку: работа не считается
    выполненной, платные действия не запускаются, пока пользователь не
    ответит.
    """

    try:
        request = _build_request(args)
    except QuestionToolError as error:
        logger.warning("Некорректный универсальный вопрос: %s", error)
        return ToolResult(
            output={
                "ok": False,
                "error": str(error),
                "instruction": (
                    "Исправь аргументы ask_user_question: укажи непустой вопрос и от "
                    "одного до трёх готовых вариантов с уникальными id и непустыми "
                    "подписями."
                ),
            },
        )

    return ToolResult(
        output=None,
        pause=ToolQuestionPause(request=request),
    )


def build_option_payloads(options: Sequence[QuestionOption]) -> dict[str, str]:
    """Возвращает соответствие идентификаторов вариантов их подписям.

    Используется при формировании окончательного результата инструмента:
    модель получает выбранный вариант структурированными данными.
    """

    return {option.option_id: option.text for option in options}
