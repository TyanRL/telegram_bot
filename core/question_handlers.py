"""Telegram-маршрутизация вопросов пользователю.

Модуль отвечает за обработку inline-кнопок вопроса, перехват текстовых и
голосовых ответов и команду отмены. Он не импортирует точку запуска
приложения: функция продолжения передаётся как зависимость, поэтому
циклических импортов нет, а Шаг 5 плана подключает её без изменения
обработчиков.

Порядок проверки нажатия:

1. подтверждение нажатия (`query.answer()`) выполняется немедленно, до любых
   проверок и до потенциально долгого продолжения;
2. проверка доступа к боту;
3. сверка владельца вопроса, чата, темы и сообщения;
4. проверка срока и статуса вопроса.

Чужое, поддельное, просроченное или повторное нажатие отклоняется безопасно
и не запускает продолжение.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from telegram import Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from core.common_types import (
    QuestionAnswer,
    QuestionAnswerKind,
    QuestionStatus,
)
from core.generation_error_mapper import (
    GenerationFailureReason,
    user_message_for_reason,
)
from core.question_state import PendingQuestion, QuestionStateStore, get_question_store
from core.state_and_commands import get_all_histories
from core.user_questions import (
    CANCEL_ACTION,
    CUSTOM_ANSWER_ACTION,
    QuestionContinuation,
    UserQuestionService,
    parse_callback_data,
)
from utils.sql import in_user_list

__all__ = [
    "QuestionContinuation",
    "handle_question_callback",
    "handle_question_text",
    "register_question_handlers",
]

logger = logging.getLogger(__name__)

# Сообщения безопасных отказов.
QUESTION_NOT_FOUND_TEXT = "Вопрос не найден или уже неактуален."
QUESTION_FOREIGN_TEXT = "Этот вопрос принадлежит другому чату или пользователю."
QUESTION_EXPIRED_TEXT = user_message_for_reason(
    GenerationFailureReason.QUESTION_EXPIRED
)
QUESTION_ALREADY_ANSWERED_TEXT = "Ответ на этот вопрос уже принят."
QUESTION_FORBIDDEN_TEXT = "Нет доступа к этому боту."
QUESTION_CANCELLED_TEXT = "Вопрос отменён."
CUSTOM_ANSWER_PROMPT = (
    "Напишите ответ текстом — он будет использован в текущей задаче. "
    "Вопрос останется активным, пока вы не ответите."
)


def get_question_service(context: ContextTypes.DEFAULT_TYPE) -> UserQuestionService:
    """Возвращает сервис вопросов из данных приложения.

    Если сервис не зарегистрирован, используется сервис с общим хранилищем
    ожиданий: обработчики остаются работоспособными, но продолжение не
    настроено.
    """

    application = getattr(context, "application", None)
    bot_data = getattr(application, "bot_data", None)
    if isinstance(bot_data, dict):
        service = bot_data.get("question_service")
        if isinstance(service, UserQuestionService):
            return service
    return UserQuestionService(store=get_question_store())


async def _safe_answer(query: Any, text: str | None = None) -> None:
    """Подтверждает нажатие, не допуская ошибку ответа Telegram."""

    try:
        if text is None:
            await query.answer()
        else:
            await query.answer(text=text, show_alert=True)
    except Exception as error:  # pragma: no cover - защита от ошибок Telegram
        logger.debug("Не удалось подтвердить нажатие: type=%s", type(error).__name__)


async def _record_history(
    user_id: int,
    question_text: str,
    answer_text: str,
) -> None:
    """Записывает вопрос и ответ в локальную историю пользователя.

    Вопрос сохраняется как сообщение ассистента, ответ — как пользовательский
    ввод. Ни один из них не становится системной инструкцией.
    """

    histories = get_all_histories()
    history = list(await histories.get(user_id, []))
    history.append({"role": "assistant", "content": question_text})
    history.append({"role": "user", "content": answer_text})
    await histories.set(user_id, history)


async def _clear_keyboard_and_show_answer(
    message: Any,
    question_text: str,
    answer_text: str,
) -> None:
    """Убирает клавиатуру и показывает выбранный ответ в сообщении вопроса."""

    if message is None:
        return
    try:
        await message.edit_text(
            f"{question_text}\n\nОтвет: {answer_text}",
            reply_markup=None,
        )
        return
    except Exception as error:
        logger.debug(
            "Не удалось обновить текст сообщения вопроса: type=%s",
            type(error).__name__,
        )
    try:
        await message.edit_reply_markup(reply_markup=None)
    except Exception as error:  # pragma: no cover - защита от ошибок Telegram
        logger.debug(
            "Не удалось убрать клавиатуру вопроса: type=%s",
            type(error).__name__,
        )


async def _show_question_hint(message: Any, hint: str | None) -> None:
    """Показывает подсказку для собственного ответа, оставляя вопрос открытым."""

    text = hint or CUSTOM_ANSWER_PROMPT
    if message is None:
        return
    try:
        await message.reply_text(text)
    except Exception as error:  # pragma: no cover - защита от ошибок Telegram
        logger.debug(
            "Не удалось отправить подсказку по вопросу: type=%s",
            type(error).__name__,
        )


async def _reject(callback_data: Any, reason: str) -> None:
    """Отклоняет нажатие и убирает клавиатуру устаревшего вопроса."""

    logger.info("Нажатие по вопросу отклонено: %s", reason)
    if callback_data is None:
        return
    try:
        await callback_data.message.edit_reply_markup(reply_markup=None)
    except Exception as error:  # pragma: no cover - защита от ошибок Telegram
        logger.debug(
            "Не удалось убрать клавиатуру отклонённого вопроса: type=%s",
            type(error).__name__,
        )


async def handle_question_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Обрабатывает нажатие inline-кнопки вопроса."""

    query = update.callback_query
    if query is None:
        return

    parsed = parse_callback_data(query.data)
    if parsed is None:
        # Кнопка не относится к вопросам: подтверждаем и не вмешиваемся.
        await _safe_answer(query)
        return

    # Подтверждение выполняется немедленно, до проверок и до продолжения.
    await _safe_answer(query)

    user = query.from_user
    message = query.message
    if user is None or message is None:
        return

    if not await in_user_list(user):
        logger.error("Нет доступа к боту: user_id=%s", user.id)
        await _safe_answer(query, QUESTION_FORBIDDEN_TEXT)
        return

    service = get_question_service(context)
    record = await service.store.get(parsed.question_id)
    if record is None:
        await _reject(query, "вопрос не найден")
        await _safe_answer(query, QUESTION_NOT_FOUND_TEXT)
        return

    if record.user_id != user.id:
        await _reject(query, "вопрос принадлежит другому пользователю")
        await _safe_answer(query, QUESTION_FOREIGN_TEXT)
        return

    chat = message.chat
    if chat is not None and record.chat_id != chat.id:
        await _reject(query, "вопрос принадлежит другому чату")
        await _safe_answer(query, QUESTION_FOREIGN_TEXT)
        return

    message_thread_id = getattr(message, "message_thread_id", None)
    if record.message_thread_id != message_thread_id:
        await _reject(query, "вопрос принадлежит другой теме")
        await _safe_answer(query, QUESTION_FOREIGN_TEXT)
        return

    if (
        record.question_message_id is not None
        and message.message_id is not None
        and record.question_message_id != message.message_id
    ):
        await _reject(query, "нажатие по устаревшему сообщению")
        await _safe_answer(query, QUESTION_NOT_FOUND_TEXT)
        return

    if record.status is not QuestionStatus.PENDING:
        await _reject(query, f"вопрос в состоянии {record.status.value}")
        await _safe_answer(query, QUESTION_ALREADY_ANSWERED_TEXT)
        return

    if await service.store.is_expired(record.question_id):
        await service.store.expire(record.question_id)
        await _reject(query, "срок ожидания истёк")
        await _safe_answer(query, QUESTION_EXPIRED_TEXT)
        return

    if parsed.action == CUSTOM_ANSWER_ACTION:
        # Кнопка «Свой ответ» только показывает подсказку и не завершает
        # ожидание: вопрос остаётся активным.
        await _show_question_hint(message, record.input_hint)
        return

    if parsed.action == CANCEL_ACTION:
        cancelled = await service.handle_cancel(record)
        if not cancelled:
            await _safe_answer(query, QUESTION_ALREADY_ANSWERED_TEXT)
            return
        await _clear_keyboard_and_show_answer(message, record.text, QUESTION_CANCELLED_TEXT)
        await _record_history(record.user_id, record.text, QUESTION_CANCELLED_TEXT)
        await _safe_answer(query, QUESTION_CANCELLED_TEXT)
        return

    if not parsed.action.isdigit():
        await _reject(query, "неизвестное действие")
        await _safe_answer(query, QUESTION_NOT_FOUND_TEXT)
        return

    answer = await service.handle_option(record, int(parsed.action))
    if answer is None:
        await _reject(query, "вариант ответа не принят")
        await _safe_answer(query, QUESTION_ALREADY_ANSWERED_TEXT)
        return

    await _clear_keyboard_and_show_answer(message, record.text, answer.text)
    await _record_history(record.user_id, record.text, answer.text)
    await _run_continuation(service, record, answer, update, context)


async def _run_continuation(
    service: UserQuestionService,
    record: PendingQuestion,
    answer: QuestionAnswer,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Запускает продолжение работы после принятого ответа.

    Продолжение выполняется вне блокировки состояния: долгая генерация не
    мешает приёму других обновлений пользователя. Ошибка продолжения
    показывается пользователю безопасным сообщением и не приводит к
    повторному платному действию.
    """

    try:
        await service.continue_after_answer(
            record,
            answer,
            update=update,
            context=context,
        )
    except Exception as error:
        logger.error(
            "Ошибка продолжения после ответа на вопрос: type=%s",
            type(error).__name__,
            exc_info=True,
        )
        await _safe_answer(getattr(update, "callback_query", None), None)
        message = getattr(update, "effective_message", None)
        if message is None:
            return
        try:
            await message.reply_text(
                "Не удалось продолжить задачу после вашего ответа. "
                "Повторный платный запрос не выполнялся.",
            )
        except Exception:  # pragma: no cover - защита от ошибок Telegram
            pass


async def handle_question_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str | None = None,
) -> bool:
    """Перехватывает текстовый ответ на активный вопрос.

    Возвращает ``True``, если текст был использован как ответ на вопрос, и
    ``False``, если активного вопроса нет и обращение нужно вести обычным
    порядком.
    """

    user = getattr(update, "effective_user", None)
    chat = getattr(update, "effective_chat", None)
    if chat is None:
        message = getattr(update, "effective_message", None) or getattr(update, "message", None)
        chat = getattr(message, "chat", None)
    if user is None or chat is None:
        return False

    service = get_question_service(context)
    record = await service.store.get_active_for_user(user.id)
    if record is None:
        return False

    if record.chat_id != chat.id:
        logger.info("Текст из другого чата не засчитан как ответ на вопрос")
        return False

    message = getattr(update, "effective_message", None)
    message_thread_id = getattr(message, "message_thread_id", None)
    if record.message_thread_id != message_thread_id:
        logger.info("Текст из другой темы не засчитан как ответ на вопрос")
        return False

    if text is None:
        text = getattr(message, "text", None)
    if not isinstance(text, str) or not text.strip():
        return False

    answer, error = await service.handle_custom_answer(record, text)
    if answer is None:
        logger.info("Ответ пользователя отклонён проверкой: %s", error)
        try:
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                error or "Ответ не принят. Попробуйте ещё раз."
            )
        except Exception:  # pragma: no cover - защита от ошибок Telegram
            pass
        return True

    await _clear_keyboard_and_show_answer(
        update.effective_message,
        record.text,
        answer.text,
    )
    await _record_history(record.user_id, record.text, answer.text)
    await _run_continuation(service, record, answer, update, context)
    return True


async def cancel_question_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Команда ``/cancel``: отменяет активный вопрос пользователя."""

    user = update.effective_user
    if user is None:
        return

    service = get_question_service(context)
    record = await service.store.cancel_active_for_user(user.id)
    if record is None:
        try:
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                "Активного вопроса нет — отменять нечего."
            )
        except Exception:  # pragma: no cover - защита от ошибок Telegram
            pass
        return

    await _clear_keyboard_and_show_answer(
        update.effective_message,
        record.text,
        QUESTION_CANCELLED_TEXT,
    )
    await _record_history(record.user_id, record.text, QUESTION_CANCELLED_TEXT)
    logger.info("Вопрос %s отменён командой пользователя", record.question_id)


def register_question_handlers(
    application: Application,
    service: UserQuestionService | None = None,
    *,
    store: QuestionStateStore | None = None,
) -> UserQuestionService:
    """Регистрирует обработчики вопросов в приложении Telegram.

    Возвращает зарегистрированный сервис вопросов, чтобы его можно было
    положить в `application.bot_data` и передать зависимостью дальше.
    """

    question_service = service if service is not None else UserQuestionService(store=store)
    application.bot_data["question_service"] = question_service
    application.add_handler(CallbackQueryHandler(handle_question_callback))
    application.add_handler(CommandHandler("cancel", cancel_question_command))
    return question_service


def pending_question_options(record: PendingQuestion) -> Sequence[Any]:
    """Возвращает варианты ответа ожидания (используется в тестах и логах)."""

    return record.options


def is_cancel_answer(answer: QuestionAnswer) -> bool:
    """Проверяет, что ответ означает отмену вопроса."""

    return answer.kind is QuestionAnswerKind.CANCEL
