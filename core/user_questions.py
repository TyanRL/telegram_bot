"""Общий сервис вопросов пользователю.

Сервис не знает ни про конкретные модели видео, ни про инструменты: он
принимает готовый список вариантов, добавляет обязательную кнопку «Свой
ответ» и отдельную служебную кнопку «Отмена», формирует компактный
`callback_data` и отправляет вопрос в Telegram.

Ограничение «до трёх готовых вариантов» проверяет вызывающий код (универсальный
инструмент вопросов), а вопрос длительности видео формируется из активного
видеопрофиля и может содержать пять значений. Сам сервис количество кнопок не
ограничивает и не ветвится по видеомодели.

Модуль не импортирует точку запуска приложения: функция продолжения
передаётся как зависимость, поэтому циклических импортов нет.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from core.common_types import (
    PendingQuestionRequest,
    QuestionAnswer,
    QuestionAnswerKind,
    QuestionOption,
)
from core.config import settings
from core.question_state import (
    PendingQuestion,
    QuestionContinuationContext,
    QuestionStateStore,
    get_question_store,
)

__all__ = [
    "CUSTOM_ANSWER_ACTION",
    "CANCEL_ACTION",
    "CALLBACK_DATA_PREFIX",
    "CUSTOM_ANSWER_TEXT",
    "CANCEL_TEXT",
    "MAX_UNIVERSAL_OPTIONS",
    "QuestionCallbackData",
    "QuestionContinuation",
    "UserQuestionService",
    "build_callback_data",
    "build_duration_question_options",
    "parse_callback_data",
    "validate_universal_option_count",
]

logger = logging.getLogger(__name__)

# Префикс callback_data отличает данные вопросов от других inline-кнопок.
CALLBACK_DATA_PREFIX = "q"

# Коды действий в callback_data.
CUSTOM_ANSWER_ACTION = "custom"
CANCEL_ACTION = "cancel"

# Ограничение Telegram: callback_data не длиннее 64 байт.
CALLBACK_DATA_MAX_BYTES = 64

CUSTOM_ANSWER_TEXT = "✍️ Свой ответ"
CANCEL_TEXT = "🚫 Отмена"

# Максимальная длина собственного ответа пользователя.
MAX_CUSTOM_ANSWER_LENGTH = 1000

# Универсальный вопрос языковой модели допускает от одного до трёх готовых
# вариантов. Ограничение проверяет вызывающий код (универсальный инструмент
# вопросов), а не сервис: вопрос длительности видео использует пять значений.
MAX_UNIVERSAL_OPTIONS = 3


@dataclass(frozen=True, slots=True)
class QuestionCallbackData:
    """Разобранные данные кнопки вопроса."""

    question_id: str
    action: str


# Функция продолжения работы после ответа пользователя. Передаётся как
# зависимость: сервис не знает, как устроен цикл модели.
QuestionContinuation = Callable[
    [PendingQuestion, QuestionAnswer, "Update | None", "ContextTypes.DEFAULT_TYPE | None"],
    Awaitable[Any],
]


def build_callback_data(question_id: str, action: str) -> str:
    """Формирует компактные данные кнопки вопроса.

    В данных кнопки попадают только непрозрачный идентификатор вопроса и код
    действия. Все реальные значения (текст варианта, payload) берутся из
    серверного состояния.
    """

    if not isinstance(question_id, str) or not question_id:
        raise ValueError("Идентификатор вопроса должен быть непустой строкой")
    if not isinstance(action, str) or not action:
        raise ValueError("Код действия должен быть непустой строкой")

    data = f"{CALLBACK_DATA_PREFIX}:{question_id}:{action}"
    if len(data.encode("utf-8")) > CALLBACK_DATA_MAX_BYTES:
        raise ValueError(
            "callback_data вопроса превышает ограничение Telegram в 64 байта: "
            f"{len(data.encode('utf-8'))} байт"
        )
    return data


def parse_callback_data(data: str | None) -> QuestionCallbackData | None:
    """Разбирает callback_data кнопки вопроса.

    Возвращает ``None``, если данные не относятся к вопросу или повреждены.
    """

    if not isinstance(data, str) or not data:
        return None
    parts = data.split(":")
    if len(parts) != 3:
        return None
    prefix, question_id, action = parts
    if prefix != CALLBACK_DATA_PREFIX:
        return None
    if not question_id or not action:
        return None
    return QuestionCallbackData(question_id=question_id, action=action)


def build_inline_keyboard(
    question_id: str,
    options: Sequence[QuestionOption],
) -> InlineKeyboardMarkup:
    """Собирает клавиатуру вопроса.

    Сначала идут переданные готовые варианты (в переданном порядке), затем
    всегда присутствует «Свой ответ» и отдельная служебная кнопка «Отмена».
    """

    rows: list[list[InlineKeyboardButton]] = []
    for index, option in enumerate(options):
        rows.append(
            [
                InlineKeyboardButton(
                    text=option.text,
                    callback_data=build_callback_data(question_id, str(index)),
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text=CUSTOM_ANSWER_TEXT,
                callback_data=build_callback_data(question_id, CUSTOM_ANSWER_ACTION),
            )
        ]
    )
    rows.append(
        [
            InlineKeyboardButton(
                text=CANCEL_TEXT,
                callback_data=build_callback_data(question_id, CANCEL_ACTION),
            )
        ]
    )
    return InlineKeyboardMarkup(rows)


class UserQuestionService:
    """Сервис вопросов пользователю.

    Отвечает за формирование сообщения с вопросом, отправку кнопок, разбор
    нажатий и проверку свободного ответа. Хранилище состояния и функция
    продолжения передаются зависимостями: сервис ничего не знает о цикле
    модели.
    """

    def __init__(
        self,
        *,
        store: QuestionStateStore | None = None,
        continuation: QuestionContinuation | None = None,
    ) -> None:
        self._store = store if store is not None else get_question_store()
        self._continuation = continuation

    @property
    def store(self) -> QuestionStateStore:
        """Хранилище ожиданий, используемое сервисом."""

        return self._store

    def set_continuation(
        self,
        continuation: QuestionContinuation | None,
    ) -> None:
        """Задаёт функцию продолжения работы после ответа пользователя."""

        self._continuation = continuation

    async def ask(
        self,
        update: Update,
        request: PendingQuestionRequest,
        *,
        continuation: QuestionContinuationContext | None = None,
        timeout_seconds: int | None = None,
    ) -> PendingQuestion:
        """Создаёт ожидание вопроса и отправляет вопрос с кнопками.

        Возвращает запись ожидания. Идентификатор сообщения вопроса
        сохраняется в состоянии, чтобы кнопки старых сообщений отклонялись.

        Вопрос отправляется в исходный чат и тему: при обновлении от
        inline-кнопки адресатом является сообщение бота, а не `update.message`.
        """

        user = update.effective_user
        chat = update.effective_chat
        if user is None or chat is None:
            raise ValueError("Вопрос нельзя отправить без пользователя и чата")

        from core.state_and_commands import get_delivery_message

        target = get_delivery_message(update)
        if target is None:
            raise ValueError("Вопрос нельзя отправить без сообщения")

        lifetime = timeout_seconds or request.timeout_seconds
        record = await self._store.create(
            user_id=user.id,
            chat_id=chat.id,
            message_thread_id=getattr(target, "message_thread_id", None),
            trigger_message_id=getattr(target, "message_id", None),
            request=request,
            continuation=continuation,
            timeout_seconds=lifetime,
        )

        keyboard = build_inline_keyboard(record.question_id, record.options)
        try:
            sent = await target.reply_text(
                request.text,
                reply_markup=keyboard,
            )
        except BaseException:
            # Не оставляем ожидание, если Telegram не принял вопрос: иначе
            # следующий обычный текст пользователя мог бы быть перехвачен как
            # ответ на вопрос, которого он не видел.
            await self._store.cancel(record.question_id)
            raise
        message_id = getattr(sent, "message_id", None)
        await self._store.set_question_message_id(record.question_id, message_id)
        record.question_message_id = message_id
        logger.info(
            "Отправлен вопрос %s пользователю %s: вариантов=%s, сообщение=%s",
            record.question_id,
            user.id,
            len(record.options),
            message_id,
        )
        return record

    async def handle_option(
        self,
        record: PendingQuestion,
        option_index: int,
    ) -> QuestionAnswer | None:
        """Принимает готовый вариант ответа.

        Возвращает принятый ответ либо ``None``, если вариант не найден или
        вопрос уже не принимает ответ.
        """

        option = record.option_by_index(option_index)
        if option is None:
            return None
        answer = QuestionAnswer(
            question_id=record.question_id,
            kind=QuestionAnswerKind.OPTION,
            text=option.text,
            option_id=option.option_id,
            payload=option.payload,
        )
        return await self._store.accept_answer(record.question_id, answer)

    async def handle_custom_answer(
        self,
        record: PendingQuestion,
        text: str,
    ) -> tuple[QuestionAnswer | None, str | None]:
        """Проверяет и принимает свободный ответ пользователя.

        Возвращает пару (принятый ответ, ошибка проверки). Если ответ
        невалиден, вопрос остаётся открытым, а ошибка описывает, что нужно
        исправить.
        """

        cleaned = (text or "").strip()
        if not cleaned:
            return None, "Ответ не может быть пустым. Пожалуйста, напишите ответ текстом."
        if len(cleaned) > MAX_CUSTOM_ANSWER_LENGTH:
            return None, (
                "Ответ слишком длинный. Максимальная длина: "
                f"{MAX_CUSTOM_ANSWER_LENGTH} символов."
            )

        error = self._validate(record, cleaned)
        if error is not None:
            return None, error

        answer = QuestionAnswer(
            question_id=record.question_id,
            kind=QuestionAnswerKind.CUSTOM,
            text=cleaned,
        )
        accepted = await self._store.accept_answer(record.question_id, answer)
        if accepted is None:
            return None, "Вопрос уже не принимает ответ. Отправьте новый запрос."
        return accepted, None

    async def handle_cancel(self, record: PendingQuestion) -> bool:
        """Отменяет вопрос по кнопке «Отмена»."""

        return await self._store.cancel(record.question_id)

    def _validate(self, record: PendingQuestion, text: str) -> str | None:
        """Применяет серверный валидатор ответа, если он задан."""

        validator = record.validator
        if validator is None:
            return None
        try:
            return validator(text)
        except Exception as error:  # pragma: no cover - защита от ошибок валидатора
            logger.error(
                "Ошибка валидатора ответа: type=%s",
                type(error).__name__,
                exc_info=True,
            )
            return "Не удалось проверить ответ. Попробуйте ещё раз."

    async def continue_after_answer(
        self,
        record: PendingQuestion,
        answer: QuestionAnswer,
        update: Update | None = None,
        context: ContextTypes.DEFAULT_TYPE | None = None,
    ) -> Any:
        """Вызывает функцию продолжения, переданную как зависимость.

        Продолжение выполняется вне блокировки состояния: долгая генерация не
        блокирует приём других обновлений пользователя.
        """

        if self._continuation is None:
            logger.error("Функция продолжения вопроса не задана")
            raise RuntimeError("Функция продолжения вопроса не настроена")
        if not await self._store.mark_continuing(record.question_id):
            logger.info(
                "Вопрос %s не переведён в продолжение: состояние изменилось",
                record.question_id,
            )
            return None
        try:
            return await self._continuation(record, answer, update, context)
        finally:
            await self._store.complete(record.question_id)


def validate_universal_option_count(count: int) -> int:
    """Проверяет число готовых вариантов универсального вопроса.

    Допускается от одного до трёх вариантов. «Свой ответ» не считается
    готовым вариантом и добавляется сервисом автоматически.
    """

    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError(
            "Число готовых вариантов должно быть целым числом, получено "
            f"{type(count).__name__}"
        )
    if count < 1 or count > MAX_UNIVERSAL_OPTIONS:
        raise ValueError(
            "Универсальный вопрос допускает от 1 до "
            f"{MAX_UNIVERSAL_OPTIONS} готовых вариантов, получено {count}"
        )
    return count


def build_duration_question_options(
    presets: Sequence[int],
) -> tuple[QuestionOption, ...]:
    """Собирает варианты ответа длительности из активного видеопрофиля.

    Значение варианта хранится в серверном состоянии (`payload`), а не в
    данных кнопки.
    """

    options: list[QuestionOption] = []
    for value in presets:
        options.append(
            QuestionOption(
                option_id=f"duration_{value}",
                text=f"{value} с",
                payload=value,
            )
        )
    return tuple(options)


def default_question_timeout_seconds() -> int:
    """Срок ожидания ответа пользователя по умолчанию."""

    return settings.application.question_timeout_seconds
