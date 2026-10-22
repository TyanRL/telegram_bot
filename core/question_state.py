"""Хранилище ожиданий ответов пользователя и контекст продолжения.

Модуль не зависит от Telegram: он хранит состояние ожидания, выполняет
атомарные переходы и освобождает ресурсы. Telegram-слой работает поверх него,
а цикл модели (Шаг 5 плана) продолжит работу через сохранённый
[`QuestionContinuationContext`](core/question_state.py:44).

Ключевые свойства хранилища:

- один активный вопрос на пользователя;
- атомарные переходы `pending` → `accepted` → `continuing` → `completed`,
  а также `cancelled` и `expired`;
- блокировка уровня пользователя удерживается только на время короткого
  перехода состояния: ни человеческое ожидание, ни долгая генерация не
  держат её;
- повторное нажатие отклоняется сразу и не получает права на повторное
  выполнение продолжения;
- просроченные записи удаляются, снимки изображений освобождаются, а сброс
  контекста инвалидирует ожидание пользователя.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

from core.common_types import (
    PendingQuestionRequest,
    PendingToolCall,
    QuestionAnswer,
    QuestionAnswerMode,
    QuestionOption,
    QuestionStatus,
)
from core.config import settings
from core.tool_registry import ImageToolTurnState, VideoToolTurnState
from core.video_requests import VideoRequestDraft

__all__ = [
    "QuestionContinuationContext",
    "PendingQuestion",
    "QuestionStateStore",
    "get_question_store",
    "invalidate_user_questions",
    "cancel_active_question",
]

logger = logging.getLogger(__name__)

# Длина идентификатора вопроса в callback_data: короткий, но достаточный
# для уникальности внутри процесса.
QUESTION_ID_LENGTH = 16


def _new_question_id() -> str:
    """Создаёт непрозрачный идентификатор вопроса."""

    return uuid4().hex[:QUESTION_ID_LENGTH]


@dataclass
class QuestionContinuationContext:
    """Серверный контекст продолжения работы после ответа пользователя.

    Контекст содержит всё, что нужно для восстановления прерванного цикла
    инструментов: идентификатор ответа OpenAI, исходные вызовы инструментов,
    уже полученные результаты, позицию текущего вызова, модель, исходный
    пользовательский текст, счётчики токенов и раундов, защитное состояние
    инструментов и, для видео, проверенный снимок заявки.

    Telegram-объекты, сетевые клиенты и временные видеофайлы здесь не
    хранятся: контекст переживает перезапуск обработчика, но не процесс.
    """

    response_id: str | None = None
    tool_calls: tuple[PendingToolCall, ...] = ()
    completed_outputs: tuple[dict[str, str], ...] = ()
    current_call_index: int = 0
    model_name: str | None = None
    original_user_text: str | None = None
    context_tokens: int = 0
    completion_tokens: int = 0
    tool_round: int = 0
    image_tool_state: ImageToolTurnState = field(default_factory=ImageToolTurnState)
    video_tool_state: VideoToolTurnState = field(default_factory=VideoToolTurnState)
    video_request: VideoRequestDraft | None = None
    video_operation_id: str | None = None
    answer_mode: QuestionAnswerMode = QuestionAnswerMode.PLAIN
    validator: Callable[[str], str | None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tool_calls = tuple(self.tool_calls)
        self.completed_outputs = tuple(self.completed_outputs)

    @property
    def is_video_request(self) -> bool:
        """Ответ применяется к сохранённой видеозаявке."""

        return self.answer_mode is QuestionAnswerMode.VIDEO_REQUEST

    def release(self) -> None:
        """Освобождает снимок изображений и прочие временные данные.

        Вызывается при завершении, отмене и истечении вопроса: байты
        изображений больше не удерживаются хранилищем.
        """

        self.video_request = None
        self.metadata.clear()
        self.tool_calls = ()
        self.completed_outputs = ()


@dataclass
class PendingQuestion:
    """Запись об ожидании ответа пользователя."""

    question_id: str
    user_id: int
    chat_id: int
    message_thread_id: int | None
    question_message_id: int | None
    trigger_message_id: int | None
    text: str
    options: tuple[QuestionOption, ...]
    input_hint: str | None
    expires_at: float
    status: QuestionStatus
    answer: QuestionAnswer | None = None
    continuation: QuestionContinuationContext | None = None
    created_at: float = 0.0
    answer_mode: QuestionAnswerMode = QuestionAnswerMode.PLAIN
    validator: Callable[[str], str | None] | None = None

    @property
    def is_active(self) -> bool:
        """Вопрос ещё принимает ответ пользователя."""

        return self.status is QuestionStatus.PENDING

    def is_expired(self, now: float) -> bool:
        """Срок ожидания истёк."""

        return self.expires_at <= now

    def find_option(self, option_id: str) -> QuestionOption | None:
        """Ищет готовый вариант по его идентификатору."""

        for option in self.options:
            if option.option_id == option_id:
                return option
        return None

    def option_by_index(self, index: int) -> QuestionOption | None:
        """Ищет готовый вариант по индексу в серверном списке."""

        if index < 0 or index >= len(self.options):
            return None
        return self.options[index]


class QuestionStateStore:
    """Хранилище ожиданий ответов пользователя в памяти процесса.

    Хранилище сериализует конкурентные webhook-обновления на уровне
    пользователя: блокировка удерживается только на время перехода
    состояния. Ожидание человека и долгая генерация её не удерживают.
    """

    def __init__(
        self,
        *,
        timeout_seconds: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        configured = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.application.question_timeout_seconds
        )
        self._timeout_seconds = int(configured)
        self._clock = clock
        self._questions: dict[str, PendingQuestion] = {}
        self._user_locks: dict[int, asyncio.Lock] = {}

    @property
    def timeout_seconds(self) -> int:
        """Срок ожидания ответа пользователя по умолчанию."""

        return self._timeout_seconds

    def _lock_for(self, user_id: int) -> asyncio.Lock:
        """Возвращает блокировку пользователя, создавая её при первом обращении."""

        lock = self._user_locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._user_locks[user_id] = lock
        return lock

    async def create(
        self,
        *,
        user_id: int,
        chat_id: int,
        message_thread_id: int | None = None,
        trigger_message_id: int | None = None,
        request: PendingQuestionRequest,
        continuation: QuestionContinuationContext | None = None,
        timeout_seconds: int | None = None,
    ) -> PendingQuestion:
        """Создаёт ожидание вопроса, вытесняя прежнее ожидание пользователя.

        Прежний активный вопрос пользователя отменяется: активный вопрос на
        пользователя всегда один.
        """

        async with self._lock_for(user_id):
            await self._cancel_active_locked(user_id)
            now = self._clock()
            lifetime = timeout_seconds or request.timeout_seconds or self._timeout_seconds
            record = PendingQuestion(
                question_id=_new_question_id(),
                user_id=user_id,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                question_message_id=None,
                trigger_message_id=trigger_message_id,
                text=request.text,
                options=tuple(request.options),
                input_hint=request.input_hint,
                expires_at=now + float(lifetime),
                status=QuestionStatus.PENDING,
                answer=None,
                continuation=continuation,
                created_at=now,
                answer_mode=request.answer_mode,
                validator=request.validator,
            )
            self._questions[record.question_id] = record
            return record

    async def get(self, question_id: str) -> PendingQuestion | None:
        """Возвращает запись вопроса по идентификатору."""

        return self._questions.get(question_id)

    async def is_expired(self, question_id: str) -> bool:
        """Проверяет, истёк ли срок ожидания вопроса."""

        record = self._questions.get(question_id)
        if record is None:
            return True
        return record.is_expired(self._clock())

    async def get_active_for_user(self, user_id: int) -> PendingQuestion | None:
        """Возвращает ожидание, которое ещё принимает ответ пользователя."""

        active_id = self._active_id(user_id)
        if active_id is None:
            return None
        record = self._questions.get(active_id)
        if record is None:
            return None
        if record.status is not QuestionStatus.PENDING:
            return None
        if record.is_expired(self._clock()):
            async with self._lock_for(user_id):
                await self._expire_locked(record)
            return None
        return record

    def _active_id(self, user_id: int) -> str | None:
        """Ищет активный вопрос пользователя без блокировки."""

        for question_id, record in self._questions.items():
            if record.user_id == user_id and record.status in (
                QuestionStatus.PENDING,
                QuestionStatus.ACCEPTED,
                QuestionStatus.CONTINUING,
            ):
                return question_id
        return None

    async def set_question_message_id(
        self,
        question_id: str,
        message_id: int | None,
    ) -> None:
        """Запоминает идентификатор сообщения с вопросом."""

        record = self._questions.get(question_id)
        if record is None:
            return
        async with self._lock_for(record.user_id):
            current = self._questions.get(question_id)
            if current is not None:
                current.question_message_id = message_id

    async def accept_answer(
        self,
        question_id: str,
        answer: QuestionAnswer,
    ) -> QuestionAnswer | None:
        """Атомарно принимает ответ пользователя.

        Возвращает принятый ответ, если он принят. Повторный, чужой или
        просроченный ответ возвращает ``None`` и не даёт права на повторное
        выполнение продолжения.
        """

        record = self._questions.get(question_id)
        if record is None:
            return None

        async with self._lock_for(record.user_id):
            current = self._questions.get(question_id)
            if current is None:
                return None
            if current.status is not QuestionStatus.PENDING:
                logger.info(
                    "Ответ на вопрос %s отклонён: состояние %s",
                    question_id,
                    current.status.value,
                )
                return None
            if current.is_expired(self._clock()):
                await self._expire_locked(current)
                return None
            current.status = QuestionStatus.ACCEPTED
            current.answer = answer
            return answer

    async def mark_continuing(self, question_id: str) -> bool:
        """Переводит принятый вопрос в состояние продолжения."""

        record = self._questions.get(question_id)
        if record is None:
            return False
        async with self._lock_for(record.user_id):
            current = self._questions.get(question_id)
            if current is None or current.status is not QuestionStatus.ACCEPTED:
                return False
            current.status = QuestionStatus.CONTINUING
            return True

    async def complete(self, question_id: str) -> None:
        """Завершает вопрос и освобождает его ресурсы."""

        record = self._questions.get(question_id)
        if record is None:
            return
        async with self._lock_for(record.user_id):
            current = self._questions.get(question_id)
            if current is None:
                return
            current.status = QuestionStatus.COMPLETED
            self._release(current)

    async def expire(self, question_id: str) -> bool:
        """Переводит вопрос в состояние истечения срока ожидания."""

        record = self._questions.get(question_id)
        if record is None:
            return False
        async with self._lock_for(record.user_id):
            current = self._questions.get(question_id)
            if current is None or current.status is not QuestionStatus.PENDING:
                return False
            await self._expire_locked(current)
            return True

    async def cancel(self, question_id: str) -> bool:
        """Отменяет вопрос пользователем."""

        record = self._questions.get(question_id)
        if record is None:
            return False
        async with self._lock_for(record.user_id):
            current = self._questions.get(question_id)
            if current is None:
                return False
            if current.status in (
                QuestionStatus.COMPLETED,
                QuestionStatus.CANCELLED,
                QuestionStatus.EXPIRED,
            ):
                return False
            current.status = QuestionStatus.CANCELLED
            self._release(current)
            return True

    async def cancel_active_for_user(self, user_id: int) -> PendingQuestion | None:
        """Отменяет активное ожидание пользователя."""

        async with self._lock_for(user_id):
            return await self._cancel_active_locked(user_id)

    async def invalidate_user_questions(self, user_id: int) -> int:
        """Инвалидирует все ожидания пользователя (например, при сбросе контекста).

        Возвращает количество инвалидированных записей.
        """

        async with self._lock_for(user_id):
            invalidated = 0
            for record in list(self._questions.values()):
                if record.user_id != user_id:
                    continue
                if record.status in (
                    QuestionStatus.COMPLETED,
                    QuestionStatus.CANCELLED,
                    QuestionStatus.EXPIRED,
                ):
                    continue
                record.status = QuestionStatus.CANCELLED
                self._release(record)
                invalidated += 1
            return invalidated

    async def cleanup_expired(self) -> int:
        """Истекает просроченные записи и удаляет завершённые.

        Возвращает количество истёкших записей. Завершённые, отменённые и
        истёкшие записи удаляются: их ресурсы уже освобождены.
        """

        now = self._clock()
        expired = 0
        for record in list(self._questions.values()):
            if record.status is not QuestionStatus.PENDING:
                continue
            if not record.is_expired(now):
                continue
            async with self._lock_for(record.user_id):
                current = self._questions.get(record.question_id)
                if current is None or current.status is not QuestionStatus.PENDING:
                    continue
                await self._expire_locked(current)
                expired += 1

        for record in list(self._questions.values()):
            if record.status in (
                QuestionStatus.COMPLETED,
                QuestionStatus.CANCELLED,
                QuestionStatus.EXPIRED,
            ):
                self._questions.pop(record.question_id, None)
        return expired

    async def _expire_locked(self, record: PendingQuestion) -> None:
        """Переводит запись в состояние истечения (вызывается под блокировкой)."""

        record.status = QuestionStatus.EXPIRED
        self._release(record)

    async def _cancel_active_locked(self, user_id: int) -> PendingQuestion | None:
        """Отменяет активный вопрос пользователя (вызывается под блокировкой)."""

        for record in list(self._questions.values()):
            if record.user_id != user_id:
                continue
            if record.status in (
                QuestionStatus.COMPLETED,
                QuestionStatus.CANCELLED,
                QuestionStatus.EXPIRED,
            ):
                continue
            record.status = QuestionStatus.CANCELLED
            self._release(record)
            return record
        return None


    @staticmethod
    def _release(record: PendingQuestion) -> None:
        """Освобождает ресурсы записи вопроса."""

        if record.continuation is not None:
            record.continuation.release()


_store = QuestionStateStore()


def get_question_store() -> QuestionStateStore:
    """Возвращает общее хранилище ожиданий приложения."""

    return _store


async def invalidate_user_questions(user_id: int) -> int:
    """Инвалидирует ожидания пользователя в общем хранилище."""

    return await _store.invalidate_user_questions(user_id)


async def cancel_active_question(user_id: int) -> PendingQuestion | None:
    """Отменяет активный вопрос пользователя в общем хранилище."""

    return await _store.cancel_active_for_user(user_id)
