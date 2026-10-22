"""Проверки Telegram-маршрутизации вопросов пользователю.

Покрываются сценарии Шага 4 плана:

- универсальный вопрос показывает до трёх готовых вариантов и «Свой ответ»;
- вопрос длительности текущей модели показывает пять значений и «Свой ответ»;
- отмена — отдельная служебная кнопка, выбирается ровно один ответ;
- данные кнопки ограничены, проверяются владелец, чат, тема и сообщение;
- чужие и старые кнопки отклоняются без запуска продолжения;
- свой текст и распознанный голос обрабатываются общим входом;
- ошибка отправки вопроса и ошибка продолжения обрабатываются безопасно.
"""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:TEST_TOKEN")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from core.common_types import (
    PendingQuestionRequest,
    QuestionAnswer,
    QuestionOption,
    QuestionStatus,
)
from core.question_handlers import (
    QUESTION_ALREADY_ANSWERED_TEXT,
    QUESTION_CANCELLED_TEXT,
    QUESTION_EXPIRED_TEXT,
    QUESTION_FOREIGN_TEXT,
    QUESTION_NOT_FOUND_TEXT,
    cancel_question_command,
    handle_question_callback,
    handle_question_text,
)
from core.question_state import QuestionStateStore, get_question_store
from core.state_and_commands import get_all_histories, reset
from core.user_questions import (
    CALLBACK_DATA_MAX_BYTES,
    CANCEL_ACTION,
    CUSTOM_ANSWER_ACTION,
    UserQuestionService,
    build_callback_data,
    build_duration_question_options,
    build_inline_keyboard,
    validate_universal_option_count,
)
from core.video_requests import VideoModelProfile, build_video_request_draft


def _profile() -> VideoModelProfile:
    return VideoModelProfile(
        model="alibaba/wan-3.0-prime",
        resolution="480p",
        allowed_durations=tuple(range(2, 31)),
        duration_presets=(5, 10, 15, 20, 30),
        allowed_aspect_ratios=("16:9", "9:16", "1:1"),
        max_references=10,
    )


def _message(
    *,
    chat_id: int = 42,
    message_id: int = 555,
    thread_id: int | None = None,
) -> SimpleNamespace:
    edits: list[dict[str, Any]] = []
    replies: list[str] = []

    async def edit_text(text: str, reply_markup: Any = None, **_kwargs: Any) -> SimpleNamespace:
        edits.append({"text": text, "reply_markup": reply_markup})
        return SimpleNamespace(message_id=message_id)

    async def edit_reply_markup(reply_markup: Any = None, **_kwargs: Any) -> bool:
        edits.append({"reply_markup": reply_markup})
        return True

    async def reply_text(text: str, reply_markup: Any = None, **_kwargs: Any) -> SimpleNamespace:
        replies.append(text)
        return SimpleNamespace(message_id=999)

    return SimpleNamespace(
        message_id=message_id,
        message_thread_id=thread_id,
        chat=SimpleNamespace(id=chat_id),
        edit_text=edit_text,
        edit_reply_markup=edit_reply_markup,
        reply_text=reply_text,
        edits=edits,
        replies=replies,
    )


def _query(
    *,
    data: str,
    user_id: int = 1001,
    message: SimpleNamespace | None = None,
) -> SimpleNamespace:
    answers: list[dict[str, Any]] = []

    async def answer(text: str | None = None, show_alert: bool = False, **_kwargs: Any) -> bool:
        answers.append({"text": text, "show_alert": show_alert})
        return True

    used_message = message if message is not None else _message()
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(id=user_id),
        message=used_message,
        answer=answer,
        answers=answers,
    )


def _update(
    *,
    user_id: int = 1001,
    chat_id: int = 42,
    text: str | None = None,
    message: SimpleNamespace | None = None,
    callback_query: SimpleNamespace | None = None,
) -> Any:
    used_message = message if message is not None else _message(chat_id=chat_id)
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id),
        effective_message=used_message,
        message=used_message,
        text=text,
        callback_query=callback_query,
    )


def _context(service: UserQuestionService) -> Any:
    return SimpleNamespace(
        application=SimpleNamespace(bot_data={"question_service": service}),
    )




def _request(
    options: tuple[QuestionOption, ...] = (),
    *,
    text: str = "Какой вариант выбираем?",
) -> PendingQuestionRequest:
    return PendingQuestionRequest(
        text=text,
        options=options,
        input_hint="Напишите ответ текстом",
    )


class QuestionCallbackHandlerTests(unittest.TestCase):
    def _service(self, store: QuestionStateStore) -> UserQuestionService:
        return UserQuestionService(store=store)

    def test_option_button_accepts_answer_and_updates_message(self) -> None:
        store = QuestionStateStore()
        service = self._service(store)
        record = asyncio.run(
            service.ask(
                _update(),
                _request((QuestionOption("a", "Первый"), QuestionOption("b", "Второй"))),
            )
        )
        message = _message(message_id=record.question_message_id or 555)
        query = _query(data=build_callback_data(record.question_id, "1"), message=message)
        update = _update(callback_query=query)
        continued: list[QuestionAnswer] = []

        async def continuation(
            record: Any,
            answer: QuestionAnswer,
            update: Any,
            context: Any,
        ) -> None:
            continued.append(answer)

        service.set_continuation(continuation)

        with patch("core.question_handlers.in_user_list", new=AsyncMock(return_value=True)):
            asyncio.run(handle_question_callback(update, _context(service)))

        self.assertEqual(len(continued), 1)
        self.assertEqual(continued[0].text, "Второй")
        self.assertTrue(any("Второй" in str(edit.get("text", "")) for edit in message.edits))
        self.assertTrue(any(edit.get("reply_markup", "x") is None for edit in message.edits))

    def test_custom_button_only_shows_hint_and_keeps_question_open(self) -> None:
        store = QuestionStateStore()
        service = self._service(store)
        record = asyncio.run(
            service.ask(_update(), _request((QuestionOption("a", "Первый"),)))
        )
        message = _message(message_id=record.question_message_id or 555)
        query = _query(
            data=build_callback_data(record.question_id, CUSTOM_ANSWER_ACTION),
            message=message,
        )
        update = _update(callback_query=query)

        with patch("core.question_handlers.in_user_list", new=AsyncMock(return_value=True)):
            asyncio.run(handle_question_callback(update, _context(service)))

        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.PENDING)
        self.assertTrue(message.replies)

    def test_cancel_button_cancels_question(self) -> None:
        store = QuestionStateStore()
        service = self._service(store)
        record = asyncio.run(service.ask(_update(), _request()))
        message = _message(message_id=record.question_message_id or 555)
        query = _query(data=build_callback_data(record.question_id, CANCEL_ACTION), message=message)
        update = _update(callback_query=query)

        with patch("core.question_handlers.in_user_list", new=AsyncMock(return_value=True)):
            asyncio.run(handle_question_callback(update, _context(service)))

        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.CANCELLED)
        self.assertTrue(
            any(QUESTION_CANCELLED_TEXT in str(edit.get("text", "")) for edit in message.edits)
        )

    def test_foreign_user_button_is_rejected(self) -> None:
        store = QuestionStateStore()
        service = self._service(store)
        record = asyncio.run(service.ask(_update(), _request()))
        query = _query(data=build_callback_data(record.question_id, "0"), user_id=2002)
        update = _update(callback_query=query)
        continued: list[QuestionAnswer] = []

        async def continuation(
            record: Any,
            answer: QuestionAnswer,
            update: Any,
            context: Any,
        ) -> None:
            continued.append(answer)

        service.set_continuation(continuation)

        with patch("core.question_handlers.in_user_list", new=AsyncMock(return_value=True)):
            asyncio.run(handle_question_callback(update, _context(service)))

        self.assertEqual(continued, [])
        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.PENDING)
        self.assertTrue(
            any(answer.get("text") == QUESTION_FOREIGN_TEXT for answer in query.answers)
        )

    def test_button_from_another_chat_is_rejected(self) -> None:
        store = QuestionStateStore()
        service = self._service(store)
        record = asyncio.run(service.ask(_update(), _request()))
        message = _message(chat_id=99, message_id=record.question_message_id or 555)
        query = _query(data=build_callback_data(record.question_id, "0"), message=message)
        update = _update(callback_query=query)

        with patch("core.question_handlers.in_user_list", new=AsyncMock(return_value=True)):
            asyncio.run(handle_question_callback(update, _context(service)))

        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.PENDING)
        self.assertTrue(
            any(answer.get("text") == QUESTION_FOREIGN_TEXT for answer in query.answers)
        )

    def test_button_from_another_topic_is_rejected(self) -> None:
        store = QuestionStateStore()
        service = self._service(store)
        source_message = _message(thread_id=77)
        record = asyncio.run(
            service.ask(_update(message=source_message), _request())
        )
        callback_message = _message(
            thread_id=88,
            message_id=record.question_message_id or 555,
        )
        query = _query(
            data=build_callback_data(record.question_id, "0"),
            message=callback_message,
        )

        with patch("core.question_handlers.in_user_list", new=AsyncMock(return_value=True)):
            asyncio.run(handle_question_callback(_update(callback_query=query), _context(service)))

        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.PENDING)
        self.assertTrue(
            any(answer.get("text") == QUESTION_FOREIGN_TEXT for answer in query.answers)
        )

    def test_button_without_topic_is_rejected_for_topic_question(self) -> None:
        store = QuestionStateStore()
        service = self._service(store)
        record = asyncio.run(
            service.ask(_update(message=_message(thread_id=77)), _request())
        )
        query = _query(
            data=build_callback_data(record.question_id, "0"),
            message=_message(message_id=record.question_message_id or 555),
        )

        with patch("core.question_handlers.in_user_list", new=AsyncMock(return_value=True)):
            asyncio.run(handle_question_callback(_update(callback_query=query), _context(service)))

        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.PENDING)
        self.assertTrue(
            any(answer.get("text") == QUESTION_FOREIGN_TEXT for answer in query.answers)
        )

    def test_stale_message_button_is_rejected(self) -> None:
        store = QuestionStateStore()
        service = self._service(store)
        record = asyncio.run(service.ask(_update(), _request()))
        message = _message(message_id=(record.question_message_id or 555) + 1000)
        query = _query(data=build_callback_data(record.question_id, "0"), message=message)
        update = _update(callback_query=query)

        with patch("core.question_handlers.in_user_list", new=AsyncMock(return_value=True)):
            asyncio.run(handle_question_callback(update, _context(service)))

        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.PENDING)
        self.assertTrue(
            any(answer.get("text") == QUESTION_NOT_FOUND_TEXT for answer in query.answers)
        )

    def test_unknown_question_button_is_rejected(self) -> None:
        store = QuestionStateStore()
        service = self._service(store)
        query = _query(data=build_callback_data("unknown-question", "0"))
        update = _update(callback_query=query)

        with patch("core.question_handlers.in_user_list", new=AsyncMock(return_value=True)):
            asyncio.run(handle_question_callback(update, _context(service)))

        self.assertTrue(query.answers)
        self.assertTrue(
            any(answer.get("text") == QUESTION_NOT_FOUND_TEXT for answer in query.answers)
        )

    def test_old_button_after_process_restart_is_rejected_by_empty_store(self) -> None:
        old_service = self._service(QuestionStateStore())
        old_record = asyncio.run(old_service.ask(_update(), _request()))

        # Новый процесс начинает с пустым хранилищем; старый callback не
        # восстанавливает и не повторяет продолжение.
        restarted_service = self._service(QuestionStateStore())
        query = _query(
            data=build_callback_data(old_record.question_id, "0"),
            message=_message(message_id=old_record.question_message_id or 555),
        )
        continued: list[QuestionAnswer] = []

        async def continuation(
            _record: Any,
            answer: QuestionAnswer,
            _update: Any,
            _context: Any,
        ) -> None:
            continued.append(answer)

        restarted_service.set_continuation(continuation)

        with patch("core.question_handlers.in_user_list", new=AsyncMock(return_value=True)):
            asyncio.run(
                handle_question_callback(
                    _update(callback_query=query),
                    _context(restarted_service),
                )
            )

        self.assertEqual(continued, [])
        self.assertIsNone(
            asyncio.run(restarted_service.store.get_active_for_user(old_record.user_id))
        )
        self.assertTrue(
            any(answer.get("text") == QUESTION_NOT_FOUND_TEXT for answer in query.answers)
        )

    def test_forbidden_user_is_rejected(self) -> None:
        store = QuestionStateStore()
        service = self._service(store)
        record = asyncio.run(service.ask(_update(), _request()))
        query = _query(data=build_callback_data(record.question_id, "0"))
        update = _update(callback_query=query)

        with patch("core.question_handlers.in_user_list", new=AsyncMock(return_value=False)):
            asyncio.run(handle_question_callback(update, _context(service)))

        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.PENDING)

    def test_repeated_button_press_does_not_run_continuation_twice(self) -> None:
        store = QuestionStateStore()
        service = self._service(store)
        record = asyncio.run(
            service.ask(_update(), _request((QuestionOption("a", "Первый"),)))
        )
        continued: list[QuestionAnswer] = []

        async def continuation(
            record: Any,
            answer: QuestionAnswer,
            update: Any,
            context: Any,
        ) -> None:
            continued.append(answer)

        service.set_continuation(continuation)

        for _ in range(2):
            message = _message(message_id=record.question_message_id or 555)
            query = _query(data=build_callback_data(record.question_id, "0"), message=message)
            update = _update(callback_query=query)
            with patch("core.question_handlers.in_user_list", new=AsyncMock(return_value=True)):
                asyncio.run(handle_question_callback(update, _context(service)))

        self.assertEqual(len(continued), 1)

    def test_button_and_text_race_runs_continuation_exactly_once(self) -> None:
        async def scenario() -> tuple[int, int, QuestionStatus]:
            store = QuestionStateStore()
            service = self._service(store)
            record = await service.ask(
                _update(),
                _request((QuestionOption("a", "Кнопочный ответ"),)),
            )
            continued: list[QuestionAnswer] = []

            async def continuation(
                _record: Any,
                answer: QuestionAnswer,
                _update: Any,
                _context: Any,
            ) -> None:
                continued.append(answer)

            service.set_continuation(continuation)
            original_option = service.handle_option
            original_custom = service.handle_custom_answer
            contenders = 0
            both_ready = asyncio.Event()

            async def rendezvous() -> None:
                nonlocal contenders
                contenders += 1
                if contenders == 2:
                    both_ready.set()
                await both_ready.wait()

            async def gated_option(question: Any, index: int) -> Any:
                await rendezvous()
                return await original_option(question, index)

            async def gated_custom(question: Any, text: str) -> Any:
                await rendezvous()
                return await original_custom(question, text)

            callback_message = _message(
                message_id=record.question_message_id or 555,
            )
            query = _query(
                data=build_callback_data(record.question_id, "0"),
                message=callback_message,
            )
            button_update = _update(callback_query=query)
            text_update = _update(text="Текстовый ответ")
            history = AsyncMock()

            with (
                patch.object(service, "handle_option", new=gated_option),
                patch.object(service, "handle_custom_answer", new=gated_custom),
                patch("core.question_handlers.in_user_list", new=AsyncMock(return_value=True)),
                patch("core.question_handlers._record_history", new=history),
            ):
                await asyncio.gather(
                    handle_question_callback(button_update, _context(service)),
                    handle_question_text(text_update, _context(service), "Текстовый ответ"),
                )

            stored = await store.get(record.question_id)
            assert stored is not None
            return len(continued), history.await_count, stored.status

        continuation_count, history_count, status = asyncio.run(scenario())

        self.assertEqual(continuation_count, 1)
        self.assertEqual(history_count, 1)
        self.assertEqual(status, QuestionStatus.COMPLETED)

    def test_expired_button_is_rejected(self) -> None:
        now = {"value": 1000.0}
        store = QuestionStateStore(timeout_seconds=10, clock=lambda: now["value"])
        service = self._service(store)
        record = asyncio.run(
            service.ask(_update(), _request((QuestionOption("a", "Первый"),)))
        )
        now["value"] += 11
        message = _message(message_id=record.question_message_id or 555)
        query = _query(data=build_callback_data(record.question_id, "0"), message=message)
        update = _update(callback_query=query)

        with patch("core.question_handlers.in_user_list", new=AsyncMock(return_value=True)):
            asyncio.run(handle_question_callback(update, _context(service)))

        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.EXPIRED)
        self.assertTrue(
            any(answer.get("text") == QUESTION_EXPIRED_TEXT for answer in query.answers)
        )

    def test_continuation_error_is_reported_safely(self) -> None:
        store = QuestionStateStore()
        service = self._service(store)
        record = asyncio.run(
            service.ask(_update(), _request((QuestionOption("a", "Первый"),)))
        )

        async def continuation(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("сбой продолжения")

        service.set_continuation(continuation)
        message = _message(message_id=record.question_message_id or 555)
        query = _query(data=build_callback_data(record.question_id, "0"), message=message)
        update = _update(callback_query=query)

        with patch("core.question_handlers.in_user_list", new=AsyncMock(return_value=True)):
            asyncio.run(handle_question_callback(update, _context(service)))

        # Пользователю сообщается о сбое продолжения; платный повтор не делается.
        self.assertTrue(update.effective_message.replies)


class QuestionTextHandlerTests(unittest.TestCase):
    def test_text_answer_is_accepted_and_recorded(self) -> None:
        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        record = asyncio.run(service.ask(_update(), _request()))
        continued: list[QuestionAnswer] = []

        async def continuation(
            record: Any,
            answer: QuestionAnswer,
            update: Any,
            context: Any,
        ) -> None:
            continued.append(answer)

        service.set_continuation(continuation)
        update = _update(text="мой ответ")

        handled = asyncio.run(handle_question_text(update, _context(service), "мой ответ"))

        self.assertTrue(handled)
        self.assertEqual(len(continued), 1)
        self.assertEqual(continued[0].text, "мой ответ")

    def test_text_without_active_question_is_not_intercepted(self) -> None:
        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        update = _update(text="обычное сообщение")

        handled = asyncio.run(handle_question_text(update, _context(service), "обычное сообщение"))

        self.assertFalse(handled)

    def test_text_from_another_topic_is_not_accepted(self) -> None:
        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        record = asyncio.run(
            service.ask(_update(message=_message(thread_id=77)), _request())
        )
        update = _update(message=_message(thread_id=88), text="чужая тема")

        handled = asyncio.run(
            handle_question_text(update, _context(service), "чужая тема")
        )

        self.assertFalse(handled)
        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.PENDING)

    def test_recognized_voice_answers_active_question_without_model_call(self) -> None:
        import main as bot_main

        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        record = asyncio.run(service.ask(_update(), _request()))
        continued: list[QuestionAnswer] = []

        async def continuation(
            _record: Any,
            answer: QuestionAnswer,
            _update: Any,
            _context: Any,
        ) -> None:
            continued.append(answer)

        service.set_continuation(continuation)
        voice_message = _message(message_id=901)
        voice_message.voice = SimpleNamespace(file_id="voice-file")
        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=1001),
            effective_chat=SimpleNamespace(id=42),
            effective_message=voice_message,
            message=voice_message,
        )
        downloaded_file = SimpleNamespace(download_to_drive=AsyncMock())
        context = SimpleNamespace(
            bot=SimpleNamespace(get_file=AsyncMock(return_value=downloaded_file)),
            application=SimpleNamespace(bot_data={"question_service": service}),
        )

        with (
            patch("main.in_user_list", new=AsyncMock(return_value=True)),
            patch("main.transcribe_audio", return_value="Ответ голосом"),
            patch("main.send_big_text", new=AsyncMock()),
            patch("main.handle_message_inner", new=AsyncMock()) as model_call,
            patch("core.question_handlers._record_history", new=AsyncMock()),
        ):
            asyncio.run(
                bot_main.handle_voice_message(
                    cast(Any, update),
                    cast(Any, context),
                )
            )

        context.bot.get_file.assert_awaited_once_with("voice-file")
        downloaded_file.download_to_drive.assert_awaited_once()
        self.assertEqual([answer.text for answer in continued], ["Ответ голосом"])
        model_call.assert_not_awaited()
        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.COMPLETED)

    def test_invalid_custom_answer_keeps_question_open(self) -> None:
        def validator(text: str) -> str | None:
            if not text.isdigit():
                return "Нужно целое число секунд."
            return None

        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        request = PendingQuestionRequest(
            text="Сколько секунд?",
            options=build_duration_question_options((5, 10, 15, 20, 30)),
            validator=validator,
        )
        record = asyncio.run(service.ask(_update(), request))
        update = _update(text="не число")

        handled = asyncio.run(handle_question_text(update, _context(service), "не число"))

        self.assertTrue(handled)
        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.PENDING)
        self.assertTrue(update.effective_message.replies)

    def test_valid_duration_answer_is_accepted(self) -> None:
        def validator(text: str) -> str | None:
            if not text.isdigit():
                return "Нужно целое число секунд."
            return None

        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        request = PendingQuestionRequest(
            text="Сколько секунд?",
            options=build_duration_question_options((5, 10, 15, 20, 30)),
            validator=validator,
        )
        record = asyncio.run(service.ask(_update(), request))
        update = _update(text="7")

        handled = asyncio.run(handle_question_text(update, _context(service), "7"))

        self.assertTrue(handled)
        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.ACCEPTED)
        assert stored.answer is not None
        self.assertEqual(stored.answer.text, "7")


class QuestionCancelCommandTests(unittest.TestCase):
    def test_cancel_command_cancels_active_question(self) -> None:
        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        record = asyncio.run(service.ask(_update(), _request()))
        update = _update()

        asyncio.run(cancel_question_command(update, _context(service)))

        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.CANCELLED)

    def test_cancel_command_without_question_reports_it(self) -> None:
        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        update = _update()

        asyncio.run(cancel_question_command(update, _context(service)))

        self.assertTrue(update.effective_message.replies)


class QuestionHistoryAndResetTests(unittest.TestCase):
    def test_question_and_answer_are_written_to_user_history(self) -> None:
        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        record = asyncio.run(
            service.ask(_update(), _request((QuestionOption("a", "Первый"),)))
        )
        message = _message(message_id=record.question_message_id or 555)
        query = _query(data=build_callback_data(record.question_id, "0"), message=message)
        update = _update(callback_query=query)

        async def continuation(*_args: Any, **_kwargs: Any) -> None:
            return None

        service.set_continuation(continuation)

        with patch("core.question_handlers.in_user_list", new=AsyncMock(return_value=True)):
            asyncio.run(handle_question_callback(update, _context(service)))

        history = asyncio.run(get_all_histories().get(1001, []))
        self.assertEqual(history[-2]["role"], "assistant")
        self.assertEqual(history[-2]["content"], "Какой вариант выбираем?")
        self.assertEqual(history[-1]["role"], "user")
        self.assertEqual(history[-1]["content"], "Первый")
        self.assertNotIn("role", "system")

    def test_reset_invalidates_active_question(self) -> None:
        # Сброс контекста работает с общим хранилищем приложения.
        store = get_question_store()
        service = UserQuestionService(store=store)
        record = asyncio.run(service.ask(_update(), _request()))

        asyncio.run(reset(_update(), cast(Any, SimpleNamespace())))

        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.CANCELLED)
        self.assertIsNone(asyncio.run(store.get_active_for_user(1001)))


class VideoDeliveryAfterButtonTests(unittest.TestCase):
    """Доставка видео и ответов после нажатия inline-кнопки.

    Обновление от кнопки не содержит `update.message`: адресатом доставки
    становится сообщение бота (`update.effective_message`), а пользователем
    остаётся автор исходного сообщения, а не бот.
    """

    def _video_message(self) -> SimpleNamespace:
        videos: list[dict[str, Any]] = []
        documents: list[dict[str, Any]] = []

        async def reply_video(**kwargs: Any) -> SimpleNamespace:
            videos.append(kwargs)
            return SimpleNamespace(message_id=1001)

        async def reply_document(**kwargs: Any) -> SimpleNamespace:
            documents.append(kwargs)
            return SimpleNamespace(message_id=1002)

        return SimpleNamespace(
            message_id=555,
            message_thread_id=None,
            chat=SimpleNamespace(id=42),
            edit_text=AsyncMock(return_value=SimpleNamespace(message_id=555)),
            edit_reply_markup=AsyncMock(return_value=True),
            reply_text=AsyncMock(return_value=SimpleNamespace(message_id=999)),
            reply_video=reply_video,
            reply_document=reply_document,
            videos=videos,
            documents=documents,
        )

    def _button_update(self, message: SimpleNamespace) -> Any:
        query = _query(data="q:0123456789abcdef:0", message=message)
        return SimpleNamespace(
            effective_user=SimpleNamespace(id=1001),
            effective_chat=SimpleNamespace(id=42),
            effective_message=message,
            message=None,
            callback_query=query,
        )

    def test_video_is_delivered_to_effective_message_after_button(self) -> None:
        from core.common_types import GeneratedVideo
        from core.tool_helpers import send_video_to_telegram

        message = self._video_message()
        update = self._button_update(message)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            path.write_bytes(b"video-bytes")
            video = GeneratedVideo(path=path)

            delivered = asyncio.run(
                send_video_to_telegram(cast(Any, update), video, message)
            )

        self.assertTrue(delivered)
        self.assertEqual(len(message.videos), 1)
        self.assertEqual(len(message.documents), 0)
        self.assertTrue(video.cleaned)

    def test_video_falls_back_to_document_after_button(self) -> None:
        from core.common_types import GeneratedVideo
        from core.tool_helpers import send_video_to_telegram

        message = self._video_message()
        message.reply_video = AsyncMock(side_effect=RuntimeError("video rejected"))
        update = self._button_update(message)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            path.write_bytes(b"video-bytes")
            video = GeneratedVideo(path=path)

            delivered = asyncio.run(
                send_video_to_telegram(cast(Any, update), video, message)
            )

        self.assertTrue(delivered)
        self.assertEqual(len(message.documents), 1)

    def test_service_answer_after_button_goes_to_effective_message(self) -> None:
        from core.state_and_commands import reply_service_text

        message = self._video_message()
        update = self._button_update(message)

        asyncio.run(reply_service_text(cast(Any, update), "Готово"))

        message.reply_text.assert_awaited_once()
        self.assertIn("Готово", str(message.reply_text.await_args.args[0]))

    def test_button_update_does_not_use_bot_author_as_user(self) -> None:
        """Пользователь ответа остаётся автором исходного сообщения."""

        from core.state_and_commands import get_delivery_message

        message = self._video_message()
        update = self._button_update(message)

        target = get_delivery_message(cast(Any, update))

        self.assertIs(target, message)
        self.assertEqual(update.effective_user.id, 1001)

    def test_video_is_generated_and_delivered_after_button_press(self) -> None:
        """После нажатия кнопки видео генерируется и уходит в исходный чат."""

        import tempfile
        from pathlib import Path

        from core.common_types import GeneratedVideo
        from core.tool_handlers.media import run_video_request
        from core.video_requests import build_video_request_draft

        store = QuestionStateStore()
        profile = _profile()
        draft = build_video_request_draft("animate", profile, duration=None)
        directory = tempfile.mkdtemp()

        async def generate_video(
            _prompt: str,
            _options: Any,
            reference_images: Any = (),
        ) -> Any:
            path = Path(directory) / "video.mp4"
            path.write_bytes(b"video-bytes")
            return GeneratedVideo(path=path, cost=0.25, metadata={"job_id": "job-9"})

        message = self._video_message()
        update = self._button_update(message)
        ctx = SimpleNamespace(
            update=update,
            openrouter_service=SimpleNamespace(generate_video=generate_video),
            context_tokens=0,
            completion_tokens=0,
            user_text="оживи картинку",
            video_tool_state=SimpleNamespace(
                attempted_video_call_ids=set(),
                completed_operation_ids=set(),
                failed_operation_ids=set(),
                retry_forbidden=False,
                mark_completed=lambda _value: None,
                mark_failed=lambda _value: None,
                is_known_operation=lambda _value: False,
            ),
        )

        with (
            patch(
                "core.tool_handlers.media.reply_service_message",
                new=AsyncMock(return_value=SimpleNamespace(edit_text=AsyncMock())),
            ),
            patch("core.tool_handlers.media.animate_service_message", new=AsyncMock()),
            patch(
                "core.tool_handlers.media.add_user_image_session_history",
                new=AsyncMock(),
            ),
        ):
            result = asyncio.run(
                run_video_request(cast(Any, ctx), draft.with_duration(10))
            )

        self.assertTrue(result.output["ok"])
        self.assertEqual(result.output["duration"], 10)
        self.assertEqual(result.output["job_id"], "job-9")
        self.assertEqual(result.output["cost"], 0.25)
        # Видео ушло в сообщение бота, к которому привязана кнопка.
        self.assertEqual(len(message.videos), 1)
        self.assertFalse(Path(directory, "video.mp4").exists())


class DurationQuestionShapeTests(unittest.TestCase):
    def test_question_buttons_respect_telegram_callback_data_limit(self) -> None:
        universal = build_inline_keyboard(
            "0123456789abcdef",
            (
                QuestionOption("a", "Первый"),
                QuestionOption("b", "Второй"),
                QuestionOption("c", "Третий"),
            ),
        )
        duration = build_inline_keyboard(
            "0123456789abcdef",
            build_duration_question_options((5, 10, 15, 20, 30)),
        )

        for keyboard in (universal, duration):
            for row in keyboard.inline_keyboard:
                for button in row:
                    callback_data = button.callback_data
                    self.assertIsInstance(callback_data, str)
                    assert isinstance(callback_data, str)
                    self.assertLessEqual(
                        len(callback_data.encode("utf-8")),
                        CALLBACK_DATA_MAX_BYTES,
                    )

    def test_duration_question_shows_five_values_and_custom(self) -> None:
        options = build_duration_question_options((5, 10, 15, 20, 30))
        request = PendingQuestionRequest(
            text="Сколько секунд должно длиться видео?",
            options=options,
        )

        self.assertEqual(len(request.options), 5)
        self.assertEqual(
            [option.payload for option in request.options],
            [5, 10, 15, 20, 30],
        )
        keyboard = build_inline_keyboard("duration-q", request.options)
        actions = []
        for row in keyboard.inline_keyboard:
            for button in row:
                callback_data = button.callback_data
                self.assertIsInstance(callback_data, str)
                assert isinstance(callback_data, str)
                actions.append(callback_data.split(":")[-1])
        self.assertEqual(len(actions), 7)
        self.assertEqual(actions[-2:], [CUSTOM_ANSWER_ACTION, CANCEL_ACTION])

    def test_universal_question_has_three_options_custom_and_separate_cancel(self) -> None:
        options = (
            QuestionOption("a", "Первый"),
            QuestionOption("b", "Второй"),
            QuestionOption("c", "Третий"),
        )

        keyboard = build_inline_keyboard("universal-q", options)
        actions = []
        for row in keyboard.inline_keyboard:
            for button in row:
                callback_data = button.callback_data
                self.assertIsInstance(callback_data, str)
                assert isinstance(callback_data, str)
                actions.append(callback_data.split(":")[-1])

        self.assertEqual(actions, ["0", "1", "2", CUSTOM_ANSWER_ACTION, CANCEL_ACTION])

    def test_universal_question_accepts_up_to_three_options(self) -> None:
        request = PendingQuestionRequest(
            text="Что выбрать?",
            options=(
                QuestionOption("a", "Первый"),
                QuestionOption("b", "Второй"),
                QuestionOption("c", "Третий"),
            ),
        )

        self.assertEqual(len(request.options), 3)

    def test_four_options_are_rejected_by_universal_validator(self) -> None:
        with self.assertRaises(ValueError):
            validate_universal_option_count(4)

    def test_duplicate_option_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PendingQuestionRequest(
                text="Что выбрать?",
                options=(
                    QuestionOption("a", "Первый"),
                    QuestionOption("a", "Второй"),
                ),
            )

    def test_video_draft_snapshot_is_independent_of_settings(self) -> None:
        profile = _profile()
        draft = build_video_request_draft(
            "котик бежит",
            profile,
            duration=None,
        )

        self.assertTrue(draft.requires_duration)
        self.assertIsNone(draft.duration)
        self.assertEqual(draft.profile.model, "alibaba/wan-3.0-prime")
        self.assertEqual(draft.profile.resolution, "480p")


class QuestionSendFailureTests(unittest.TestCase):
    def test_failed_question_send_cancels_pending_state_and_reports_error(self) -> None:
        import main as bot_main

        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        message = _message()
        message.reply_text = AsyncMock(side_effect=RuntimeError("telegram unavailable"))
        update = _update(message=message)
        context = _context(service)
        created_records: list[Any] = []
        create = store.create

        async def capture_create(**kwargs: Any) -> Any:
            record = await create(**kwargs)
            created_records.append(record)
            return record

        with (
            patch.object(store, "create", new=capture_create),
            patch("main.reply_service_text", new=AsyncMock()) as report_error,
        ):
            asyncio.run(
                bot_main._ask_user_question(
                    update,
                    context,
                    _request((QuestionOption("a", "Первый"),)),
                    None,
                )
            )

        self.assertEqual(len(created_records), 1)
        self.assertEqual(created_records[0].status, QuestionStatus.CANCELLED)
        self.assertIsNone(asyncio.run(store.get_active_for_user(1001)))
        report_error.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
