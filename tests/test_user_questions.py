"""Проверки сервиса вопросов и атомарного хранилища ожиданий.

Покрываются сценарии Шагов 3, 4 и 6 плана:

- универсальный вопрос принимает один-три готовых варианта и отклоняет четыре;
- «Свой ответ» нельзя отключить, отмена — отдельная служебная кнопка;
- сервис отображает и три, и пять вариантов без привязки к видеомодели;
- атомарное принятие ответа, повторные нажатия, срок, отмена и очистка;
- ошибка продолжения не приводит к повторному платному действию;
- обработчик `ask_user_question` проверяет структуру на сервере и возвращает
  типизированную приостановку.
"""

import asyncio
import os
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:TEST_TOKEN")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from core.common_types import (
    ModelAnswer,
    PendingQuestionRequest,
    QuestionAnswer,
    QuestionAnswerKind,
    QuestionAnswerMode,
    QuestionOption,
    QuestionStatus,
)
from core.question_state import (
    QuestionContinuationContext,
    QuestionStateStore,
)
from core.video_requests import VideoModelProfile, build_video_request_draft
from core.user_questions import (
    CALLBACK_DATA_MAX_BYTES,
    CUSTOM_ANSWER_ACTION,
    UserQuestionService,
    build_callback_data,
    build_duration_question_options,
    build_inline_keyboard,
    parse_callback_data,
    validate_universal_option_count,
)


def _profile() -> VideoModelProfile:
    return VideoModelProfile(
        model="alibaba/wan-3.0-prime",
        resolution="480p",
        allowed_durations=tuple(range(2, 31)),
        duration_presets=(5, 10, 15, 20, 30),
        allowed_aspect_ratios=("16:9", "9:16", "1:1"),
        max_references=10,
    )


def _option(option_id: str, text: str, payload: Any = None) -> QuestionOption:
    return QuestionOption(option_id=option_id, text=text, payload=payload)


def _request(
    *,
    options: tuple[QuestionOption, ...] = (),
    validator: Any = None,
    text: str = "Какой вариант выбираем?",
    answer_mode: QuestionAnswerMode = QuestionAnswerMode.PLAIN,
) -> PendingQuestionRequest:
    return PendingQuestionRequest(
        text=text,
        options=options,
        input_hint="Напишите ответ текстом",
        validator=validator,
        answer_mode=answer_mode,
    )


def _message() -> SimpleNamespace:
    """Тестовое сообщение Telegram с записью отправки."""

    sent: dict[str, Any] = {}

    async def reply_text(text: str, reply_markup: Any = None, **_kwargs: Any) -> SimpleNamespace:
        sent["text"] = text
        sent["reply_markup"] = reply_markup
        return SimpleNamespace(message_id=777)

    return SimpleNamespace(
        message_id=555,
        message_thread_id=None,
        chat=SimpleNamespace(id=42),
        reply_text=reply_text,
        sent=sent,
    )


def _update(user_id: int = 1001) -> Any:
    message = _message()
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=42),
        effective_message=message,
        message=message,
    )


class CallbackDataTests(unittest.TestCase):
    def test_callback_data_contains_only_question_id_and_action(self) -> None:
        data = build_callback_data("abcdef0123456789", "0")

        self.assertEqual(data, "q:abcdef0123456789:0")
        self.assertLessEqual(len(data.encode("utf-8")), CALLBACK_DATA_MAX_BYTES)
        self.assertNotIn("10", data.split(":")[-1])

    def test_callback_data_respects_telegram_limit(self) -> None:
        with self.assertRaises(ValueError):
            build_callback_data("x" * 100, "0")

    def test_parse_callback_data_rejects_foreign_data(self) -> None:
        self.assertIsNone(parse_callback_data("other:1:2"))
        self.assertIsNone(parse_callback_data("q:1"))
        self.assertIsNone(parse_callback_data(None))
        self.assertIsNone(parse_callback_data(""))

    def test_parse_callback_data_reads_action(self) -> None:
        parsed = parse_callback_data("q:abc:custom")

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.question_id, "abc")
        self.assertEqual(parsed.action, CUSTOM_ANSWER_ACTION)


class UniversalOptionLimitTests(unittest.TestCase):
    def test_universal_question_accepts_one_to_three_options(self) -> None:
        for count in (1, 2, 3):
            with self.subTest(count=count):
                self.assertEqual(validate_universal_option_count(count), count)

    def test_universal_question_rejects_four_options(self) -> None:
        with self.assertRaises(ValueError):
            validate_universal_option_count(4)

    def test_universal_question_rejects_zero_options(self) -> None:
        with self.assertRaises(ValueError):
            validate_universal_option_count(0)

    def test_service_does_not_limit_duration_question_to_three_buttons(self) -> None:
        options = build_duration_question_options((5, 10, 15, 20, 30))
        keyboard = build_inline_keyboard("qid", options)

        self.assertEqual(len(keyboard.inline_keyboard), 7)


class KeyboardTests(unittest.TestCase):
    def test_three_options_plus_custom_and_cancel(self) -> None:
        keyboard = build_inline_keyboard(
            "qid",
            (
                _option("a", "Первый"),
                _option("b", "Второй"),
                _option("c", "Третий"),
            ),
        )

        self.assertEqual(len(keyboard.inline_keyboard), 5)
        self.assertEqual(keyboard.inline_keyboard[0][0].text, "Первый")
        self.assertEqual(keyboard.inline_keyboard[2][0].text, "Третий")
        self.assertEqual(
            keyboard.inline_keyboard[3][0].callback_data,
            "q:qid:custom",
        )
        self.assertEqual(
            keyboard.inline_keyboard[4][0].callback_data,
            "q:qid:cancel",
        )

    def test_five_duration_options_are_rendered_without_limit(self) -> None:
        options = build_duration_question_options((5, 10, 15, 20, 30))

        self.assertEqual(len(options), 5)
        keyboard = build_inline_keyboard("qid", options)
        self.assertEqual(len(keyboard.inline_keyboard), 7)
        self.assertEqual(keyboard.inline_keyboard[0][0].text, "5 с")
        self.assertEqual(keyboard.inline_keyboard[4][0].text, "30 с")

    def test_single_option_keyboard_still_has_custom_and_cancel(self) -> None:
        keyboard = build_inline_keyboard("qid", (_option("a", "Один"),))

        self.assertEqual(len(keyboard.inline_keyboard), 3)


class QuestionServiceTests(unittest.TestCase):
    def test_ask_sends_question_and_remembers_message_id(self) -> None:
        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        update = _update()

        record = asyncio.run(
            service.ask(
                update,
                _request(options=(_option("a", "Первый"), _option("b", "Второй"))),
            )
        )

        self.assertEqual(record.status, QuestionStatus.PENDING)
        self.assertEqual(record.question_message_id, 777)
        self.assertEqual(record.user_id, 1001)
        self.assertEqual(record.chat_id, 42)
        self.assertEqual(len(record.options), 2)
        self.assertIsNotNone(update.message.sent["reply_markup"])

    def test_option_answer_is_accepted_atomically(self) -> None:
        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        update = _update()
        record = asyncio.run(
            service.ask(update, _request(options=(_option("a", "Первый"),)))
        )

        answer = asyncio.run(service.handle_option(record, 0))

        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertEqual(answer.kind, QuestionAnswerKind.OPTION)
        self.assertEqual(answer.text, "Первый")
        self.assertEqual(answer.option_id, "a")

        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.ACCEPTED)

    def test_repeated_answer_is_rejected(self) -> None:
        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        record = asyncio.run(
            service.ask(_update(), _request(options=(_option("a", "Первый"),)))
        )

        first = asyncio.run(service.handle_option(record, 0))
        second = asyncio.run(service.handle_option(record, 0))

        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_option_index_out_of_range_is_rejected(self) -> None:
        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        record = asyncio.run(
            service.ask(_update(), _request(options=(_option("a", "Первый"),)))
        )

        self.assertIsNone(asyncio.run(service.handle_option(record, 5)))

    def test_custom_answer_is_accepted_and_question_closes(self) -> None:
        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        record = asyncio.run(service.ask(_update(), _request()))

        answer, error = asyncio.run(service.handle_custom_answer(record, "  мой ответ  "))

        self.assertIsNone(error)
        assert answer is not None
        self.assertEqual(answer.kind, QuestionAnswerKind.CUSTOM)
        self.assertEqual(answer.text, "мой ответ")

        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.ACCEPTED)

    def test_invalid_custom_answer_keeps_question_open(self) -> None:
        def validator(text: str) -> str | None:
            if not text.isdigit():
                return "Нужно целое число секунд."
            return None

        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        record = asyncio.run(service.ask(_update(), _request(validator=validator)))

        answer, error = asyncio.run(service.handle_custom_answer(record, "не число"))

        self.assertIsNone(answer)
        self.assertIsNotNone(error)
        self.assertIn("целое число", error or "")

        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.PENDING)

    def test_empty_custom_answer_is_rejected(self) -> None:
        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        record = asyncio.run(service.ask(_update(), _request()))

        answer, error = asyncio.run(service.handle_custom_answer(record, "   "))

        self.assertIsNone(answer)
        self.assertIsNotNone(error)

    def test_cancel_closes_question(self) -> None:
        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        record = asyncio.run(service.ask(_update(), _request()))

        self.assertTrue(asyncio.run(service.handle_cancel(record)))
        self.assertFalse(asyncio.run(service.handle_cancel(record)))

        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.CANCELLED)

    def test_expired_question_does_not_accept_answer(self) -> None:
        now = {"value": 1000.0}
        store = QuestionStateStore(timeout_seconds=10, clock=lambda: now["value"])
        service = UserQuestionService(store=store)
        record = asyncio.run(
            service.ask(_update(), _request(options=(_option("a", "Первый"),)))
        )

        now["value"] += 11
        answer = asyncio.run(service.handle_option(record, 0))

        self.assertIsNone(answer)
        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.EXPIRED)
        self.assertIsNone(asyncio.run(store.get_active_for_user(1001)))

    def test_new_question_replaces_previous_one(self) -> None:
        store = QuestionStateStore()
        service = UserQuestionService(store=store)

        first = asyncio.run(service.ask(_update(), _request()))
        second = asyncio.run(service.ask(_update(), _request()))

        first_stored = asyncio.run(store.get(first.question_id))
        assert first_stored is not None
        self.assertEqual(first_stored.status, QuestionStatus.CANCELLED)
        self.assertEqual(
            asyncio.run(store.get_active_for_user(1001)).question_id,  # type: ignore[union-attr]
            second.question_id,
        )

    def test_invalidate_user_questions_cancels_active_wait(self) -> None:
        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        record = asyncio.run(service.ask(_update(), _request()))

        invalidated = asyncio.run(store.invalidate_user_questions(1001))

        self.assertEqual(invalidated, 1)
        self.assertIsNone(asyncio.run(store.get_active_for_user(1001)))
        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.CANCELLED)

    def test_cleanup_expired_removes_records(self) -> None:
        now = {"value": 1000.0}
        store = QuestionStateStore(timeout_seconds=10, clock=lambda: now["value"])
        service = UserQuestionService(store=store)
        record = asyncio.run(service.ask(_update(), _request()))

        now["value"] += 11
        removed = asyncio.run(store.cleanup_expired())

        self.assertEqual(removed, 1)
        self.assertIsNone(asyncio.run(store.get(record.question_id)))

    def test_continuation_failure_does_not_repeat_execution(self) -> None:
        store = QuestionStateStore()
        calls: list[str] = []

        async def continuation(
            record: Any,
            answer: QuestionAnswer,
            update: Any,
            context: Any,
        ) -> None:
            calls.append(answer.text)
            raise RuntimeError("продолжение упало")

        service = UserQuestionService(store=store, continuation=continuation)
        record = asyncio.run(
            service.ask(_update(), _request(options=(_option("a", "Первый"),)))
        )
        answer = asyncio.run(service.handle_option(record, 0))
        assert answer is not None

        with self.assertRaises(RuntimeError):
            asyncio.run(
                service.continue_after_answer(
                    record,
                    answer,
                    update=None,
                    context=None,
                )
            )

        # Продолжение вызвано ровно один раз: сбой не приводит к повтору.
        self.assertEqual(calls, ["Первый"])
        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.COMPLETED)

    def test_continuation_is_called_once_with_accepted_answer(self) -> None:
        store = QuestionStateStore()
        calls: list[QuestionAnswer] = []

        async def continuation(
            record: Any,
            answer: QuestionAnswer,
            update: Any,
            context: Any,
        ) -> None:
            calls.append(answer)

        service = UserQuestionService(store=store, continuation=continuation)
        record = asyncio.run(
            service.ask(_update(), _request(options=(_option("a", "Первый"),)))
        )
        answer = asyncio.run(service.handle_option(record, 0))
        assert answer is not None

        asyncio.run(
            service.continue_after_answer(record, answer, update=None, context=None)
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].text, "Первый")
        stored = asyncio.run(store.get(record.question_id))
        assert stored is not None
        self.assertEqual(stored.status, QuestionStatus.COMPLETED)


class ContinuationContextTests(unittest.TestCase):
    def test_release_drops_video_snapshot(self) -> None:
        draft = build_video_request_draft(
            "котик бежит",
            _profile(),
            duration=None,
        )
        context = QuestionContinuationContext(
            response_id="resp-1",
            original_user_text="сделай видео",
            video_request=draft,
        )

        context.release()

        self.assertIsNone(context.video_request)
        self.assertEqual(context.metadata, {})

    def test_continuation_keeps_technical_fields(self) -> None:
        context = QuestionContinuationContext(
            response_id="resp-2",
            model_name="gpt-5.6-terra",
            original_user_text="опиши картинки",
            context_tokens=100,
            completion_tokens=20,
            tool_round=2,
            answer_mode=QuestionAnswerMode.PLAIN,
        )

        self.assertEqual(context.response_id, "resp-2")
        self.assertEqual(context.model_name, "gpt-5.6-terra")
        self.assertEqual(context.original_user_text, "опиши картинки")
        self.assertEqual(context.context_tokens, 100)
        self.assertEqual(context.completion_tokens, 20)
        self.assertEqual(context.tool_round, 2)
        self.assertFalse(context.is_video_request)
        self.assertIsNotNone(context.image_tool_state)
        self.assertIsNotNone(context.video_tool_state)


class AskUserQuestionHandlerTests(unittest.TestCase):
    """Проверки обработчика универсального вопроса."""

    def _ctx(self) -> Any:
        return SimpleNamespace(
            update=SimpleNamespace(effective_user=SimpleNamespace(id=1001)),
            context=SimpleNamespace(),
            model_name="gpt-5.6-terra",
            recursion_depth=0,
            openrouter_service=SimpleNamespace(),
            context_tokens=0,
            completion_tokens=0,
            user_text="сделай картинку",
        )

    def _args(
        self,
        options: list[dict[str, Any]],
        question: str = "Какой вариант выбираем?",
        **extra: Any,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"question": question, "options": options}
        args.update(extra)
        return args

    def test_one_to_three_options_are_accepted(self) -> None:
        from core.tool_handlers.questions import handle_ask_user_question

        for count in (1, 2, 3):
            with self.subTest(count=count):
                options = [
                    {"id": f"opt-{index}", "label": f"Вариант {index}"}
                    for index in range(count)
                ]
                result = asyncio.run(
                    handle_ask_user_question(self._ctx(), self._args(options))
                )

                self.assertIsNotNone(result.pause)
                assert result.pause is not None
                self.assertEqual(len(result.pause.request.options), count)
                self.assertEqual(result.pause.request.answer_mode, QuestionAnswerMode.PLAIN)
                self.assertEqual(result.pause.request.text, "Какой вариант выбираем?")

    def test_four_options_are_rejected(self) -> None:
        from core.tool_handlers.questions import handle_ask_user_question

        options = [
            {"id": f"opt-{index}", "label": f"Вариант {index}"}
            for index in range(4)
        ]
        result = asyncio.run(handle_ask_user_question(self._ctx(), self._args(options)))

        self.assertIsNone(result.pause)
        self.assertFalse(result.output["ok"])
        self.assertIn("1", str(result.output["error"]))
        self.assertIn("3", str(result.output["error"]))

    def test_zero_options_are_rejected(self) -> None:
        from core.tool_handlers.questions import handle_ask_user_question

        result = asyncio.run(handle_ask_user_question(self._ctx(), self._args([])))

        self.assertIsNone(result.pause)
        self.assertFalse(result.output["ok"])

    def test_empty_question_is_rejected(self) -> None:
        from core.tool_handlers.questions import handle_ask_user_question

        result = asyncio.run(
            handle_ask_user_question(
                self._ctx(),
                self._args([{"id": "a", "label": "Первый"}], question="   "),
            )
        )

        self.assertIsNone(result.pause)
        self.assertFalse(result.output["ok"])

    def test_empty_option_label_is_rejected(self) -> None:
        from core.tool_handlers.questions import handle_ask_user_question

        result = asyncio.run(
            handle_ask_user_question(
                self._ctx(),
                self._args([{"id": "a", "label": "  "}]),
            )
        )

        self.assertIsNone(result.pause)
        self.assertFalse(result.output["ok"])

    def test_empty_option_id_is_rejected(self) -> None:
        from core.tool_handlers.questions import handle_ask_user_question

        result = asyncio.run(
            handle_ask_user_question(
                self._ctx(),
                self._args([{"id": "", "label": "Первый"}]),
            )
        )

        self.assertIsNone(result.pause)
        self.assertFalse(result.output["ok"])

    def test_duplicate_option_ids_are_rejected(self) -> None:
        from core.tool_handlers.questions import handle_ask_user_question

        result = asyncio.run(
            handle_ask_user_question(
                self._ctx(),
                self._args(
                    [
                        {"id": "same", "label": "Первый"},
                        {"id": "same", "label": "Второй"},
                    ]
                ),
            )
        )

        self.assertIsNone(result.pause)
        self.assertFalse(result.output["ok"])

    def test_text_field_is_accepted_as_option_label(self) -> None:
        from core.tool_handlers.questions import handle_ask_user_question

        result = asyncio.run(
            handle_ask_user_question(
                self._ctx(),
                self._args([{"id": "a", "text": "Первый"}]),
            )
        )

        self.assertIsNotNone(result.pause)
        assert result.pause is not None
        self.assertEqual(result.pause.request.options[0].text, "Первый")

    def test_server_only_parameters_are_rejected(self) -> None:
        from core.tool_handlers.questions import handle_ask_user_question

        for key in ("user_id", "chat_id", "duration", "model", "continuation"):
            with self.subTest(key=key):
                result = asyncio.run(
                    handle_ask_user_question(
                        self._ctx(),
                        self._args([{"id": "a", "label": "Первый"}], **{key: "значение"}),
                    )
                )

                self.assertIsNone(result.pause)
                self.assertFalse(result.output["ok"])

    def test_custom_input_hint_is_optional_and_preserved(self) -> None:
        from core.tool_handlers.questions import handle_ask_user_question

        result = asyncio.run(
            handle_ask_user_question(
                self._ctx(),
                self._args(
                    [{"id": "a", "label": "Первый"}],
                    custom_input_hint="Напишите свой вариант",
                ),
            )
        )

        self.assertIsNotNone(result.pause)
        assert result.pause is not None
        self.assertEqual(
            result.pause.request.input_hint,
            "Напишите свой вариант",
        )

    def test_custom_answer_option_is_always_available(self) -> None:
        from core.tool_handlers.questions import handle_ask_user_question

        result = asyncio.run(
            handle_ask_user_question(
                self._ctx(),
                self._args([{"id": "a", "label": "Первый"}]),
            )
        )

        assert result.pause is not None
        keyboard = build_inline_keyboard(
            "qid",
            result.pause.request.options,
        )
        callback_actions = [
            button.callback_data.split(":")[-1]
            for row in keyboard.inline_keyboard
            for button in row
        ]
        self.assertIn(CUSTOM_ANSWER_ACTION, callback_actions)
        self.assertIn("cancel", callback_actions)

    def test_question_check_uses_same_validation(self) -> None:
        from core.tool_handlers.questions import check_ask_user_question

        check = asyncio.run(
            check_ask_user_question(
                self._ctx(),
                self._args([{"id": "a", "label": "Первый"}]),
            )
        )

        self.assertIsNotNone(check)
        assert check is not None
        self.assertEqual(len(check.request.options), 1)

        with self.assertRaises(Exception):
            asyncio.run(
                check_ask_user_question(
                    self._ctx(),
                    self._args([{"id": "a", "label": "Первый"}], user_id="1"),
                )
            )


class BotReplyQuestionIntegrationTests(unittest.TestCase):
    """Связка `get_bot_reply` с сервисом вопросов."""

    def test_pending_question_is_sent_and_turn_is_finished(self) -> None:
        import main as bot_main

        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        request = _request(options=(_option("a", "Первый"),))
        continuation = QuestionContinuationContext(
            response_id="resp-1",
            original_user_text="исходный текст",
        )
        answer = ModelAnswer(
            None,
            pending_question=request,
            pending_continuation=continuation,
        )
        update = _update()
        context = SimpleNamespace(
            application=SimpleNamespace(bot_data={"question_service": service}),
        )

        with patch(
            "main.get_model_answer",
            new=AsyncMock(return_value=answer),
        ):
            bot_reply, _tokens = asyncio.run(
                bot_main.get_bot_reply(
                    cast(Any, update),
                    cast(Any, context),
                    "уточни",
                    cast(Any, SimpleNamespace()),
                )
            )

        # Обычный текст ответа не отправляется: вопрос уходит кнопками.
        self.assertIsNone(bot_reply)
        self.assertEqual(update.message.sent["text"], request.text)
        self.assertIsNotNone(update.message.sent["reply_markup"])

        record = asyncio.run(store.get_active_for_user(1001))
        assert record is not None
        self.assertIs(record.continuation, continuation)

    def test_question_is_not_duplicated_as_plain_text(self) -> None:
        import main as bot_main

        store = QuestionStateStore()
        service = UserQuestionService(store=store)
        request = _request(options=(_option("a", "Первый"),))
        answer = ModelAnswer(
            None,
            pending_question=request,
            pending_continuation=QuestionContinuationContext(response_id="resp-1"),
        )
        update = _update()
        context = SimpleNamespace(
            application=SimpleNamespace(bot_data={"question_service": service}),
        )

        with patch(
            "main.get_model_answer",
            new=AsyncMock(return_value=answer),
        ):
            asyncio.run(
                bot_main.get_bot_reply(
                    cast(Any, update),
                    cast(Any, context),
                    "уточни",
                    cast(Any, SimpleNamespace()),
                )
            )

        # Текст вопроса отправлен ровно один раз.
        self.assertEqual(update.message.sent["text"], request.text)
        self.assertEqual(len(update.message.sent), 2)


if __name__ == "__main__":
    unittest.main()
