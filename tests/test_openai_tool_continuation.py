"""Проверки continuation Responses API после function calls.

Покрываются сценарии Шага 5 плана:

- пауза без следующего обращения к модели;
- правильные идентификаторы продолжения (`previous_response_id`) и `call_id`;
- полный пакет результатов нескольких инструментов;
- приоритет вопроса перед зависимыми платными инструментами одного пакета;
- отсутствие повторных эффектов уже выполненных инструментов;
- сохранение исходного текста, лимитов раундов, токенов и защитного состояния;
- безопасный отказ при `store_responses=False`.
"""

import asyncio
import os
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:TEST_TOKEN")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from openai.types.responses import Response
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from core.common_types import (
    GeneratedImage,
    PendingQuestionRequest,
    PendingToolCall,
    QuestionAnswer,
    QuestionAnswerKind,
    QuestionAnswerMode,
    QuestionOption,
    ToolQuestionCheck,
    ToolQuestionPause,
    ToolResult,
)
from core.config import settings
from core.openai_api import get_model_answer, resume_model_answer
from core.question_state import QuestionContinuationContext
from core.tool_registry import ImageToolTurnState, VideoToolTurnState


def response_with_output(response_id: str, output: list[object], text: str = "") -> Response:
    """Создаёт минимальный SDK Response без сетевого вызова."""

    if text:
        output = [
            ResponseOutputMessage(
                id="msg-final",
                content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ]

    return Response.model_construct(
        id=response_id,
        object="response",
        created_at=0,
        status="completed",
        error=None,
        incomplete_details=None,
        instructions=None,
        max_output_tokens=None,
        model="gpt-5.6-terra",
        output=output,
        parallel_tool_calls=True,
        temperature=None,
        tool_choice="auto",
        tools=[],
        top_p=None,
        truncation="disabled",
        usage=None,
        user=None,
        metadata={},
    )


def function_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def question_request(
    *,
    text: str = "Какой вариант?",
    option_ids: tuple[str, ...] = ("a", "b"),
) -> PendingQuestionRequest:
    """Готовый запрос вопроса с заданными вариантами."""

    return PendingQuestionRequest(
        text=text,
        options=tuple(
            QuestionOption(option_id=option_id, text=f"Вариант {option_id}")
            for option_id in option_ids
        ),
    )


def settings_without_stored_responses() -> Any:
    """Возвращает настройки с отключённым сохранением ответов OpenAI."""

    from dataclasses import replace

    import core.openai_api as api_module

    return replace(
        api_module.settings,
        openai=replace(api_module.settings.openai, store_responses=False),
    )


class FakeRegistry:
    """Тестовый реестр инструментов с предварительными проверками."""

    def __init__(
        self,
        handlers: dict[str, Any] | None = None,
        question_checks: dict[str, Any] | None = None,
    ) -> None:
        self._handlers = dict(handlers or {})
        self._question_checks = dict(question_checks or {})

    def get(self, name: str) -> Any:
        return self._handlers.get(name)

    def get_question_check(self, name: str) -> Any:
        return self._question_checks.get(name)


class OpenAIToolContinuationTests(unittest.TestCase):
    def test_user_text_is_preserved_in_every_tool_continuation_round(self) -> None:
        first_response = response_with_output(
            "resp-first-tool",
            [function_call("call-first", "test_first", "{}")],
        )
        second_response = response_with_output(
            "resp-second-tool",
            [function_call("call-second", "test_second", "{}")],
        )
        final_response = response_with_output("resp-final", [], "Готово")
        create = AsyncMock(side_effect=[first_response, second_response, final_response])
        captured_user_text: list[str | None] = []

        async def first_handler(ctx, _args):
            captured_user_text.append(ctx.user_text)
            return ToolResult({"ok": True})

        async def second_handler(ctx, _args):
            captured_user_text.append(ctx.user_text)
            return ToolResult({"ok": True})

        registry = SimpleNamespace(
            get=lambda name: {
                "test_first": first_handler,
                "test_second": second_handler,
            }.get(name)
        )
        update: Any = SimpleNamespace(effective_user=SimpleNamespace(id=1))
        context: Any = SimpleNamespace()
        openrouter_service: Any = SimpleNamespace()
        original_user_text = 'Измени изображение и сохрани надпись "HELLO"'

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_user_model", new=AsyncMock(return_value="gpt-5.6-terra")),
            patch("core.openai_api.get_simple_answer", side_effect=create),
        ):
            answer = asyncio.run(
                get_model_answer(
                    update,
                    context,
                    [{"role": "user", "content": "техническая сводка сессии"}],
                    openrouter_service=openrouter_service,
                    user_text=original_user_text,
                )
            )

        self.assertEqual(answer.bot_reply, "Готово")
        self.assertEqual(captured_user_text, [original_user_text, original_user_text])

    def test_all_calls_are_returned_with_previous_response_id(self) -> None:
        first_response = response_with_output(
            "resp-tool",
            [
                SimpleNamespace(type="reasoning", id="rs_1"),
                function_call("call-weather", "test_weather", '{"city":"Moscow"}'),
                function_call("call-second", "test_echo", '{"value":42}'),
            ],
        )
        final_response = response_with_output(
            "resp-final",
            [],
            "Готово",
        )
        create = AsyncMock(side_effect=[first_response, final_response])

        async def weather_handler(_ctx, args):
            self.assertEqual(args, {"city": "Moscow"})
            return ToolResult({"ok": True, "temperature": 20})

        async def echo_handler(_ctx, args):
            self.assertEqual(args, {"value": 42})
            return ToolResult({"ok": True, "echo": args["value"]})

        registry = SimpleNamespace(
            get=lambda name: {
                "test_weather": weather_handler,
                "test_echo": echo_handler,
            }.get(name)
        )
        update: Any = SimpleNamespace(effective_user=SimpleNamespace(id=1))
        context: Any = SimpleNamespace()
        openrouter_service: Any = SimpleNamespace()

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_user_model", new=AsyncMock(return_value="gpt-5.6-terra")),
            patch("core.openai_api.get_simple_answer", side_effect=create) as get_answer,
        ):
            answer = asyncio.run(
                get_model_answer(
                    update,
                    context,
                    [{"role": "user", "content": "Погода?"}],
                    openrouter_service=openrouter_service,
                )
            )

        self.assertEqual(answer.bot_reply, "Готово")
        self.assertEqual(get_answer.await_count, 2)
        first_call = get_answer.await_args_list[0]
        second_call = get_answer.await_args_list[1]
        self.assertEqual(first_call.args[0], [{"role": "user", "content": "Погода?"}])
        self.assertEqual(second_call.kwargs["previous_response_id"], "resp-tool")
        self.assertEqual(
            second_call.args[0],
            [
                {
                    "type": "function_call_output",
                    "call_id": "call-weather",
                    "output": '{"ok": true, "temperature": 20}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-second",
                    "output": '{"ok": true, "echo": 42}',
                },
            ],
        )

    def test_handler_error_is_returned_as_function_call_output(self) -> None:
        first_response = response_with_output(
            "resp-error",
            [function_call("call-error", "test_error", "{}")],
        )
        final_response = response_with_output("resp-final", [], "Ошибка обработана")
        create = AsyncMock(side_effect=[first_response, final_response])

        async def broken_handler(_ctx, _args):
            raise RuntimeError("private details")

        registry = SimpleNamespace(get=lambda name: broken_handler if name == "test_error" else None)
        update: Any = SimpleNamespace(effective_user=SimpleNamespace(id=1))
        context: Any = SimpleNamespace()
        openrouter_service: Any = SimpleNamespace()

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_user_model", new=AsyncMock(return_value="gpt-5.6-terra")),
            patch("core.openai_api.get_simple_answer", side_effect=create) as get_answer,
        ):
            answer = asyncio.run(
                get_model_answer(
                    update,
                    context,
                    [{"role": "user", "content": "Вызови tool"}],
                    openrouter_service=openrouter_service,
                )
            )

        self.assertEqual(answer.bot_reply, "Ошибка обработана")
        output = get_answer.await_args_list[1].args[0][0]["output"]
        self.assertEqual(output, '{"ok": false, "error": "Не удалось выполнить инструмент \'test_error\'."}')
        self.assertNotIn("private details", output)

    def test_failed_image_operation_blocks_new_tool_call_ids_for_same_user_turn(self) -> None:
        first_response = response_with_output(
            "resp-image-failed",
            [function_call("image-call-first", "generate_image", '{"prompt":"first"}')],
        )
        second_response = response_with_output(
            "resp-retry-attempt",
            [
                function_call("image-call-new-id", "generate_image_from_image", '{"prompt":"retry"}'),
                function_call("weather-call", "test_weather", "{}"),
            ],
        )
        final_response = response_with_output("resp-final", [], "Готово")
        get_answer = AsyncMock(side_effect=[first_response, second_response, final_response])
        image_api_calls = 0
        weather_calls = 0

        async def image_handler(ctx, _args):
            nonlocal image_api_calls
            image_api_calls += 1
            return ToolResult(
                {
                    "ok": False,
                    "error": "provider failed",
                    "retry_forbidden": True,
                    "instruction": "Не повторять платный вызов в этом ходе.",
                },
                meta={"retry_forbidden": True},
            )

        async def other_handler(_ctx, _args):
            nonlocal weather_calls
            weather_calls += 1
            return ToolResult({"ok": True, "weather": "sunny"})

        registry = SimpleNamespace(
            get=lambda name: {
                "generate_image": image_handler,
                "generate_image_from_image": image_handler,
                "test_weather": other_handler,
            }.get(name)
        )
        update: Any = SimpleNamespace(effective_user=SimpleNamespace(id=5))

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_user_model", new=AsyncMock(return_value="gpt-5.6-terra")),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
        ):
            answer = asyncio.run(
                get_model_answer(
                    update,
                    SimpleNamespace(),
                    [{"role": "user", "content": "Создай картинку"}],
                    openrouter_service=SimpleNamespace(),
                    user_text="Создай картинку",
                )
            )

        self.assertEqual(answer.bot_reply, "Готово")
        self.assertEqual(image_api_calls, 1)
        self.assertEqual(weather_calls, 1)
        self.assertEqual(get_answer.await_count, 3)
        blocked_and_weather = get_answer.await_args_list[2].args[0]
        self.assertIn('"blocked_after_image_failure": true', blocked_and_weather[0]["output"])
        self.assertIn('"retry_forbidden": true', blocked_and_weather[0]["output"])
        self.assertIn('"weather": "sunny"', blocked_and_weather[1]["output"])

    def test_explicit_image_calls_in_same_response_remain_distinct_after_failure(self) -> None:
        first_response = response_with_output(
            "resp-two-explicit-images",
            [
                function_call("image-call-failed", "generate_image", '{"prompt":"first"}'),
                function_call(
                    "image-call-explicit-second",
                    "generate_image",
                    '{"prompt":"second"}',
                ),
            ],
        )
        next_response = response_with_output(
            "resp-autonomous-retry",
            [
                function_call(
                    "image-call-new-continuation-id",
                    "generate_image",
                    '{"prompt":"retry"}',
                )
            ],
        )
        final_response = response_with_output("resp-final", [], "Готово")
        get_answer = AsyncMock(side_effect=[first_response, next_response, final_response])
        attempted_prompts: list[str] = []

        async def image_handler(_ctx, args):
            attempted_prompts.append(args["prompt"])
            if args["prompt"] == "first":
                return ToolResult(
                    {"ok": False, "retry_forbidden": True},
                    meta={"retry_forbidden": True},
                )
            return ToolResult({"ok": True, "delivered": True})

        registry = SimpleNamespace(
            get=lambda name: image_handler if name == "generate_image" else None
        )
        update: Any = SimpleNamespace(effective_user=SimpleNamespace(id=7))

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_user_model", new=AsyncMock(return_value="gpt-5.6-terra")),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
        ):
            answer = asyncio.run(
                get_model_answer(
                    update,
                    cast(Any, SimpleNamespace()),
                    [{"role": "user", "content": "Создай две картинки"}],
                    openrouter_service=cast(Any, SimpleNamespace()),
                    user_text="Создай две картинки: первое и второе изображения",
                )
            )

        self.assertEqual(answer.bot_reply, "Готово")
        self.assertEqual(attempted_prompts, ["first", "second"])
        self.assertEqual(get_answer.await_count, 3)
        next_round_output = get_answer.await_args_list[2].args[0][0]["output"]
        self.assertIn('"blocked_after_image_failure": true', next_round_output)

    def test_repeated_image_call_id_is_blocked_even_after_success(self) -> None:
        repeated_call = function_call(
            "same-image-call-id",
            "generate_image",
            '{"prompt":"A single image"}',
        )
        first_response = response_with_output("resp-first-image", [repeated_call])
        repeated_response = response_with_output("resp-repeat-image", [repeated_call])
        final_response = response_with_output("resp-final", [], "Готово")
        get_answer = AsyncMock(side_effect=[first_response, repeated_response, final_response])
        image_calls = 0

        async def image_handler(_ctx, _args):
            nonlocal image_calls
            image_calls += 1
            return ToolResult({"ok": True, "delivered": True})

        registry = SimpleNamespace(
            get=lambda name: image_handler if name == "generate_image" else None
        )
        update: Any = SimpleNamespace(effective_user=SimpleNamespace(id=8))

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_user_model", new=AsyncMock(return_value="gpt-5.6-terra")),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
        ):
            answer = asyncio.run(
                get_model_answer(
                    update,
                    cast(Any, SimpleNamespace()),
                    [{"role": "user", "content": "Создай одну картинку"}],
                    openrouter_service=cast(Any, SimpleNamespace()),
                    user_text="Создай одну картинку",
                )
            )

        self.assertEqual(answer.bot_reply, "Готово")
        self.assertEqual(image_calls, 1)
        self.assertEqual(get_answer.await_count, 3)
        blocked_output = get_answer.await_args_list[2].args[0][0]["output"]
        self.assertIn('"blocked_after_image_failure": true', blocked_output)

    def test_multiple_successful_explicit_image_requests_remain_allowed_in_same_turn(self) -> None:
        first_response = response_with_output(
            "resp-two-images",
            [
                function_call("cat-call", "generate_image", '{"prompt":"A cat"}'),
                function_call("dog-call", "generate_image", '{"prompt":"A dog"}'),
            ],
        )
        final_response = response_with_output("resp-final", [], "Обе картинки готовы")
        get_answer = AsyncMock(side_effect=[first_response, final_response])
        prompts: list[str] = []

        async def image_handler(_ctx, args):
            prompts.append(args["prompt"])
            return ToolResult({"ok": True, "delivered": True})

        registry = SimpleNamespace(get=lambda name: image_handler if name == "generate_image" else None)
        update: Any = SimpleNamespace(effective_user=SimpleNamespace(id=6))

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_user_model", new=AsyncMock(return_value="gpt-5.6-terra")),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
        ):
            answer = asyncio.run(
                get_model_answer(
                    update,
                    SimpleNamespace(),
                    [{"role": "user", "content": "Сделай две картинки: кота и собаку"}],
                    openrouter_service=SimpleNamespace(),
                    user_text="Сделай две картинки: кота и собаку",
                )
            )

        self.assertEqual(answer.bot_reply, "Обе картинки готовы")
        self.assertEqual(prompts, ["A cat", "A dog"])


class ToolPauseTests(unittest.TestCase):
    """Приостановка цикла инструментов до ответа пользователя."""

    def _update(self) -> Any:
        return SimpleNamespace(effective_user=SimpleNamespace(id=11))

    def test_pause_does_not_call_model_again(self) -> None:
        from core.tool_handlers.questions import handle_ask_user_question

        first_response = response_with_output(
            "resp-question",
            [
                function_call(
                    "call-question",
                    "ask_user_question",
                    '{"question":"Какой цвет?", "options":[{"id":"red","label":"Красный"}]}',
                )
            ],
        )
        get_answer = AsyncMock(side_effect=[first_response])
        registry = FakeRegistry(
            handlers={"ask_user_question": handle_ask_user_question},
        )

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_user_model", new=AsyncMock(return_value="gpt-5.6-terra")),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
        ):
            answer = asyncio.run(
                get_model_answer(
                    cast(Any, self._update()),
                    cast(Any, SimpleNamespace()),
                    [{"role": "user", "content": "уточни"}],
                    openrouter_service=cast(Any, SimpleNamespace()),
                    user_text="уточни",
                )
            )

        # Модель вызвана ровно один раз: после паузы следующего запроса нет.
        self.assertEqual(get_answer.await_count, 1)
        self.assertIsNone(answer.bot_reply)
        self.assertIsNotNone(answer.pending_question)
        self.assertIsNotNone(answer.pending_continuation)

    def test_pause_keeps_continuation_identifiers(self) -> None:
        first_response = response_with_output(
            "resp-question-id",
            [function_call("call-question-id", "ask_user_question", '{"question":"?"}')],
        )
        get_answer = AsyncMock(side_effect=[first_response])
        request = question_request()
        registry = FakeRegistry(
            question_checks={
                "ask_user_question": AsyncMock(
                    return_value=ToolQuestionCheck(request=request)
                )
            },
        )

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_user_model", new=AsyncMock(return_value="gpt-5.6-terra")),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
        ):
            answer = asyncio.run(
                get_model_answer(
                    cast(Any, self._update()),
                    cast(Any, SimpleNamespace()),
                    [{"role": "user", "content": "уточни"}],
                    openrouter_service=cast(Any, SimpleNamespace()),
                    user_text="уточни",
                )
            )

        continuation = answer.pending_continuation
        self.assertIsNotNone(continuation)
        assert continuation is not None
        self.assertEqual(continuation.response_id, "resp-question-id")
        self.assertEqual(continuation.current_call_index, 0)
        self.assertEqual(len(continuation.tool_calls), 1)
        self.assertEqual(continuation.tool_calls[0].call_id, "call-question-id")
        self.assertEqual(continuation.tool_calls[0].name, "ask_user_question")
        self.assertEqual(continuation.original_user_text, "уточни")

    def test_question_precedes_dependent_paid_tool_in_same_batch(self) -> None:
        """Платный инструмент после вопроса не выполняется до ответа."""

        first_response = response_with_output(
            "resp-mixed",
            [
                function_call("call-paid", "generate_video", '{"prompt":"video"}'),
                function_call("call-question", "ask_user_question", '{"question":"?"}'),
            ],
        )
        get_answer = AsyncMock(side_effect=[first_response])
        paid_calls: list[str] = []

        async def paid_handler(_ctx: Any, _args: Any) -> ToolResult:
            paid_calls.append("paid")
            return ToolResult({"ok": True})

        registry = FakeRegistry(
            handlers={"generate_video": paid_handler},
            question_checks={
                "ask_user_question": AsyncMock(
                    return_value=ToolQuestionCheck(request=question_request())
                )
            },
        )

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_user_model", new=AsyncMock(return_value="gpt-5.6-terra")),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
        ):
            answer = asyncio.run(
                get_model_answer(
                    cast(Any, self._update()),
                    cast(Any, SimpleNamespace()),
                    [{"role": "user", "content": "сделай видео"}],
                    openrouter_service=cast(Any, SimpleNamespace()),
                    user_text="сделай видео",
                )
            )

        # Вопрос задан до платного вызова: платный инструмент не запускался.
        self.assertEqual(paid_calls, [])
        self.assertIsNotNone(answer.pending_question)
        continuation = answer.pending_continuation
        assert continuation is not None
        # Вопрос становится первым вызовом пакета: платный инструмент ждёт
        # ответа пользователя и выполняется только после него.
        self.assertEqual(continuation.current_call_index, 0)
        self.assertEqual(continuation.completed_outputs, ())
        self.assertEqual(
            [call.call_id for call in continuation.tool_calls],
            ["call-question", "call-paid"],
        )

    def test_pause_without_stored_responses_is_rejected_safely(self) -> None:
        first_response = response_with_output(
            "resp-not-stored",
            [function_call("call-question", "ask_user_question", '{"question":"?"}')],
        )
        get_answer = AsyncMock(side_effect=[first_response])
        registry = FakeRegistry(
            question_checks={
                "ask_user_question": AsyncMock(
                    return_value=ToolQuestionCheck(request=question_request())
                )
            },
        )

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_user_model", new=AsyncMock(return_value="gpt-5.6-terra")),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
            patch("core.openai_api.settings", settings_without_stored_responses()),
        ):
            answer = asyncio.run(
                get_model_answer(
                    cast(Any, self._update()),
                    cast(Any, SimpleNamespace()),
                    [{"role": "user", "content": "уточни"}],
                    openrouter_service=cast(Any, SimpleNamespace()),
                    user_text="уточни",
                )
            )

        # Повисшего ожидания не создаётся: пользователь получает безопасный отказ.
        self.assertIsNone(answer.pending_question)
        self.assertIsNone(answer.pending_continuation)
        self.assertIsNotNone(answer.bot_reply)
        self.assertEqual(get_answer.await_count, 1)

    def test_pause_preserves_state_for_continuation(self) -> None:
        first_response = response_with_output(
            "resp-state",
            [function_call("call-question", "ask_user_question", '{"question":"?"}')],
        )
        get_answer = AsyncMock(side_effect=[first_response])
        registry = FakeRegistry(
            question_checks={
                "ask_user_question": AsyncMock(
                    return_value=ToolQuestionCheck(request=question_request())
                )
            },
        )

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_user_model", new=AsyncMock(return_value="gpt-5.6-terra")),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
        ):
            answer = asyncio.run(
                get_model_answer(
                    cast(Any, self._update()),
                    cast(Any, SimpleNamespace()),
                    [{"role": "user", "content": "уточни"}],
                    openrouter_service=cast(Any, SimpleNamespace()),
                    user_text="Исходный текст задачи",
                )
            )

        continuation = answer.pending_continuation
        assert continuation is not None
        self.assertEqual(continuation.original_user_text, "Исходный текст задачи")
        self.assertEqual(continuation.model_name, "gpt-5.6-terra")
        self.assertIsInstance(continuation.image_tool_state, ImageToolTurnState)
        self.assertIsInstance(continuation.video_tool_state, VideoToolTurnState)
        self.assertEqual(continuation.tool_round, 0)


class ToolResumeTests(unittest.TestCase):
    """Продолжение цикла инструментов после ответа пользователя."""

    def _continuation(
        self,
        *,
        completed_outputs: tuple[dict[str, str], ...] = (),
        current_call_index: int = 0,
        answer_mode: Any = None,
        tool_calls: tuple[PendingToolCall, ...] | None = None,
    ) -> Any:
        from core.common_types import QuestionAnswerMode

        return SimpleNamespace(
            response_id="resp-cont",
            tool_calls=tool_calls
            or (
                PendingToolCall(call_id="call-current", name="ask_user_question", arguments={}),
            ),
            completed_outputs=completed_outputs,
            current_call_index=current_call_index,
            model_name="gpt-5.6-terra",
            original_user_text="Исходный текст задачи",
            context_tokens=120,
            completion_tokens=30,
            tool_round=2,
            image_tool_state=ImageToolTurnState(retry_forbidden=True),
            video_tool_state=VideoToolTurnState(),
            video_request=None,
            video_operation_id=None,
            answer_mode=answer_mode or QuestionAnswerMode.PLAIN,
            validator=None,
        )

    def test_resume_uses_previous_response_id_and_call_id(self) -> None:
        final_response = response_with_output("resp-final", [], "Готово")
        get_answer = AsyncMock(side_effect=[final_response])
        registry = FakeRegistry()
        answer = QuestionAnswer(
            question_id="q1",
            kind=QuestionAnswerKind.OPTION,
            text="Вариант a",
            option_id="a",
        )

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer) as mocked,
        ):
            result = asyncio.run(
                resume_model_answer(
                    cast(Any, SimpleNamespace(effective_user=SimpleNamespace(id=11))),
                    cast(Any, SimpleNamespace()),
                    self._continuation(),
                    answer,
                    openrouter_service=cast(Any, SimpleNamespace()),
                    question=question_request(),
                )
            )

        self.assertEqual(result.bot_reply, "Готово")
        self.assertEqual(mocked.await_count, 1)
        call_kwargs = mocked.await_args_list[0].kwargs
        self.assertEqual(call_kwargs["previous_response_id"], "resp-cont")
        outputs = mocked.await_args_list[0].args[0]
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["call_id"], "call-current")
        self.assertEqual(outputs[0]["type"], "function_call_output")

    def test_resume_sends_full_batch_of_outputs(self) -> None:
        final_response = response_with_output("resp-final", [], "Готово")
        get_answer = AsyncMock(side_effect=[final_response])
        registry = FakeRegistry()
        answer = QuestionAnswer(
            question_id="q1",
            kind=QuestionAnswerKind.CUSTOM,
            text="мой ответ",
        )
        continuation = self._continuation(
            completed_outputs=(
                {"type": "function_call_output", "call_id": "call-done", "output": '{"ok": true}'},
            ),
            current_call_index=1,
            tool_calls=(
                PendingToolCall(call_id="call-done", name="test_first", arguments={}),
                PendingToolCall(call_id="call-current", name="ask_user_question", arguments={}),
            ),
        )

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer) as mocked,
        ):
            result = asyncio.run(
                resume_model_answer(
                    cast(Any, SimpleNamespace(effective_user=SimpleNamespace(id=11))),
                    cast(Any, SimpleNamespace()),
                    continuation,
                    answer,
                    openrouter_service=cast(Any, SimpleNamespace()),
                    question=question_request(),
                )
            )

        self.assertEqual(result.bot_reply, "Готово")
        outputs = mocked.await_args_list[0].args[0]
        self.assertEqual(
            [item["call_id"] for item in outputs],
            ["call-done", "call-current"],
        )

    def test_resume_runs_remaining_calls_and_does_not_repeat_finished(self) -> None:
        final_response = response_with_output("resp-final", [], "Готово")
        get_answer = AsyncMock(side_effect=[final_response])
        executed: list[str] = []

        async def remaining_handler(_ctx: Any, _args: Any) -> ToolResult:
            executed.append("remaining")
            return ToolResult({"ok": True, "value": 1})

        registry = FakeRegistry(handlers={"test_rest": remaining_handler})
        answer = QuestionAnswer(
            question_id="q1",
            kind=QuestionAnswerKind.OPTION,
            text="Вариант a",
            option_id="a",
        )
        continuation = self._continuation(
            current_call_index=0,
            tool_calls=(
                PendingToolCall(call_id="call-current", name="ask_user_question", arguments={}),
                PendingToolCall(call_id="call-rest", name="test_rest", arguments={}),
            ),
        )

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer) as mocked,
        ):
            result = asyncio.run(
                resume_model_answer(
                    cast(Any, SimpleNamespace(effective_user=SimpleNamespace(id=11))),
                    cast(Any, SimpleNamespace()),
                    continuation,
                    answer,
                    openrouter_service=cast(Any, SimpleNamespace()),
                    question=question_request(),
                )
            )

        self.assertEqual(result.bot_reply, "Готово")
        self.assertEqual(executed, ["remaining"])
        outputs = mocked.await_args_list[0].args[0]
        self.assertEqual(
            [item["call_id"] for item in outputs],
            ["call-current", "call-rest"],
        )

    def test_resume_pauses_again_on_next_question(self) -> None:
        get_answer = AsyncMock()
        request = question_request(text="Второй вопрос")

        async def next_handler(_ctx: Any, _args: Any) -> ToolResult:
            return ToolResult(pause=ToolQuestionPause(request=request))

        registry = FakeRegistry(handlers={"test_next": next_handler})
        answer = QuestionAnswer(
            question_id="q1",
            kind=QuestionAnswerKind.OPTION,
            text="Вариант a",
            option_id="a",
        )
        continuation = self._continuation(
            current_call_index=0,
            tool_calls=(
                PendingToolCall(call_id="call-current", name="ask_user_question", arguments={}),
                PendingToolCall(call_id="call-next", name="test_next", arguments={}),
            ),
        )

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
        ):
            result = asyncio.run(
                resume_model_answer(
                    cast(Any, SimpleNamespace(effective_user=SimpleNamespace(id=11))),
                    cast(Any, SimpleNamespace()),
                    continuation,
                    answer,
                    openrouter_service=cast(Any, SimpleNamespace()),
                    question=question_request(),
                )
            )

        # Неполный пакет в API не отправляется: следующий вопрос ожидает ответа.
        self.assertEqual(get_answer.await_count, 0)
        self.assertIsNotNone(result.pending_question)
        next_continuation = result.pending_continuation
        assert next_continuation is not None
        self.assertEqual(next_continuation.current_call_index, 1)
        self.assertEqual(len(next_continuation.completed_outputs), 1)
        self.assertEqual(
            next_continuation.completed_outputs[0]["call_id"],
            "call-current",
        )

    def test_resume_without_stored_responses_is_rejected(self) -> None:
        get_answer = AsyncMock()
        answer = QuestionAnswer(
            question_id="q1",
            kind=QuestionAnswerKind.OPTION,
            text="Вариант a",
            option_id="a",
        )
        continuation = self._continuation()
        continuation.response_id = "resp-cont"

        with (
            patch("core.openai_api.get_tool_registry", return_value=FakeRegistry()),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
            patch("core.openai_api.settings", settings_without_stored_responses()),
        ):
            result = asyncio.run(
                resume_model_answer(
                    cast(Any, SimpleNamespace(effective_user=SimpleNamespace(id=11))),
                    cast(Any, SimpleNamespace()),
                    continuation,
                    answer,
                    openrouter_service=cast(Any, SimpleNamespace()),
                    question=question_request(),
                )
            )

        self.assertEqual(get_answer.await_count, 0)
        self.assertIsNone(result.pending_question)
        self.assertIsNotNone(result.bot_reply)

    def test_resume_preserves_original_text_and_protection_state(self) -> None:
        final_response = response_with_output("resp-final", [], "Готово")
        get_answer = AsyncMock(side_effect=[final_response])
        captured: dict[str, Any] = {}

        async def remaining_handler(ctx: Any, _args: Any) -> ToolResult:
            captured["user_text"] = ctx.user_text
            captured["context_tokens"] = ctx.context_tokens
            captured["completion_tokens"] = ctx.completion_tokens
            captured["recursion_depth"] = ctx.recursion_depth
            captured["image_state"] = ctx.image_tool_state
            captured["video_state"] = ctx.video_tool_state
            return ToolResult({"ok": True})

        registry = FakeRegistry(handlers={"test_rest": remaining_handler})
        answer = QuestionAnswer(
            question_id="q1",
            kind=QuestionAnswerKind.CUSTOM,
            text="10",
        )
        continuation = self._continuation(
            current_call_index=0,
            tool_calls=(
                PendingToolCall(call_id="call-current", name="ask_user_question", arguments={}),
                PendingToolCall(call_id="call-rest", name="test_rest", arguments={}),
            ),
        )

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
        ):
            result = asyncio.run(
                resume_model_answer(
                    cast(Any, SimpleNamespace(effective_user=SimpleNamespace(id=11))),
                    cast(Any, SimpleNamespace()),
                    continuation,
                    answer,
                    openrouter_service=cast(Any, SimpleNamespace()),
                    question=question_request(),
                )
            )

        self.assertEqual(result.bot_reply, "Готово")
        # Ответ «10» не подменяет исходную задачу и не обнуляет защиту.
        self.assertEqual(captured["user_text"], "Исходный текст задачи")
        self.assertEqual(captured["context_tokens"], 120)
        self.assertEqual(captured["completion_tokens"], 30)
        self.assertEqual(captured["recursion_depth"], 2)
        self.assertEqual(result.ctx_token, 120)
        self.assertEqual(result.completion_token, 30)
        self.assertTrue(captured["image_state"].retry_forbidden)
        self.assertIs(captured["image_state"], continuation.image_tool_state)
        self.assertIs(captured["video_state"], continuation.video_tool_state)


class VideoDurationQuestionTests(unittest.TestCase):
    """Обязательный вопрос о длительности видео перед платным запуском."""

    def _update(self) -> Any:
        return SimpleNamespace(effective_user=SimpleNamespace(id=42))

    def test_video_without_duration_pauses_before_paid_call(self) -> None:
        from core.tool_registry import get_tool_registry

        first_response = response_with_output(
            "resp-video",
            [function_call("call-video", "generate_video", '{"prompt":"animate"}')],
        )
        get_answer = AsyncMock(side_effect=[first_response])
        registry = get_tool_registry()
        paid_calls: list[str] = []

        async def paid_handler(_ctx: Any, _args: Any) -> ToolResult:
            paid_calls.append("paid")
            return ToolResult({"ok": True})

        registry._handlers["generate_video"] = paid_handler

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_user_model", new=AsyncMock(return_value="gpt-5.6-terra")),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
            patch(
                "core.tool_handlers.media.get_user_image_edit_session",
                new=AsyncMock(return_value=None),
            ),
        ):
            answer = asyncio.run(
                get_model_answer(
                    cast(Any, self._update()),
                    cast(Any, SimpleNamespace()),
                    [{"role": "user", "content": "оживи картинку"}],
                    openrouter_service=cast(Any, SimpleNamespace()),
                    user_text="оживи картинку",
                )
            )

        self.assertEqual(paid_calls, [])
        self.assertIsNotNone(answer.pending_question)
        request = answer.pending_question
        assert request is not None
        # Варианты длительности берутся из активного видеопрофиля.
        self.assertEqual(len(request.options), 5)
        self.assertEqual(request.options[0].payload, 5)
        self.assertEqual(request.options[-1].payload, 30)

    def test_video_with_model_invented_duration_still_pauses(self) -> None:
        """Значение duration, придуманное моделью, согласием пользователя не является."""

        from core.tool_registry import get_tool_registry

        first_response = response_with_output(
            "resp-video-invented",
            [
                function_call(
                    "call-video-invented",
                    "generate_video",
                    '{"prompt":"animate","duration":15}',
                )
            ],
        )
        get_answer = AsyncMock(side_effect=[first_response])
        registry = get_tool_registry()
        paid_calls: list[str] = []

        async def paid_handler(_ctx: Any, _args: Any) -> ToolResult:
            paid_calls.append("paid")
            return ToolResult({"ok": True})

        registry._handlers["generate_video"] = paid_handler

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_user_model", new=AsyncMock(return_value="gpt-5.6-terra")),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
            patch(
                "core.tool_handlers.media.get_user_image_edit_session",
                new=AsyncMock(return_value=None),
            ),
        ):
            answer = asyncio.run(
                get_model_answer(
                    cast(Any, self._update()),
                    cast(Any, SimpleNamespace()),
                    [{"role": "user", "content": "оживи картинку"}],
                    openrouter_service=cast(Any, SimpleNamespace()),
                    user_text="оживи картинку",
                )
            )

        self.assertEqual(paid_calls, [])
        self.assertIsNotNone(answer.pending_question)
        request = answer.pending_question
        assert request is not None
        self.assertEqual(len(request.options), 5)

    def test_video_with_explicit_duration_is_not_interrupted(self) -> None:
        from core.tool_registry import get_tool_registry

        first_response = response_with_output(
            "resp-video-explicit",
            [function_call("call-video", "generate_video", '{"prompt":"animate"}')],
        )
        final_response = response_with_output("resp-final", [], "Готово")
        get_answer = AsyncMock(side_effect=[first_response, final_response])
        registry = get_tool_registry()

        async def paid_handler(_ctx: Any, _args: Any) -> ToolResult:
            return ToolResult({"ok": True})

        registry._handlers["generate_video"] = paid_handler

        with (
            patch("core.openai_api.get_tool_registry", return_value=registry),
            patch("core.openai_api.get_user_model", new=AsyncMock(return_value="gpt-5.6-terra")),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
            patch(
                "core.tool_handlers.media.get_user_image_edit_session",
                new=AsyncMock(return_value=None),
            ),
        ):
            answer = asyncio.run(
                get_model_answer(
                    cast(Any, self._update()),
                    cast(Any, SimpleNamespace()),
                    [{"role": "user", "content": "оживи картинку"}],
                    openrouter_service=cast(Any, SimpleNamespace()),
                    user_text="оживи картинку на 12 секунд",
                )
            )

        self.assertIsNone(answer.pending_question)
        self.assertEqual(answer.bot_reply, "Готово")


class VideoResumeTests(unittest.TestCase):
    """Продолжение видеозаявки после ответа на вопрос длительности."""

    def _continuation(self) -> Any:
        from core.common_types import QuestionAnswerMode
        from core.video_requests import build_video_request_draft

        profile = settings.video_profile
        draft = build_video_request_draft("animate", profile, duration=None)
        return QuestionContinuationContext(
            response_id="resp-video-cont",
            tool_calls=(
                PendingToolCall(call_id="call-video", name="generate_video", arguments={}),
            ),
            current_call_index=0,
            model_name="gpt-5.6-terra",
            original_user_text="оживи картинку",
            video_request=draft,
            answer_mode=QuestionAnswerMode.VIDEO_REQUEST,
        )

    def test_video_continuation_uses_checked_duration(self) -> None:
        answer = QuestionAnswer(
            question_id="q1",
            kind=QuestionAnswerKind.OPTION,
            text="10 с",
            option_id="duration_10",
            payload=10,
        )
        final_response = response_with_output("resp-final", [], "Видео готово")
        get_answer = AsyncMock(side_effect=[final_response])
        generated: list[Any] = []

        async def generate_video(
            _prompt: str,
            options: Any,
            reference_images: Any = (),
        ) -> Any:
            generated.append(options)
            return SimpleNamespace(cleanup=lambda: None)

        service = SimpleNamespace(generate_video=generate_video)

        with (
            patch("core.openai_api.get_tool_registry", return_value=FakeRegistry()),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
            patch(
                "core.tool_handlers.media.reply_service_message",
                new=AsyncMock(return_value=SimpleNamespace(edit_text=AsyncMock())),
            ),
            patch("core.tool_handlers.media.animate_service_message", new=AsyncMock()),
            patch(
                "core.tool_handlers.media.send_video_to_telegram",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "core.tool_handlers.media.add_user_image_session_history",
                new=AsyncMock(),
            ),
        ):
            result = asyncio.run(
                resume_model_answer(
                    cast(Any, SimpleNamespace(effective_user=SimpleNamespace(id=42))),
                    cast(Any, SimpleNamespace()),
                    self._continuation(),
                    answer,
                    openrouter_service=cast(Any, service),
                )
            )

        self.assertEqual(result.bot_reply, "Видео готово")
        # Генерация выполнена ровно один раз с проверенной длительностью.
        self.assertEqual(len(generated), 1)
        profile = settings.video_profile
        self.assertEqual(generated[0].duration, 10)
        self.assertEqual(generated[0].model, profile.model)
        self.assertEqual(generated[0].resolution, profile.resolution)

    def test_video_continuation_keeps_prompt_references_and_profile(self) -> None:
        """После ответа меняется только длительность: заявка в остальном прежняя."""

        from core.video_requests import build_video_request_draft

        profile = settings.video_profile
        images = [
            GeneratedImage(content=b"third", mime_type="image/png"),
            GeneratedImage(content=b"first", mime_type="image/png"),
        ]
        draft = build_video_request_draft(
            "animate",
            profile,
            reference_images=images,
            reference_indices=(3, 1),
            duration=None,
        )
        continuation = QuestionContinuationContext(
            response_id="resp-video-cont",
            tool_calls=(
                PendingToolCall(call_id="call-video", name="generate_video", arguments={}),
            ),
            current_call_index=0,
            model_name="gpt-5.6-terra",
            original_user_text="оживи картинку",
            video_request=draft,
            answer_mode=QuestionAnswerMode.VIDEO_REQUEST,
        )
        answer = QuestionAnswer(
            question_id="q1",
            kind=QuestionAnswerKind.OPTION,
            text="10 с",
            option_id="duration_10",
            payload=10,
        )
        final_response = response_with_output("resp-final", [], "Видео готово")
        get_answer = AsyncMock(side_effect=[final_response])
        generated: list[Any] = []
        references: list[Any] = []

        async def generate_video(
            _prompt: str,
            options: Any,
            reference_images: Any = (),
        ) -> Any:
            generated.append(options)
            references.append(list(reference_images))
            return SimpleNamespace(cleanup=lambda: None)

        service = SimpleNamespace(generate_video=generate_video)

        with (
            patch("core.openai_api.get_tool_registry", return_value=FakeRegistry()),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer),
            patch(
                "core.tool_handlers.media.reply_service_message",
                new=AsyncMock(return_value=SimpleNamespace(edit_text=AsyncMock())),
            ),
            patch("core.tool_handlers.media.animate_service_message", new=AsyncMock()),
            patch(
                "core.tool_handlers.media.send_video_to_telegram",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "core.tool_handlers.media.add_user_image_session_history",
                new=AsyncMock(),
            ),
        ):
            result = asyncio.run(
                resume_model_answer(
                    cast(Any, SimpleNamespace(effective_user=SimpleNamespace(id=42))),
                    cast(Any, SimpleNamespace()),
                    continuation,
                    answer,
                    openrouter_service=cast(Any, service),
                )
            )

        self.assertEqual(result.bot_reply, "Видео готово")
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0].duration, 10)
        self.assertEqual(generated[0].model, profile.model)
        self.assertEqual(generated[0].resolution, profile.resolution)
        # Порядок и содержимое референсов не меняются.
        self.assertEqual(
            [image.content for image in references[0]],
            [b"third", b"first"],
        )

    def test_video_continuation_rejects_out_of_range_duration(self) -> None:
        answer = QuestionAnswer(
            question_id="q1",
            kind=QuestionAnswerKind.CUSTOM,
            text="45",
        )
        final_response = response_with_output("resp-final", [], "Готово")
        get_answer = AsyncMock(side_effect=[final_response])
        generated: list[Any] = []

        async def generate_video(
            _prompt: str,
            options: Any,
            reference_images: Any = (),
        ) -> Any:
            generated.append(options)
            return SimpleNamespace(cleanup=lambda: None)

        service = SimpleNamespace(generate_video=generate_video)

        with (
            patch("core.openai_api.get_tool_registry", return_value=FakeRegistry()),
            patch("core.openai_api.get_simple_answer", side_effect=get_answer) as mocked,
        ):
            result = asyncio.run(
                resume_model_answer(
                    cast(Any, SimpleNamespace(effective_user=SimpleNamespace(id=42))),
                    cast(Any, SimpleNamespace()),
                    self._continuation(),
                    answer,
                    openrouter_service=cast(Any, service),
                )
            )

        # Недопустимая длительность не запускает платную генерацию.
        self.assertEqual(generated, [])
        self.assertEqual(result.bot_reply, "Готово")
        outputs = mocked.await_args_list[0].args[0]
        self.assertEqual(outputs[0]["call_id"], "call-video")
        self.assertIn("не поддерживается", outputs[0]["output"])


class VideoOperationProtectionTests(unittest.TestCase):
    """Защита видео от повторного платного запуска того же вызова."""

    def test_completed_operation_blocks_repeated_call(self) -> None:
        state = VideoToolTurnState()
        state.mark_completed("op-1")

        self.assertTrue(state.is_known_operation("op-1"))
        self.assertFalse(state.is_known_operation("op-2"))
        self.assertFalse(state.retry_forbidden)

    def test_failed_operation_forbids_paid_retry(self) -> None:
        state = VideoToolTurnState()
        state.mark_failed("op-1")

        self.assertTrue(state.is_known_operation("op-1"))
        self.assertTrue(state.retry_forbidden)

    def test_image_protection_rules_are_unchanged(self) -> None:
        """Защита изображений сохраняет прежние правила."""

        image_state = ImageToolTurnState()

        self.assertFalse(image_state.retry_forbidden)
        self.assertFalse(image_state.image_call_attempted)
        self.assertEqual(image_state.attempted_image_call_ids, set())


if __name__ == "__main__":
    unittest.main()
