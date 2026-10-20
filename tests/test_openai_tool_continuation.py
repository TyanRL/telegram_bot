"""Проверки continuation Responses API после function calls."""

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

from core.common_types import ToolResult
from core.openai_api import get_model_answer


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


if __name__ == "__main__":
    unittest.main()
