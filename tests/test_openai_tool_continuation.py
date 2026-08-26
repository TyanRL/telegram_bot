"""Проверки continuation Responses API после function calls."""

import asyncio
import os
import unittest
from types import SimpleNamespace
from typing import Any
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


if __name__ == "__main__":
    unittest.main()
