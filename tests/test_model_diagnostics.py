"""Диагностика ошибок OpenAI API и валидация смены модели."""

import asyncio
import os
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:TEST_TOKEN")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import httpx
from openai import APIConnectionError, NotFoundError, OpenAI

from core.openai_api import (
    _describe_api_error,
    get_model_answer,
    verify_default_model_available,
)
from core.tool_handlers.model import handle_change_model
from core.tool_registry import ToolExecutionContext


def _http_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return httpx.Response(status_code=status_code, request=request)


def _not_found_error(message: str = "Model not found", code: str | None = None) -> NotFoundError:
    body = {"error": {"message": message, "code": code}}
    error = NotFoundError(message, response=_http_response(404), body=body)
    return error


class DescribeApiErrorTests(unittest.TestCase):
    def test_describe_contains_type_status_code_and_model(self) -> None:
        error = _not_found_error("Model 'gpt-6-sol' not found", code="model_not_found")

        description = _describe_api_error(error, "gpt-6-sol")

        self.assertIn("type=NotFoundError", description)
        self.assertIn("model=gpt-6-sol", description)
        self.assertIn("status_code=404", description)
        self.assertIn("code=model_not_found", description)
        self.assertIn("Model 'gpt-6-sol' not found", description)

    def test_describe_handles_plain_exception_without_model(self) -> None:
        description = _describe_api_error(ValueError("что-то сломалось"), None)

        self.assertIn("type=ValueError", description)
        self.assertIn("model=unknown", description)
        self.assertIn("что-то сломалось", description)
        self.assertNotIn("status_code=", description)

    def test_describe_truncates_long_message(self) -> None:
        long_message = "x" * 5000

        description = _describe_api_error(ValueError(long_message), "m")

        self.assertLess(len(description), 3000)
        self.assertIn("<обрезано>", description)


class GetModelAnswerErrorLoggingTests(unittest.TestCase):
    def test_api_error_is_logged_with_details_and_returns_fallback(self) -> None:
        update: Any = SimpleNamespace(effective_user=SimpleNamespace(id=1))
        context: Any = SimpleNamespace()

        with (
            patch("core.openai_api.get_user_model", new=AsyncMock(return_value="gpt-6-sol")),
            patch(
                "core.openai_api.get_simple_answer",
                new=AsyncMock(side_effect=_not_found_error("Model not found")),
            ) as get_answer,
        ):
            answer = asyncio.run(
                get_model_answer(
                    update,
                    context,
                    [{"role": "user", "content": "Привет"}],
                    openrouter_service=SimpleNamespace(),
                )
            )

        self.assertEqual(get_answer.await_count, 1)
        self.assertEqual(answer.bot_reply, "Произошла ошибка при обработке запроса.")


class VerifyDefaultModelTests(unittest.TestCase):
    def test_available_model_returns_true(self) -> None:
        with patch("core.openai_api.openai_client.models.retrieve") as retrieve:
            asyncio.run(verify_default_model_available())

        retrieve.assert_called_once_with("gpt-6-sol")

    def test_not_found_model_returns_false(self) -> None:
        with patch(
            "core.openai_api.openai_client.models.retrieve",
            side_effect=_not_found_error("Model not found"),
        ):
            result = asyncio.run(verify_default_model_available())

        self.assertFalse(result)

    def test_temporary_error_does_not_block_startup(self) -> None:
        connection_error = APIConnectionError(request=_http_response(500).request)
        with patch(
            "core.openai_api.openai_client.models.retrieve",
            side_effect=connection_error,
        ):
            result = asyncio.run(verify_default_model_available())

        self.assertTrue(result)


def _tool_ctx(model_name: str = "gpt-6-sol") -> ToolExecutionContext:
    update: Any = SimpleNamespace(
        effective_user=SimpleNamespace(id=1),
        message=SimpleNamespace(reply_text=AsyncMock()),
    )
    return ToolExecutionContext(
        update=update,
        context=SimpleNamespace(),
        model_name=model_name,
        recursion_depth=0,
        openrouter_service=SimpleNamespace(),
    )


class ChangeModelValidationTests(unittest.TestCase):
    def test_unknown_model_is_rejected(self) -> None:
        ctx = _tool_ctx()
        result = asyncio.run(handle_change_model(ctx, {"model": "unknown-model"}))

        self.assertEqual(result.output["ok"], False)
        self.assertIn("недоступна", result.output["error"])

    def test_known_model_is_applied(self) -> None:
        ctx = _tool_ctx(model_name="gpt-5.6-terra")
        with (
            patch("core.tool_handlers.model.set_user_model", new=AsyncMock()) as set_model,
            patch("core.tool_handlers.model.reply_service_text", new=AsyncMock()) as reply,
        ):
            result = asyncio.run(
                handle_change_model(ctx, {"model": "gpt-6-sol"})
            )

        self.assertEqual(result.output["ok"], True)
        set_model.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
