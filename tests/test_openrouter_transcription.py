"""Проверки STT через настоящий SDK и подменённый HTTP-транспорт."""

import base64
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from openrouter import OpenRouter

from utils.openrouter_client import (
    OpenRouterConfig,
    OpenRouterService,
    OpenRouterAuthenticationError,
    OpenRouterRateLimitError,
    OpenRouterResponseError,
    OpenRouterTimeoutError,
)


class TranscriptionTests(unittest.IsolatedAsyncioTestCase):
    async def transcribe(self, handler, content=b"OggS-test-audio"):
        transport = httpx.MockTransport(handler)
        def factory(**kwargs):
            return OpenRouter(
                **kwargs,
                retry_config=None,
                client=httpx.Client(transport=transport, trust_env=False),
                async_client=httpx.AsyncClient(transport=transport, trust_env=False),
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.ogg"
            path.write_bytes(content)
            async with OpenRouterService(
                OpenRouterConfig(api_key="test-openrouter-key"),
                client_factory=factory,
            ) as service:
                return await service.transcribe_audio(str(path))

    async def test_sdk_sends_audio_model_and_openrouter_key(self):
        def handler(request):
            self.assertEqual(str(request.url), "https://openrouter.ai/api/v1/audio/transcriptions")
            self.assertEqual(request.headers["authorization"], "Bearer test-openrouter-key")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "qwen/qwen3-asr-1.7b")
            self.assertEqual(payload["input_audio"]["format"], "ogg")
            self.assertEqual(base64.b64decode(payload["input_audio"]["data"]), b"OggS-test-audio")
            self.assertNotIn("language", payload)
            return httpx.Response(200, json={"text": " Привет, мир! \n"})
        self.assertEqual(await self.transcribe(handler), "Привет, мир!")

    async def test_silence_returns_empty_string(self):
        self.assertEqual(await self.transcribe(lambda _: httpx.Response(200, json={"text": "  "})), "")

    async def test_empty_audio_does_not_call_api(self):
        def handler(_):
            self.fail("Empty audio must not be sent")
        with self.assertRaises(OpenRouterResponseError):
            await self.transcribe(handler, content=b"")

    async def test_auth_and_rate_limit_are_not_transcripts(self):
        for status, error in [(401, OpenRouterAuthenticationError), (429, OpenRouterRateLimitError)]:
            with self.subTest(status=status), self.assertRaises(error):
                await self.transcribe(lambda _: httpx.Response(status, json={"error": {"message": "Denied", "code": status}}))

    async def test_timeout_is_translated(self):
        def handler(request):
            raise httpx.ReadTimeout("Timed out", request=request)
        with self.assertRaises(OpenRouterTimeoutError):
            await self.transcribe(handler)
