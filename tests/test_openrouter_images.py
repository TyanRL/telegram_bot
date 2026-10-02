"""Тесты payload OpenRouter Images API для FLUX.3."""

import asyncio
import base64
import os
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:TEST_TOKEN")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from core.common_types import (
    GeneratedImage,
    ImageEditOptions,
    ImageGenerationOptions,
)
from utils import openrouter_images
from utils.openrouter_images import OpenRouter


FLUX_MODEL = "black-forest-labs/flux-3-image"

UNSUPPORTED_FIELDS = {
    "output_format",
    "output_compression",
    "quality",
    "background",
    "size",
    "stream",
}


def _png_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"source-image-bytes"


def _source_image() -> GeneratedImage:
    return GeneratedImage(content=_png_bytes(), mime_type="image/png")


def _client() -> Any:
    images = SimpleNamespace(generate_async=AsyncMock())
    return SimpleNamespace(images=images)


def _response(content: bytes, *, mime_type: str = "image/png") -> SimpleNamespace:
    encoded = base64.b64encode(content).decode("ascii")
    image = SimpleNamespace(b64_json=encoded, media_type=mime_type)
    return SimpleNamespace(
        data=[image],
        created=1748372400,
        usage=SimpleNamespace(cost=0.04),
    )


class SdkImageOptionsTests(unittest.TestCase):
    def test_only_flux_supported_fields_are_passed(self) -> None:
        options = ImageGenerationOptions(aspect_ratio="16:9", resolution="2K", seed=42)

        payload = openrouter_images._sdk_image_options(options)

        self.assertEqual(
            payload,
            {"aspect_ratio": "16:9", "resolution": "2K", "seed": 42},
        )

    def test_unsupported_fields_are_dropped(self) -> None:
        payload = openrouter_images._sdk_image_options(ImageEditOptions())

        self.assertEqual(payload, {"aspect_ratio": None, "resolution": None, "seed": None})
        self.assertFalse(UNSUPPORTED_FIELDS & set(payload))


class GeneratePayloadTests(unittest.TestCase):
    def test_generate_payload_matches_flux_contract(self) -> None:
        client = _client()
        client.images.generate_async.return_value = _response(b"generated-image")

        result = asyncio.run(
            openrouter_images.generate(
                "a cozy storefront",
                ImageGenerationOptions(aspect_ratio="16:9", resolution="2K", seed=7),
                client=cast(OpenRouter, client),
            )
        )

        self.assertEqual(client.images.generate_async.await_count, 1)
        kwargs: dict[str, Any] = client.images.generate_async.await_args.kwargs
        self.assertEqual(kwargs.get("model"), FLUX_MODEL)
        self.assertEqual(kwargs.get("prompt"), "a cozy storefront")
        self.assertEqual(kwargs.get("n"), 1)
        self.assertEqual(kwargs.get("aspect_ratio"), "16:9")
        self.assertEqual(kwargs.get("resolution"), "2K")
        self.assertEqual(kwargs.get("seed"), 7)
        self.assertFalse(UNSUPPORTED_FIELDS & set(kwargs))
        self.assertNotIn("input_references", kwargs)

        self.assertEqual(result.content, b"generated-image")
        self.assertEqual(result.mime_type, "image/png")
        self.assertEqual(result.metadata["model"], FLUX_MODEL)
        self.assertEqual(result.metadata["operation"], "generate")
        self.assertEqual(result.metadata["cost"], 0.04)

    def test_generate_rejects_response_without_data(self) -> None:
        client = _client()
        client.images.generate_async.return_value = SimpleNamespace(data=[])

        with self.assertRaises(ValueError):
            asyncio.run(
                openrouter_images.generate(
                    "prompt", None, client=cast(OpenRouter, client)
                )
            )


class EditPayloadTests(unittest.TestCase):
    def test_edit_payload_uses_data_url_reference(self) -> None:
        client = _client()
        client.images.generate_async.return_value = _response(b"edited-image")

        result = asyncio.run(
            openrouter_images.edit(
                _source_image(),
                "make it a cozy storefront",
                ImageEditOptions(aspect_ratio="1:1", resolution="1K"),
                client=cast(OpenRouter, client),
            )
        )

        self.assertEqual(client.images.generate_async.await_count, 1)
        kwargs: dict[str, Any] = client.images.generate_async.await_args.kwargs
        self.assertEqual(kwargs.get("model"), FLUX_MODEL)
        self.assertEqual(kwargs.get("prompt"), "make it a cozy storefront")
        self.assertEqual(kwargs.get("n"), 1)
        self.assertEqual(kwargs.get("aspect_ratio"), "1:1")
        self.assertEqual(kwargs.get("resolution"), "1K")
        self.assertFalse(UNSUPPORTED_FIELDS & set(kwargs))

        references: Any = kwargs.get("input_references")
        self.assertIsInstance(references, list)
        self.assertEqual(len(references), 1)
        entry: Any = references[0]
        self.assertEqual(entry["type"], "image_url")
        expected_data_url = (
            "data:image/png;base64,"
            + base64.b64encode(_png_bytes()).decode("ascii")
        )
        self.assertEqual(entry["image_url"]["url"], expected_data_url)

        self.assertEqual(result.content, b"edited-image")
        self.assertEqual(result.metadata["model"], FLUX_MODEL)
        self.assertEqual(result.metadata["operation"], "edit")

    def test_edit_without_options_uses_defaults(self) -> None:
        client = _client()
        client.images.generate_async.return_value = _response(b"edited-image")

        asyncio.run(
            openrouter_images.edit(
                _source_image(),
                "recolor",
                None,
                client=cast(OpenRouter, client),
            )
        )

        kwargs: dict[str, Any] = client.images.generate_async.await_args.kwargs
        self.assertEqual(kwargs.get("model"), FLUX_MODEL)
        self.assertEqual(kwargs.get("n"), 1)
        self.assertIsNone(kwargs.get("aspect_ratio"))
        self.assertIsNone(kwargs.get("resolution"))
        self.assertFalse(UNSUPPORTED_FIELDS & set(kwargs))


if __name__ == "__main__":
    unittest.main()
