"""Тесты payload OpenRouter Images API для FLUX.3."""

import asyncio
import base64
from io import BytesIO
import os
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from PIL import Image

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:TEST_TOKEN")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from core.common_types import (
    GeneratedImage,
    ImageEditOptions,
    ImageGenerationOptions,
)
from core.config import settings
from utils import openrouter_images
from utils.openrouter_client import OpenRouterService
from utils.openrouter_images import OpenRouter


FLUX_MODEL = "black-forest-labs/flux-3-image"
KLEIN_MODEL = "black-forest-labs/flux-3-image-klein"

UNSUPPORTED_FIELDS = {
    "output_format",
    "output_compression",
    "quality",
    "background",
    "size",
    "stream",
}


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (1, 1), color=(10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


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
        options = ImageGenerationOptions(aspect_ratio="16:9", resolution="2K")

        payload = openrouter_images._sdk_image_options(options)

        self.assertEqual(payload, {"aspect_ratio": "16:9", "resolution": "2K"})
        self.assertFalse(UNSUPPORTED_FIELDS & set(payload))

    def test_unset_optional_fields_are_omitted(self) -> None:
        payload = openrouter_images._sdk_image_options(ImageEditOptions())

        self.assertEqual(payload, {})
        self.assertFalse(UNSUPPORTED_FIELDS & set(payload))

    def test_seed_is_rejected_as_unsupported_by_images_api(self) -> None:
        with self.assertRaisesRegex(ValueError, "seed не поддерживается"):
            openrouter_images._sdk_image_options(ImageGenerationOptions(seed=42))


class GeneratePayloadTests(unittest.TestCase):
    def test_generate_payload_matches_flux_contract(self) -> None:
        client = _client()
        client.images.generate_async.return_value = _response(_png_bytes())

        result = asyncio.run(
            openrouter_images.generate(
                "a cozy storefront",
                ImageGenerationOptions(aspect_ratio="16:9", resolution="2K"),
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
        self.assertNotIn("seed", kwargs)
        self.assertFalse(UNSUPPORTED_FIELDS & set(kwargs))
        self.assertNotIn("input_references", kwargs)

        self.assertEqual(result.content, _png_bytes())
        self.assertEqual(result.mime_type, "image/png")
        self.assertEqual(result.metadata["model"], FLUX_MODEL)
        self.assertEqual(result.metadata["requested_model"], FLUX_MODEL)
        self.assertEqual(result.metadata["operation"], "generate")
        self.assertEqual(result.metadata["cost"], 0.04)

    def test_generate_accepts_model_override_and_blank_uses_pro_default(self) -> None:
        client = _client()
        client.images.generate_async.return_value = _response(_png_bytes())

        result = asyncio.run(
            openrouter_images.generate(
                "prompt",
                ImageGenerationOptions(model=KLEIN_MODEL),
                client=cast(OpenRouter, client),
            )
        )

        self.assertEqual(client.images.generate_async.await_args.kwargs["model"], KLEIN_MODEL)
        self.assertEqual(result.metadata["requested_model"], KLEIN_MODEL)

        client.images.generate_async.reset_mock()
        client.images.generate_async.return_value = _response(_png_bytes())
        result = asyncio.run(
            openrouter_images.generate(
                "prompt",
                ImageGenerationOptions(model="   "),
                client=cast(OpenRouter, client),
            )
        )

        self.assertEqual(
            client.images.generate_async.await_args.kwargs["model"],
            settings.media.image_generation_model,
        )
        self.assertEqual(client.images.generate_async.await_args.kwargs["resolution"], "1K")
        self.assertEqual(result.metadata["requested_model"], settings.media.image_generation_model)

    def test_generate_rejects_seed_and_multiple_results_before_network(self) -> None:
        client = _client()

        for options, expected_message in (
            (ImageGenerationOptions(seed=11), "seed не поддерживается"),
            (ImageGenerationOptions(n=2), "n == 1"),
        ):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, expected_message):
                    asyncio.run(
                        openrouter_images.generate(
                            "prompt", options, client=cast(OpenRouter, client)
                        )
                    )

        client.images.generate_async.assert_not_awaited()

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
        client.images.generate_async.return_value = _response(_png_bytes())

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
        self.assertNotIn("seed", kwargs)
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

        self.assertEqual(result.content, _png_bytes())
        self.assertEqual(result.metadata["model"], FLUX_MODEL)
        self.assertEqual(result.metadata["requested_model"], FLUX_MODEL)
        self.assertEqual(result.metadata["operation"], "edit")

    def test_edit_without_options_uses_defaults(self) -> None:
        client = _client()
        client.images.generate_async.return_value = _response(_png_bytes())

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
        self.assertNotIn("aspect_ratio", kwargs)
        self.assertEqual(kwargs.get("resolution"), "1K")
        self.assertNotIn("seed", kwargs)
        self.assertFalse(UNSUPPORTED_FIELDS & set(kwargs))

    def test_edit_accepts_model_override_and_blank_uses_pro_default(self) -> None:
        client = _client()
        client.images.generate_async.return_value = _response(_png_bytes())

        result = asyncio.run(
            openrouter_images.edit(
                _source_image(),
                "recolor",
                ImageEditOptions(model=KLEIN_MODEL),
                client=cast(OpenRouter, client),
            )
        )

        self.assertEqual(client.images.generate_async.await_args.kwargs["model"], KLEIN_MODEL)
        self.assertEqual(result.metadata["requested_model"], KLEIN_MODEL)

        client.images.generate_async.reset_mock()
        client.images.generate_async.return_value = _response(_png_bytes())
        result = asyncio.run(
            openrouter_images.edit(
                _source_image(),
                "recolor",
                ImageEditOptions(model=""),
                client=cast(OpenRouter, client),
            )
        )

        self.assertEqual(
            client.images.generate_async.await_args.kwargs["model"],
            settings.media.image_edit_model,
        )
        self.assertEqual(result.metadata["requested_model"], settings.media.image_edit_model)

    def test_edit_rejects_seed_and_multiple_results_before_network(self) -> None:
        client = _client()

        for options, expected_message in (
            (ImageEditOptions(seed=11), "seed не поддерживается"),
            (ImageEditOptions(n=2), "n == 1"),
        ):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, expected_message):
                    asyncio.run(
                        openrouter_images.edit(
                            _source_image(),
                            "recolor",
                            options,
                            client=cast(OpenRouter, client),
                        )
                    )

        client.images.generate_async.assert_not_awaited()

    def test_edit_payload_supports_multiple_reference_images(self) -> None:
        client = _client()
        client.images.generate_async.return_value = _response(_png_bytes())
        second_image = GeneratedImage(
            content=b"second-source-image",
            mime_type="image/jpeg",
        )

        asyncio.run(
            openrouter_images.edit(
                [_source_image(), second_image],
                "combine the two references",
                client=cast(OpenRouter, client),
            )
        )

        references: Any = client.images.generate_async.await_args.kwargs["input_references"]
        self.assertEqual(len(references), 2)
        expected_urls = [
            "data:image/png;base64," + base64.b64encode(_png_bytes()).decode("ascii"),
            "data:image/jpeg;base64," + base64.b64encode(b"second-source-image").decode("ascii"),
        ]
        self.assertEqual(
            [entry["image_url"]["url"] for entry in references],
            expected_urls,
        )

    def test_edit_rejects_more_than_ten_reference_images(self) -> None:
        client = _client()
        with self.assertRaisesRegex(ValueError, "не более 10"):
            asyncio.run(
                openrouter_images.edit(
                    [_source_image() for _ in range(11)],
                    "combine all references",
                    client=cast(OpenRouter, client),
                )
            )

        client.images.generate_async.assert_not_awaited()


class ResponseValidationTests(unittest.TestCase):
    def test_corrupt_image_keeps_safe_attempt_and_cost_metadata(self) -> None:
        client = _client()
        client.images.generate_async.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(
                    b64_json=base64.b64encode(b"not-a-real-image").decode("ascii"),
                    media_type="image/png",
                )
            ],
            created=1748372400,
            usage=SimpleNamespace(cost=0.123),
        )

        with self.assertRaises(openrouter_images.InvalidImageResponseError) as caught:
            asyncio.run(
                openrouter_images.generate(
                    "prompt",
                    ImageGenerationOptions(model=KLEIN_MODEL),
                    client=cast(OpenRouter, client),
                )
            )

        error = caught.exception
        self.assertEqual(error.cost, 0.123)
        self.assertEqual(error.model, KLEIN_MODEL)
        self.assertEqual(error.requested_model, KLEIN_MODEL)
        self.assertEqual(error.operation, "generate")
        self.assertEqual(error.attempt_metadata["cost"], 0.123)
        self.assertEqual(error.attempt_metadata["requested_model"], KLEIN_MODEL)

        service_error = OpenRouterService._translate_exception(error, "image")
        self.assertEqual(service_error.cost, 0.123)
        self.assertEqual(service_error.model, KLEIN_MODEL)
        self.assertEqual(service_error.requested_model, KLEIN_MODEL)
        self.assertEqual(service_error.attempt_metadata["cost"], 0.123)


class ConcurrentModelSelectionTests(unittest.TestCase):
    def test_concurrent_generation_models_do_not_mutate_global_settings(self) -> None:
        client = _client()
        client.images.generate_async.return_value = _response(_png_bytes())
        original_settings = (
            settings.media.image_generation_model,
            settings.media.image_generation_fast_model,
            settings.media.image_edit_model,
            settings.media.image_edit_fast_model,
        )

        async def call_both() -> tuple[GeneratedImage, GeneratedImage]:
            return await asyncio.gather(
                openrouter_images.generate(
                    "first prompt",
                    ImageGenerationOptions(model=FLUX_MODEL),
                    client=cast(OpenRouter, client),
                ),
                openrouter_images.generate(
                    "second prompt",
                    ImageGenerationOptions(model=KLEIN_MODEL),
                    client=cast(OpenRouter, client),
                ),
            )

        first, second = asyncio.run(call_both())

        self.assertEqual(
            [call.kwargs["model"] for call in client.images.generate_async.await_args_list],
            [FLUX_MODEL, KLEIN_MODEL],
        )
        self.assertEqual(first.metadata["requested_model"], FLUX_MODEL)
        self.assertEqual(second.metadata["requested_model"], KLEIN_MODEL)
        self.assertEqual(
            (
                settings.media.image_generation_model,
                settings.media.image_generation_fast_model,
                settings.media.image_edit_model,
                settings.media.image_edit_fast_model,
            ),
            original_settings,
        )


class ImageOptionCompatibilityTests(unittest.TestCase):
    def test_model_is_keyword_only_without_changing_positional_options(self) -> None:
        generation = ImageGenerationOptions("16:9", "2K", None, 1, model=KLEIN_MODEL)
        edit = ImageEditOptions("1:1", "1K", None, 1, model=FLUX_MODEL)

        self.assertEqual(generation.aspect_ratio, "16:9")
        self.assertEqual(generation.resolution, "2K")
        self.assertEqual(generation.model, KLEIN_MODEL)
        self.assertEqual(edit.aspect_ratio, "1:1")
        self.assertEqual(edit.resolution, "1K")
        self.assertEqual(edit.model, FLUX_MODEL)
        with self.assertRaises(TypeError):
            ImageGenerationOptions("16:9", "2K", None, 1, KLEIN_MODEL)  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
