"""Проверки формирования списка reference images для OpenRouter Video API."""

import asyncio
import base64
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from core.common_types import GeneratedImage
from utils import openrouter_videos
from utils.openrouter_videos import OpenRouter


class VideoReferencePayloadTests(unittest.TestCase):
    def test_generate_passes_multiple_reference_images(self) -> None:
        response = SimpleNamespace(
            headers={"content-type": "video/mp4"},
            aread=AsyncMock(return_value=b"video-bytes"),
            aclose=AsyncMock(),
        )
        video_generation = SimpleNamespace(
            generate_async=AsyncMock(return_value=SimpleNamespace(id="job-1")),
            get_generation_async=AsyncMock(return_value=SimpleNamespace(status="completed")),
            get_video_content_async=AsyncMock(return_value=response),
        )
        client = SimpleNamespace(video_generation=video_generation)
        images = [
            GeneratedImage(content=b"first", mime_type="image/png"),
            GeneratedImage(content=b"second", mime_type="image/jpeg"),
        ]

        video = asyncio.run(
            openrouter_videos.generate(
                "animate these references",
                images,
                client=cast(OpenRouter, client),
            )
        )
        try:
            request: dict[str, Any] = video_generation.generate_async.await_args.kwargs
            references = request["input_references"]
            self.assertEqual(
                [item["image_url"]["url"] for item in references],
                [
                    "data:image/png;base64," + base64.b64encode(b"first").decode("ascii"),
                    "data:image/jpeg;base64," + base64.b64encode(b"second").decode("ascii"),
                ],
            )
        finally:
            video.cleanup()


if __name__ == "__main__":
    unittest.main()
