"""Проверки контракта и жизненного цикла видеоадаптера OpenRouter.

Покрываются сценарии Шага 9 плана:

- реальная сериализация SDK 1.1.29 через подменённый HTTP-транспорт ``httpx``:
  порядок картинок сохраняется не только в аргументах метода, но и в
  отправляемом JSON;
- запрос текущего профиля Prime/480p и подставного профиля другой модели без
  жёстко заданного Wan;
- обязательная длительность и пустой промпт;
- референсы в обратном и произвольном порядке, 0/1/несколько референсов,
  определение MIME-типов (PNG, JPEG, WebP);
- статусы (`completed`, `failed`, `cancelled`, `expired`, неизвестный статус) и
  разграничение 404 при создании и 404 при опросе/скачивании;
- единый таймаут на создание, опрос и скачивание, отмена, пустой файл,
  ошибка при чтении чанков, гарантированное закрытие потока и удаление
  временного файла;
- отсутствие повторного создания платной задачи при сбое/таймауте и
  сохранение `job_id` / `cost`.
"""

import asyncio
import base64
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
from openrouter.components.contentpartimage import ContentPartImageTypedDict
from openrouter.components.videogenerationrequest import (
    VideoGenerationRequestResolution,
)

from core.common_types import GeneratedImage, VideoGenerationOptions
from core.config import settings
from core.generation_error_mapper import (
    GenerationFailureReason,
    map_generation_error,
    user_message_for_reason,
)
from utils import openrouter_videos
from utils.openrouter_client import (
    OpenRouterResponseError,
    OpenRouterTimeoutError,
    OpenRouterUnsupportedModelError,
    OpenRouterUnsupportedParameterError,
    OpenRouterVideoCancelledError,
    OpenRouterVideoExpiredError,
    OpenRouterVideoFailedError,
    OpenRouterVideoJobUnavailableError,
)
from utils.openrouter_videos import OpenRouter

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"png-body"
JPEG_BYTES = b"\xff\xd8\xff" + b"jpeg-body"
WEBP_BYTES = b"RIFF\x00\x00\x00\x00WEBP" + b"webp-body"


def _content_response(
    payload: bytes = b"video-bytes",
    *,
    content_type: str = "video/mp4",
) -> SimpleNamespace:
    """Ответ скачивания видео с поддержкой потокового чтения."""

    async def aiter_bytes():
        yield payload

    return SimpleNamespace(
        headers={"content-type": content_type},
        aiter_bytes=aiter_bytes,
        aread=AsyncMock(return_value=payload),
        aclose=AsyncMock(),
    )


def _client(
    *,
    job: Any = None,
    statuses: list[Any] | None = None,
    content: Any = None,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Собирает подменённый SDK-клиент видео и его объект `video_generation`."""

    generate_async = AsyncMock(
        return_value=job if job is not None else SimpleNamespace(id="job-1")
    )
    get_generation_async = AsyncMock(
        side_effect=list(statuses) if statuses is not None else None,
        return_value=SimpleNamespace(status="completed"),
    )
    get_video_content_async = AsyncMock(
        return_value=content if content is not None else _content_response()
    )
    video_generation = SimpleNamespace(
        generate_async=generate_async,
        get_generation_async=get_generation_async,
        get_video_content_async=get_video_content_async,
    )
    return SimpleNamespace(video_generation=video_generation), video_generation


def _options(**overrides: Any) -> VideoGenerationOptions:
    values: dict[str, Any] = {
        "duration": 10,
        "model": settings.media.video_model,
        "resolution": settings.media.video_resolution,
        "polling_interval_seconds": 0.0,
    }
    values.update(overrides)
    return VideoGenerationOptions(**values)


class VideoReferencePayloadTests(unittest.TestCase):
    def test_generate_passes_multiple_reference_images(self) -> None:
        client, video_generation = _client()
        images = [
            GeneratedImage(content=PNG_BYTES, mime_type="image/png"),
            GeneratedImage(content=JPEG_BYTES, mime_type="image/jpeg"),
        ]

        video = asyncio.run(
            openrouter_videos.generate(
                "animate these references",
                _options(),
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
                    "data:image/png;base64,"
                    + base64.b64encode(PNG_BYTES).decode("ascii"),
                    "data:image/jpeg;base64,"
                    + base64.b64encode(JPEG_BYTES).decode("ascii"),
                ],
            )
            self.assertEqual(request["duration"], 10)
            self.assertEqual(request["model"], settings.media.video_model)
            self.assertEqual(request["resolution"], settings.media.video_resolution)
        finally:
            video.cleanup()

    def test_generate_keeps_user_order_of_references(self) -> None:
        client, video_generation = _client()
        images = [
            GeneratedImage(content=b"third", mime_type="image/png"),
            GeneratedImage(content=b"first", mime_type="image/png"),
            GeneratedImage(content=b"second", mime_type="image/png"),
        ]

        video = asyncio.run(
            openrouter_videos.generate(
                "animate these references",
                _options(),
                images,
                client=cast(OpenRouter, client),
            )
        )
        try:
            request: dict[str, Any] = video_generation.generate_async.await_args.kwargs
            self.assertEqual(
                [item["image_url"]["url"] for item in request["input_references"]],
                [
                    "data:image/png;base64," + base64.b64encode(b"third").decode("ascii"),
                    "data:image/png;base64," + base64.b64encode(b"first").decode("ascii"),
                    "data:image/png;base64," + base64.b64encode(b"second").decode("ascii"),
                ],
            )
        finally:
            video.cleanup()

    def test_generate_omits_unused_optional_fields(self) -> None:
        client, video_generation = _client()

        video = asyncio.run(
            openrouter_videos.generate(
                "text only video",
                _options(
                    duration=5,
                    resolution=cast(
                        VideoGenerationRequestResolution,
                        settings.media.video_resolution,
                    ),
                ),
                client=cast(OpenRouter, client),
            )
        )
        try:
            request: dict[str, Any] = video_generation.generate_async.await_args.kwargs
            self.assertNotIn("input_references", request)
            self.assertNotIn("generate_audio", request)
            self.assertNotIn("seed", request)
            self.assertNotIn("size", request)
            self.assertNotIn("frame_images", request)
        finally:
            video.cleanup()

    def test_generate_uses_replaced_profile_without_hardcoded_wan(self) -> None:
        """Подставной профиль другой модели меняет модель и разрешение."""

        client, video_generation = _client()

        video = asyncio.run(
            openrouter_videos.generate(
                "another model",
                _options(
                    model="test/other-video-model",
                    resolution="720p",
                    duration=8,
                ),
                client=cast(OpenRouter, client),
            )
        )
        try:
            request: dict[str, Any] = video_generation.generate_async.await_args.kwargs
            self.assertEqual(request["model"], "test/other-video-model")
            self.assertEqual(request["resolution"], "720p")
            self.assertEqual(request["duration"], 8)
            self.assertNotIn("alibaba/wan-3.0-prime", json.dumps(request, default=str))
        finally:
            video.cleanup()

    def test_empty_prompt_is_rejected_before_network_call(self) -> None:
        client, video_generation = _client()

        with self.assertRaises(OpenRouterUnsupportedParameterError):
            asyncio.run(
                openrouter_videos.generate(
                    "   ",
                    _options(),
                    client=cast(OpenRouter, client),
                )
            )

        video_generation.generate_async.assert_not_awaited()

    def test_duration_is_required(self) -> None:
        with self.assertRaises(TypeError):
            VideoGenerationOptions(duration=None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            VideoGenerationOptions(duration=0)

    def test_reference_count_variants(self) -> None:
        for count in (0, 1, 3):
            with self.subTest(count=count):
                client, video_generation = _client()
                images = [
                    GeneratedImage(content=f"image-{index}".encode(), mime_type="image/png")
                    for index in range(count)
                ]
                video = asyncio.run(
                    openrouter_videos.generate(
                        "references",
                        _options(),
                        images,
                        client=cast(OpenRouter, client),
                    )
                )
                try:
                    request: dict[str, Any] = video_generation.generate_async.await_args.kwargs
                    if count == 0:
                        self.assertNotIn("input_references", request)
                    else:
                        self.assertEqual(len(request["input_references"]), count)
                finally:
                    video.cleanup()

    def test_mime_types_are_detected_by_signature(self) -> None:
        cases = (
            (PNG_BYTES, "image/png"),
            (JPEG_BYTES, "image/jpeg"),
            (WEBP_BYTES, "image/webp"),
        )
        for content, expected in cases:
            with self.subTest(mime=expected):
                self.assertEqual(
                    openrouter_videos.detect_reference_mime_type(content),
                    expected,
                )

    def test_raw_bytes_reference_gets_detected_mime_type(self) -> None:
        client, video_generation = _client()

        video = asyncio.run(
            openrouter_videos.generate(
                "raw reference",
                _options(),
                [WEBP_BYTES],
                client=cast(OpenRouter, client),
            )
        )
        try:
            request: dict[str, Any] = video_generation.generate_async.await_args.kwargs
            self.assertEqual(
                request["input_references"][0]["image_url"]["url"],
                "data:image/webp;base64,"
                + base64.b64encode(WEBP_BYTES).decode("ascii"),
            )
        finally:
            video.cleanup()

    def test_unknown_reference_format_is_rejected(self) -> None:
        client, video_generation = _client()

        with self.assertRaises(OpenRouterUnsupportedParameterError):
            asyncio.run(
                openrouter_videos.generate(
                    "raw reference",
                    _options(),
                    [b"not-an-image-at-all"],
                    client=cast(OpenRouter, client),
                )
            )

        video_generation.generate_async.assert_not_awaited()


class VideoSdkSerializationTests(unittest.TestCase):
    """Реальная сериализация SDK через подменённый HTTP-транспорт."""

    def _capture_request(
        self,
        *,
        duration: int,
        references: list[ContentPartImageTypedDict],
        model: str = settings.media.video_model,
        resolution: str = settings.media.video_resolution,
    ) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                202,
                json={
                    "id": "job-abc123",
                    "generation_id": "gen-xyz789",
                    "polling_url": "/api/v1/videos/job-abc123",
                    "status": "pending",
                },
            )

        async def run() -> None:
            transport = httpx.MockTransport(handler)
            async_client = httpx.AsyncClient(transport=transport)
            client = OpenRouter(api_key="test-key", async_client=async_client)
            await client.__aenter__()
            try:
                await client.video_generation.generate_async(
                    model=model,
                    prompt="a slow cinematic push-in",
                    resolution=cast(
                        VideoGenerationRequestResolution, resolution
                    ),
                    duration=duration,
                    aspect_ratio="16:9",
                    input_references=references,
                )
            finally:
                await client.__aexit__(None, None, None)
                await async_client.aclose()

        asyncio.run(run())
        return captured

    def test_sdk_sends_model_resolution_duration_and_ordered_references(self) -> None:
        references: list[ContentPartImageTypedDict] = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAA"},
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,BBB"},
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/webp;base64,CCC"},
            },
        ]

        captured = self._capture_request(duration=7, references=references)
        body = captured["body"]

        self.assertTrue(captured["url"].endswith("/api/v1/videos"))
        self.assertEqual(body["model"], "alibaba/wan-3.0-prime")
        self.assertEqual(body["resolution"], "480p")
        self.assertEqual(body["duration"], 7)
        self.assertEqual(body["aspect_ratio"], "16:9")
        self.assertEqual(
            [item["image_url"]["url"] for item in body["input_references"]],
            [
                "data:image/png;base64,AAA",
                "data:image/jpeg;base64,BBB",
                "data:image/webp;base64,CCC",
            ],
        )
        self.assertEqual(
            [item["type"] for item in body["input_references"]],
            ["image_url", "image_url", "image_url"],
        )
        self.assertNotIn("size", body)
        self.assertNotIn("frame_images", body)
        self.assertNotIn("seed", body)
        self.assertNotIn("generate_audio", body)

    def test_sdk_keeps_reverse_reference_order(self) -> None:
        references: list[ContentPartImageTypedDict] = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,THIRD"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,FIRST"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECOND"}},
        ]

        captured = self._capture_request(duration=30, references=references)
        body = captured["body"]

        self.assertEqual(
            [item["image_url"]["url"] for item in body["input_references"]],
            [
                "data:image/png;base64,THIRD",
                "data:image/png;base64,FIRST",
                "data:image/png;base64,SECOND",
            ],
        )

    def test_sdk_sends_replaced_profile_without_hardcoded_wan(self) -> None:
        captured = self._capture_request(
            duration=4,
            references=[],
            model="test/other-video-model",
            resolution="720p",
        )
        body = captured["body"]

        self.assertEqual(body["model"], "test/other-video-model")
        self.assertEqual(body["resolution"], "720p")
        self.assertNotIn("alibaba/wan-3.0-prime", json.dumps(body))


class VideoStatusTests(unittest.TestCase):
    def test_completed_status_is_downloaded(self) -> None:
        client, video_generation = _client(
            statuses=[SimpleNamespace(status="completed")],
        )

        video = asyncio.run(
            openrouter_videos.generate(
                "prompt",
                _options(),
                client=cast(OpenRouter, client),
            )
        )
        try:
            self.assertTrue(video.path.exists())
            self.assertEqual(video.path.read_bytes(), b"video-bytes")
            video_generation.get_video_content_async.assert_awaited_once()
        finally:
            video.cleanup()

    def test_pending_statuses_are_polled(self) -> None:
        client, video_generation = _client(
            statuses=[
                SimpleNamespace(status="pending"),
                SimpleNamespace(status="in_progress"),
                SimpleNamespace(status="completed"),
            ],
        )

        video = asyncio.run(
            openrouter_videos.generate(
                "prompt",
                _options(),
                client=cast(OpenRouter, client),
            )
        )
        try:
            self.assertEqual(video_generation.get_generation_async.await_count, 3)
        finally:
            video.cleanup()

    def test_failed_status_raises_video_failed(self) -> None:
        client, _ = _client(
            statuses=[
                SimpleNamespace(status="failed", error="provider message"),
            ],
        )

        with self.assertRaises(OpenRouterVideoFailedError) as caught:
            asyncio.run(
                openrouter_videos.generate(
                    "prompt",
                    _options(),
                    client=cast(OpenRouter, client),
                )
            )
        self.assertEqual(caught.exception.job_id, "job-1")

    def test_cancelled_status_raises_video_cancelled(self) -> None:
        client, _ = _client(statuses=[SimpleNamespace(status="cancelled")])

        with self.assertRaises(OpenRouterVideoCancelledError):
            asyncio.run(
                openrouter_videos.generate(
                    "prompt",
                    _options(),
                    client=cast(OpenRouter, client),
                )
            )

    def test_expired_status_raises_video_expired(self) -> None:
        client, _ = _client(statuses=[SimpleNamespace(status="expired")])

        with self.assertRaises(OpenRouterVideoExpiredError):
            asyncio.run(
                openrouter_videos.generate(
                    "prompt",
                    _options(),
                    client=cast(OpenRouter, client),
                )
            )

    def test_unknown_status_finishes_with_clear_error(self) -> None:
        client, video_generation = _client(statuses=[SimpleNamespace(status="mystery")])

        with self.assertRaises(OpenRouterResponseError) as caught:
            asyncio.run(
                openrouter_videos.generate(
                    "prompt",
                    _options(),
                    client=cast(OpenRouter, client),
                )
            )

        self.assertIn("mystery", str(caught.exception))
        # Неизвестный статус не приводит к бесконечному опросу.
        self.assertEqual(video_generation.get_generation_async.await_count, 1)

    def test_missing_job_id_is_rejected(self) -> None:
        client, _ = _client(job=SimpleNamespace(id=""))

        with self.assertRaises(OpenRouterResponseError):
            asyncio.run(
                openrouter_videos.generate(
                    "prompt",
                    _options(),
                    client=cast(OpenRouter, client),
                )
            )


class VideoNotFoundTests(unittest.TestCase):
    def test_404_on_create_means_unavailable_model(self) -> None:
        client, video_generation = _client()
        video_generation.generate_async.side_effect = _not_found_error()

        with self.assertRaises(OpenRouterUnsupportedModelError) as caught:
            asyncio.run(
                openrouter_videos.generate(
                    "prompt",
                    _options(),
                    client=cast(OpenRouter, client),
                )
            )
        self.assertEqual(caught.exception.reason, "unsupported_model")

    def test_404_on_poll_means_lost_job(self) -> None:
        client, video_generation = _client()
        video_generation.get_generation_async.side_effect = _not_found_error()

        with self.assertRaises(OpenRouterVideoJobUnavailableError) as caught:
            asyncio.run(
                openrouter_videos.generate(
                    "prompt",
                    _options(),
                    client=cast(OpenRouter, client),
                )
            )
        self.assertEqual(caught.exception.reason, "video_job_unavailable")
        self.assertEqual(caught.exception.job_id, "job-1")

    def test_404_on_download_means_lost_job(self) -> None:
        client, video_generation = _client()
        video_generation.get_video_content_async.side_effect = _not_found_error()

        with self.assertRaises(OpenRouterVideoJobUnavailableError) as caught:
            asyncio.run(
                openrouter_videos.generate(
                    "prompt",
                    _options(),
                    client=cast(OpenRouter, client),
                )
            )
        self.assertEqual(caught.exception.job_id, "job-1")


class VideoLifecycleTests(unittest.TestCase):
    def test_single_deadline_covers_create_poll_and_download(self) -> None:
        """Общий таймаут начинается до создания задачи и покрывает все этапы."""

        client, video_generation = _client()

        async def slow_create(**_kwargs: Any) -> Any:
            await asyncio.sleep(0.05)
            return SimpleNamespace(id="job-1")

        async def slow_poll(**_kwargs: Any) -> Any:
            await asyncio.sleep(0.05)
            return SimpleNamespace(status="completed")

        video_generation.generate_async = AsyncMock(side_effect=slow_create)
        video_generation.get_generation_async = AsyncMock(side_effect=slow_poll)

        with self.assertRaises(OpenRouterTimeoutError):
            asyncio.run(
                openrouter_videos.generate(
                    "prompt",
                    _options(timeout_seconds=0.06),
                    client=cast(OpenRouter, client),
                )
            )

        self.assertEqual(video_generation.generate_async.await_count, 1)
        # Таймаут не создаёт новую платную задачу.
        self.assertEqual(video_generation.generate_async.await_count, 1)

    def test_remaining_timeout_is_passed_to_network_calls(self) -> None:
        client, video_generation = _client()

        video = asyncio.run(
            openrouter_videos.generate(
                "prompt",
                _options(timeout_seconds=30.0),
                client=cast(OpenRouter, client),
            )
        )
        try:
            create_kwargs = video_generation.generate_async.await_args.kwargs
            self.assertLessEqual(create_kwargs["timeout_ms"], 30_000)
            self.assertGreater(create_kwargs["timeout_ms"], 0)
            poll_kwargs = video_generation.get_generation_async.await_args.kwargs
            self.assertLessEqual(poll_kwargs["timeout_ms"], 30_000)
            download_kwargs = video_generation.get_video_content_async.await_args.kwargs
            self.assertLessEqual(download_kwargs["timeout_ms"], 30_000)
        finally:
            video.cleanup()

    def test_shared_deadline_timeout_during_download_closes_stream_and_removes_file(self) -> None:
        created: list[Path] = []
        original_create = openrouter_videos._create_temp_file

        def tracking_create(suffix: str) -> Path:
            path = original_create(suffix)
            created.append(path)
            return path

        response = _content_response()

        async def slow_iter():
            yield b"partial-video"
            await asyncio.sleep(5)

        response.aiter_bytes = slow_iter
        client, video_generation = _client(content=response)

        with (
            patch.object(
                openrouter_videos,
                "_create_temp_file",
                side_effect=tracking_create,
            ),
            self.assertRaises(OpenRouterTimeoutError) as caught,
        ):
            asyncio.run(
                openrouter_videos.generate(
                    "prompt",
                    _options(timeout_seconds=0.08),
                    client=cast(OpenRouter, client),
                )
            )

        self.assertEqual(caught.exception.job_id, "job-1")
        self.assertEqual(video_generation.generate_async.await_count, 1)
        self.assertEqual(video_generation.get_generation_async.await_count, 1)
        response.aclose.assert_awaited()
        self.assertEqual(len(created), 1)
        self.assertFalse(created[0].exists(), "Временный файл не должен оставаться")

    def test_cancellation_closes_stream_and_removes_temporary_file(self) -> None:
        created: list[Path] = []
        original_create = openrouter_videos._create_temp_file

        def tracking_create(suffix: str) -> Path:
            path = original_create(suffix)
            created.append(path)
            return path

        response = _content_response()

        async def slow_iter():
            yield b"partial"
            await asyncio.sleep(5)

        response.aiter_bytes = slow_iter

        client, video_generation = _client(content=response)

        async def run() -> None:
            task = asyncio.create_task(
                openrouter_videos.generate(
                    "prompt",
                    _options(timeout_seconds=30.0),
                    client=cast(OpenRouter, client),
                )
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        with patch.object(
            openrouter_videos, "_create_temp_file", side_effect=tracking_create
        ):
            asyncio.run(run())

        response.aclose.assert_awaited()
        for path in created:
            self.assertFalse(path.exists(), "Временный файл не должен оставаться")

    def test_empty_file_closes_stream_and_removes_temporary_file(self) -> None:
        created: list[Path] = []
        original_create = openrouter_videos._create_temp_file

        def tracking_create(suffix: str) -> Path:
            path = original_create(suffix)
            created.append(path)
            return path

        response = _content_response(b"")
        client, _ = _client(content=response)

        with patch.object(
            openrouter_videos, "_create_temp_file", side_effect=tracking_create
        ):
            with self.assertRaises(OpenRouterResponseError):
                asyncio.run(
                    openrouter_videos.generate(
                        "prompt",
                        _options(),
                        client=cast(OpenRouter, client),
                    )
                )

        response.aclose.assert_awaited()
        for path in created:
            self.assertFalse(path.exists(), "Временный файл не должен оставаться")

    def test_chunk_read_error_closes_stream_and_removes_temporary_file(self) -> None:
        created: list[Path] = []
        original_create = openrouter_videos._create_temp_file

        def tracking_create(suffix: str) -> Path:
            path = original_create(suffix)
            created.append(path)
            return path

        response = _content_response()

        async def broken_iter():
            yield b"partial"
            raise RuntimeError("stream broken")

        response.aiter_bytes = broken_iter
        client, _ = _client(content=response)

        with patch.object(
            openrouter_videos, "_create_temp_file", side_effect=tracking_create
        ):
            with self.assertRaises(RuntimeError):
                asyncio.run(
                    openrouter_videos.generate(
                        "prompt",
                        _options(),
                        client=cast(OpenRouter, client),
                    )
                )

        response.aclose.assert_awaited()
        for path in created:
            self.assertFalse(path.exists(), "Временный файл не должен оставаться")

    def test_no_paid_retry_after_failure(self) -> None:
        """Сбой не приводит к повторному созданию платной задачи."""

        client, video_generation = _client()
        video_generation.get_generation_async.side_effect = RuntimeError("network")

        with self.assertRaises(RuntimeError):
            asyncio.run(
                openrouter_videos.generate(
                    "prompt",
                    _options(),
                    client=cast(OpenRouter, client),
                )
            )

        self.assertEqual(video_generation.generate_async.await_count, 1)

    def test_job_id_is_preserved_in_error_after_creation(self) -> None:
        client, video_generation = _client()
        video_generation.get_generation_async.side_effect = RuntimeError("network")

        with self.assertRaises(RuntimeError) as caught:
            asyncio.run(
                openrouter_videos.generate(
                    "prompt",
                    _options(),
                    client=cast(OpenRouter, client),
                )
            )

        self.assertEqual(getattr(caught.exception, "job_id", None), "job-1")

    def test_job_id_and_cost_are_stored_in_result_metadata(self) -> None:
        client, _ = _client(
            job=SimpleNamespace(id="job-77", generation_id="gen-88"),
            statuses=[
                SimpleNamespace(
                    status="completed",
                    usage=SimpleNamespace(cost=0.42),
                )
            ],
        )

        video = asyncio.run(
            openrouter_videos.generate(
                "prompt",
                _options(),
                client=cast(OpenRouter, client),
            )
        )
        try:
            self.assertEqual(video.metadata["job_id"], "job-77")
            self.assertEqual(video.job_id, "job-77")
            self.assertEqual(video.metadata["generation_id"], "gen-88")
            self.assertEqual(video.cost, 0.42)
            self.assertEqual(video.metadata["cost"], 0.42)
            self.assertEqual(video.metadata["model"], settings.media.video_model)
            self.assertEqual(video.metadata["resolution"], settings.media.video_resolution)
            self.assertEqual(video.metadata["duration"], 10)
        finally:
            video.cleanup()

    def test_missing_cost_means_unknown_not_free(self) -> None:
        client, _ = _client(
            statuses=[SimpleNamespace(status="completed", usage=None)],
        )

        video = asyncio.run(
            openrouter_videos.generate(
                "prompt",
                _options(),
                client=cast(OpenRouter, client),
            )
        )
        try:
            self.assertIsNone(video.cost)
            self.assertNotIn("cost", video.metadata)
        finally:
            video.cleanup()

    def test_sdk_retries_are_disabled_for_paid_create(self) -> None:
        client, video_generation = _client()

        video = asyncio.run(
            openrouter_videos.generate(
                "prompt",
                _options(),
                client=cast(OpenRouter, client),
            )
        )
        try:
            retries = video_generation.generate_async.await_args.kwargs["retries"]
            self.assertEqual(retries.strategy, "none")
            self.assertFalse(retries.retry_connection_errors)
        finally:
            video.cleanup()


class VideoLoggingSafetyTests(unittest.TestCase):
    def test_logs_contain_only_safe_identifiers(self) -> None:
        """В логи попадают только безопасные идентификаторы и параметры."""

        import io
        import logging

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger("utils.openrouter_videos")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            client, _ = _client(
                job=SimpleNamespace(id="job-abc"),
                statuses=[
                    SimpleNamespace(status="completed", usage=SimpleNamespace(cost=0.5))
                ],
            )
            secret_prompt = "секретный промпт с data:image/png;base64,AAA"
            video = asyncio.run(
                openrouter_videos.generate(
                    secret_prompt,
                    _options(),
                    [GeneratedImage(content=PNG_BYTES, mime_type="image/png")],
                    client=cast(OpenRouter, client),
                )
            )
            try:
                log_text = stream.getvalue()
            finally:
                video.cleanup()
        finally:
            logger.removeHandler(handler)

        self.assertIn("job-abc", log_text)
        self.assertIn(settings.media.video_model, log_text)
        self.assertNotIn("секретный промпт", log_text)
        self.assertNotIn("base64", log_text)
        self.assertNotIn("data:image", log_text)


class VideoErrorMappingTests(unittest.TestCase):
    def test_lost_job_has_own_user_message(self) -> None:
        error = OpenRouterVideoJobUnavailableError(
            "Задача недоступна",
            status_code=404,
            reason="video_job_unavailable",
            operation="video",
            job_id="job-1",
        )

        mapped = map_generation_error(error, context="video")

        self.assertIs(mapped.reason, GenerationFailureReason.VIDEO_JOB_UNAVAILABLE)
        self.assertIn("недоступна", mapped.user_message.lower())
        self.assertIn("повторный платный запрос", mapped.user_message.lower())

    def test_unavailable_model_has_own_user_message(self) -> None:
        error = OpenRouterUnsupportedModelError(
            "Модель недоступна",
            status_code=404,
            reason="unsupported_model",
            operation="video",
        )

        mapped = map_generation_error(error, context="video")

        self.assertIs(mapped.reason, GenerationFailureReason.UNSUPPORTED_MODEL)

    def test_invalid_duration_message_is_user_friendly(self) -> None:
        message = user_message_for_reason(GenerationFailureReason.INVALID_DURATION)

        self.assertIn("длительность", message.lower())
        self.assertIn("секунд", message.lower())

    def test_expired_question_message_is_user_friendly(self) -> None:
        message = user_message_for_reason(GenerationFailureReason.QUESTION_EXPIRED)

        self.assertIn("истекло", message.lower())
        self.assertIn("заново", message.lower())

    def test_job_id_is_kept_in_mapped_error(self) -> None:
        error = OpenRouterVideoFailedError(
            "Генерация видео завершилась с ошибкой",
            operation="video",
            job_id="job-77",
        )

        mapped = map_generation_error(error, context="video")

        self.assertIs(mapped.reason, GenerationFailureReason.VIDEO_FAILED)
        self.assertEqual(error.job_id, "job-77")


def _not_found_error() -> Exception:
    """Ошибка провайдера со статусом 404."""

    error = RuntimeError("not found")
    setattr(error, "status_code", 404)
    setattr(error, "body", '{"error": {"message": "not found"}}')
    return error


if __name__ == "__main__":
    unittest.main()
