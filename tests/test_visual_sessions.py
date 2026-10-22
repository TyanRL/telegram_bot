"""Проверки хранения, выбора и сброса изображений visual session."""

import asyncio
import base64
import json
from copy import deepcopy
import os
import unittest
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:TEST_TOKEN")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import main as bot_main
from core.state_and_commands import (
    append_user_image_to_session,
    clear_user_image_edit_session,
    get_all_histories,
    get_user_image_edit_session,
    reset,
    set_user_image_edit_session,
)
from core.tool_helpers import (
    append_image_to_session,
    build_image_dict_from_image,
    describe_image_provenance,
    generated_image_from_session_dict,
    get_session_images,
    resolve_selected_images,
)
from core.common_types import GeneratedImage, ModelAnswer
from core.config import settings
from core.image_routing import resolve_image_route
from core.image_observability import aggregate_image_cost_report
from core.tool_handlers.media import (
    _sdk_aspect_ratio,
    handle_generate_image,
    handle_generate_image_from_image,
    handle_generate_video,
)
from core.video_requests import validate_duration
from PIL import Image
from utils.openrouter_client import OpenRouterResponseError


def _image(number: int) -> dict[str, object]:
    return {
        "image_type": "image/png",
        "image": base64.b64encode(f"image-content-{number}".encode("ascii")).decode("ascii"),
        "width": 100,
        "height": 100,
        "aspect_ratio": 1.0,
    }


def _png_bytes(width: int = 4, height: int = 3) -> bytes:
    image_buffer = BytesIO()
    Image.new("RGB", (width, height), color=(12, 34, 56)).save(
        image_buffer,
        format="PNG",
    )
    return image_buffer.getvalue()


def _auto_settings():
    return replace(
        settings,
        media=replace(
            settings.media,
            image_routing_mode="auto",
            image_default_resolution="1K",
        ),
    )


def _media_context(user_id: int, service: Any, user_text: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        update=SimpleNamespace(effective_user=SimpleNamespace(id=user_id)),
        openrouter_service=service,
        context_tokens=0,
        completion_tokens=0,
        user_text=user_text,
    )


def _successful_generated_image(model: str, cost: float = 0.2) -> GeneratedImage:
    return GeneratedImage(
        content=_png_bytes(),
        mime_type="image/png",
        metadata={
            "model": model,
            "requested_model": model,
            "cost": cost,
            "total_known_cost": cost,
            "fallback_used": False,
            "billing_unknown": False,
            "attempts": [
                {
                    "model": model,
                    "status": 200,
                    "status_code": 200,
                    "duration_ms": 12.0,
                    "cost": cost,
                    "billing_unknown": False,
                    "error_reason": None,
                }
            ],
        },
    )


class SessionImageTests(unittest.TestCase):
    def test_append_and_default_selection_use_all_images(self) -> None:
        session: dict[str, object] = {"images": [], "history": []}

        first_number = append_image_to_session(session, _image(1), "uploaded")
        second_number = append_image_to_session(session, _image(2), "generated")

        self.assertEqual((first_number, second_number), (1, 2))
        self.assertEqual(get_session_images(session), [_image(1) | {"source_kind": "uploaded"}, _image(2) | {"source_kind": "generated"}])

    def test_append_stores_only_the_unified_images_list(self) -> None:
        session: dict[str, object] = {"images": [], "history": []}

        append_image_to_session(session, _image(1), "uploaded")

        self.assertEqual(set(session), {"images", "history"})
        self.assertNotIn("current_image", session)
        self.assertNotIn("original_image", session)

    def test_selection_uses_one_based_indices_and_deduplicates(self) -> None:
        session: dict[str, object] = {
            "images": [_image(1), _image(2), _image(3)],
        }

        selected = get_session_images(session, [3, 1, 3])

        self.assertEqual(
            [image["image"] for image in selected],
            [_image(3)["image"], _image(1)["image"]],
        )

    def test_selection_rejects_out_of_range_indices(self) -> None:
        session: dict[str, object] = {"images": [_image(1)]}

        with self.assertRaisesRegex(ValueError, "вне диапазона"):
            get_session_images(session, [2])

    def test_generation_metadata_and_cost_round_trip_with_image(self) -> None:
        image = GeneratedImage(
            content=b"generated-image",
            mime_type="image/png",
            metadata={
                "model": settings.media.image_generation_model,
                "requested_model": settings.media.image_generation_fast_model,
                "operation": "edit",
                "cost": 0.04,
                "fallback_used": True,
            },
        )
        encoded = base64.b64encode(b"telegram-jpeg").decode("ascii")

        image_dict = build_image_dict_from_image(
            image,
            encoded,
            640,
            480,
            4 / 3,
            generation={
                "operation_id": "operation-123",
                "operation": "edit",
                "prompt": "Сохранить форму, изменить цвет на синий",
                "source_image_indices": [8, 2, 8, 11],
                "aspect_ratio": "4:3",
                "resolution": "1K",
                "route_reason": "quality_hint",
                "total_known_cost": 0.09,
                "billing_unknown": True,
            },
        )

        restored = generated_image_from_session_dict(image_dict)

        self.assertEqual(image_dict["metadata"], {"cost": 0.04})
        self.assertEqual(image_dict["generation"]["operation_id"], "operation-123")
        self.assertEqual(image_dict["generation"]["task_id"], "operation-123")
        self.assertEqual(image_dict["generation"]["model"], settings.media.image_generation_model)
        self.assertEqual(image_dict["generation"]["requested_model"], settings.media.image_generation_fast_model)
        self.assertEqual(image_dict["generation"]["quality"], "pro")
        self.assertTrue(image_dict["generation"]["fallback_used"])
        self.assertEqual(
            image_dict["generation"]["source_image_indices"],
            [8, 2, 11],
        )
        self.assertEqual(image_dict["generation"]["total_known_cost"], 0.09)
        self.assertTrue(image_dict["generation"]["billing_unknown"])
        self.assertEqual(restored.metadata["cost"], 0.04)
        self.assertEqual(restored.metadata["generation"]["prompt"], "Сохранить форму, изменить цвет на синий")
        self.assertEqual(restored.metadata["generation"]["task_id"], "operation-123")
        self.assertEqual(restored.metadata["generation"]["source_image_indices"], [8, 2, 11])

    def test_legacy_session_image_without_generation_stays_unknown(self) -> None:
        restored = generated_image_from_session_dict(_image(1))

        self.assertNotIn("generation", restored.metadata)
        self.assertNotIn("model", restored.metadata)
        self.assertNotIn("quality", restored.metadata)

    def test_resolved_selected_images_keep_real_indices_after_dedup_and_limit(self) -> None:
        session = {"images": [_image(index) for index in range(1, 16)]}
        requested_indices = [15, 2, 4, 2, 9, 13, 1, 6, 3, 8, 12, 5, 14, 7, 11, 10, 15]

        selected, selected_indices = resolve_selected_images(
            session,
            requested_indices,
            max_images=10,
        )

        self.assertEqual(selected_indices, [1, 6, 3, 8, 12, 5, 14, 7, 11, 10])
        self.assertEqual(
            [image.content for image in selected],
            [
                base64.b64decode(_image(index)["image"])
                for index in selected_indices
            ],
        )

    def test_provenance_summary_has_context_without_cost_or_binary_data(self) -> None:
        images = [
            _image(1) | {"source_kind": "uploaded"},
            _image(2)
            | {
                "source_kind": "edit",
                "metadata": {"cost": 0.123},
                "generation": {
                    "operation": "edit",
                    "model": "black-forest-labs/flux-3-image",
                    "quality": "pro",
                    "prompt": "Изменить фон на лес",
                    "source_image_indices": [1],
                },
            },
        ]

        summary = describe_image_provenance(images)

        self.assertIn("загружено пользователем", summary)
        self.assertIn("последний успешный результат", summary)
        self.assertIn("уровень pro", summary)
        self.assertIn("источники №1", summary)
        self.assertIn("Изменить фон на лес", summary)
        self.assertNotIn("0.123", summary)
        self.assertNotIn(images[1]["image"], summary)

    def test_session_without_images_list_has_no_images(self) -> None:
        """Единственный актуальный формат сессии — список ``images``."""

        self.assertEqual(get_session_images({"current_image": _image(1)}), [])

    def test_rollback_route_keeps_session_images_readable(self) -> None:
        legacy_image = _image(1) | {"source_kind": "generated"}
        current_image = _image(2) | {
            "source_kind": "edit",
            "metadata": {"cost": 0.42},
            "generation": {
                "operation_id": "0123456789abcdef0123456789abcdef",
                "task_id": "fedcba9876543210fedcba9876543210",
                "operation": "edit",
                "model": settings.media.image_edit_model,
                "quality": "pro",
                "requested_model": settings.media.image_edit_fast_model,
                "fallback_used": True,
                "prompt": "Preserve the original line engraving",
                "source_image_indices": [1],
                "aspect_ratio": "16:9",
                "resolution": "1K",
                "route_reason": "text_marker",
                "total_known_cost": 0.57,
                "billing_unknown": False,
            },
        }
        session = {
            "images": [legacy_image, current_image],
            "history": [{"kind": "edit", "prompt": "Preserve the engraving"}],
        }
        original_session = deepcopy(session)
        rollback_media = replace(
            _auto_settings().media,
            image_routing_mode="pro_only",
            image_default_resolution=None,
        )

        decision = resolve_image_route(
            rollback_media,
            "generate",
            quality="standard",
            prompt="A simple new landscape",
        )
        restored_legacy = generated_image_from_session_dict(session["images"][0])
        restored_current = generated_image_from_session_dict(session["images"][1])

        self.assertEqual(decision.model, rollback_media.image_generation_model)
        self.assertEqual(decision.quality, "pro")
        self.assertIsNone(decision.resolution)
        self.assertNotIn("generation", restored_legacy.metadata)
        self.assertEqual(restored_current.metadata["generation"]["model"], settings.media.image_edit_model)
        self.assertEqual(restored_current.metadata["generation"]["source_image_indices"], [1])
        self.assertEqual(restored_current.metadata["generation"]["prompt"], "Preserve the original line engraving")
        self.assertEqual(restored_current.metadata["cost"], 0.42)
        self.assertEqual(session, original_session)

    def test_concurrent_uploads_receive_unique_ordered_numbers(self) -> None:
        user_id = 9_876_543_214

        async def scenario() -> tuple[list[tuple[int, int]], dict | None]:
            await clear_user_image_edit_session(user_id)
            first_number = await append_user_image_to_session(
                user_id,
                _image(0),
                "uploaded",
            )
            numbers = await asyncio.gather(
                *[
                    append_generated(index)
                    for index in range(1, 21)
                ]
            )
            session = await get_user_image_edit_session(user_id)
            await clear_user_image_edit_session(user_id)

            self.assertEqual(first_number, 1)
            return numbers, session

        async def append_generated(index: int) -> tuple[int, int]:
            image_dict = _image(index)
            image_dict["generation"] = {
                "operation_id": f"operation-{index}",
                "prompt": f"prompt-{index}",
                "source_image_indices": [1],
            }
            number = await append_user_image_to_session(
                user_id,
                image_dict,
                "generated",
                f"prompt-{index}",
            )
            return index, number

        numbers, session = asyncio.run(scenario())

        self.assertEqual(sorted(number for _, number in numbers), list(range(2, 22)))
        images = get_session_images(session)  # type: ignore[arg-type]
        self.assertEqual(len(images), 21)
        self.assertEqual(images[0]["source_kind"], "uploaded")
        for operation_index, image_number in numbers:
            stored = images[image_number - 1]
            self.assertEqual(
                stored["generation"]["operation_id"],
                f"operation-{operation_index}",
            )


class ResetVisualSessionTests(unittest.TestCase):
    def test_reset_removes_visual_session_and_legacy_image_file(self) -> None:
        user_id = 9_876_543_210
        histories = get_all_histories()
        asyncio.run(
            set_user_image_edit_session(
                user_id,
                {"images": [_image(1), _image(2)], "history": []},
            )
        )
        asyncio.run(
            histories.set(
                user_id,
                [{"role": "user", "content": [{"type": "input_image", "image": "stale"}]}],
            )
        )
        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=user_id),
        )
        context = SimpleNamespace()

        with (
            patch("core.state_and_commands.reply_service_text", new=AsyncMock()),
            patch("core.state_and_commands.Path") as path_factory,
        ):
            asyncio.run(reset(update, context))

        self.assertIsNone(asyncio.run(get_user_image_edit_session(user_id)))
        self.assertEqual(asyncio.run(histories.get(user_id, [])), [])
        path_factory.assert_called_once_with(f"user_{user_id}_image.jpg")
        path_factory.return_value.unlink.assert_called_once_with(missing_ok=True)


class PhotoAndVisionTests(unittest.TestCase):
    def test_photo_uploads_accumulate_and_caption_is_processed(self) -> None:
        user_id = 9_876_543_212
        image_buffer = BytesIO()
        Image.new("RGB", (2, 2), color="red").save(image_buffer, format="PNG")
        image_bytes = image_buffer.getvalue()
        user = SimpleNamespace(id=user_id)

        def make_update(caption: str | None = None) -> SimpleNamespace:
            telegram_file = SimpleNamespace(
                download_as_bytearray=AsyncMock(return_value=bytearray(image_bytes)),
            )
            photo = SimpleNamespace(get_file=AsyncMock(return_value=telegram_file))
            message = SimpleNamespace(photo=[photo], caption=caption)
            return SimpleNamespace(effective_user=user, message=message)

        async def scenario() -> None:
            await clear_user_image_edit_session(user_id)
            with (
                patch("main.in_user_list", new=AsyncMock(return_value=True)),
                patch("main.reply_service_text", new=AsyncMock()) as reply_service,
                patch("main.handle_message_inner", new=AsyncMock()) as handle_message,
            ):
                await bot_main.handle_photo(make_update(), SimpleNamespace())
                await bot_main.handle_photo(make_update(), SimpleNamespace())
                await bot_main.handle_photo(make_update("Опиши эти картинки"), SimpleNamespace())

            session = await get_user_image_edit_session(user_id)
            self.assertIsNotNone(session)
            self.assertEqual(len(get_session_images(session)), 3)  # type: ignore[arg-type]
            self.assertEqual(reply_service.await_count, 2)
            handle_message.assert_awaited_once()
            self.assertEqual(
                handle_message.await_args.args[2],
                "Опиши эти картинки",
            )
            await clear_user_image_edit_session(user_id)

        asyncio.run(scenario())

    def test_recognized_voice_text_is_forwarded_as_the_user_message(self) -> None:
        user_id = 9_876_543_217
        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=user_id),
            message=SimpleNamespace(voice=SimpleNamespace(file_id="voice-file")),
        )
        downloaded_file = SimpleNamespace(download_to_drive=AsyncMock())
        context = SimpleNamespace(
            bot=SimpleNamespace(get_file=AsyncMock(return_value=downloaded_file))
        )
        recognized_text = 'Создай вывеску с точной надписью «СЕВЕР»'

        with (
            patch("main.in_user_list", new=AsyncMock(return_value=True)),
            patch("main.transcribe_audio", return_value=recognized_text) as transcribe,
            patch("main.send_big_text", new=AsyncMock()),
            patch("main.handle_message_inner", new=AsyncMock()) as handle_message,
        ):
            async def process_voice() -> None:
                await bot_main.handle_voice_message(
                    cast(Any, update),
                    cast(Any, context),
                )

            asyncio.run(process_voice())

        context.bot.get_file.assert_awaited_once_with("voice-file")
        downloaded_file.download_to_drive.assert_awaited_once()
        transcribe.assert_called_once()
        handle_message.assert_awaited_once_with(update, context, recognized_text)

    def test_document_upload_normalizes_and_resizes_large_rgba_image(self) -> None:
        user_id = 9_876_543_215
        image_buffer = BytesIO()
        Image.new("RGBA", (2000, 1000), color=(255, 0, 0, 128)).save(
            image_buffer,
            format="PNG",
        )
        image_bytes = image_buffer.getvalue()
        telegram_file = SimpleNamespace(
            download_as_bytearray=AsyncMock(return_value=bytearray(image_bytes)),
        )
        document = SimpleNamespace(
            mime_type="image/png",
            get_file=AsyncMock(return_value=telegram_file),
        )
        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=user_id),
            message=SimpleNamespace(photo=[], document=document, caption=None),
        )

        async def scenario() -> None:
            await clear_user_image_edit_session(user_id)
            try:
                with (
                    patch("main.in_user_list", new=AsyncMock(return_value=True)),
                    patch("main.reply_service_text", new=AsyncMock()),
                ):
                    await bot_main.handle_photo(update, SimpleNamespace())

                document.get_file.assert_awaited_once_with()
                telegram_file.download_as_bytearray.assert_awaited_once_with()
                session = await get_user_image_edit_session(user_id)
                images = get_session_images(session)  # type: ignore[arg-type]
                self.assertEqual(len(images), 1)
                stored_image = images[0]
                self.assertEqual(stored_image["image_type"], "image/jpeg")

                normalized_bytes = base64.b64decode(stored_image["image"])
                with Image.open(BytesIO(normalized_bytes)) as normalized_image:
                    self.assertEqual(normalized_image.format, "JPEG")
                    self.assertEqual(normalized_image.mode, "RGB")
                    self.assertLessEqual(
                        max(normalized_image.size),
                        settings.media.telegram_image_max_size,
                    )
                    self.assertEqual(
                        (stored_image["width"], stored_image["height"]),
                        normalized_image.size,
                    )
            finally:
                await clear_user_image_edit_session(user_id)

        asyncio.run(scenario())

    def test_vision_request_contains_every_session_image(self) -> None:
        user_id = 9_876_543_213
        session = {"images": [_image(1), _image(2), _image(3)], "history": []}
        update = SimpleNamespace(effective_user=SimpleNamespace(id=user_id))
        answer = ModelAnswer(bot_reply="Вижу все картинки")

        async def scenario() -> None:
            await bot_main.user_histories.delete(user_id)
            await set_user_image_edit_session(user_id, session)
            with (
                patch(
                    "main.get_user_image_edit_session",
                    new=AsyncMock(return_value=session),
                ),
                patch("main.get_system_message", return_value={"role": "system", "content": "test"}),
                patch("main.get_model_answer", new=AsyncMock(return_value=answer)) as get_answer,
            ):
                await bot_main.get_bot_reply(
                    update,
                    SimpleNamespace(),
                    "Сравни",
                    SimpleNamespace(),
                )

            messages = get_answer.await_args.args[2]
            self.assertEqual(get_answer.await_args.kwargs["user_text"], "Сравни")
            user_content = messages[-1]["content"]
            attached_images = [part for part in user_content if part["type"] == "input_image"]
            self.assertEqual(len(attached_images), 3)
            self.assertIn("№1, №2, №3", user_content[0]["text"])
            stored_history = await bot_main.user_histories.get(user_id, [])
            self.assertEqual(len(stored_history), 2)
            self.assertIsInstance(stored_history[0]["content"], str)
            await bot_main.user_histories.delete(user_id)
            await clear_user_image_edit_session(user_id)

        asyncio.run(scenario())

    def test_vision_request_limits_session_images_to_the_latest_ten(self) -> None:
        user_id = 9_876_543_216
        image_count = settings.media.max_session_images + 2
        session = {
            "images": [_image(index) for index in range(1, image_count + 1)],
            "history": [],
        }
        update = SimpleNamespace(effective_user=SimpleNamespace(id=user_id))
        answer = ModelAnswer(bot_reply="Вижу последние картинки")
        first_attached_index = image_count - settings.media.max_session_images + 1
        expected_labels = ", ".join(
            f"№{index}"
            for index in range(first_attached_index, image_count + 1)
        )

        async def scenario() -> None:
            await bot_main.user_histories.delete(user_id)
            try:
                with (
                    patch(
                        "main.get_user_image_edit_session",
                        new=AsyncMock(return_value=session),
                    ),
                    patch(
                        "main.get_system_message",
                        return_value={"role": "system", "content": "test"},
                    ),
                    patch(
                        "main.get_model_answer",
                        new=AsyncMock(return_value=answer),
                    ) as get_answer,
                ):
                    await bot_main.get_bot_reply(
                        update,
                        SimpleNamespace(),
                        "Сравни последние картинки",
                        SimpleNamespace(),
                    )

                messages = get_answer.await_args.args[2]
                user_content = messages[-1]["content"]
                attached_images = [
                    part for part in user_content if part["type"] == "input_image"
                ]
                self.assertEqual(
                    len(attached_images),
                    settings.media.max_session_images,
                )
                self.assertIn(expected_labels, user_content[0]["text"])
                self.assertEqual(
                    [part["image"].content for part in attached_images],
                    [
                        base64.b64decode(_image(index)["image"])
                        for index in range(first_attached_index, image_count + 1)
                    ],
                )
            finally:
                await bot_main.user_histories.delete(user_id)
                await clear_user_image_edit_session(user_id)

        asyncio.run(scenario())

    def test_explicit_session_clear_deletes_storage_entry(self) -> None:
        user_id = 9_876_543_211
        asyncio.run(set_user_image_edit_session(user_id, {"images": [_image(1)]}))

        asyncio.run(clear_user_image_edit_session(user_id))

        self.assertIsNone(asyncio.run(get_user_image_edit_session(user_id)))


class AspectRatioTests(unittest.TestCase):
    def test_sdk_aspect_ratio_supports_flux_formats_and_auto(self) -> None:
        for value in ("4:5", "5:4", "2:1", "auto"):
            with self.subTest(value=value):
                self.assertEqual(_sdk_aspect_ratio(value), value)

        self.assertIsNone(_sdk_aspect_ratio("unknown-ratio"))


class MediaToolImageSelectionTests(unittest.TestCase):
    def test_generate_image_is_allowed_with_an_existing_session(self) -> None:
        user_id = 122
        image = GeneratedImage(content=b"generated", mime_type="image/png")
        service = SimpleNamespace(generate=AsyncMock(return_value=image))
        ctx = SimpleNamespace(
            update=SimpleNamespace(effective_user=SimpleNamespace(id=user_id)),
            openrouter_service=service,
            context_tokens=0,
            completion_tokens=0,
        )

        with (
            patch(
                "core.tool_handlers.media.send_image_to_telegram",
                new=AsyncMock(return_value=("encoded", 10, 10, 1.0)),
            ),
            patch(
                "core.tool_handlers.media.append_user_image_to_session",
                new=AsyncMock(return_value=2),
            ) as append_image,
            patch(
                "core.tool_handlers.media.get_user_image_edit_session",
                new=AsyncMock(return_value={"images": ["existing"]}),
            ),
        ):
            result = asyncio.run(handle_generate_image(ctx, {"prompt": "new image"}))

        self.assertTrue(result.output["ok"])
        self.assertIn("№2", result.output["message"])
        append_image.assert_awaited_once()

    def test_generate_image_passes_explicit_aspect_ratio_and_deletes_status(self) -> None:
        user_id = 127
        image = GeneratedImage(content=b"generated", mime_type="image/png")
        service = SimpleNamespace(generate=AsyncMock(return_value=image))
        status_message = SimpleNamespace(delete=AsyncMock())
        update = SimpleNamespace(effective_user=SimpleNamespace(id=user_id))
        ctx = SimpleNamespace(
            update=update,
            openrouter_service=service,
            context_tokens=0,
            completion_tokens=0,
        )

        with (
            patch(
                "core.tool_handlers.media.reply_service_message",
                new=AsyncMock(return_value=status_message),
            ) as reply_status,
            patch("core.tool_handlers.media.animate_service_message", new=AsyncMock()),
            patch(
                "core.tool_handlers.media.send_image_to_telegram",
                new=AsyncMock(return_value=("encoded", 100, 50, 2.0)),
            ) as send_image,
            patch(
                "core.tool_handlers.media.append_user_image_to_session",
                new=AsyncMock(return_value=1),
            ),
        ):
            result = asyncio.run(
                handle_generate_image(
                    ctx,
                    {"prompt": "wide landscape", "aspect_ratio": "2:1"},
                )
            )

        self.assertTrue(result.output["ok"])
        self.assertEqual(service.generate.await_args.args[1].aspect_ratio, "2:1")
        reply_status.assert_awaited_once_with(
            update,
            "Изображение генерируется, подождите",
        )
        send_image.assert_awaited_once_with(update, image)
        status_message.delete.assert_awaited_once_with()

    def test_edit_tool_passes_only_selected_reference_images(self) -> None:
        source_images = [
            _image(1),
            _image(2),
            _image(3),
        ]
        session = {"images": source_images, "history": []}
        service = SimpleNamespace(
            edit=AsyncMock(
                return_value=GeneratedImage(content=b"result", mime_type="image/png")
            )
        )
        ctx = SimpleNamespace(
            update=SimpleNamespace(effective_user=SimpleNamespace(id=123)),
            openrouter_service=service,
            context_tokens=0,
            completion_tokens=0,
        )
        selected_image_b64 = _image(2)["image"]

        with (
            patch(
                "core.tool_handlers.media.get_user_image_edit_session",
                new=AsyncMock(return_value=session),
            ),
            patch(
                "core.tool_handlers.media.send_image_to_telegram",
                new=AsyncMock(return_value=("normalized", 100, 100, 1.0)),
            ),
            patch(
                "core.tool_handlers.media.append_user_image_to_session",
                new=AsyncMock(return_value=4),
            ) as append_image,
        ):
            result = asyncio.run(
                handle_generate_image_from_image(
                    ctx,
                    {"prompt": "use only image 2", "image_indices": [2]},
                )
            )

        passed_images = service.edit.await_args.args[0]
        self.assertEqual(len(passed_images), 1)
        self.assertEqual(passed_images[0].content, base64.b64decode(selected_image_b64))
        stored_image_dict = append_image.await_args.args[1]
        self.assertEqual(
            stored_image_dict["generation"]["source_image_indices"],
            [2],
        )
        self.assertEqual(stored_image_dict["generation"]["operation"], "edit")
        self.assertTrue(result.output["ok"])

    def test_edit_tool_uses_explicit_aspect_ratio_without_prompt_or_padding(self) -> None:
        image = GeneratedImage(content=b"edited", mime_type="image/png")
        service = SimpleNamespace(edit=AsyncMock(return_value=image))
        status_message = SimpleNamespace(delete=AsyncMock())
        update = SimpleNamespace(effective_user=SimpleNamespace(id=128))
        ctx = SimpleNamespace(
            update=update,
            openrouter_service=service,
            context_tokens=0,
            completion_tokens=0,
        )
        prompt = "Создай вертикальную композицию"

        with (
            patch(
                "core.tool_handlers.media.get_user_image_edit_session",
                new=AsyncMock(return_value={"images": [_image(1)], "history": []}),
            ),
            patch(
                "core.tool_handlers.media.reply_service_message",
                new=AsyncMock(return_value=status_message),
            ),
            patch("core.tool_handlers.media.animate_service_message", new=AsyncMock()),
            patch(
                "core.tool_handlers.media.send_image_to_telegram",
                new=AsyncMock(return_value=("normalized", 100, 125, 0.8)),
            ) as send_image,
            patch(
                "core.tool_handlers.media.append_user_image_to_session",
                new=AsyncMock(return_value=2),
            ),
        ):
            result = asyncio.run(
                handle_generate_image_from_image(
                    ctx,
                    {"prompt": prompt, "aspect_ratio": "4:5"},
                )
            )

        self.assertTrue(result.output["ok"])
        self.assertEqual(service.edit.await_args.args[1], prompt)
        self.assertNotIn(
            "preserve the original image aspect ratio",
            service.edit.await_args.args[1],
        )
        self.assertEqual(service.edit.await_args.args[2].aspect_ratio, "4:5")
        send_image.assert_awaited_once_with(update, image)
        status_message.delete.assert_awaited_once_with()

    def _video_context(
        self,
        user_id: int,
        service: Any,
        user_text: str | None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            update=SimpleNamespace(effective_user=SimpleNamespace(id=user_id)),
            openrouter_service=service,
            context_tokens=0,
            completion_tokens=0,
            user_text=user_text,
        )

    def _run_video_tool(
        self,
        ctx: SimpleNamespace,
        args: dict[str, Any],
        session: dict[str, Any] | None,
        service: Any,
    ):
        with (
            patch(
                "core.tool_handlers.media.get_user_image_edit_session",
                new=AsyncMock(return_value=session),
            ),
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
            return asyncio.run(handle_generate_video(ctx, args))

    def test_video_tool_passes_selected_reference_images(self) -> None:
        session = {"images": [_image(1), _image(2), _image(3)], "history": []}
        video = SimpleNamespace(cleanup=Mock())
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        ctx = self._video_context(
            124,
            service,
            "Оживи картинки 1 и 3 на 10 секунд",
        )

        result = self._run_video_tool(
            ctx,
            {"prompt": "animate selected", "image_indices": [1, 3]},
            session,
            service,
        )

        passed_options = service.generate_video.await_args.args[1]
        passed_images = service.generate_video.await_args.kwargs["reference_images"]
        self.assertEqual(passed_options.duration, 10)
        self.assertEqual(passed_options.model, settings.media.video_model)
        self.assertEqual(passed_options.resolution, settings.media.video_resolution)
        self.assertEqual(
            [image.content for image in passed_images],
            [
                base64.b64decode(_image(1)["image"]),
                base64.b64decode(_image(3)["image"]),
            ],
        )
        self.assertTrue(result.output["ok"])
        video.cleanup.assert_called_once_with()

    def test_video_tool_keeps_user_order_of_references(self) -> None:
        session = {"images": [_image(1), _image(2), _image(3)], "history": []}
        video = SimpleNamespace(cleanup=Mock())
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        ctx = self._video_context(
            132,
            service,
            "Оживи картинки 3, 1, 2 на 10 секунд",
        )

        result = self._run_video_tool(
            ctx,
            {"prompt": "animate selected", "image_indices": [1, 2, 3]},
            session,
            service,
        )

        passed_images = service.generate_video.await_args.kwargs["reference_images"]
        self.assertEqual(
            [image.content for image in passed_images],
            [
                base64.b64decode(_image(3)["image"]),
                base64.b64decode(_image(1)["image"]),
                base64.b64decode(_image(2)["image"]),
            ],
        )
        self.assertTrue(result.output["ok"])

    def test_video_tool_deduplicates_repeated_reference_numbers(self) -> None:
        session = {"images": [_image(1), _image(2), _image(3)], "history": []}
        video = SimpleNamespace(cleanup=Mock())
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        ctx = self._video_context(
            133,
            service,
            "Оживи картинки 3, 1, 3, 2 на 10 секунд",
        )

        result = self._run_video_tool(
            ctx,
            {"prompt": "animate selected", "image_indices": [3, 1, 3, 2]},
            session,
            service,
        )

        passed_images = service.generate_video.await_args.kwargs["reference_images"]
        self.assertEqual(
            [image.content for image in passed_images],
            [
                base64.b64decode(_image(3)["image"]),
                base64.b64decode(_image(1)["image"]),
                base64.b64decode(_image(2)["image"]),
            ],
        )
        self.assertTrue(result.output["ok"])

    def test_video_tool_rejects_reference_limit_without_truncation(self) -> None:
        max_references = settings.media.video_max_references
        image_count = max_references + 2
        session = {
            "images": [_image(index) for index in range(1, image_count + 1)],
            "history": [],
        }
        video = SimpleNamespace(cleanup=Mock())
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        ctx = self._video_context(
            129,
            service,
            "Оживи все картинки на 5 секунд",
        )

        result = self._run_video_tool(ctx, {"prompt": "animate"}, session, service)

        service.generate_video.assert_not_awaited()
        self.assertFalse(result.output["ok"])
        self.assertIn("Уменьшите набор явно", result.output["message"])

    def test_video_tool_requires_explicit_duration(self) -> None:
        session = {"images": [_image(1)], "history": []}
        video = SimpleNamespace(cleanup=Mock())
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        ctx = self._video_context(131, service, "Оживи картинку")

        result = self._run_video_tool(ctx, {"prompt": "animate"}, session, service)

        service.generate_video.assert_not_awaited()
        self.assertFalse(result.output["ok"])
        self.assertTrue(result.output["requires_duration"])

    def test_video_tool_ignores_duration_invented_by_the_model(self) -> None:
        session = {"images": [_image(1)], "history": []}
        video = SimpleNamespace(cleanup=Mock())
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        ctx = self._video_context(134, service, "Оживи картинку")

        result = self._run_video_tool(
            ctx,
            {"prompt": "animate", "duration": 15},
            session,
            service,
        )

        service.generate_video.assert_not_awaited()
        self.assertFalse(result.output["ok"])
        self.assertTrue(result.output["requires_duration"])

    def test_video_tool_rejects_unsupported_explicit_duration(self) -> None:
        session = {"images": [_image(1)], "history": []}
        video = SimpleNamespace(cleanup=Mock())
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        ctx = self._video_context(135, service, "Оживи картинку на 45 секунд")

        result = self._run_video_tool(ctx, {"prompt": "animate"}, session, service)

        service.generate_video.assert_not_awaited()
        self.assertFalse(result.output["ok"])
        self.assertIn("45", result.output["message"])

    def test_video_tool_asks_duration_for_ambiguous_request(self) -> None:
        session = {"images": [_image(1)], "history": []}
        video = SimpleNamespace(cleanup=Mock())
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        ctx = self._video_context(136, service, "Оживи картинку примерно на 10 секунд")

        result = self._run_video_tool(ctx, {"prompt": "animate"}, session, service)

        service.generate_video.assert_not_awaited()
        self.assertFalse(result.output["ok"])
        self.assertTrue(result.output["requires_duration"])

    def test_video_tool_rejects_invalid_reference_numbers(self) -> None:
        session = {"images": [_image(1)], "history": []}
        video = SimpleNamespace(cleanup=Mock())
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        ctx = self._video_context(137, service, "Оживи картинку на 10 секунд")

        result = self._run_video_tool(
            ctx,
            {"prompt": "animate", "image_indices": [0]},
            session,
            service,
        )

        service.generate_video.assert_not_awaited()
        self.assertFalse(result.output["ok"])

    def test_video_tool_rejects_boolean_and_fractional_reference_numbers(self) -> None:
        session = {"images": [_image(1)], "history": []}
        video = SimpleNamespace(cleanup=Mock())
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        ctx = self._video_context(138, service, "Оживи картинку на 10 секунд")

        for indices in ([True], [1.5], ["1"]):
            with self.subTest(indices=indices):
                result = self._run_video_tool(
                    ctx,
                    {"prompt": "animate", "image_indices": indices},
                    session,
                    service,
                )

                service.generate_video.assert_not_awaited()
                self.assertFalse(result.output["ok"])

    def test_video_tool_rejects_corrupted_reference_image(self) -> None:
        broken = _image(1) | {"image": "not-a-valid-base64-image!!"}
        session = {"images": [broken], "history": []}
        video = SimpleNamespace(cleanup=Mock())
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        ctx = self._video_context(139, service, "Оживи картинку на 10 секунд")

        result = self._run_video_tool(ctx, {"prompt": "animate"}, session, service)

        service.generate_video.assert_not_awaited()
        self.assertFalse(result.output["ok"])

    def test_video_tool_without_images_uses_text_mode(self) -> None:
        video = SimpleNamespace(cleanup=Mock())
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        ctx = self._video_context(140, service, "Сделай видео на 10 секунд")

        result = self._run_video_tool(ctx, {"prompt": "animate"}, None, service)

        passed_images = service.generate_video.await_args.kwargs["reference_images"]
        self.assertEqual(list(passed_images), [])
        self.assertTrue(result.output["ok"])

    def test_video_tool_result_reports_actual_profile_and_references(self) -> None:
        """Структурированный результат содержит фактические параметры профиля."""

        session = {"images": [_image(1), _image(2), _image(3)], "history": []}
        video = SimpleNamespace(
            cleanup=Mock(),
            cost=0.31,
            metadata={"job_id": "job-42"},
        )
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        ctx = self._video_context(144, service, "Оживи картинки 3, 1, 2 на 10 секунд")

        result = self._run_video_tool(
            ctx,
            {"prompt": "animate", "image_indices": [1, 2, 3]},
            session,
            service,
        )

        output = result.output
        self.assertEqual(output["model"], settings.media.video_model)
        self.assertEqual(output["resolution"], settings.media.video_resolution)
        self.assertEqual(output["duration"], 10)
        self.assertEqual(output["reference_indices"], [3, 1, 2])
        self.assertEqual(output["job_id"], "job-42")
        self.assertEqual(output["cost"], 0.31)
        self.assertFalse(output["billing_unknown"])
        # Бинарное содержимое картинок и data URL в результат не попадают.
        serialized = json.dumps(output, default=str)
        self.assertNotIn("data:image", serialized)
        self.assertNotIn("base64", serialized)

    def test_video_tool_result_marks_unknown_cost(self) -> None:
        session = {"images": [_image(1)], "history": []}
        video = SimpleNamespace(cleanup=Mock(), cost=None, metadata={})
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        ctx = self._video_context(145, service, "Оживи картинку на 10 секунд")

        result = self._run_video_tool(
            ctx,
            {"prompt": "animate"},
            session,
            service,
        )

        self.assertIsNone(result.output["cost"])
        self.assertTrue(result.output["billing_unknown"])
        self.assertIsNone(result.output["job_id"])

    def test_video_delivery_failure_does_not_trigger_regeneration(self) -> None:
        session = {"images": [_image(1)], "history": []}
        video = SimpleNamespace(cleanup=Mock(), cost=None, metadata={})
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        ctx = self._video_context(146, service, "Оживи картинку на 10 секунд")

        with (
            patch(
                "core.tool_handlers.media.get_user_image_edit_session",
                new=AsyncMock(return_value=session),
            ),
            patch(
                "core.tool_handlers.media.reply_service_message",
                new=AsyncMock(return_value=SimpleNamespace(edit_text=AsyncMock())),
            ),
            patch("core.tool_handlers.media.animate_service_message", new=AsyncMock()),
            patch(
                "core.tool_handlers.media.send_video_to_telegram",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "core.tool_handlers.media.add_user_image_session_history",
                new=AsyncMock(),
            ),
        ):
            result = asyncio.run(
                handle_generate_video(ctx, {"prompt": "animate"})
            )

        self.assertEqual(service.generate_video.await_count, 1)
        self.assertFalse(result.output["ok"])
        self.assertEqual(result.output["delivery"], "failed")
        video.cleanup.assert_called_once_with()


class VideoDraftSnapshotTests(unittest.TestCase):
    """Снимок заявки не меняется от новых загрузок и инвалидируется сбросом."""

    def _prepare(
        self,
        user_id: int,
        session: dict[str, Any] | None,
        user_text: str,
    ):
        from core.tool_handlers.media import _prepare_video_draft

        service = SimpleNamespace(generate_video=AsyncMock())
        ctx = _media_context(user_id, service, user_text)
        with patch(
            "core.tool_handlers.media.get_user_image_edit_session",
            new=AsyncMock(return_value=session),
        ):
            preparation = asyncio.run(
                _prepare_video_draft(
                    ctx,
                    {"prompt": "animate", "image_indices": [1]},
                )
            )
        return preparation

    def test_new_upload_does_not_change_pending_draft(self) -> None:
        session = {"images": [_image(1)], "history": []}
        preparation = self._prepare(141, session, "Оживи картинку")

        self.assertIsNotNone(preparation.draft)
        draft = preparation.draft
        assert draft is not None
        self.assertEqual(draft.reference_indices, (1,))

        # Новая загрузка пополняет сессию, но не меняет уже созданную заявку.
        append_image_to_session(session, _image(2), "uploaded")

        self.assertEqual(draft.reference_indices, (1,))
        self.assertEqual(len(draft.reference_images), 1)
        self.assertEqual(
            [image["image"] for image in get_session_images(session)],
            [_image(1)["image"], _image(2)["image"]],
        )

    def test_session_numbers_do_not_change_during_waiting(self) -> None:
        session = {"images": [_image(1)], "history": []}
        preparation = self._prepare(142, session, "Оживи картинку")

        draft = preparation.draft
        assert draft is not None
        append_image_to_session(session, _image(2), "uploaded")

        # Номера уже выбранных картинок в сессии остаются прежними.
        self.assertEqual(draft.reference_indices, (1,))
        self.assertEqual(len(get_session_images(session)), 2)

    def test_reset_invalidates_pending_video_draft(self) -> None:
        user_id = 143
        session = {"images": [_image(1)], "history": []}
        preparation = self._prepare(user_id, session, "Оживи картинку")

        draft = preparation.draft
        assert draft is not None
        self.assertIsNotNone(draft.operation_id)

        async def scenario() -> None:
            await clear_user_image_edit_session(user_id)
            with (
                patch("core.state_and_commands.reply_service_text", new=AsyncMock()),
                patch("core.state_and_commands.Path"),
            ):
                await reset(
                    SimpleNamespace(effective_user=SimpleNamespace(id=user_id)),
                    SimpleNamespace(),
                )

        asyncio.run(scenario())

        self.assertIsNone(asyncio.run(get_user_image_edit_session(user_id)))
        self.assertEqual(get_session_images(session), [_image(1)])

    def test_replaced_profile_changes_validation_and_question_options(self) -> None:
        from core.tool_handlers.media import build_duration_question_request
        from core.video_requests import VideoModelProfile

        profile = VideoModelProfile(
            model="vendor/other-video-model",
            resolution="720p",
            allowed_durations=(4, 8),
            duration_presets=(4, 8),
            allowed_aspect_ratios=("16:9",),
            max_references=2,
        )

        request = build_duration_question_request(profile)

        self.assertEqual(
            [option.payload for option in request.options],
            [4, 8],
        )
        with self.assertRaises(ValueError):
            validate_duration(12, profile)
        self.assertEqual(validate_duration(8, profile), 8)

    def test_replaced_video_profile_flows_through_question_continuation_and_delivery(self) -> None:
        """Профиль меняется данными, а вопросы, continuation и доставка остаются общими."""

        from openai.types.responses import Response, ResponseOutputMessage, ResponseOutputText

        from core.common_types import (
            PendingToolCall,
            QuestionAnswer,
            QuestionAnswerKind,
            QuestionAnswerMode,
        )
        from core.openai_api import resume_model_answer
        from core.question_state import QuestionContinuationContext
        from core.tool_handlers.media import (
            _prepare_video_draft,
            check_generate_video,
        )

        new_model = "vendor/compatible-video-model"
        new_media = replace(
            settings.media,
            video_model=new_model,
            video_resolution="720p",
            video_aspect_ratio="16:9",
            video_allowed_durations=(4, 8),
            video_duration_presets=(4, 8),
            video_allowed_aspect_ratios=("16:9",),
            video_max_references=2,
        )
        replacement_settings = replace(settings, media=new_media).validate()
        session = {"images": [_image(1), _image(2), _image(3)], "history": []}
        service = SimpleNamespace(generate_video=AsyncMock())
        ctx = _media_context(
            147,
            service,
            "Оживи картинки 3, 1",
        )

        async def get_session(_user_id: int) -> dict[str, Any]:
            return session

        with (
            patch("core.tool_handlers.media.settings", replacement_settings),
            patch("core.tool_handlers.media.get_user_image_edit_session", new=AsyncMock(side_effect=get_session)),
        ):
            check = asyncio.run(
                check_generate_video(
                    cast(Any, ctx),
                    {"prompt": "animate", "image_indices": [1, 2]},
                )
            )
            profile = replacement_settings.video_profile
            self.assertIsNotNone(check)
            assert check is not None and check.video_request is not None
            draft = check.video_request

            self.assertEqual(profile.model, new_model)
            self.assertEqual(profile.resolution, "720p")
            self.assertEqual([option.payload for option in check.request.options], [4, 8])
            self.assertEqual(draft.profile, profile)
            self.assertEqual(draft.reference_indices, (3, 1))
            self.assertEqual(
            [image.content for image in draft.reference_images],
            [
                    base64.b64decode(cast(str, _image(3)["image"])),
                    base64.b64decode(cast(str, _image(1)["image"])),
                ],
            )
            with self.assertRaises(ValueError):
                validate_duration(10, profile)
            self.assertEqual(validate_duration(8, profile), 8)

            over_limit = asyncio.run(
                _prepare_video_draft(
                    _media_context(148, service, "Оживи все картинки"),
                    {"prompt": "animate"},
                )
            )
            self.assertIsNone(over_limit.draft)
            self.assertIn("не более 2", over_limit.error or "")

            video = SimpleNamespace(
                cleanup=Mock(),
                cost=0.44,
                metadata={"job_id": "profile-job"},
            )
            service.generate_video.return_value = video
            continuation = QuestionContinuationContext(
                response_id="resp-profile",
                tool_calls=(
                    PendingToolCall(
                        call_id="call-profile-video",
                        name="generate_video",
                        arguments={"prompt": "animate"},
                    ),
                ),
                model_name="gpt-5.6-terra",
                original_user_text="Оживи картинки 3, 1",
                context_tokens=120,
                completion_tokens=24,
                tool_round=1,
                video_request=draft,
                video_operation_id=draft.operation_id,
                answer_mode=QuestionAnswerMode.VIDEO_REQUEST,
            )
            answer = QuestionAnswer(
                question_id="profile-question",
                kind=QuestionAnswerKind.OPTION,
                text="8 с",
                option_id="duration_8",
                payload=8,
            )
            final_response = Response.model_construct(
                id="resp-profile-final",
                object="response",
                created_at=0,
                status="completed",
                error=None,
                incomplete_details=None,
                instructions=None,
                max_output_tokens=None,
                model="gpt-5.6-terra",
                output=[
                    ResponseOutputMessage(
                        id="msg-profile-final",
                        content=[
                            ResponseOutputText(
                                annotations=[],
                                text="Видео готово",
                                type="output_text",
                            )
                        ],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                ],
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
            update = SimpleNamespace(effective_user=SimpleNamespace(id=147))

            with (
                patch(
                    "core.openai_api.get_simple_answer",
                    new=AsyncMock(return_value=final_response),
                ) as get_answer,
                patch(
                    "core.tool_handlers.media.reply_service_message",
                    new=AsyncMock(return_value=SimpleNamespace(edit_text=AsyncMock())),
                ),
                patch("core.tool_handlers.media.animate_service_message", new=AsyncMock()),
                patch(
                    "core.tool_handlers.media.send_video_to_telegram",
                    new=AsyncMock(return_value=True),
                ) as deliver,
                patch(
                    "core.tool_handlers.media.add_user_image_session_history",
                    new=AsyncMock(),
                ),
            ):
                result = asyncio.run(
                    resume_model_answer(
                        cast(Any, update),
                        cast(Any, SimpleNamespace()),
                        continuation,
                        answer,
                        openrouter_service=cast(Any, service),
                        question=check.request,
                    )
                )

            self.assertEqual(result.bot_reply, "Видео готово")
            self.assertEqual(get_answer.await_count, 1)
            self.assertEqual(
                get_answer.await_args.kwargs["previous_response_id"],
                "resp-profile",
            )
            service.generate_video.assert_awaited_once()
            generated_options = service.generate_video.await_args.args[1]
            self.assertEqual(generated_options.model, new_model)
            self.assertEqual(generated_options.resolution, "720p")
            self.assertEqual(generated_options.duration, 8)
            self.assertEqual(
                [image.content for image in service.generate_video.await_args.kwargs["reference_images"]],
                [
                    base64.b64decode(cast(str, _image(3)["image"])),
                    base64.b64decode(cast(str, _image(1)["image"])),
                ],
            )
            deliver.assert_awaited_once()
            video.cleanup.assert_called_once_with()


class ImageHandlerRoutingAndObservabilityTests(unittest.TestCase):
    def _call_tool(
        self,
        tool: str,
        *,
        user_id: int,
        args: dict[str, Any],
        user_text: str,
        session: dict[str, Any] | None,
        image: GeneratedImage,
    ):
        service = SimpleNamespace(
            generate=AsyncMock(return_value=image),
            edit=AsyncMock(return_value=image),
        )
        ctx = _media_context(user_id, service, user_text)
        status = SimpleNamespace(delete=AsyncMock())
        append = AsyncMock(return_value=1)
        update = ctx.update

        with (
            patch("core.tool_handlers.media.settings", _auto_settings()),
            patch(
                "core.tool_handlers.media.get_user_image_edit_session",
                new=AsyncMock(return_value=session),
            ),
            patch(
                "core.tool_handlers.media.reply_service_message",
                new=AsyncMock(return_value=status),
            ) as send_status,
            patch("core.tool_handlers.media.animate_service_message", new=AsyncMock()),
            patch(
                "core.tool_handlers.media.send_image_to_telegram",
                new=AsyncMock(return_value=("normalized-image", 4, 3, 4 / 3)),
            ) as send_image,
            patch(
                "core.tool_handlers.media.append_user_image_to_session",
                new=append,
            ),
        ):
            if tool == "generate":
                result = asyncio.run(handle_generate_image(ctx, args))
            else:
                result = asyncio.run(handle_generate_image_from_image(ctx, args))

        return result, service, append, status, send_status, send_image, update

    def test_auto_text_to_image_routes_to_klein_1k_and_saves_provenance(self) -> None:
        image = _successful_generated_image(settings.media.image_generation_fast_model)
        result, service, append, status, send_status, send_image, update = self._call_tool(
            "generate",
            user_id=50_001,
            args={"prompt": "A simple blue ceramic teapot"},
            user_text="Нарисуй простой синий чайник",
            session=None,
            image=image,
        )

        options = service.generate.await_args.args[1]
        saved_generation = append.await_args.args[1]["generation"]
        self.assertEqual(options.model, settings.media.image_generation_fast_model)
        self.assertEqual(options.resolution, "1K")
        self.assertEqual(result.output["quality"], "standard")
        self.assertEqual(saved_generation["quality"], "standard")
        self.assertEqual(saved_generation["model"], settings.media.image_generation_fast_model)
        self.assertEqual(saved_generation["requested_model"], settings.media.image_generation_fast_model)
        self.assertEqual(saved_generation["resolution"], "1K")
        self.assertEqual(saved_generation["route_reason"], "standard_default")
        self.assertEqual(saved_generation["total_known_cost"], 0.2)
        self.assertFalse(saved_generation["billing_unknown"])
        self.assertEqual(saved_generation["operation"], "generate")
        self.assertEqual(saved_generation["source_image_indices"], [])
        self.assertTrue(saved_generation["operation_id"])
        self.assertTrue(result.output["delivered"])
        send_image.assert_awaited_once()
        send_status.assert_awaited_once_with(update, "Изображение генерируется, подождите")
        status.delete.assert_awaited_once_with()

    def test_text_marker_or_quotes_override_standard_quality_hint(self) -> None:
        for user_text, prompt in (
            ("Добавь надпись на плакат", "A colorful poster"),
            ("Нарисуй вывеску", 'A storefront sign reading "OPEN"'),
        ):
            with self.subTest(user_text=user_text):
                image = _successful_generated_image(settings.media.image_generation_model)
                result, service, _, _, _, _, _ = self._call_tool(
                    "generate",
                    user_id=50_002,
                    args={"prompt": prompt, "quality": "standard"},
                    user_text=user_text,
                    session=None,
                    image=image,
                )
                options = service.generate.await_args.args[1]
                self.assertEqual(options.model, settings.media.image_generation_model)
                self.assertEqual(result.output["quality"], "pro")

    def test_edit_route_counts_selected_references_after_selection(self) -> None:
        session = {"images": [_image(index) for index in range(1, 6)], "history": []}
        for indices, expected_model, expected_count in (
            ([1, 2], settings.media.image_edit_fast_model, 2),
            ([1, 3, 5], settings.media.image_edit_model, 3),
        ):
            with self.subTest(indices=indices):
                image = _successful_generated_image(expected_model)
                result, service, append, _, _, _, _ = self._call_tool(
                    "edit",
                    user_id=50_003,
                    args={"prompt": "Change the background color", "image_indices": indices},
                    user_text="Измени цвет фона",
                    session=session,
                    image=image,
                )
                options = service.edit.await_args.args[2]
                self.assertEqual(len(service.edit.await_args.args[0]), expected_count)
                self.assertEqual(options.model, expected_model)
                self.assertEqual(options.resolution, "1K")
                self.assertEqual(
                    append.await_args.args[1]["generation"]["source_image_indices"],
                    indices,
                )
                self.assertTrue(result.output["ok"])

    def test_stage_eight_edit_fixture_covers_reference_counts_and_aspect_ratios(self) -> None:
        session = {"images": [_image(index) for index in range(1, 11)], "history": []}
        aspect_ratios = ("9:16", "1:1", "16:9", "21:9")

        for reference_count in (1, 2, 3, 10):
            indices = list(range(1, reference_count + 1))
            expected_quality = "standard" if reference_count <= 2 else "pro"
            expected_model = (
                settings.media.image_edit_fast_model
                if expected_quality == "standard"
                else settings.media.image_edit_model
            )
            for aspect_ratio in aspect_ratios:
                with self.subTest(reference_count=reference_count, aspect_ratio=aspect_ratio):
                    result, service, append, _, _, _, _ = self._call_tool(
                        "edit",
                        user_id=51_000 + reference_count,
                        args={
                            "prompt": "Change only the background color",
                            "image_indices": indices,
                            "aspect_ratio": aspect_ratio,
                        },
                        user_text="Измени только цвет фона",
                        session=session,
                        image=_successful_generated_image(expected_model),
                    )

                    passed_images = service.edit.await_args.args[0]
                    options = service.edit.await_args.args[2]
                    saved_generation = append.await_args.args[1]["generation"]
                    self.assertEqual(len(passed_images), reference_count)
                    self.assertEqual(options.model, expected_model)
                    self.assertEqual(options.resolution, "1K")
                    self.assertEqual(options.aspect_ratio, aspect_ratio)
                    self.assertEqual(saved_generation["source_image_indices"], indices)
                    self.assertEqual(saved_generation["aspect_ratio"], aspect_ratio)
                    self.assertEqual(result.output["quality"], expected_quality)
                    self.assertTrue(result.output["ok"])

    def test_ordinary_edit_without_indices_keeps_all_session_images(self) -> None:
        session = {"images": [_image(index) for index in range(1, 4)], "history": []}
        image = _successful_generated_image(settings.media.image_edit_model)
        result, service, append, _, _, _, _ = self._call_tool(
            "edit",
            user_id=50_022,
            args={"prompt": "Blend the three reference photos into one scene"},
            user_text="Объедини загруженные изображения",
            session=session,
            image=image,
        )

        self.assertEqual(len(service.edit.await_args.args[0]), 3)
        self.assertEqual(
            append.await_args.args[1]["generation"]["source_image_indices"],
            [1, 2, 3],
        )
        self.assertEqual(service.edit.await_args.args[2].model, settings.media.image_edit_model)
        self.assertTrue(result.output["ok"])

    def test_complaint_retry_is_pro_but_unrelated_generation_returns_to_klein(self) -> None:
        session = {
            "images": [
                _image(1)
                | {
                    "source_kind": "text2image",
                    "generation": {
                        "operation_id": "0123456789abcdef0123456789abcdef",
                        "operation": "generate",
                        "model": settings.media.image_generation_fast_model,
                        "quality": "standard",
                        "prompt": "A red fox in a snowy forest",
                        "source_image_indices": [],
                    },
                },
                _image(2) | {"source_kind": "uploaded"},
            ],
            "history": [],
        }
        service = SimpleNamespace(
            generate=AsyncMock(
                side_effect=[
                    _successful_generated_image(settings.media.image_generation_model),
                    _successful_generated_image(settings.media.image_generation_fast_model),
                ]
            ),
            edit=AsyncMock(),
        )
        ctx = _media_context(50_004, service)
        status = SimpleNamespace(delete=AsyncMock())

        async def call(text: str, prompt: str):
            ctx.user_text = text
            with (
                patch("core.tool_handlers.media.settings", _auto_settings()),
                patch(
                    "core.tool_handlers.media.get_user_image_edit_session",
                    new=AsyncMock(return_value=session),
                ),
                patch(
                    "core.tool_handlers.media.reply_service_message",
                    new=AsyncMock(return_value=status),
                ),
                patch("core.tool_handlers.media.animate_service_message", new=AsyncMock()),
                patch(
                    "core.tool_handlers.media.send_image_to_telegram",
                    new=AsyncMock(return_value=("normalized", 4, 3, 4 / 3)),
                ),
                patch(
                    "core.tool_handlers.media.append_user_image_to_session",
                    new=AsyncMock(return_value=2),
                ),
            ):
                return await handle_generate_image(ctx, {"prompt": prompt, "quality": "standard"})

        first = asyncio.run(call("Плохо получилось, переделай заново", "Keep the fox and scarf"))
        second = asyncio.run(call("Нарисуй новый простой синий шар", "A simple blue sphere"))

        self.assertEqual(
            [call.args[1].model for call in service.generate.await_args_list],
            [settings.media.image_generation_model, settings.media.image_generation_fast_model],
        )
        self.assertEqual(first.output["quality"], "pro")
        self.assertEqual(second.output["quality"], "standard")
        self.assertIn("A red fox in a snowy forest", service.generate.await_args_list[0].args[0])
        self.assertIn("Keep the fox and scarf", service.generate.await_args_list[0].args[0])
        self.assertNotIn("Плохо получилось", service.generate.await_args_list[0].args[0])
        self.assertTrue(first.output["ok"] and second.output["ok"])

    def test_ordinary_color_change_does_not_upgrade_because_result_is_generated(self) -> None:
        session = {
            "images": [
                _image(1)
                | {
                    "source_kind": "text2image",
                    "generation": {
                        "operation": "generate",
                        "model": settings.media.image_generation_fast_model,
                        "quality": "standard",
                        "prompt": "A small blue house",
                        "source_image_indices": [],
                    },
                }
            ],
            "history": [],
        }
        image = _successful_generated_image(settings.media.image_edit_fast_model)
        result, service, _, _, _, _, _ = self._call_tool(
            "edit",
            user_id=50_005,
            args={"prompt": "Change the roof to green"},
            user_text="Поменяй цвет крыши на зелёный",
            session=session,
            image=image,
        )

        self.assertEqual(len(service.edit.await_args.args[0]), 1)
        self.assertEqual(service.edit.await_args.args[2].model, settings.media.image_edit_fast_model)
        self.assertEqual(result.output["quality"], "standard")

    def test_mistaken_edit_tool_for_generation_retry_is_redirected_without_bad_reference(self) -> None:
        session = {
            "images": [
                _image(1)
                | {
                    "source_kind": "text2image",
                    "generation": {
                        "operation_id": "1123456789abcdef0123456789abcdef",
                        "operation": "generate",
                        "model": settings.media.image_generation_fast_model,
                        "quality": "standard",
                        "prompt": "A red fox in a snowy forest",
                        "source_image_indices": [],
                    },
                }
            ],
            "history": [],
        }
        image = _successful_generated_image(settings.media.image_generation_model)
        result, service, append, _, _, _, _ = self._call_tool(
            "edit",
            user_id=50_006,
            args={"prompt": "A red fox with a clearly visible scarf"},
            user_text="Плохо получилось, переделай заново",
            session=session,
            image=image,
        )

        service.edit.assert_not_awaited()
        service.generate.assert_awaited_once()
        self.assertEqual(service.generate.await_args.args[1].model, settings.media.image_generation_model)
        self.assertIn("A red fox in a snowy forest", service.generate.await_args.args[0])
        self.assertEqual(append.await_args.args[1]["generation"]["source_image_indices"], [])
        self.assertTrue(result.output["ok"])

    def test_mistaken_generation_tool_for_edit_retry_restores_original_sources(self) -> None:
        session = {
            "images": [
                _image(1) | {"source_kind": "uploaded"},
                _image(2)
                | {
                    "source_kind": "edit",
                    "generation": {
                        "operation_id": "2123456789abcdef0123456789abcdef",
                        "operation": "edit",
                        "model": settings.media.image_edit_fast_model,
                        "quality": "standard",
                        "prompt": "Turn the portrait into a watercolor",
                        "source_image_indices": [1],
                    },
                },
                _image(3) | {"source_kind": "uploaded"},
            ],
            "history": [],
        }
        image = _successful_generated_image(settings.media.image_edit_model)
        result, service, append, _, _, _, _ = self._call_tool(
            "generate",
            user_id=50_007,
            args={"prompt": "Make the watercolor cleaner"},
            user_text="Плохо получилось, переделай редактирование",
            session=session,
            image=image,
        )

        service.generate.assert_not_awaited()
        service.edit.assert_awaited_once()
        self.assertEqual(len(service.edit.await_args.args[0]), 1)
        self.assertEqual(service.edit.await_args.args[0][0].content, base64.b64decode(_image(1)["image"]))
        self.assertIn("Turn the portrait into a watercolor", service.edit.await_args.args[1])
        self.assertEqual(append.await_args.args[1]["generation"]["source_image_indices"], [1])
        self.assertEqual(service.edit.await_args.args[2].model, settings.media.image_edit_model)
        self.assertTrue(result.output["ok"])

    def test_explicit_nonlatest_reference_wins_over_newer_upload(self) -> None:
        session = {
            "images": [
                _image(1) | {"source_kind": "uploaded"},
                _image(2)
                | {
                    "source_kind": "text2image",
                    "generation": {
                        "operation": "generate",
                        "model": settings.media.image_generation_fast_model,
                        "quality": "standard",
                        "prompt": "A generated castle",
                        "source_image_indices": [],
                    },
                },
                _image(3) | {"source_kind": "uploaded"},
            ],
            "history": [],
        }
        image = _successful_generated_image(settings.media.image_edit_fast_model)
        _, service, append, _, _, _, _ = self._call_tool(
            "edit",
            user_id=50_008,
            args={"prompt": "Change the selected image background"},
            user_text="Измени фон изображения №1",
            session=session,
            image=image,
        )

        self.assertEqual(len(service.edit.await_args.args[0]), 1)
        self.assertEqual(service.edit.await_args.args[0][0].content, base64.b64decode(_image(1)["image"]))
        self.assertEqual(append.await_args.args[1]["generation"]["source_image_indices"], [1])

    def test_missing_retry_sources_and_legacy_retry_are_not_silently_substituted(self) -> None:
        unavailable_session = {
            "images": [
                _image(1)
                | {
                    "source_kind": "edit",
                    "generation": {
                        "operation": "edit",
                        "quality": "standard",
                        "model": settings.media.image_edit_fast_model,
                        "prompt": "Original edit",
                        "source_image_indices": [8],
                    },
                },
                _image(2) | {"source_kind": "uploaded"},
            ],
            "history": [],
        }
        _, service, _, _, _, _, _ = self._call_tool(
            "edit",
            user_id=50_009,
            args={"prompt": "Try again"},
            user_text="Плохой результат, переделай",
            session=unavailable_session,
            image=_successful_generated_image(settings.media.image_edit_model),
        )
        service.edit.assert_not_awaited()

        legacy_session = {
            "images": [
                _image(1)
                | {
                    "source_kind": "edit",
                    "generation": {"operation": "edit", "prompt": "Legacy operation"},
                }
            ],
            "history": [],
        }
        result, legacy_service, _, _, _, _, _ = self._call_tool(
            "edit",
            user_id=50_010,
            args={"prompt": "Try again"},
            user_text="Плохой результат, повтори",
            session=legacy_session,
            image=_successful_generated_image(settings.media.image_edit_model),
        )
        legacy_service.edit.assert_not_awaited()
        self.assertFalse(result.output["ok"])
        self.assertIn("нет данных об исходниках", result.output["message"])

        old_generated_result = {
            "images": [_image(3) | {"source_kind": "generated"}],
            "history": [],
        }
        old_result, old_service, _, _, _, _, _ = self._call_tool(
            "edit",
            user_id=50_023,
            args={"prompt": "Try again"},
            user_text="Плохой результат, повтори",
            session=old_generated_result,
            image=_successful_generated_image(settings.media.image_edit_model),
        )
        old_service.edit.assert_not_awaited()
        self.assertFalse(old_result.output["ok"])

    def test_delivery_failure_logs_cost_once_and_removes_single_status(self) -> None:
        image = _successful_generated_image(settings.media.image_generation_fast_model, cost=0.27)
        service = SimpleNamespace(generate=AsyncMock(return_value=image), edit=AsyncMock())
        ctx = _media_context(50_011, service, "Сгенерируй секретный-пrompt-456")
        status = SimpleNamespace(delete=AsyncMock())

        with (
            self.assertLogs("core.image_observability", level="INFO") as logs,
            patch("core.tool_handlers.media.settings", _auto_settings()),
            patch(
                "core.tool_handlers.media.get_user_image_edit_session",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "core.tool_handlers.media.reply_service_message",
                new=AsyncMock(return_value=status),
            ) as send_status,
            patch("core.tool_handlers.media.animate_service_message", new=AsyncMock()),
            patch(
                "core.tool_handlers.media.send_image_to_telegram",
                new=AsyncMock(side_effect=RuntimeError("telegram delivery failed")),
            ),
            patch(
                "core.tool_handlers.media.append_user_image_to_session",
                new=AsyncMock(),
            ) as append,
        ):
            result = asyncio.run(
                handle_generate_image(ctx, {"prompt": "secret prompt body must not be logged"})
            )

        log_text = "\n".join(logs.output)
        self.assertIn("image_generation_attempt", log_text)
        self.assertIn("image_operation_summary", log_text)
        self.assertIn('"cost":0.27', log_text)
        self.assertIn('"delivered":false', log_text)
        self.assertNotIn("secret prompt body", log_text)
        self.assertNotIn("secretный-prompt", log_text)
        self.assertNotIn(base64.b64encode(image.content).decode("ascii"), log_text)
        self.assertTrue(result.meta["retry_forbidden"])
        self.assertTrue(result.output["retry_forbidden"])
        self.assertIn("Не вызывай платные инструменты", result.output["instruction"])
        self.assertEqual(service.generate.await_count, 1)
        append.assert_not_awaited()
        send_status.assert_awaited_once()
        status.delete.assert_awaited_once_with()

    def test_session_write_failure_does_not_redeliver_or_regenerate(self) -> None:
        image = _successful_generated_image(settings.media.image_generation_fast_model)
        service = SimpleNamespace(generate=AsyncMock(return_value=image), edit=AsyncMock())
        ctx = _media_context(50_024, service, "Нарисуй простую сцену")
        status = SimpleNamespace(delete=AsyncMock())

        with (
            patch("core.tool_handlers.media.settings", _auto_settings()),
            patch(
                "core.tool_handlers.media.get_user_image_edit_session",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "core.tool_handlers.media.reply_service_message",
                new=AsyncMock(return_value=status),
            ),
            patch("core.tool_handlers.media.animate_service_message", new=AsyncMock()),
            patch(
                "core.tool_handlers.media.send_image_to_telegram",
                new=AsyncMock(return_value=("normalized", 4, 3, 4 / 3)),
            ) as send_image,
            patch(
                "core.tool_handlers.media.append_user_image_to_session",
                new=AsyncMock(side_effect=RuntimeError("session unavailable")),
            ),
        ):
            result = asyncio.run(
                handle_generate_image(ctx, {"prompt": "A full instruction"})
            )

        self.assertTrue(result.output["delivered"])
        self.assertIsNone(result.output["image_number"])
        self.assertTrue(result.output["retry_forbidden"])
        self.assertEqual(service.generate.await_count, 1)
        send_image.assert_awaited_once()
        status.delete.assert_awaited_once_with()

    def test_technical_fallback_metadata_reports_actual_pro_model(self) -> None:
        fast_model = settings.media.image_generation_fast_model
        pro_model = settings.media.image_generation_model
        image = GeneratedImage(
            content=_png_bytes(),
            mime_type="image/png",
            metadata={
                "model": pro_model,
                "requested_model": fast_model,
                "fallback_used": True,
                "cost": 0.41,
                "total_known_cost": 0.41,
                "billing_unknown": False,
                "attempts": [
                    {
                        "model": fast_model,
                        "status": 502,
                        "status_code": 502,
                        "duration_ms": 12.0,
                        "cost": 0.0,
                        "billing_unknown": False,
                        "error_reason": "generation_failed",
                    },
                    {
                        "model": pro_model,
                        "status": 200,
                        "status_code": 200,
                        "duration_ms": 18.0,
                        "cost": 0.41,
                        "billing_unknown": False,
                        "error_reason": None,
                    },
                ],
            },
        )
        with self.assertLogs("core.image_observability", level="INFO") as logs:
            result, service, append, status, send_status, _, update = self._call_tool(
                "generate",
                user_id=50_013,
                args={"prompt": "A simple landscape", "quality": "standard"},
                user_text="Нарисуй простой пейзаж",
                session=None,
                image=image,
            )

        generation = append.await_args.args[1]["generation"]
        self.assertEqual(service.generate.await_args.args[1].model, fast_model)
        self.assertEqual(generation["requested_model"], fast_model)
        self.assertEqual(generation["model"], pro_model)
        self.assertEqual(generation["quality"], "pro")
        self.assertTrue(generation["fallback_used"])
        self.assertEqual(result.output["actual_model"], pro_model)
        self.assertEqual(result.output["quality"], "pro")
        self.assertTrue(result.output["fallback_used"])
        event_log = "\n".join(logs.output)
        self.assertEqual(event_log.count("image_generation_attempt"), 2)
        self.assertEqual(event_log.count("image_operation_summary"), 1)
        self.assertIn(f'"requested_model":"{fast_model}"', event_log)
        self.assertIn(f'"actual_model":"{pro_model}"', event_log)
        self.assertIn('"attempt_count":2', event_log)
        self.assertIn('"fallback_used":true', event_log)
        send_status.assert_awaited_once_with(update, "Изображение генерируется, подождите")
        status.delete.assert_awaited_once_with()

    def test_explicit_request_to_improve_generated_result_uses_pro_reference(self) -> None:
        session = {
            "images": [
                _image(1)
                | {
                    "source_kind": "text2image",
                    "generation": {
                        "operation": "generate",
                        "model": settings.media.image_generation_fast_model,
                        "quality": "standard",
                        "prompt": "A glass greenhouse",
                        "source_image_indices": [],
                    },
                },
                _image(2) | {"source_kind": "uploaded"},
            ],
            "history": [],
        }
        result, service, append, _, _, _, _ = self._call_tool(
            "edit",
            user_id=50_014,
            args={"prompt": "Refine the composition"},
            user_text="Доработай именно этот последний результат, сохрани композицию",
            session=session,
            image=_successful_generated_image(settings.media.image_edit_model),
        )

        self.assertEqual(len(service.edit.await_args.args[0]), 1)
        self.assertEqual(service.edit.await_args.args[2].model, settings.media.image_edit_model)
        self.assertEqual(result.output["quality"], "pro")
        self.assertEqual(append.await_args.args[1]["generation"]["source_image_indices"], [1])

    def test_decode_failure_cost_and_safe_error_codes_are_logged(self) -> None:
        model = settings.media.image_generation_fast_model
        error = OpenRouterResponseError(
            "OpenRouter вернул некорректный media-результат",
            status_code=200,
            reason="invalid_response",
            provider_message=(
                "secret_token=do-not-log "
                "https://openrouter.test/image?token=private-url-token "
                "data:image/png;base64,private-binary"
            ),
            operation="image",
            model=model,
            requested_model=model,
            cost=0.33,
            attempt_metadata={"model": model, "operation": "generate"},
            billing_unknown=False,
            attempts=[
                {
                    "model": model,
                    "status": 200,
                    "status_code": 200,
                    "duration_ms": 33.0,
                    "cost": 0.33,
                    "billing_unknown": False,
                    "error_reason": "invalid_response",
                }
            ],
            total_known_cost=0.33,
        )
        service = SimpleNamespace(generate=AsyncMock(side_effect=error), edit=AsyncMock())
        ctx = _media_context(50_018, service, "Секретное описание изображения")
        status = SimpleNamespace(delete=AsyncMock())

        with (
            self.assertLogs("core.image_observability", level="INFO") as logs,
            patch("core.tool_handlers.media.settings", _auto_settings()),
            patch(
                "core.tool_handlers.media.get_user_image_edit_session",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "core.tool_handlers.media.reply_service_message",
                new=AsyncMock(return_value=status),
            ),
            patch("core.tool_handlers.media.animate_service_message", new=AsyncMock()),
            patch("core.tool_handlers.media.reply_service_text", new=AsyncMock()),
        ):
            result = asyncio.run(
                handle_generate_image(ctx, {"prompt": "private prompt body 789"})
            )

        log_text = "\n".join(logs.output)
        self.assertIn('"status":"error"', log_text)
        self.assertIn('"status_code":200', log_text)
        self.assertIn('"cost":0.33', log_text)
        self.assertIn('"total_known_cost":0.33', log_text)
        self.assertNotIn("private prompt", log_text)
        self.assertNotIn("Секретное описание", log_text)
        self.assertNotIn("do-not-log", log_text)
        self.assertNotIn("private-url-token", log_text)
        self.assertNotIn("private-binary", log_text)
        self.assertEqual(service.generate.await_count, 1)
        self.assertTrue(result.meta["retry_forbidden"])
        status.delete.assert_awaited_once_with()

    def test_cancelled_generation_logs_attempt_and_cleans_status(self) -> None:
        async def cancel_generation(*_args: Any, **_kwargs: Any):
            raise asyncio.CancelledError

        service = SimpleNamespace(generate=AsyncMock(side_effect=cancel_generation), edit=AsyncMock())
        ctx = _media_context(50_019, service, "Нарисуй изображение")
        status = SimpleNamespace(delete=AsyncMock())

        with (
            self.assertLogs("core.image_observability", level="INFO") as logs,
            patch("core.tool_handlers.media.settings", _auto_settings()),
            patch(
                "core.tool_handlers.media.get_user_image_edit_session",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "core.tool_handlers.media.reply_service_message",
                new=AsyncMock(return_value=status),
            ),
            patch("core.tool_handlers.media.animate_service_message", new=AsyncMock()),
        ):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(handle_generate_image(ctx, {"prompt": "A full instruction"}))

        log_text = "\n".join(logs.output)
        self.assertIn('"error_code":"cancelled"', log_text)
        self.assertIn('"delivered":false', log_text)
        status.delete.assert_awaited_once_with()
        service.generate.assert_awaited_once()

    def test_concurrent_users_keep_model_routes_and_prompts_isolated(self) -> None:
        fast_model = settings.media.image_generation_fast_model
        pro_model = settings.media.image_generation_model
        services = [
            SimpleNamespace(generate=AsyncMock(return_value=_successful_generated_image(fast_model))),
            SimpleNamespace(generate=AsyncMock(return_value=_successful_generated_image(pro_model))),
        ]
        contexts = [
            _media_context(50_020, services[0], "Нарисуй обычную гору"),
            _media_context(50_021, services[1], 'Нарисуй вывеску "СЕВЕР"'),
        ]
        status_messages = [SimpleNamespace(delete=AsyncMock()) for _ in contexts]

        async def run_both():
            with (
                patch("core.tool_handlers.media.settings", _auto_settings()),
                patch(
                    "core.tool_handlers.media.get_user_image_edit_session",
                    new=AsyncMock(return_value=None),
                ),
                patch(
                    "core.tool_handlers.media.reply_service_message",
                    new=AsyncMock(side_effect=status_messages),
                ),
                patch("core.tool_handlers.media.animate_service_message", new=AsyncMock()),
                patch(
                    "core.tool_handlers.media.send_image_to_telegram",
                    new=AsyncMock(return_value=("normalized", 4, 3, 4 / 3)),
                ),
                patch(
                    "core.tool_handlers.media.append_user_image_to_session",
                    new=AsyncMock(return_value=1),
                ),
            ):
                return await asyncio.gather(
                    handle_generate_image(contexts[0], {"prompt": "A quiet mountain"}),
                    handle_generate_image(
                        contexts[1],
                        {"prompt": 'A sign reading "СЕВЕР"', "quality": "standard"},
                    ),
                )

        results = asyncio.run(run_both())
        selected_models = [service.generate.await_args.args[1].model for service in services]
        self.assertEqual(selected_models, [fast_model, pro_model])
        self.assertEqual([result.output["quality"] for result in results], ["standard", "pro"])
        self.assertNotEqual(
            services[0].generate.await_args.args[0],
            services[1].generate.await_args.args[0],
        )
        for status_message in status_messages:
            status_message.delete.assert_awaited_once_with()

    def test_report_aggregates_quality_retries_cost_and_latency(self) -> None:
        report = aggregate_image_cost_report(
            [
                {
                    "operation_id": "operation-1",
                    "task_id": "task-1",
                    "quality": "standard",
                    "route_reason": "standard_default",
                    "delivered": True,
                    "fallback_used": False,
                    "is_complaint_retry": False,
                    "total_duration_ms": 100,
                    "result_cost": 0.2,
                    "total_known_cost": 0.2,
                    "billing_unknown": False,
                },
                {
                    "operation_id": "operation-2",
                    "task_id": "task-1",
                    "quality": "pro",
                    "route_reason": "retry_after_fast",
                    "delivered": True,
                    "fallback_used": False,
                    "is_complaint_retry": True,
                    "total_duration_ms": 300,
                    "result_cost": 0.4,
                    "total_known_cost": 0.4,
                    "billing_unknown": False,
                },
                {
                    "operation_id": "operation-3",
                    "task_id": "task-2",
                    "quality": "pro",
                    "route_reason": "text_marker",
                    "delivered": False,
                    "fallback_used": True,
                    "is_complaint_retry": False,
                    "total_duration_ms": 900,
                    "result_cost": None,
                    "total_known_cost": 0.0,
                    "billing_unknown": True,
                },
            ],
            klein_cost=0.1,
            pro_cost=1.0,
            text_context_cost=0.02,
        )

        self.assertEqual(report["klein_share"], 1 / 3)
        self.assertEqual(report["server_promotions_by_reason"]["text_marker"]["count"], 1)
        self.assertEqual(report["technical_fallback_share"], 1 / 3)
        self.assertEqual(report["complaint_retry_share"], 1 / 3)
        self.assertEqual(report["duration_ms"]["median"], 200)
        self.assertEqual(report["duration_ms"]["p95"], 290)
        self.assertAlmostEqual(report["delivered_result_cost"]["known_total"], 0.6)
        self.assertAlmostEqual(report["accepted_task_cost"]["known_subtotal_total"], 0.6)
        self.assertEqual(report["accepted_task_cost"]["accepted_task_count"], 1)
        self.assertEqual(report["all_operations_cost"]["unknown_operation_count"], 1)
        self.assertAlmostEqual(report["estimated_cost_formula"]["per_task"], 1.12)

    def test_report_separates_unknown_billing_for_delivered_accepted_task(self) -> None:
        report = aggregate_image_cost_report(
            [
                {
                    "operation_id": "operation-unknown",
                    "task_id": "task-unknown",
                    "requested_quality": "standard",
                    "quality": "standard",
                    "delivered": True,
                    "result_cost": None,
                    "total_known_cost": 0.0,
                    "billing_unknown": True,
                    "fallback_used": False,
                    "is_complaint_retry": False,
                }
            ],
            klein_cost=0.1,
            pro_cost=1.0,
        )

        self.assertEqual(report["delivered_result_cost"]["unknown_count"], 1)
        self.assertEqual(report["delivered_result_cost"]["unknown_share"], 1.0)
        self.assertEqual(report["accepted_task_cost"]["unknown_task_count"], 1)
        self.assertEqual(report["accepted_task_cost"]["unknown_task_share"], 1.0)


if __name__ == "__main__":
    unittest.main()
