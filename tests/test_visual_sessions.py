"""Проверки хранения, выбора и сброса изображений visual session."""

import asyncio
import base64
import os
import unittest
from io import BytesIO
from types import SimpleNamespace
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
from core.tool_helpers import append_image_to_session, get_session_images
from core.common_types import GeneratedImage, ModelAnswer
from core.config import settings
from core.tool_handlers.media import (
    _sdk_aspect_ratio,
    handle_generate_image,
    handle_generate_image_from_image,
    handle_generate_video,
)
from PIL import Image


def _image(number: int) -> dict[str, object]:
    return {
        "image_type": "image/png",
        "image": base64.b64encode(f"image-content-{number}".encode("ascii")).decode("ascii"),
        "width": 100,
        "height": 100,
        "aspect_ratio": 1.0,
    }


class SessionImageTests(unittest.TestCase):
    def test_append_and_default_selection_use_all_images(self) -> None:
        session: dict[str, object] = {"images": [], "history": []}

        first_number = append_image_to_session(session, _image(1), "uploaded")
        second_number = append_image_to_session(session, _image(2), "generated")

        self.assertEqual((first_number, second_number), (1, 2))
        self.assertEqual(get_session_images(session), [_image(1) | {"source_kind": "uploaded"}, _image(2) | {"source_kind": "generated"}])

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

    def test_legacy_current_image_session_remains_readable(self) -> None:
        legacy_image = _image(1)

        self.assertEqual(
            get_session_images({"current_image": legacy_image}),
            [legacy_image],
        )

    def test_concurrent_uploads_receive_unique_ordered_numbers(self) -> None:
        user_id = 9_876_543_214

        async def scenario() -> tuple[list[int], dict | None]:
            await clear_user_image_edit_session(user_id)
            numbers = await asyncio.gather(
                *[
                    append_user_image_to_session(user_id, _image(index), "uploaded")
                    for index in range(1, 21)
                ]
            )
            session = await get_user_image_edit_session(user_id)
            await clear_user_image_edit_session(user_id)
            return numbers, session

        numbers, session = asyncio.run(scenario())

        self.assertEqual(sorted(numbers), list(range(1, 21)))
        self.assertEqual(len(get_session_images(session)), 20)  # type: ignore[arg-type]


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
            ),
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

    def test_video_tool_passes_selected_reference_images(self) -> None:
        session = {"images": [_image(1), _image(2), _image(3)], "history": []}
        video = SimpleNamespace(cleanup=Mock())
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        status_message = SimpleNamespace(edit_text=AsyncMock())
        ctx = SimpleNamespace(
            update=SimpleNamespace(effective_user=SimpleNamespace(id=124)),
            openrouter_service=service,
            context_tokens=0,
            completion_tokens=0,
        )

        with (
            patch(
                "core.tool_handlers.media.get_user_image_edit_session",
                new=AsyncMock(return_value=session),
            ),
            patch(
                "core.tool_handlers.media.reply_service_message",
                new=AsyncMock(return_value=status_message),
            ),
            patch("core.tool_handlers.media.animate_service_message", new=AsyncMock()),
            patch(
                "core.tool_handlers.media.send_video_to_telegram",
                new=AsyncMock(return_value=True),
            ),
            patch("core.tool_handlers.media.add_user_image_session_history", new=AsyncMock()),
        ):
            result = asyncio.run(
                handle_generate_video(
                    ctx,
                    {"prompt": "animate selected", "image_indices": [1, 3]},
                )
            )

        passed_images = service.generate_video.await_args.args[1]
        self.assertEqual(
            [image.content for image in passed_images],
            [
                base64.b64decode(_image(1)["image"]),
                base64.b64decode(_image(3)["image"]),
            ],
        )
        self.assertTrue(result.output["ok"])
        video.cleanup.assert_called_once_with()

    def test_video_tool_limits_references_to_the_latest_ten(self) -> None:
        max_references = min(10, settings.media.max_session_images)
        image_count = max_references + 2
        session = {
            "images": [_image(index) for index in range(1, image_count + 1)],
            "history": [],
        }
        video = SimpleNamespace(cleanup=Mock())
        service = SimpleNamespace(generate_video=AsyncMock(return_value=video))
        status_message = SimpleNamespace(edit_text=AsyncMock())
        ctx = SimpleNamespace(
            update=SimpleNamespace(effective_user=SimpleNamespace(id=129)),
            openrouter_service=service,
            context_tokens=0,
            completion_tokens=0,
        )

        with (
            patch(
                "core.tool_handlers.media.get_user_image_edit_session",
                new=AsyncMock(return_value=session),
            ),
            patch(
                "core.tool_handlers.media.reply_service_message",
                new=AsyncMock(return_value=status_message),
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
            result = asyncio.run(handle_generate_video(ctx, {"prompt": "animate"}))

        passed_images = service.generate_video.await_args.args[1]
        first_reference_index = image_count - max_references + 1
        self.assertEqual(len(passed_images), max_references)
        self.assertEqual(
            [image.content for image in passed_images],
            [
                base64.b64decode(_image(index)["image"])
                for index in range(first_reference_index, image_count + 1)
            ],
        )
        self.assertTrue(result.output["ok"])
        video.cleanup.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
