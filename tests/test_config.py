"""Проверки единого загрузчика конфигурации."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from core.config import Settings, SettingsError


FLUX_MODEL = "black-forest-labs/flux-3-image"
KLEIN_MODEL = "black-forest-labs/flux-3-image-klein"
VIDEO_MODEL = "alibaba/wan-3.0-prime"
VIDEO_RESOLUTION = "480p"
VIDEO_DURATION_PRESETS = (5, 10, 15, 20, 30)
VIDEO_ASPECT_RATIOS = ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9")


class SettingsTests(unittest.TestCase):
    def _load(self, content: str, environ: dict[str, str] | None = None) -> Settings:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(content, encoding="utf-8")
            return Settings.from_yaml(path, environ=environ or {})

    def test_loads_yaml_and_secrets_from_different_sources(self) -> None:
        loaded = self._load(
            """
application:
  version: "31.0"
  max_tool_rounds: 4
openai:
  default_model: "test/model"
openrouter:
  timeout_ms: 5000
secrets:
  openai_api_key: "must-not-be-read"
""",
            {
                "OPENAI_API_KEY": "from-environment",
                "TELEGRAM_BOT_TOKEN": "telegram-token",
            },
        )

        self.assertEqual(loaded.application.version, "31.0")
        self.assertEqual(loaded.application.max_tool_rounds, 4)
        self.assertEqual(loaded.openai.default_model, "test/model")
        self.assertEqual(loaded.openrouter.timeout_ms, 5000)
        self.assertEqual(loaded.secrets.openai_api_key, "from-environment")

    def test_defaults_are_applied(self) -> None:
        loaded = self._load("{}")

        self.assertEqual(loaded.application.version, "29.3")
        self.assertEqual(loaded.openai.speech_model, "whisper-1")
        self.assertEqual(loaded.media.video_timeout_seconds, 360.0)
        self.assertEqual(
            loaded.media.image_generation_model, "black-forest-labs/flux-3-image"
        )
        self.assertEqual(loaded.media.image_edit_model, "black-forest-labs/flux-3-image")
        self.assertEqual(loaded.media.image_generation_fast_model, KLEIN_MODEL)
        self.assertEqual(loaded.media.image_edit_fast_model, KLEIN_MODEL)
        self.assertEqual(loaded.media.image_routing_mode, "pro_only")
        self.assertIsNone(loaded.media.image_default_resolution)
        self.assertEqual(loaded.media.image_multi_reference_pro_threshold, 2)
        self.assertTrue(loaded.media.image_fallback_enabled)

    def test_legacy_media_configuration_keeps_pro_route_and_provider_resolution(self) -> None:
        loaded = self._load(
            f"""
media:
  image_generation_model: "{FLUX_MODEL}"
  image_edit_model: "{FLUX_MODEL}"
"""
        )

        self.assertEqual(loaded.media.image_generation_model, FLUX_MODEL)
        self.assertEqual(loaded.media.image_edit_model, FLUX_MODEL)
        self.assertEqual(loaded.media.image_routing_mode, "pro_only")
        self.assertIsNone(loaded.media.image_default_resolution)

    def test_reads_new_image_settings(self) -> None:
        loaded = self._load(
            """
media:
  image_generation_fast_model: "custom/fast-generation"
  image_edit_fast_model: "custom/fast-edit"
  image_routing_mode: "auto"
  image_default_resolution: "2K"
  image_multi_reference_pro_threshold: 0
  image_fallback_enabled: false
"""
        )

        self.assertEqual(loaded.media.image_generation_fast_model, "custom/fast-generation")
        self.assertEqual(loaded.media.image_edit_fast_model, "custom/fast-edit")
        self.assertEqual(loaded.media.image_routing_mode, "auto")
        self.assertEqual(loaded.media.image_default_resolution, "2K")
        self.assertEqual(loaded.media.image_multi_reference_pro_threshold, 0)
        self.assertFalse(loaded.media.image_fallback_enabled)

    def test_working_config_uses_target_hybrid_image_route(self) -> None:
        loaded = Settings.from_yaml(environ={})

        self.assertEqual(loaded.media.image_generation_model, FLUX_MODEL)
        self.assertEqual(loaded.media.image_edit_model, FLUX_MODEL)
        self.assertEqual(loaded.media.image_generation_fast_model, KLEIN_MODEL)
        self.assertEqual(loaded.media.image_edit_fast_model, KLEIN_MODEL)
        self.assertEqual(loaded.media.image_routing_mode, "auto")
        self.assertEqual(loaded.media.image_default_resolution, "1K")
        self.assertEqual(loaded.media.image_multi_reference_pro_threshold, 2)
        self.assertTrue(loaded.media.image_fallback_enabled)
        self.assertEqual(loaded.media.image_generation_count, 1)
        self.assertEqual(loaded.media.image_edit_count, 1)

    def test_invalid_image_settings_are_rejected(self) -> None:
        invalid_media_sections = (
            "image_generation_model: ''",
            "image_edit_model: ''",
            "image_generation_fast_model: '   '",
            "image_edit_fast_model: ''",
            "image_edit_fast_model: 123",
            "image_routing_mode: fast_only",
            "image_routing_mode: []",
            "image_default_resolution: 1024",
            "image_default_resolution: true",
            "image_multi_reference_pro_threshold: -1",
            "image_multi_reference_pro_threshold: true",
            "image_fallback_enabled: 'true'",
            "image_generation_count: 2",
            "image_generation_count: true",
            "image_edit_count: 0",
        )

        for media_setting in invalid_media_sections:
            with self.subTest(media_setting=media_setting):
                with self.assertRaises(SettingsError):
                    self._load(f"media:\n  {media_setting}\n")

    def test_validate_rejects_bool_threshold_even_for_direct_settings(self) -> None:
        loaded = self._load("{}")
        invalid_media = replace(
            loaded.media,
            image_multi_reference_pro_threshold=True,
        )

        with self.assertRaises(SettingsError):
            replace(loaded, media=invalid_media).validate()

    def test_invalid_timeout_is_rejected(self) -> None:
        with self.assertRaises(SettingsError):
            self._load(
                """
telegram:
  read_timeout: 0
"""
            )

    def test_invalid_section_type_is_rejected(self) -> None:
        with self.assertRaises(SettingsError):
            self._load("media: []")

    def test_missing_config_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                Settings.from_yaml(Path(directory) / "missing.yaml", environ={})


class VideoProfileSettingsTests(unittest.TestCase):
    def _load(self, content: str) -> Settings:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(content, encoding="utf-8")
            return Settings.from_yaml(path, environ={})

    def test_default_profile_is_prime_480p_with_five_presets(self) -> None:
        loaded = self._load("{}")

        self.assertEqual(loaded.media.video_model, VIDEO_MODEL)
        self.assertEqual(loaded.media.video_resolution, VIDEO_RESOLUTION)
        self.assertEqual(loaded.media.video_duration_presets, VIDEO_DURATION_PRESETS)
        self.assertEqual(loaded.media.video_allowed_aspect_ratios, VIDEO_ASPECT_RATIOS)
        self.assertEqual(loaded.media.video_max_references, 10)
        self.assertEqual(loaded.media.video_allowed_durations, tuple(range(2, 31)))
        self.assertEqual(loaded.application.question_timeout_seconds, 900)

    def test_active_profile_is_exposed_from_settings(self) -> None:
        loaded = self._load("{}")

        profile = loaded.video_profile

        self.assertEqual(profile.model, VIDEO_MODEL)
        self.assertEqual(profile.resolution, VIDEO_RESOLUTION)
        self.assertEqual(profile.allowed_durations, tuple(range(2, 31)))
        self.assertEqual(profile.duration_presets, VIDEO_DURATION_PRESETS)
        self.assertEqual(profile.allowed_aspect_ratios, VIDEO_ASPECT_RATIOS)
        self.assertEqual(profile.max_references, 10)

    def test_working_config_uses_prime_480p(self) -> None:
        loaded = Settings.from_yaml(environ={})

        self.assertEqual(loaded.media.video_model, VIDEO_MODEL)
        self.assertEqual(loaded.media.video_resolution, VIDEO_RESOLUTION)
        self.assertEqual(loaded.media.video_duration_presets, VIDEO_DURATION_PRESETS)
        self.assertEqual(loaded.application.question_timeout_seconds, 900)
        self.assertEqual(loaded.video_profile.model, VIDEO_MODEL)
        self.assertEqual(loaded.video_profile.resolution, VIDEO_RESOLUTION)

    def test_removed_video_fields_are_rejected(self) -> None:
        removed_fields = (
            "video_duration: 15",
            "video_duration: null",
            "video_size: '1280x720'",
            "video_size: null",
        )

        for media_setting in removed_fields:
            with self.subTest(media_setting=media_setting):
                with self.assertRaises(SettingsError):
                    self._load(f"media:\n  {media_setting}\n")

    def test_unknown_video_fields_are_rejected(self) -> None:
        with self.assertRaises(SettingsError):
            self._load("media:\n  video_frame_images: 2\n")

    def test_legacy_video_fields_error_mentions_replacement(self) -> None:
        with self.assertRaises(SettingsError) as caught:
            self._load("media:\n  video_duration: 15\n")

        message = str(caught.exception)
        self.assertIn("video_duration", message)
        self.assertIn("video_allowed_durations", message)

    def test_invalid_video_profile_values_are_rejected(self) -> None:
        invalid_media_sections = (
            "video_model: ''",
            "video_model: 123",
            "video_resolution: '240p'",
            "video_resolution: 480",
            "video_aspect_ratio: '5:4'",
            "video_aspect_ratio: 16",
            "video_allowed_durations: []",
            "video_allowed_durations: [5, 10.5]",
            "video_allowed_durations: [5, true]",
            "video_allowed_durations: [5, 0]",
            "video_allowed_durations: [5, -5]",
            "video_allowed_durations: [5, 5]",
            "video_allowed_durations: 5",
            "video_duration_presets: []",
            "video_duration_presets: [5, 10.5]",
            "video_duration_presets: [5, true]",
            "video_duration_presets: [5, 45]",
            "video_allowed_aspect_ratios: []",
            "video_allowed_aspect_ratios: ['5:4']",
            "video_allowed_aspect_ratios: [16]",
            "video_allowed_aspect_ratios: '16:9'",
            "video_max_references: 0",
            "video_max_references: -1",
            "video_max_references: true",
            "video_max_references: 1.5",
        )

        for media_setting in invalid_media_sections:
            with self.subTest(media_setting=media_setting):
                with self.assertRaises(SettingsError):
                    self._load(f"media:\n  {media_setting}\n")

    def test_duration_presets_must_be_subset_of_allowed_durations(self) -> None:
        loaded = self._load("{}")

        with self.assertRaises(SettingsError):
            replace(
                loaded,
                media=replace(
                    loaded.media,
                    video_allowed_durations=(4, 8),
                    video_duration_presets=(4, 8, 16),
                ),
            ).validate()

    def test_replaced_profile_changes_validation_without_code_changes(self) -> None:
        loaded = self._load("{}")

        replaced = replace(
            loaded,
            media=replace(
                loaded.media,
                video_model="test/other-video-model",
                video_resolution="720p",
                video_allowed_durations=(4, 8),
                video_duration_presets=(4, 8),
                video_allowed_aspect_ratios=("16:9",),
                video_max_references=3,
            ),
        ).validate()

        profile = replaced.video_profile
        self.assertEqual(profile.model, "test/other-video-model")
        self.assertEqual(profile.resolution, "720p")
        self.assertEqual(profile.allowed_durations, (4, 8))
        self.assertEqual(profile.duration_presets, (4, 8))
        self.assertEqual(profile.allowed_aspect_ratios, ("16:9",))
        self.assertEqual(profile.max_references, 3)

    def test_invalid_question_timeout_is_rejected(self) -> None:
        for application_setting in (
            "question_timeout_seconds: 0",
            "question_timeout_seconds: -1",
            "question_timeout_seconds: true",
            "question_timeout_seconds: 1.5",
        ):
            with self.subTest(application_setting=application_setting):
                with self.assertRaises(SettingsError):
                    self._load(f"application:\n  {application_setting}\n")

    def test_valid_video_profile_is_read_from_yaml(self) -> None:
        loaded = self._load(
            """
media:
  video_model: "test/other-video-model"
  video_resolution: "720p"
  video_allowed_durations: [4, 8]
  video_duration_presets: [4, 8]
  video_allowed_aspect_ratios: ["16:9"]
  video_max_references: 3
"""
        )

        self.assertEqual(loaded.media.video_model, "test/other-video-model")
        self.assertEqual(loaded.media.video_resolution, "720p")
        self.assertEqual(loaded.media.video_allowed_durations, (4, 8))
        self.assertEqual(loaded.media.video_duration_presets, (4, 8))
        self.assertEqual(loaded.media.video_allowed_aspect_ratios, ("16:9",))
        self.assertEqual(loaded.media.video_max_references, 3)


if __name__ == "__main__":
    unittest.main()
