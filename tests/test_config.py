"""Проверки единого загрузчика конфигурации."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from core.config import Settings, SettingsError


FLUX_MODEL = "black-forest-labs/flux-3-image"
KLEIN_MODEL = "black-forest-labs/flux-3-image-klein"


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


if __name__ == "__main__":
    unittest.main()
