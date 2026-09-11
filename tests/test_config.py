"""Проверки единого загрузчика конфигурации."""

import tempfile
import unittest
from pathlib import Path

from core.config import Settings, SettingsError


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
        self.assertEqual(loaded.openrouter.speech_model, "qwen/qwen3-asr-1.7b")
        self.assertEqual(loaded.media.video_timeout_seconds, 360.0)

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
