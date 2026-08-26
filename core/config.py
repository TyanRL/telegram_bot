"""Единая конфигурация приложения.

Несекретные параметры читаются из ``config.yaml``, а секреты — только из
переменных окружения (или локального ``.env``). Остальные модули не должны
самостоятельно читать окружение или хранить собственные значения настроек.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DEFAULT_SYSTEM_PROMPT = """Вы — личный помощник, который СЖАТО И КРАТКО отвечает на вопросы пользователя. Время по Москве — {local_time}.
1. Если по доступному в диалоге контексту видно, что пользователь просит сгенерировать изображение только по текстовому описанию (с нуля), используй функцию generate_image.
2. Если по доступному в диалоге контексту видно, что у пользователя есть текущая картинка для редактирования и он просит изменить, стилизовать, перерисовать, улучшить или сделать вариацию, используй функцию generate_image_from_image.
3. Если пользователь просит сгенерировать видео, анимацию, оживить картинку или сделать видео на основе изображения — используй функцию generate_video.
4. Промпты для генерации видео и изображений создавай на английском языке, если результат генерации требует наличие текста, то этот текст не обязательно должен быть на английском.
5. Если функция недоступна для текущего состояния сессии, не придумывай обходной путь и следуй ограничениям, которые вернет приложение.
"""


class SettingsError(ValueError):
    """Ошибка структуры или значения конфигурационного файла."""


def _section(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SettingsError(f"Секция '{name}' в config.yaml должна быть объектом")
    return dict(value)


def _value(
    section: Mapping[str, Any],
    name: str,
    default: Any,
    expected_type: type | tuple[type, ...],
) -> Any:
    value = section.get(name, default)
    if value is None and default is None:
        return None
    is_bool_for_integer = isinstance(value, bool) and (
        expected_type is int
        or (isinstance(expected_type, tuple) and int in expected_type and bool not in expected_type)
    )
    if not isinstance(value, expected_type) or is_bool_for_integer:
        expected_name = (
            ", ".join(item.__name__ for item in expected_type)
            if isinstance(expected_type, tuple)
            else expected_type.__name__
        )
        raise SettingsError(
            f"Параметр '{name}' должен иметь тип {expected_name}, "
            f"получен {type(value).__name__}"
        )
    return value


def _optional_string(section: Mapping[str, Any], name: str, default: str | None = None) -> str | None:
    value = section.get(name, default)
    if value is not None and not isinstance(value, str):
        raise SettingsError(
            f"Параметр '{name}' должен иметь тип str или null, получен {type(value).__name__}"
        )
    return value


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise SettingsError(f"Параметр '{name}' должен быть положительным")


def _non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise SettingsError(f"Параметр '{name}' не может быть отрицательным")


def _env(environ: Mapping[str, str], name: str) -> str:
    return str(environ.get(name, "")).strip()


@dataclass(frozen=True, slots=True)
class SecretSettings:
    """Секреты приложения, намеренно не читаемые из YAML."""

    telegram_bot_token: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    mysql_password: str = ""
    elastic_access_key: str = ""
    elastic_secret_key: str = ""
    yandex_geocoder_api_key: str = ""
    openweather_api_key: str = ""
    weatherstack_api_key: str = ""
    google_api_key: str = ""
    google_search_engine_id: str = ""

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "SecretSettings":
        values = environ if environ is not None else os.environ
        return cls(
            telegram_bot_token=_env(values, "TELEGRAM_BOT_TOKEN"),
            openai_api_key=_env(values, "OPENAI_API_KEY"),
            openrouter_api_key=_env(values, "OPENROUTER_API_KEY"),
            mysql_password=_env(values, "MYSQL_ADDON_PASSWORD"),
            elastic_access_key=_env(values, "ELASTIC_ACCESS_KEY"),
            elastic_secret_key=_env(values, "ELASTIC_SECRET_KEY"),
            yandex_geocoder_api_key=_env(values, "YMAPS_GEOCODER"),
            openweather_api_key=_env(values, "OPENWEATHERMAP_API_KEY"),
            weatherstack_api_key=_env(values, "WEATHERSTACK_API_KEY"),
            google_api_key=_env(values, "GOOGLE_API_KEY"),
            google_search_engine_id=_env(values, "GOOGLE_SEARCH_ENGINE_ID"),
        )


@dataclass(frozen=True, slots=True)
class ApplicationSettings:
    version: str = "29.3"
    timezone: str = "Europe/Moscow"
    max_history_length: int = 15
    history_warning_thresholds: tuple[int, ...] = (8, 14)
    max_tool_rounds: int = 10
    concurrency_limit: int = 10
    log_level: str = "INFO"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass(frozen=True, slots=True)
class AccessSettings:
    allowed_user_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ServerSettings:
    webhook_url: str = "https://telegram-bot-xmj4.onrender.com/telegram-webhook"
    host: str = "0.0.0.0"
    port: int = 8443


@dataclass(frozen=True, slots=True)
class TelegramSettings:
    parse_mode: str = "MarkdownV2"
    max_message_length: int = 4096
    connection_pool_size: int = 8
    read_timeout: float = 120.0
    write_timeout: float = 180.0
    connect_timeout: float = 30.0
    pool_timeout: float = 30.0
    service_animation_interval_seconds: float = 1.5
    video_write_timeout: float = 180.0
    video_read_timeout: float = 120.0
    video_connect_timeout: float = 30.0
    video_pool_timeout: float = 30.0


@dataclass(frozen=True, slots=True)
class OpenAISettings:
    default_model: str = "gpt-5.6-terra"
    speech_model: str = "whisper-1"
    max_output_tokens: int = 16_384
    verbosity: str = "low"
    reasoning_effort: str = "medium"
    store_responses: bool = True


@dataclass(frozen=True, slots=True)
class OpenRouterSettings:
    http_referer: str | None = None
    title: str | None = None
    categories: str | None = None
    timeout_ms: int = 120_000


@dataclass(frozen=True, slots=True)
class MediaSettings:
    image_generation_model: str = "qwen/qwen-image-3-pro"
    image_edit_model: str = "google/gemini-3.1-flash-image"
    image_generation_count: int = 1
    image_edit_count: int = 1
    image_edit_output_format: str = "png"
    video_model: str = "alibaba/wan-3.0"
    video_resolution: str = "480p"
    video_aspect_ratio: str | None = None
    video_duration: int | None = 15
    video_generate_audio: bool | None = None
    video_size: str | None = None
    video_timeout_seconds: float = 360.0
    video_polling_interval_seconds: float = 5.0
    video_index: int = 0
    telegram_image_max_size: int = 1280
    telegram_image_jpeg_quality: int = 88


@dataclass(frozen=True, slots=True)
class MySQLSettings:
    host: str = ""
    database: str = ""
    user: str = ""
    port: int = 3306
    connection_retries: int = 10
    retry_delay_seconds: float = 5.0
    user_ids_table_name: str = "user_ids"
    last_session_table_name: str = "last_session_big_int"


@dataclass(frozen=True, slots=True)
class ElasticSettings:
    url: str = ""
    notes_index_name: str = "user_notes_index"
    search_top_k: int = 10


@dataclass(frozen=True, slots=True)
class IntegrationSettings:
    weather_url: str = "https://api.open-meteo.com/v1/forecast"
    weather_request_timeout_seconds: float = 30.0
    yandex_geocoder_url: str = "https://geocode-maps.yandex.ru/1.x/"
    yandex_request_timeout_seconds: float = 10.0
    google_search_result_count: int = 10
    web_content_request_timeout_seconds: float = 30.0
    web_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )


@dataclass(frozen=True, slots=True)
class Settings:
    """Типизированные настройки всего приложения."""

    application: ApplicationSettings
    access: AccessSettings
    server: ServerSettings
    telegram: TelegramSettings
    openai: OpenAISettings
    openrouter: OpenRouterSettings
    media: MediaSettings
    mysql: MySQLSettings
    elastic: ElasticSettings
    integrations: IntegrationSettings
    secrets: SecretSettings

    @classmethod
    def from_yaml(
        cls,
        path: str | Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "Settings":
        """Загружает YAML и секреты окружения в один объект настроек."""

        config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        config_path = config_path.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

        # Локальный .env удобен для запуска на рабочей машине, но не имеет
        # приоритета над уже заданными переменными окружения.
        load_dotenv(dotenv_path=config_path.parent / ".env", override=False)

        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise SettingsError(f"Не удалось разобрать {config_path.name}: {exc}") from exc
        except OSError as exc:
            raise SettingsError(f"Не удалось прочитать {config_path}: {exc}") from exc

        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            raise SettingsError("Корень config.yaml должен быть объектом")

        env = environ if environ is not None else os.environ
        return cls(
            application=cls._application(_section(loaded, "application")),
            access=cls._access(_section(loaded, "access")),
            server=cls._server(_section(loaded, "server")),
            telegram=cls._telegram(_section(loaded, "telegram")),
            openai=cls._openai(_section(loaded, "openai")),
            openrouter=cls._openrouter(_section(loaded, "openrouter")),
            media=cls._media(_section(loaded, "media")),
            mysql=cls._mysql(_section(loaded, "mysql")),
            elastic=cls._elastic(_section(loaded, "elastic")),
            integrations=cls._integrations(_section(loaded, "integrations")),
            secrets=SecretSettings.from_env(env),
        ).validate()

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "Settings":
        """Понятный синоним ``from_yaml`` для точек запуска и тестов."""

        return cls.from_yaml(path, environ=environ)

    def validate(self) -> "Settings":
        """Проверяет ограничения, общие для всех способов запуска."""

        positive_ints = {
            "application.max_history_length": self.application.max_history_length,
            "application.max_tool_rounds": self.application.max_tool_rounds,
            "application.concurrency_limit": self.application.concurrency_limit,
            "server.port": self.server.port,
            "telegram.max_message_length": self.telegram.max_message_length,
            "telegram.connection_pool_size": self.telegram.connection_pool_size,
            "openai.max_output_tokens": self.openai.max_output_tokens,
            "media.image_generation_count": self.media.image_generation_count,
            "media.image_edit_count": self.media.image_edit_count,
            "media.telegram_image_max_size": self.media.telegram_image_max_size,
            "mysql.connection_retries": self.mysql.connection_retries,
            "elastic.search_top_k": self.elastic.search_top_k,
            "integrations.google_search_result_count": self.integrations.google_search_result_count,
        }
        for name, value in positive_ints.items():
            _positive(name, value)

        positive_numbers = {
            "telegram.read_timeout": self.telegram.read_timeout,
            "telegram.write_timeout": self.telegram.write_timeout,
            "telegram.connect_timeout": self.telegram.connect_timeout,
            "telegram.pool_timeout": self.telegram.pool_timeout,
            "telegram.service_animation_interval_seconds": self.telegram.service_animation_interval_seconds,
            "telegram.video_write_timeout": self.telegram.video_write_timeout,
            "telegram.video_read_timeout": self.telegram.video_read_timeout,
            "telegram.video_connect_timeout": self.telegram.video_connect_timeout,
            "telegram.video_pool_timeout": self.telegram.video_pool_timeout,
            "openrouter.timeout_ms": self.openrouter.timeout_ms,
            "media.video_timeout_seconds": self.media.video_timeout_seconds,
            "media.video_polling_interval_seconds": self.media.video_polling_interval_seconds,
            "mysql.retry_delay_seconds": self.mysql.retry_delay_seconds,
            "integrations.weather_request_timeout_seconds": self.integrations.weather_request_timeout_seconds,
            "integrations.yandex_request_timeout_seconds": self.integrations.yandex_request_timeout_seconds,
            "integrations.web_content_request_timeout_seconds": self.integrations.web_content_request_timeout_seconds,
        }
        for name, value in positive_numbers.items():
            _positive(name, value)

        _non_negative("media.telegram_image_jpeg_quality", self.media.telegram_image_jpeg_quality)
        if self.media.telegram_image_jpeg_quality > 100:
            raise SettingsError("Параметр 'media.telegram_image_jpeg_quality' не может быть больше 100")
        if self.media.video_index < 0:
            raise SettingsError("Параметр 'media.video_index' не может быть отрицательным")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.application.history_warning_thresholds
        ):
            raise SettingsError(
                "Параметр 'application.history_warning_thresholds' должен содержать "
                "неотрицательные целые числа"
            )
        return self

    def require_runtime_secrets(self) -> None:
        """Проверяет секреты, без которых основной процесс не может стартовать."""

        required_secrets = {
            "TELEGRAM_BOT_TOKEN": self.secrets.telegram_bot_token,
            "OPENAI_API_KEY": self.secrets.openai_api_key,
            "OPENROUTER_API_KEY": self.secrets.openrouter_api_key,
            "MYSQL_ADDON_PASSWORD": self.secrets.mysql_password,
            "ELASTIC_ACCESS_KEY": self.secrets.elastic_access_key,
            "ELASTIC_SECRET_KEY": self.secrets.elastic_secret_key,
            "YMAPS_GEOCODER": self.secrets.yandex_geocoder_api_key,
        }
        missing = [name for name, value in required_secrets.items() if not value]
        if missing:
            raise EnvironmentError(
                "Не заданы обязательные настройки окружения: " + ", ".join(missing)
            )

        required_config = {
            "mysql.host": self.mysql.host,
            "mysql.database": self.mysql.database,
            "mysql.user": self.mysql.user,
            "elastic.url": self.elastic.url,
        }
        missing_config = [name for name, value in required_config.items() if not value]
        if missing_config:
            raise SettingsError(
                "Не заданы обязательные параметры config.yaml: " + ", ".join(missing_config)
            )

    @staticmethod
    def _application(section: Mapping[str, Any]) -> ApplicationSettings:
        thresholds = section.get("history_warning_thresholds", [8, 14])
        if not isinstance(thresholds, list):
            raise SettingsError(
                "Параметр 'history_warning_thresholds' должен быть списком"
            )
        return ApplicationSettings(
            version=_value(section, "version", "29.3", str),
            timezone=_value(section, "timezone", "Europe/Moscow", str),
            max_history_length=_value(section, "max_history_length", 15, int),
            history_warning_thresholds=tuple(thresholds),
            max_tool_rounds=_value(section, "max_tool_rounds", 10, int),
            concurrency_limit=_value(section, "concurrency_limit", 10, int),
            log_level=_value(section, "log_level", "INFO", str),
            system_prompt=_value(section, "system_prompt", DEFAULT_SYSTEM_PROMPT, str),
        )

    @staticmethod
    def _access(section: Mapping[str, Any]) -> AccessSettings:
        values = section.get("allowed_user_ids", [])
        if not isinstance(values, list):
            raise SettingsError("Параметр 'allowed_user_ids' должен быть списком")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise SettingsError("Все значения 'allowed_user_ids' должны быть целыми числами")
        return AccessSettings(allowed_user_ids=tuple(values))

    @staticmethod
    def _server(section: Mapping[str, Any]) -> ServerSettings:
        defaults = ServerSettings()
        return ServerSettings(
            webhook_url=_value(section, "webhook_url", defaults.webhook_url, str),
            host=_value(section, "host", defaults.host, str),
            port=_value(section, "port", defaults.port, int),
        )

    @staticmethod
    def _telegram(section: Mapping[str, Any]) -> TelegramSettings:
        defaults = TelegramSettings()
        return TelegramSettings(
            parse_mode=_value(section, "parse_mode", defaults.parse_mode, str),
            max_message_length=_value(section, "max_message_length", defaults.max_message_length, int),
            connection_pool_size=_value(section, "connection_pool_size", defaults.connection_pool_size, int),
            read_timeout=_value(section, "read_timeout", defaults.read_timeout, (int, float)),
            write_timeout=_value(section, "write_timeout", defaults.write_timeout, (int, float)),
            connect_timeout=_value(section, "connect_timeout", defaults.connect_timeout, (int, float)),
            pool_timeout=_value(section, "pool_timeout", defaults.pool_timeout, (int, float)),
            service_animation_interval_seconds=_value(
                section,
                "service_animation_interval_seconds",
                defaults.service_animation_interval_seconds,
                (int, float),
            ),
            video_write_timeout=_value(section, "video_write_timeout", defaults.video_write_timeout, (int, float)),
            video_read_timeout=_value(section, "video_read_timeout", defaults.video_read_timeout, (int, float)),
            video_connect_timeout=_value(section, "video_connect_timeout", defaults.video_connect_timeout, (int, float)),
            video_pool_timeout=_value(section, "video_pool_timeout", defaults.video_pool_timeout, (int, float)),
        )

    @staticmethod
    def _openai(section: Mapping[str, Any]) -> OpenAISettings:
        defaults = OpenAISettings()
        return OpenAISettings(
            default_model=_value(section, "default_model", defaults.default_model, str),
            speech_model=_value(section, "speech_model", defaults.speech_model, str),
            max_output_tokens=_value(section, "max_output_tokens", defaults.max_output_tokens, int),
            verbosity=_value(section, "verbosity", defaults.verbosity, str),
            reasoning_effort=_value(section, "reasoning_effort", defaults.reasoning_effort, str),
            store_responses=_value(section, "store_responses", defaults.store_responses, bool),
        )

    @staticmethod
    def _openrouter(section: Mapping[str, Any]) -> OpenRouterSettings:
        defaults = OpenRouterSettings()
        return OpenRouterSettings(
            http_referer=_optional_string(section, "http_referer", defaults.http_referer),
            title=_optional_string(section, "title", defaults.title),
            categories=_optional_string(section, "categories", defaults.categories),
            timeout_ms=_value(section, "timeout_ms", defaults.timeout_ms, int),
        )

    @staticmethod
    def _media(section: Mapping[str, Any]) -> MediaSettings:
        defaults = MediaSettings()
        return MediaSettings(
            image_generation_model=_value(section, "image_generation_model", defaults.image_generation_model, str),
            image_edit_model=_value(section, "image_edit_model", defaults.image_edit_model, str),
            image_generation_count=_value(section, "image_generation_count", defaults.image_generation_count, int),
            image_edit_count=_value(section, "image_edit_count", defaults.image_edit_count, int),
            image_edit_output_format=_value(section, "image_edit_output_format", defaults.image_edit_output_format, str),
            video_model=_value(section, "video_model", defaults.video_model, str),
            video_resolution=_value(section, "video_resolution", defaults.video_resolution, str),
            video_aspect_ratio=_optional_string(section, "video_aspect_ratio", defaults.video_aspect_ratio),
            video_duration=_value(section, "video_duration", defaults.video_duration, (int, type(None))),
            video_generate_audio=_value(section, "video_generate_audio", defaults.video_generate_audio, (bool, type(None))),
            video_size=_optional_string(section, "video_size", defaults.video_size),
            video_timeout_seconds=_value(section, "video_timeout_seconds", defaults.video_timeout_seconds, (int, float)),
            video_polling_interval_seconds=_value(
                section,
                "video_polling_interval_seconds",
                defaults.video_polling_interval_seconds,
                (int, float),
            ),
            video_index=_value(section, "video_index", defaults.video_index, int),
            telegram_image_max_size=_value(section, "telegram_image_max_size", defaults.telegram_image_max_size, int),
            telegram_image_jpeg_quality=_value(
                section,
                "telegram_image_jpeg_quality",
                defaults.telegram_image_jpeg_quality,
                int,
            ),
        )

    @staticmethod
    def _mysql(section: Mapping[str, Any]) -> MySQLSettings:
        defaults = MySQLSettings()
        return MySQLSettings(
            host=_value(section, "host", defaults.host, str),
            database=_value(section, "database", defaults.database, str),
            user=_value(section, "user", defaults.user, str),
            port=_value(section, "port", defaults.port, int),
            connection_retries=_value(section, "connection_retries", defaults.connection_retries, int),
            retry_delay_seconds=_value(section, "retry_delay_seconds", defaults.retry_delay_seconds, (int, float)),
            user_ids_table_name=_value(
                section,
                "user_ids_table_name",
                defaults.user_ids_table_name,
                str,
            ),
            last_session_table_name=_value(
                section,
                "last_session_table_name",
                defaults.last_session_table_name,
                str,
            ),
        )

    @staticmethod
    def _elastic(section: Mapping[str, Any]) -> ElasticSettings:
        defaults = ElasticSettings()
        return ElasticSettings(
            url=_value(section, "url", defaults.url, str),
            notes_index_name=_value(section, "notes_index_name", defaults.notes_index_name, str),
            search_top_k=_value(section, "search_top_k", defaults.search_top_k, int),
        )

    @staticmethod
    def _integrations(section: Mapping[str, Any]) -> IntegrationSettings:
        defaults = IntegrationSettings()
        return IntegrationSettings(
            weather_url=_value(section, "weather_url", defaults.weather_url, str),
            weather_request_timeout_seconds=_value(
                section,
                "weather_request_timeout_seconds",
                defaults.weather_request_timeout_seconds,
                (int, float),
            ),
            yandex_geocoder_url=_value(section, "yandex_geocoder_url", defaults.yandex_geocoder_url, str),
            yandex_request_timeout_seconds=_value(
                section,
                "yandex_request_timeout_seconds",
                defaults.yandex_request_timeout_seconds,
                (int, float),
            ),
            google_search_result_count=_value(
                section,
                "google_search_result_count",
                defaults.google_search_result_count,
                int,
            ),
            web_content_request_timeout_seconds=_value(
                section,
                "web_content_request_timeout_seconds",
                defaults.web_content_request_timeout_seconds,
                (int, float),
            ),
            web_user_agent=_value(section, "web_user_agent", defaults.web_user_agent, str),
        )


def load_settings(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Загружает настройки приложения."""

    return Settings.from_yaml(path, environ=environ)


settings = load_settings()
