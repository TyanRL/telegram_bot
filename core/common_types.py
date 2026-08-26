
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    """Бинарный результат генерации изображения.

    Внутри приложения изображение передаётся как bytes. Кодирование в base64
    выполняется только при сохранении visual session.
    """

    content: bytes
    mime_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes):
            raise TypeError("GeneratedImage.content должен иметь тип bytes")
        if not self.mime_type or "/" not in self.mime_type:
            raise ValueError("GeneratedImage.mime_type должен быть MIME-типом")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(slots=True)
class GeneratedVideo:
    """Готовый видеофайл, которым владеет вызывающий код."""

    path: Path
    cleanup_on_exit: bool = True
    _cleaned: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    @property
    def file_path(self) -> Path:
        """Явное имя пути для кода доставки."""

        return self.path

    @property
    def cleaned(self) -> bool:
        return self._cleaned

    def cleanup(self) -> None:
        """Удаляет временный файл не более одного раза."""

        if self._cleaned:
            return
        try:
            if self.cleanup_on_exit:
                self.path.unlink(missing_ok=True)
        finally:
            self._cleaned = True

    close = cleanup


@dataclass(frozen=True, slots=True)
class ImageGenerationOptions:
    """Параметры операции text-to-image."""

    aspect_ratio: str | None = None
    resolution: str | None = None
    output_format: str | None = None
    output_compression: int | None = None
    quality: str | None = None
    background: str | None = None
    size: str | None = None
    seed: int | None = None
    n: int = 1


@dataclass(frozen=True, slots=True)
class ImageEditOptions:
    """Параметры операции редактирования изображения."""

    aspect_ratio: str | None = None
    resolution: str | None = None
    output_format: str = "png"
    output_compression: int | None = None
    quality: str | None = None
    background: str | None = None
    size: str | None = None
    seed: int | None = None
    n: int = 1


@dataclass(frozen=True, slots=True)
class VideoGenerationOptions:
    """Параметры запуска и ожидания генерации видео."""

    model: str = "alibaba/wan-3.0"
    resolution: str = "480p"
    aspect_ratio: str | None = None
    duration: int | None = 15 #для wan-3.0
    #duration: int | None = None
    generate_audio: bool | None = None
    seed: int | None = None
    size: str | None = None
    timeout_seconds: float = 360.0 # для 10 секунд таймоут должен быть 180 сек
    polling_interval_seconds: float = 5.0
    index: int = 0


@dataclass
class ModelAnswer:
    bot_reply: str | None = None
    additional_system_messages: list[dict] = field(default_factory=list)
    ctx_token: int = 0
    completion_token: int = 0
    recurse: bool = False


@dataclass
class ToolResult:
    """Результат выполнения инструмента до продолжения Responses API.

    `output` отправляется модели в поле `function_call_output`. Поле
    `additional_system_messages` предназначено только для фактов, которые
    нужно сохранить в локальной истории следующего пользовательского хода;
    текущий ответ модели получает данные исключительно через `output`.
    """

    output: Any
    additional_system_messages: list[dict] = field(default_factory=list)
    ctx_token: int = 0
    completion_token: int = 0


class SafeDict:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.data = {}

    async def set(self, key, value):
        async with self.lock:
            self.data[key] = value

    async def get(self, key, default_value):
        async with self.lock:
            return self.data.get(key, default_value)
    
    async def delete(self, key):
        async with self.lock:
            if key in self.data:
                del self.data[key]

class SafeList:
    def __init__(self, list: list):
        self.lock = asyncio.Lock()
        self.data = list

    async def append(self, value):
        async with self.lock:
            self.data.append(value)

    async def get(self, index):
        async with self.lock:
            return self.data[index]
        
    async def remove(self, value):
        async with self.lock:
            if value in self.data:
                self.data.remove(value)
    
    async def get_all(self):
        async with self.lock:
            return list(self.data)
        

def dict_to_markdown(d, indent=0):
    result = []
    for key, value in d.items():
        indentation = '  ' * indent
        if isinstance(value, dict):
            result.append(f"{indentation}## {key}")
            result.append(dict_to_markdown(value, indent + 1))
        elif isinstance(value, list):
            result.append(f"{indentation}### {key}")
            for item in value:
                result.append(f"{indentation}- {item}")
        else:
            result.append(f"{indentation}- **{key}**: {value}")
    return "\n".join(result)
