
"""Общие типы приложения: изображения, видео, инструменты и вопросы.

Модуль содержит доменные типы, которыми обмениваются слои приложения.
Ожидание пользовательского ответа описывается явно: тип
[`ToolQuestionPause`](core/common_types.py:176) и
[`PendingQuestionRequest`](core/common_types.py:186) означают приостановку
работы до ответа человека, а не обычный успешный результат инструмента.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

from core.config import settings

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from core.question_state import QuestionContinuationContext
    from core.tool_registry import ImageToolTurnState, VideoToolTurnState
    from core.video_requests import VideoModelProfile, VideoRequestDraft


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
    """Готовый видеофайл, которым владеет вызывающий код.

    `metadata` хранит безопасные сведения об операции: идентификатор задачи
    провайдера (`job_id`), фактические модель, разрешение и длительность,
    число референсов и известные метаданные использования. Бинарное
    содержимое и data URL здесь не хранятся.

    `cost` — известная стоимость генерации в долларах. Отсутствие значения
    не означает бесплатную генерацию: стоимость считается неизвестной.
    """

    path: Path
    cleanup_on_exit: bool = True
    cost: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _cleaned: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.metadata = dict(self.metadata)
        if self.cost is not None:
            if isinstance(self.cost, bool) or not isinstance(self.cost, (int, float)):
                raise TypeError("GeneratedVideo.cost должен быть числом или None")
            self.cost = float(self.cost)

    @property
    def job_id(self) -> str | None:
        """Идентификатор задачи провайдера, если он известен."""

        job_id = self.metadata.get("job_id")
        return job_id if isinstance(job_id, str) and job_id else None

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
    """Параметры операции text-to-image.

    Параметры разрешённого Images API: `aspect_ratio`, `resolution` и `n`.
    """

    aspect_ratio: str | None = None
    resolution: str | None = None
    seed: int | None = None
    n: int = settings.media.image_generation_count
    model: Optional[str] = field(default=None, kw_only=True)


@dataclass(frozen=True, slots=True)
class ImageEditOptions:
    """Параметры операции редактирования изображения (image-to-image).

    Параметры разрешённого Images API: `aspect_ratio`, `resolution` и `n`.
    """

    aspect_ratio: str | None = None
    resolution: str | None = None
    seed: int | None = None
    n: int = settings.media.image_edit_count
    model: Optional[str] = field(default=None, kw_only=True)


@dataclass(frozen=True, slots=True)
class VideoGenerationOptions:
    """Параметры запуска и ожидания генерации видео.

    Это готовые параметры запуска: длительность обязательна и выбирается
    пользователем явно. Проверка принадлежности длительности допустимому
    набору активного профиля выполняется в
    [`validate_duration()`](core/video_requests.py:104); черновик заявки без
    длительности живёт в [`VideoRequestDraft`](core/video_requests.py:141).
    """

    duration: int
    model: str = settings.media.video_model
    resolution: str = settings.media.video_resolution
    aspect_ratio: str | None = settings.media.video_aspect_ratio
    generate_audio: bool | None = settings.media.video_generate_audio
    seed: int | None = None
    timeout_seconds: float = settings.media.video_timeout_seconds
    polling_interval_seconds: float = settings.media.video_polling_interval_seconds
    index: int = settings.media.video_index

    def __post_init__(self) -> None:
        if isinstance(self.duration, bool) or not isinstance(self.duration, int):
            raise TypeError(
                "VideoGenerationOptions.duration должен быть целым числом секунд, "
                f"получено {type(self.duration).__name__}"
            )
        if self.duration <= 0:
            raise ValueError(
                "VideoGenerationOptions.duration должен быть положительным числом секунд"
            )


class QuestionStatus(str, Enum):
    """Жизненный цикл ожидания ответа пользователя.

    Переходы выполняются атомарно в
    [`QuestionStateStore`](core/question_state.py:213):

    `PENDING` → `ACCEPTED` → `CONTINUING` → `COMPLETED`,
    а также терминальные `CANCELLED` и `EXPIRED`.
    """

    PENDING = "pending"
    ACCEPTED = "accepted"
    CONTINUING = "continuing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class QuestionAnswerKind(str, Enum):
    """Происхождение принятого ответа."""

    OPTION = "option"
    CUSTOM = "custom"
    CANCEL = "cancel"


class QuestionAnswerMode(str, Enum):
    """Режим интерпретации принятого ответа.

    Режим задаёт вызывающим кодом, а не определяется по тексту сообщения.
    `PLAIN` означает, что окончательным результатом инструмента становится
    выбранный вариант или собственный текст. `VIDEO_REQUEST` означает, что
    ответ применяется к сохранённой видеозаявке, которая затем выполняется.
    """

    PLAIN = "plain"
    VIDEO_REQUEST = "video_request"


@dataclass(frozen=True, slots=True)
class QuestionOption:
    """Готовый вариант ответа, показываемый кнопкой.

    `option_id` — уникальный внутри вопроса идентификатор, `payload` — серверное
    значение варианта (например, длительность видео в секундах). Текст кнопки и
    payload не попадают в `callback_data`: кнопка несёт только идентификатор
    вопроса и код действия.
    """

    option_id: str
    text: str
    payload: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.option_id, str) or not self.option_id.strip():
            raise ValueError("QuestionOption.option_id должен быть непустой строкой")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("QuestionOption.text должен быть непустой строкой")


@dataclass(frozen=True, slots=True)
class PendingQuestionRequest:
    """Запрос вопроса пользователю.

    Описывает только то, что нужно показать пользователю: текст вопроса,
    готовые варианты (любое количество), подсказку для собственного ответа и
    срок ожидания. Проверка и интерпретация ответа выполняются на сервере.
    """

    text: str
    options: tuple[QuestionOption, ...] = ()
    input_hint: str | None = None
    timeout_seconds: int | None = None
    answer_mode: QuestionAnswerMode = QuestionAnswerMode.PLAIN
    validator: Callable[[str], str | None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("PendingQuestionRequest.text должен быть непустой строкой")
        object.__setattr__(self, "options", tuple(self.options))
        option_ids = [option.option_id for option in self.options]
        if len(set(option_ids)) != len(option_ids):
            raise ValueError(
                "PendingQuestionRequest.options должны иметь уникальные option_id"
            )
        if self.input_hint is not None and (
            not isinstance(self.input_hint, str) or not self.input_hint.strip()
        ):
            raise ValueError(
                "PendingQuestionRequest.input_hint должен быть непустой строкой или None"
            )
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or self.timeout_seconds <= 0
        ):
            raise ValueError(
                "PendingQuestionRequest.timeout_seconds должен быть положительным "
                "целым числом или None"
            )


@dataclass(frozen=True, slots=True)
class PendingToolCall:
    """Исходный вызов инструмента, который должен получить один результат.

    Сохраняется в контексте продолжения, чтобы после ответа пользователя
    каждому вызоду был отправлен ровно один `function_call_output`.
    """

    call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id:
            raise ValueError("PendingToolCall.call_id должен быть непустой строкой")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("PendingToolCall.name должен быть непустой строкой")
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True, slots=True)
class QuestionAnswer:
    """Принятый ответ пользователя на вопрос."""

    question_id: str
    kind: QuestionAnswerKind
    text: str
    option_id: str | None = None
    payload: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.question_id, str) or not self.question_id:
            raise ValueError("QuestionAnswer.question_id должен быть непустой строкой")
        if not isinstance(self.kind, QuestionAnswerKind):
            raise TypeError("QuestionAnswer.kind должен быть QuestionAnswerKind")
        if not isinstance(self.text, str):
            raise TypeError("QuestionAnswer.text должен быть строкой")


@dataclass(frozen=True, slots=True)
class ToolQuestionPause:
    """Явная приостановка инструмента до ответа пользователя.

    Инструмент возвращает такой объект вместо обычного успешного результата:
    работа не считается завершённой, платные действия не выполняются, пока
    пользователь не ответит. `continuation` хранит серверный контекст
    продолжения, который восстанавливается после ответа.

    `video_request` заполняется, если приостановка относится к сохранённой
    видеозаявке: после ответа заявка выполняется с проверенным значением.
    """

    request: PendingQuestionRequest
    continuation: "QuestionContinuationContext | None" = None
    video_request: "VideoRequestDraft | None" = None


@dataclass(frozen=True, slots=True)
class ToolQuestionCheck:
    """Результат предварительной проверки инструмента.

    Проверка выполняется до выполнения пакета вызовов: она позволяет
    уточнить недостающий параметр у пользователя до того, как произойдёт
    зависимая платная операция. Поле `video_request` заполняется, если
    вопрос относится к сохранённой видеозаявке: после ответа заявка
    выполняется с проверенным значением.
    """

    request: PendingQuestionRequest
    video_request: "VideoRequestDraft | None" = None
    video_operation_id: str | None = None


@dataclass
class ModelAnswer:
    """Ответ слоя модели.

    `pending_question` заполняется, если работа остановлена до ответа
    пользователя: в этом случае `bot_reply` не содержит обычного текста
    ответа, а вопрос отправляется пользователю отдельным сообщением с
    кнопками. `pending_continuation` хранит серверный контекст, который
    позволяет продолжить прерванный цикл инструментов после ответа.
    """

    bot_reply: str | None = None
    additional_system_messages: list[dict] = field(default_factory=list)
    ctx_token: int = 0
    completion_token: int = 0
    recurse: bool = False
    pending_question: PendingQuestionRequest | None = None
    question_id: str | None = None
    pending_continuation: "QuestionContinuationContext | None" = None


@dataclass
class ToolResult:
    """Результат выполнения инструмента до продолжения Responses API.

    `output` отправляется модели в поле `function_call_output`. Поле
    `additional_system_messages` предназначено только для фактов, которые
    нужно сохранить в локальной истории следующего пользовательского хода;
    текущий ответ модели получает данные исключительно через `output`.

    `pause` означает явную приостановку до ответа пользователя: в этом случае
    результат не считается выполненным, а `output` не должен интерпретироваться
    как результат инструмента.
    """

    output: Any = None
    additional_system_messages: list[dict] = field(default_factory=list)
    ctx_token: int = 0
    completion_token: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    pause: ToolQuestionPause | None = None


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
