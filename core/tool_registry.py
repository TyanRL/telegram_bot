from dataclasses import dataclass, field
from uuid import uuid4
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from telegram import Update
from telegram.ext import ContextTypes

from core.common_types import QuestionAnswer, ToolQuestionCheck, ToolResult
from utils.openrouter_client import OpenRouterService

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from core.video_requests import VideoRequestDraft


@dataclass
class ImageToolTurnState:
    """Защита от автоматического платного повтора в одном user turn."""

    turn_id: str = field(default_factory=lambda: uuid4().hex)
    retry_forbidden: bool = False
    image_call_attempted: bool = False
    attempted_image_call_ids: set[str] = field(default_factory=set)
    failure_round: int | None = None


@dataclass
class VideoToolTurnState:
    """Защита от повторного платного запуска генерации видео.

    Состояние переживает приостановку цикла: ожидание ответа пользователя не
    обнуляет защиту, поэтому уже выполненный платный вызов не повторяется.

    `completed_operation_ids` содержит идентификаторы уже выполненных
    операций: повторный запуск той же операции блокируется даже при новом
    идентификаторе вызова. `failed_operation_ids` содержит операции, завершившиеся
    ошибкой: после них автоматический платный повтор запрещён.
    """

    turn_id: str = field(default_factory=lambda: uuid4().hex)
    attempted_video_call_ids: set[str] = field(default_factory=set)
    completed_operation_ids: set[str] = field(default_factory=set)
    failed_operation_ids: set[str] = field(default_factory=set)
    retry_forbidden: bool = False

    def mark_completed(self, operation_id: str | None) -> None:
        """Отмечает операцию как выполненную."""

        if operation_id:
            self.completed_operation_ids.add(operation_id)

    def mark_failed(self, operation_id: str | None) -> None:
        """Отмечает операцию как завершившуюся ошибкой."""

        if operation_id:
            self.failed_operation_ids.add(operation_id)
            self.retry_forbidden = True

    def is_known_operation(self, operation_id: str | None) -> bool:
        """Проверяет, выполнялась ли уже операция с таким идентификатором."""

        if not operation_id:
            return False
        return operation_id in self.completed_operation_ids or (
            operation_id in self.failed_operation_ids
        )


@dataclass
class ToolExecutionContext:
    """Контекст выполнения tool handler.

    Контекст хранит состояние видеозаявки и проверенный ответ пользователя:
    после ответа кнопкой продолжение выполняет именно сохранённую заявку,
    а не создаёт новую. Идентификатор операции используется для защиты от
    повторного платного запуска.
    """

    update: Update
    context: ContextTypes.DEFAULT_TYPE
    model_name: str
    recursion_depth: int
    openrouter_service: OpenRouterService
    context_tokens: int = 0
    completion_tokens: int = 0
    user_text: str | None = None
    image_tool_state: ImageToolTurnState = field(default_factory=ImageToolTurnState)
    video_tool_state: VideoToolTurnState = field(default_factory=VideoToolTurnState)
    video_request: "VideoRequestDraft | None" = None
    video_operation_id: str | None = None
    video_answer: "QuestionAnswer | None" = None


ToolHandler = Callable[[ToolExecutionContext, dict[str, Any]], Awaitable[ToolResult]]

# Предварительная проверка инструмента: вызывается до выполнения пакета
# вызовов и сообщает, что инструменту требуется ответ пользователя.
ToolQuestionCheckHandler = Callable[
    [ToolExecutionContext, dict[str, Any]],
    Awaitable[ToolQuestionCheck | None],
]

# Описания функций для OpenAI API
TOOLS_SCHEMA = [
    {
        "type": "function",
        "name": "generate_image",
        "description": "Сгенерировать новое изображение с нуля по текстовому описанию. Можно вызывать и при наличии картинок в сессии: результат добавится в конец списка и получит следующий номер.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Запрос пользователя, по которому сгенерируется картинка",
                },
                "quality": {
                    "type": "string",
                    "enum": ["standard", "pro"],
                    "description": "Необязательный уровень. standard — обычная генерация, стилизация и простые изменения. pro — точные надписи/текст на изображении, сложная композиция, портретное сходство, несколько значимых ограничений, высокая точность сохранения деталей и исправление/повтор неудачного результата.",
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": [
                        "21:9", "2:1", "16:9", "3:2", "7:5", "4:3", "5:4",
                        "1:1", "4:5", "3:4", "5:7", "2:3", "9:16", "1:2", "9:21", "auto"
                    ],
                    "description": "Необязательное соотношение сторон результата (указывай, если пользователь явно просит формат/ориентацию или при объединении картинок).",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "type": "function",
        "name": "generate_image_from_image",
        "description": "Сгенерировать или преобразовать изображения с учетом текстовой инструкции. Используй для изменения, стилизации, перерисовки, улучшения, объединения или вариации. Без image_indices используются все изображения сессии; после генерации результат добавляется в конец списка, исходники остаются доступными.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Инструкция, как преобразовать или объединить выбранные изображения",
                },
                "quality": {
                    "type": "string",
                    "enum": ["standard", "pro"],
                    "description": "Необязательный уровень. standard — обычная стилизация и простые изменения. pro — точные надписи/текст на изображении, сложная композиция, портретное сходство, несколько значимых ограничений, высокая точность сохранения деталей и исправление/повтор неудачного результата.",
                },
                "image_indices": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "description": "Необязательные 1-based номера изображений сессии. Если параметр не задан, используй все изображения.",
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": [
                        "21:9", "2:1", "16:9", "3:2", "7:5", "4:3", "5:4",
                        "1:1", "4:5", "3:4", "5:7", "2:3", "9:16", "1:2", "9:21", "auto"
                    ],
                    "description": "Необязательное соотношение сторон результата (указывай, если пользователь явно просит формат/ориентацию или при объединении картинок).",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "type": "function",
        "name": "generate_video",
        "description": (
            "Сгенерировать видео или анимацию по тексту. Без image_indices, если картинки "
            "сессии есть, используй все как референсы; иначе создай видео только по тексту. "
            "Можно указать конкретные картинки по их 1-based номерам. Порядок image_indices "
            "значим: референсы передаются провайдеру именно в указанном порядке, а не в "
            "порядке номеров. Первому и последнему референсу автоматически не назначается "
            "роль начального или конечного кадра. Генерация видео не изменяет изображения "
            "сессии."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Запрос пользователя, по которому сгенерируется видео",
                },
                "image_indices": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "description": (
                        "Необязательные 1-based номера изображений сессии в порядке "
                        "пользователя. Если параметр не задан, используются все изображения "
                        "сессии в её порядке. Порядок списка значим и передаётся провайдеру "
                        "как есть."
                    ),
                },
                "duration": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Необязательная длительность видео в секундах. Указывай только "
                        "длительность, явно названную пользователем в его сообщении; "
                        "значение проверяется по активной видеомодели. Если пользователь "
                        "длительность не назвал, не подставляй свою: сервер задаст вопрос."
                    ),
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "type": "function",
        "name": "ask_user_question",
        "description": (
            "Задать пользователю уточняющий вопрос и дождаться ответа. "
            "Покажи от одного до трёх готовых вариантов ответа с короткими "
            "уникальными идентификаторами и понятными подписями; кнопка "
            "«Свой ответ» добавляется приложением автоматически и не может "
            "быть отключена. Используй, только когда для выполнения задачи не "
            "хватает существенного параметра, который пользователь не указал. "
            "Не придумывай ответ за пользователя и не продолжай задачу без его "
            "ответа."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Текст вопроса пользователю: что нужно уточнить и "
                        "почему это важно для выполнения задачи."
                    ),
                },
                "options": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "description": (
                        "От одного до трёх готовых вариантов ответа. "
                        "Не дополняй список бессмысленными вариантами ради "
                        "ровно трёх."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": (
                                    "Короткий уникальный идентификатор варианта "
                                    "внутри вопроса (например, «red»)."
                                ),
                            },
                            "label": {
                                "type": "string",
                                "description": "Текст кнопки с вариантом ответа.",
                            },
                            "text": {
                                "type": "string",
                                "description": (
                                    "Альтернативное имя поля с текстом кнопки "
                                    "вместо label."
                                ),
                            },
                        },
                        "required": ["id"],
                    },
                },
                "custom_input_hint": {
                    "type": "string",
                    "description": (
                        "Необязательная подсказка, как ввести свой ответ "
                        "(например, «Напишите цвет или его код»)."
                    ),
                },
            },
            "required": ["question"],
        },
    },
    {
        "type": "function",
        "name": "get_weather_description",
        "description": "Получить текущую погоду по координатам",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
            },
            "required": ["latitude", "longitude"],
        },
    },
    {
        "type": "function",
        "name": "get_weekly_forecast",
        "description": "Получить прогноз погоды на неделю по координатам",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
            },
            "required": ["latitude", "longitude"],
        },
    },
    {
        "type": "function",
        "name": "request_geolocation",
        "description": "Запросить у пользователя геолокацию",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "change_model",
        "description": "Изменить модель ИИ",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Название модели"},
            },
            "required": ["model"],
        },
    },
    {
        "type": "function",
        "name": "get_location_by_address",
        "description": "Получить координаты по адресу",
        "parameters": {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "Адрес для геокодирования"},
            },
            "required": ["address"],
        },
    },
    {
        "type": "function",
        "name": "add_note",
        "description": "Добавить заметку",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "body", "tags"],
        },
    },
    {
        "type": "function",
        "name": "get_all_user_notes",
        "description": "Получить все заметки пользователя",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "get_notes_by_query",
        "description": "Найти заметки по запросу",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {"type": "string"},
                "start_created_date": {"type": "string"},
                "end_created_date": {"type": "string"},
            },
            "required": ["search_query"],
        },
    },
    {
        "type": "function",
        "name": "remove_notes",
        "description": "Удалить заметки по ID",
        "parameters": {
            "type": "object",
            "properties": {
                "note_ids": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["note_ids"],
        },
    },
    {"type": "web_search"},
]


class ToolRegistry:
    """Реестр обработчиков tool calls."""

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}
        self._question_checks: dict[str, ToolQuestionCheckHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        self._handlers[name] = handler

    def get(self, name: str) -> ToolHandler | None:
        return self._handlers.get(name)

    def has(self, name: str) -> bool:
        return name in self._handlers

    def register_question_check(
        self,
        name: str,
        handler: ToolQuestionCheckHandler,
    ) -> None:
        """Регистрирует предварительную проверку инструмента.

        Проверка вызывается до выполнения пакета вызовов: она позволяет
        задать уточняющий вопрос до зависимых платных операций.
        """

        self._question_checks[name] = handler

    def get_question_check(self, name: str) -> ToolQuestionCheckHandler | None:
        """Возвращает предварительную проверку инструмента, если она есть."""

        return self._question_checks.get(name)


# Глобальный реестр
_registry = ToolRegistry()


def register_tool(name: str) -> Callable[[ToolHandler], ToolHandler]:
    """Декоратор для регистрации tool handler."""

    def decorator(handler: ToolHandler) -> ToolHandler:
        _registry.register(name, handler)
        return handler

    return decorator


def register_tool_question_check(
    name: str,
) -> Callable[[ToolQuestionCheckHandler], ToolQuestionCheckHandler]:
    """Декоратор для регистрации предварительной проверки инструмента."""

    def decorator(handler: ToolQuestionCheckHandler) -> ToolQuestionCheckHandler:
        _registry.register_question_check(name, handler)
        return handler

    return decorator


def get_tool_registry() -> ToolRegistry:
    return _registry
