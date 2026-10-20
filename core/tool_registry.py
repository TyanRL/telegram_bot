from dataclasses import dataclass, field
from uuid import uuid4
from typing import Any, Awaitable, Callable

from telegram import Update
from telegram.ext import ContextTypes

from core.common_types import ToolResult
from utils.openrouter_client import OpenRouterService


@dataclass
class ImageToolTurnState:
    """Защита от автоматического платного повтора в одном user turn."""

    turn_id: str = field(default_factory=lambda: uuid4().hex)
    retry_forbidden: bool = False
    image_call_attempted: bool = False
    attempted_image_call_ids: set[str] = field(default_factory=set)
    failure_round: int | None = None


@dataclass
class ToolExecutionContext:
    """Контекст выполнения tool handler."""

    update: Update
    context: ContextTypes.DEFAULT_TYPE
    model_name: str
    recursion_depth: int
    openrouter_service: OpenRouterService
    context_tokens: int = 0
    completion_tokens: int = 0
    user_text: str | None = None
    image_tool_state: ImageToolTurnState = field(default_factory=ImageToolTurnState)


ToolHandler = Callable[[ToolExecutionContext, dict[str, Any]], Awaitable[ToolResult]]

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
        "description": "Сгенерировать видео или анимацию по тексту. Без image_indices, если картинки сессии есть, используй все как референсы; иначе создай видео только по тексту. Можно указать конкретные картинки по их 1-based номерам. Генерация видео не изменяет изображения сессии.",
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
                    "description": "Необязательные 1-based номера изображений сессии. Если параметр не задан, используй все изображения.",
                },
            },
            "required": ["prompt"],
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

    def register(self, name: str, handler: ToolHandler) -> None:
        self._handlers[name] = handler

    def get(self, name: str) -> ToolHandler | None:
        return self._handlers.get(name)

    def has(self, name: str) -> bool:
        return name in self._handlers


# Глобальный реестр
_registry = ToolRegistry()


def register_tool(name: str) -> Callable[[ToolHandler], ToolHandler]:
    """Декоратор для регистрации tool handler."""

    def decorator(handler: ToolHandler) -> ToolHandler:
        _registry.register(name, handler)
        return handler

    return decorator


def get_tool_registry() -> ToolRegistry:
    return _registry
