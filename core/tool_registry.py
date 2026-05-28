from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from telegram import Update
from telegram.ext import ContextTypes

from core.common_types import ModelAnswer


@dataclass
class ToolExecutionContext:
    """Контекст выполнения tool handler."""

    update: Update
    context: ContextTypes.DEFAULT_TYPE
    messages: list[dict]
    model_name: str
    recursion_depth: int
    context_tokens: int = 0
    completion_tokens: int = 0
    additional_system_messages: list[dict] = field(default_factory=list)


ToolHandler = Callable[[ToolExecutionContext, dict[str, Any]], Awaitable[ModelAnswer]]

# Описания функций для OpenAI API
TOOLS_SCHEMA = [
    {
        "type": "function",
        "name": "generate_image",
        "description": "Сгенерировать изображение только по текстовому описанию пользователя (с нуля). Используй ТОЛЬКО когда у пользователя НЕТ активной сессии редактирования изображения. Если сессия уже есть, сообщи пользователю, что нужно сначала выполнить /reset.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Запрос пользователя, по которому сгенерируется картинка",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "type": "function",
        "name": "generate_image_from_image",
        "description": "Сгенерировать или преобразовать изображение на основе ТЕКУЩЕЙ картинки из активной сессии редактирования с учетом текстовой инструкции. Используй, когда пользователь просит изменить, стилизовать, перерисовать, улучшить или сделать вариацию. Текущая картинка берётся из сессии автоматически — повторная загрузка изображения не требуется. После успешной генерации текущая картинка обновляется результатом.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Инструкция, как преобразовать текущее изображение",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "type": "function",
        "name": "generate_video",
        "description": "Сгенерировать видео или анимацию по текстовому описанию пользователя. Если есть активная сессия редактирования, используй текущую картинку сессии как основу для видео. Генерация видео НЕ изменяет текущую картинку — после видео можно продолжать редактировать ту же картинку. Используй, когда пользователь просит видео, анимацию, оживить картинку или сделать видео на основе изображения.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Запрос пользователя, по которому сгенерируется видео",
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
