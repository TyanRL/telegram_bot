# OpenRouter Media API

Приложение использует SDK `openrouter` напрямую. Отдельные URL OpenRouter в
конфигурации приложения не задаются: адреса операций принадлежат SDK.

## Lifecycle

Создаётся один долгоживущий сервис на время работы приложения:

```python
from utils.openrouter_client import OpenRouterService

media_service = OpenRouterService.from_env()
async with media_service:
    # Обработка обновлений приложения.
    ...
```

Клиент SDK создаётся внутри `__aenter__` сервиса и закрывается внутри
`__aexit__`. SDK-клиент не создаётся во время импорта модулей и не создаётся
отдельно в media handler-ах.

Обязательная переменная окружения — `OPENROUTER_API_KEY`. Дополнительные
параметры приложения: `OPENROUTER_HTTP_REFERER`,
`OPENROUTER_X_OPEN_ROUTER_TITLE`, `OPENROUTER_X_OPEN_ROUTER_CATEGORIES`,
`OPENROUTER_TIMEOUT_MS`, `OPENROUTER_VIDEO_TIMEOUT_SECONDS` и
`OPENROUTER_VIDEO_POLL_INTERVAL_SECONDS`.

## Изображения

У сервиса есть две разные операции:

```python
image = await media_service.generate(prompt, options)
edited = await media_service.edit(source_image, instruction, options)
```

- `generate` всегда использует `qwen/qwen-image-3-pro`;
- `edit` всегда использует `google/gemini-3.1-flash-image`;
- обе операции вызывают `client.images.generate_async`;
- редактирование передаёт исходник через `input_references`;
- ответ `b64_json` декодируется адаптером в `GeneratedImage`.

`GeneratedImage` содержит бинарное поле `content`, MIME-тип `mime_type` и
технические метаданные. Data URL не является контрактом приложения: адаптер
создаёт его только непосредственно перед формированием `input_references`.

## Видео

Операция видео возвращает готовый управляемый временный ресурс:

```python
video = await media_service.generate_video(prompt, reference_image, options)
try:
    await deliver(video)
finally:
    video.cleanup()
```

Адаптер выполняет следующие операции SDK:

1. `video_generation.generate_async` запускает задачу;
2. `video_generation.get_generation_async` проверяет статус;
3. между проверками вызывается `asyncio.sleep`;
4. `video_generation.get_video_content_async` получает содержимое;
5. ответ читается через `aread`, затем закрывается через `aclose`;
6. запись во временный файл выполняется через `asyncio.to_thread`.

Статусы `failed`, `cancelled` и `expired` преобразуются в отдельные доменные
ошибки. Возврат polling URL, unsigned URL и ручное скачивание по URL не входят
в контракт приложения.

## Ошибки и логирование

Все ошибки media SDK преобразуются в иерархию `OpenRouterMediaError` с
техническим статусом, причиной и безопасным коротким сообщением провайдера.
Пользовательские сообщения формируются через `map_generation_error`.

Логи не содержат base64, полные ответы OpenRouter или содержимое промптов.

## Распознавание голосовых (версия 30.5)

Голосовые сообщения распознаёт `qwen/qwen3-asr-1.7b` через OpenRouter.
Модель задаётся в `config.yaml` → `openrouter.speech_model`; ключ — существующий
`OPENROUTER_API_KEY`. Параметр `openai.speech_model` удалён. OpenAI продолжает
использоваться для текстовых ответов, поэтому его ключ по-прежнему нужен.

Установите зависимости из `requirements.txt`: для STT используется проверенная
версия SDK `openrouter>=1.1.137,<2`. При запуске через Pixi обновите окружение
командой `pixi install` (старый lock-файл уже не соответствует манифесту).

`await service.transcribe_audio(path)` передаёт Telegram OGG/Opus как base64
через `client.stt.create_transcription_async` в `/api/v1/audio/transcriptions`.
Язык определяется автоматически. Используются общий клиент и таймаут
`openrouter.timeout_ms`. Чтение файла вынесено в рабочий поток; сетевой запрос
асинхронный. Пустой ответ и ошибки не передаются в диалог. Автоматического
возврата к Whisper нет. Команда `/info` показывает новую модель и провайдера.

Контракт API: https://openrouter.ai/docs/guides/overview/multimodal/stt
Модель: https://openrouter.ai/qwen/qwen3-asr-1.7b
