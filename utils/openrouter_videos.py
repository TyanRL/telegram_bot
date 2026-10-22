"""OpenRouter Video Generation API adapter.

Адаптер реализует единый актуальный контракт общего Video API OpenRouter и
работает с любым активным видеопрофилем: переданная модель не подменяется
жёстко заданным Wan.

Жизненный цикл операции состоит из трёх этапов, и все они укладываются в
**один общий таймаут** (`deadline`), который начинается до создания задачи:

1. `create` — создание платной задачи (`videos.generate`);
2. `poll` — опрос статуса задачи до терминального статуса;
3. `download` — скачивание содержимого готовой задачи.

Особенности контракта:

- запрос формируется из снимка активного профиля (`model`, `resolution`,
  `aspect_ratio`), промпта, выбранной длительности и упорядоченных
  `input_references`;
- картинки кодируются в data URL с собственным MIME-типом (по сигнатуре
  байтов либо по `GeneratedImage.mime_type`) только на границе API;
- неиспользуемые необязательные поля (`size`, `frame_images`, `generate_audio`,
  `seed`) в запрос не попадают;
- обычные референсы не заменяют обязательный текстовый промпт: пустой промпт
  отклоняется до сетевого вызова;
- повторы SDK на платном POST отключены: повторная отправка создала бы вторую
  платную задачу; опрос статуса при этом идёт в управляемом цикле до
  `deadline`;
- при отмене, таймауте, пустом файле и любой ошибке скачивания поток
  закрывается, а незавершённый временный файл удаляется;
- известный `job_id` сохраняется в метаданных результата и в исключении,
  возникшем после создания задачи.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import logging
import math
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Sequence, cast

from openrouter import OpenRouter
from openrouter import utils as openrouter_utils
from openrouter.components.contentpartimage import ContentPartImageTypedDict
from openrouter.components.videogenerationrequest import (
    VideoGenerationRequestAspectRatio,
    VideoGenerationRequestResolution,
)

from core.common_types import GeneratedImage, GeneratedVideo, VideoGenerationOptions
from utils.openrouter_client import (
    OpenRouterResponseError,
    OpenRouterTimeoutError,
    OpenRouterUnsupportedModelError,
    OpenRouterUnsupportedParameterError,
    OpenRouterVideoCancelledError,
    OpenRouterVideoExpiredError,
    OpenRouterVideoFailedError,
    OpenRouterVideoJobUnavailableError,
)

logger = logging.getLogger(__name__)

# Успешный терминальный статус.
COMPLETED_VIDEO_STATUS = "completed"

# Документированные терминальные статусы, означающие отказ.
TERMINAL_VIDEO_STATUSES = frozenset({"failed", "cancelled", "expired"})

# Документированные нетерминальные статусы: по ним опрос продолжается.
PENDING_VIDEO_STATUSES = frozenset({"pending", "in_progress"})

# Повторы SDK на платном POST отключены: повторная отправка запроса создала
# бы вторую платную задачу. Опрос статуса выполняется управляемым циклом.
NO_SDK_RETRIES = openrouter_utils.RetryConfig(
    "none",
    openrouter_utils.BackoffStrategy(500, 60000, 1.5, 3600000),
    False,
)

# Известные MIME-типы референсов, определяемые по сигнатуре байтов.
_MIME_TYPE_BY_SIGNATURE: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _status_value(status: object) -> str:
    return str(getattr(status, "value", status))


def _known_cost(value: Any) -> float | None:
    """Возвращает известную неотрицательную стоимость или ``None``.

    Отсутствие стоимости не означает бесплатную генерацию: в этом случае
    стоимость считается неизвестной.
    """

    if isinstance(value, bool) or value is None:
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


def detect_reference_mime_type(content: bytes) -> str | None:
    """Определяет MIME-тип изображения по сигнатуре байтов.

    Поддерживаются PNG, JPEG, WebP и GIF. Для неизвестного формата
    возвращается ``None``: такой референс не отправляется провайдеру.
    """

    if not content:
        return None
    for signature, mime_type in _MIME_TYPE_BY_SIGNATURE:
        if content.startswith(signature):
            return mime_type
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return None


def _reference_to_data_url(reference: GeneratedImage | bytes | str) -> str:
    """Приводит референс к data URL на границе API.

    Изображение сохраняет свой MIME-тип, строка уже является готовым
    data URL, сырые байты кодируются с MIME-типом, определённым по сигнатуре.
    Порядок референсов не меняется.
    """

    if isinstance(reference, GeneratedImage):
        if not reference.content:
            raise OpenRouterUnsupportedParameterError(
                "Референс видео не должен быть пустым",
                reason="invalid_source_image",
                operation="video",
            )
        mime_type = reference.mime_type
        if not mime_type.startswith("image/"):
            raise OpenRouterUnsupportedParameterError(
                f"Референс видео должен быть изображением, получено {mime_type!r}",
                reason="invalid_source_image",
                operation="video",
            )
        encoded = base64.b64encode(reference.content).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
    if isinstance(reference, str):
        if not reference.strip():
            raise OpenRouterUnsupportedParameterError(
                "Ссылка на изображение не должна быть пустой",
                reason="invalid_source_image",
                operation="video",
            )
        return reference
    if not isinstance(reference, (bytes, bytearray)):
        raise OpenRouterUnsupportedParameterError(
            "Референс видео должен быть байтами или строкой data URL",
            reason="invalid_source_image",
            operation="video",
        )
    content = bytes(reference)
    if not content:
        raise OpenRouterUnsupportedParameterError(
            "Референс видео не должен быть пустым",
            reason="invalid_source_image",
            operation="video",
        )
    mime_type = detect_reference_mime_type(content)
    if mime_type is None:
        raise OpenRouterUnsupportedParameterError(
            "Не удалось определить формат изображения-референса видео",
            reason="invalid_source_image",
            operation="video",
        )
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_input_references(
    references: Sequence[GeneratedImage | bytes | str],
) -> list[ContentPartImageTypedDict] | None:
    """Готовит упорядоченный список `input_references` для запроса."""

    if not references:
        return None
    return [
        cast(
            ContentPartImageTypedDict,
            {
                "type": "image_url",
                "image_url": {"url": _reference_to_data_url(reference)},
            },
        )
        for reference in references
    ]


def _video_suffix(response: Any) -> str:
    headers = getattr(response, "headers", {}) or {}
    content_type = str(headers.get("content-type", headers.get("Content-Type", ""))).lower()
    if "webm" in content_type:
        return ".webm"
    if "quicktime" in content_type or "mov" in content_type:
        return ".mov"
    return ".mp4"


def _create_temp_file(suffix: str) -> Path:
    """Создаёт пустой временный файл под видео."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
        return Path(temporary_file.name)


def _append_chunk(path: Path, chunk: bytes) -> None:
    with path.open("ab") as handle:
        handle.write(chunk)


def _remove_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as error:  # pragma: no cover - защита от ошибок ФС
        logger.warning(
            "Не удалось удалить временный видеофайл: type=%s",
            type(error).__name__,
        )


def _stage_error(exc: Exception, stage: str, job_id: str | None) -> Exception:
    """Различает 404 модели при создании и 404 задачи при опросе/скачивании.

    Отсутствие модели означает, что платная задача не была создана, а
    отсутствие уже принятой задачи — что задача потеряна у провайдера.
    """

    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        raw_response = getattr(exc, "raw_response", None)
        candidate = getattr(raw_response, "status_code", None)
        status_code = candidate if isinstance(candidate, int) else None
    if status_code != 404:
        return exc

    provider_message = getattr(exc, "body", None) or getattr(exc, "message", None)
    if stage == "create":
        return OpenRouterUnsupportedModelError(
            "Выбранная видеомодель недоступна у провайдера",
            status_code=404,
            reason="unsupported_model",
            provider_message=provider_message,
            operation="video",
            job_id=job_id,
        )
    return OpenRouterVideoJobUnavailableError(
        "Задача генерации видео недоступна у провайдера",
        status_code=404,
        reason="video_job_unavailable",
        provider_message=provider_message,
        operation="video",
        job_id=job_id,
    )


def _attach_job_id(error: BaseException, job_id: str | None) -> None:
    """Сохраняет известный идентификатор задачи в исключении."""

    if job_id is None:
        return
    try:
        setattr(error, "job_id", job_id)
    except Exception:  # pragma: no cover - защита от нестандартных исключений
        logger.debug("Не удалось сохранить job_id в исключении")


def _extract_job_id(job: Any) -> str:
    job_id = getattr(job, "id", None)
    if isinstance(job_id, str) and job_id.strip():
        return job_id
    raise OpenRouterResponseError(
        "OpenRouter не вернул идентификатор видео-задачи",
        operation="video",
    )


def _extract_usage_cost(result: Any) -> float | None:
    usage = getattr(result, "usage", None)
    if usage is None:
        return None
    cost = getattr(usage, "cost", None)
    if cost is None and isinstance(usage, dict):
        cost = usage.get("cost")
    return _known_cost(cost)


async def _iter_response_chunks(response: Any) -> AsyncIterator[bytes]:
    """Читает содержимое ответа чанками."""

    aiter_bytes = getattr(response, "aiter_bytes", None)
    if callable(aiter_bytes):
        async for chunk in aiter_bytes():
            if chunk:
                yield chunk
        return
    aread = getattr(response, "aread", None)
    if callable(aread):
        content = await aread()
        if content:
            yield content
        return
    raise OpenRouterResponseError(
        "OpenRouter вернул ответ без доступного содержимого",
        operation="video",
    )


async def _write_response_chunks(response: Any, path: Path) -> int:
    """Записывает содержимое ответа в файл чанками и возвращает размер."""

    written = 0
    async for chunk in _iter_response_chunks(response):
        written += len(chunk)
        await asyncio.to_thread(_append_chunk, path, chunk)
    return written


async def _close_response(response: Any) -> None:
    """Гарантированно закрывает поток скачивания."""

    for name in ("aclose", "close"):
        closer = getattr(response, name, None)
        if not callable(closer):
            continue
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as error:  # pragma: no cover - защита от ошибок закрытия
            logger.debug(
                "Не удалось закрыть поток скачивания видео: type=%s",
                type(error).__name__,
            )
        return


async def _create_job(
    client: OpenRouter,
    *,
    prompt: str,
    options: VideoGenerationOptions,
    input_references: list[ContentPartImageTypedDict] | None,
    deadline: float,
    loop: asyncio.AbstractEventLoop,
) -> Any:
    """Создаёт платную задачу провайдера."""

    remaining = deadline - loop.time()
    if remaining <= 0:
        raise OpenRouterTimeoutError(
            "Общий лимит времени генерации видео исчерпан",
            operation="video",
        )

    request: dict[str, Any] = {
        "model": options.model,
        "prompt": prompt,
        "duration": options.duration,
        "resolution": cast(
            VideoGenerationRequestResolution | None,
            options.resolution,
        ),
        "retries": NO_SDK_RETRIES,
        "timeout_ms": max(1, math.floor(remaining * 1000)),
    }
    if options.aspect_ratio:
        request["aspect_ratio"] = cast(
            VideoGenerationRequestAspectRatio,
            options.aspect_ratio,
        )
    if options.generate_audio is not None:
        request["generate_audio"] = options.generate_audio
    if options.seed is not None:
        request["seed"] = options.seed
    if input_references:
        request["input_references"] = input_references

    logger.info(
        "Видео: этап=create, model=%s, resolution=%s, duration=%s, "
        "references=%s, timeout_ms=%s",
        options.model,
        options.resolution,
        options.duration,
        len(input_references or ()),
        request["timeout_ms"],
    )
    try:
        return await asyncio.wait_for(
            client.video_generation.generate_async(**request),
            timeout=remaining,
        )
    except asyncio.TimeoutError as error:
        raise OpenRouterTimeoutError(
            "Генерация видео не завершилась вовремя",
            operation="video",
        ) from error
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise _stage_error(exc, "create", None) from exc


async def _poll_job(
    client: OpenRouter,
    *,
    job_id: str,
    options: VideoGenerationOptions,
    deadline: float,
    loop: asyncio.AbstractEventLoop,
) -> Any:
    """Опрашивает статус задачи до терминального статуса."""

    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise OpenRouterTimeoutError(
                "Генерация видео не завершилась вовремя",
                operation="video",
                job_id=job_id,
            )
        try:
            result = await asyncio.wait_for(
                client.video_generation.get_generation_async(
                    job_id=job_id,
                    timeout_ms=max(1, math.floor(remaining * 1000)),
                ),
                timeout=remaining,
            )
        except asyncio.TimeoutError as error:
            raise OpenRouterTimeoutError(
                "Генерация видео не завершилась вовремя",
                operation="video",
                job_id=job_id,
            ) from error
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _stage_error(exc, "poll", job_id) from exc

        status = _status_value(getattr(result, "status", None))
        logger.info(
            "Видео: этап=poll, job_id=%s, status=%s",
            job_id,
            status,
        )
        if status == COMPLETED_VIDEO_STATUS:
            return result
        if status == "failed":
            raise OpenRouterVideoFailedError(
                "Генерация видео завершилась с ошибкой",
                provider_message=getattr(result, "error", None),
                operation="video",
                job_id=job_id,
            )
        if status == "cancelled":
            raise OpenRouterVideoCancelledError(
                "Генерация видео отменена провайдером",
                operation="video",
                job_id=job_id,
            )
        if status == "expired":
            raise OpenRouterVideoExpiredError(
                "Срок действия задачи генерации видео истёк",
                operation="video",
                job_id=job_id,
            )
        if status not in PENDING_VIDEO_STATUSES:
            # Неизвестный статус: бесконечный опрос запрещён.
            raise OpenRouterResponseError(
                f"Неизвестный статус видео-задачи: {status!r}",
                reason="invalid_response",
                operation="video",
                job_id=job_id,
            )

        await asyncio.sleep(min(options.polling_interval_seconds, max(0.0, remaining)))


async def _download_content(
    client: OpenRouter,
    *,
    job_id: str,
    options: VideoGenerationOptions,
    deadline: float,
    loop: asyncio.AbstractEventLoop,
) -> Path:
    """Скачивает содержимое готовой задачи во временный файл.

    Остаток общего таймаута ограничивает и открытие потока, и его чтение:
    скачивание не может выйти за пределы `deadline`.
    """

    remaining = deadline - loop.time()
    if remaining <= 0:
        raise OpenRouterTimeoutError(
            "Общий лимит времени генерации видео исчерпан",
            operation="video",
            job_id=job_id,
        )

    logger.info(
        "Видео: этап=download, job_id=%s, index=%s",
        job_id,
        options.index,
    )
    try:
        response = await asyncio.wait_for(
            client.video_generation.get_video_content_async(
                job_id=job_id,
                index=options.index,
                timeout_ms=max(1, math.floor(remaining * 1000)),
            ),
            timeout=remaining,
        )
    except asyncio.TimeoutError as error:
        raise OpenRouterTimeoutError(
            "Генерация видео не завершилась вовремя",
            operation="video",
            job_id=job_id,
        ) from error
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise _stage_error(exc, "download", job_id) from exc

    path: Path | None = None
    try:
        path = await asyncio.to_thread(_create_temp_file, _video_suffix(response))
        written = await asyncio.wait_for(
            _write_response_chunks(response, path),
            timeout=max(0.0, deadline - loop.time()),
        )
        if written == 0:
            raise OpenRouterResponseError(
                "OpenRouter вернул пустой видеофайл",
                operation="video",
                job_id=job_id,
            )
        return path
    except asyncio.TimeoutError as error:
        _remove_file(path)
        raise OpenRouterTimeoutError(
            "Генерация видео не завершилась вовремя",
            operation="video",
            job_id=job_id,
        ) from error
    except BaseException:
        # Отмена, таймаут, пустой файл и любая ошибка скачивания удаляют
        # незавершённый временный файл.
        _remove_file(path)
        raise
    finally:
        await _close_response(response)


async def generate(
    prompt: str,
    options: VideoGenerationOptions,
    reference_images: Sequence[GeneratedImage | bytes | str] = (),
    *,
    client: OpenRouter,
) -> GeneratedVideo:
    """Генерирует видео, ожидает готовность и возвращает временный файл.

    Единый актуальный контракт: упорядоченная последовательность референсов
    и обязательные готовые параметры запуска, включая длительность. Порядок
    референсов совпадает с порядком пользователя и сохраняется до запроса.

    Общий таймаут охватывает создание задачи, опрос статуса и скачивание
    содержимого: отсчёт начинается до создания задачи.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise OpenRouterUnsupportedParameterError(
            "Запрос на генерацию видео пуст",
            reason="empty_prompt",
            operation="video",
        )

    selected_options = options
    ordered_references = list(reference_images or ())
    input_references = _build_input_references(ordered_references)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + float(selected_options.timeout_seconds)

    job = await _create_job(
        client,
        prompt=prompt,
        options=selected_options,
        input_references=input_references,
        deadline=deadline,
        loop=loop,
    )
    job_id = _extract_job_id(job)
    logger.info(
        "Видео: этап=create, job_id=%s, model=%s, resolution=%s, duration=%s, "
        "references=%s",
        job_id,
        selected_options.model,
        selected_options.resolution,
        selected_options.duration,
        len(ordered_references),
    )

    metadata: dict[str, Any] = {
        "operation": "video",
        "job_id": job_id,
        "model": selected_options.model,
        "resolution": selected_options.resolution,
        "duration": selected_options.duration,
        "aspect_ratio": selected_options.aspect_ratio,
        "reference_count": len(ordered_references),
        "status": COMPLETED_VIDEO_STATUS,
    }
    generation_id = getattr(job, "generation_id", None)
    if isinstance(generation_id, str) and generation_id:
        metadata["generation_id"] = generation_id

    try:
        result = await _poll_job(
            client,
            job_id=job_id,
            options=selected_options,
            deadline=deadline,
            loop=loop,
        )
        cost = _extract_usage_cost(result)
        if cost is not None:
            metadata["cost"] = cost
        path = await _download_content(
            client,
            job_id=job_id,
            options=selected_options,
            deadline=deadline,
            loop=loop,
        )
    except BaseException as error:
        _attach_job_id(error, job_id)
        raise

    logger.info(
        "Видео: этап=download, job_id=%s, результат=готов, cost=%s",
        job_id,
        cost if cost is not None else "unknown",
    )
    return GeneratedVideo(path=path, cost=cost, metadata=metadata)
