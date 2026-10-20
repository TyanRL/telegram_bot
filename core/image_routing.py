"""Чистые правила выбора модели изображений и анализа связанных повторов.

Модуль намеренно не импортирует Telegram, сетевые клиенты или глобальное
хранилище. Все решения принимаются только по переданным настройкам и данным.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import MediaSettings, Settings


@dataclass(frozen=True, slots=True)
class ImageRoutingDecision:
    """Итог выбора маршрута для одной операции с изображением."""

    model: str
    quality: str
    resolution: str | None
    reason: str
    is_retry: bool = False
    retry_target_index: int | None = None


@dataclass(frozen=True, slots=True)
class ImageRetryContext:
    """Детерминированный анализ повтора и подготовленный контекст операции.

    ``reference_image_indices`` содержит только проверенные номера. Для
    неудачного редактирования при отсутствии исходников список остаётся
    пустым, а ``source_indices_available`` становится ``False``: вызывающий
    код не должен молча заменять источники всеми изображениями сессии.
    """

    is_retry: bool = False
    is_complaint: bool = False
    is_improvement: bool = False
    target_index: int | None = None
    target_found: bool = False
    target_is_generated: bool = False
    target_missing: bool = False
    ambiguous_target: bool = False
    target_operation: str | None = None
    target_operation_id: str | None = None
    target_task_id: str | None = None
    target_model: str | None = None
    target_quality: str | None = None
    origin_known: bool = False
    original_prompt: str | None = None
    source_image_indices: tuple[int, ...] = ()
    operation: str | None = None
    prompt: str | None = None
    reference_image_indices: tuple[int, ...] = ()
    use_result_as_reference: bool = False
    source_indices_available: bool | None = None
    missing_source_indices: tuple[int, ...] = ()


_PAIRED_QUOTES: tuple[tuple[str, str], ...] = (
    ('"', '"'),
    ("'", "'"),
    ("«", "»"),
    ("“", "”"),
    ("„", "“"),
    ("‘", "’"),
)

_TEXT_MARKER_RE = re.compile(
    r"(?<!\w)(?:"
    r"надпис(?:ь|и|ью|ей|ями|ях)|"
    r"вывеск(?:а|е|у|и|ой|ою|ам|ами|ах)|"
    r"знак(?:а|е|у|ом|и|ов|ам|ами|ах)?|"
    r"текст(?:а|у|ом|е|ы|ов|ами|ах)?|"
    r"типографик(?:а|и|е|у|ой|ою)|"
    r"typograph(?:y|ic(?:al)?)|"
    r"lettering|captions?|signs?|texts?|"
    r"words?|quotes?|labels?|banners?|titles?|inscriptions?|"
    r"(?:логотип\w*\s+(?:с|со)\s+текст\w*)|"
    r"(?:logo\s+(?:with|containing)\s+text)|"
    r"(?:text\s+on\s+(?:the\s+)?logo)"
    r")(?!\w)",
    re.IGNORECASE,
)

_EXPLICIT_PRO_RE = re.compile(
    r"(?<!\w)(?:"
    r"максимальн\w*\s+качеств\w*|"
    r"(?:наилучш\w*|лучш\w*|высочайш\w*|профессиональн\w*)\s+качеств\w*|"
    r"(?:самое|максимальное)\s+качество|"
    r"(?:maximum|highest|best(?:\s+possible)?|professional)\s+quality"
    r")(?!\w)",
    re.IGNORECASE,
)

_RETRY_RE = re.compile(
    r"(?<!\w)(?:"
    r"передел\w*|перегенерир\w*|пересозда\w*|"
    r"повтор(?:и|ите|ить|но)|заново|"
    r"(?:еще|ещё)\s+раз|"
    r"попробу\w*\s+(?:снова|заново|(?:еще|ещё)\s+раз)|"
    r"redo(?:\s+it)?|regenerate|try\s+again|one\s+more\s+time|"
    r"start\s+over|generate\s+again|do\s+it\s+again"
    r")(?!\w)",
    re.IGNORECASE,
)

_COMPLAINT_RE = re.compile(
    r"(?<!\w)(?:"
    r"(?:плох\w*|низк\w*|ужасн\w*)\s+(?:качеств\w*|результат\w*)|"
    r"(?:плохо|неудачно)\s+(?:получил\w*|вышл\w*|выгляд\w*)|"
    r"не\s+получил\w*|"
    r"качество\s+(?:плох\w*|низк\w*|ужасн\w*)|"
    r"размы\w*|мыльн\w*|артефакт\w*|искаж\w*|деформир\w*|"
    r"не\s*похож\w*|непохож\w*|"
    r"(?:крив\w*|неверн\w*|неправильн\w*|ошибочн\w*)\s+"
    r"(?:надпис\w*|текст\w*|букв\w*|логотип\w*)|"
    r"(?:надпис\w*|текст\w*|букв\w*|логотип\w*)[^.!?\n]{0,40}"
    r"(?:крив\w*|неверн\w*|неправильн\w*|с\s+ошибк\w*)|"
    r"исправ\w*[^.!?\n]{0,40}(?:надпис\w*|текст\w*|букв\w*|качеств\w*)|"
    r"лишн\w*\s+(?:пальц\w*|артефакт\w*)|"
    r"(?:bad|poor|low|awful)\s+(?:image\s+)?quality|"
    r"blurr\w*|artifact\w*|distort\w*|"
    r"(?:does\s+not|doesn['’]t|not)\s+(?:look\s+like|resembl\w*)|"
    r"not\s+similar|not\s+what\s+(?:i|I)\s+asked|"
    r"(?:wrong|incorrect|misspelled|garbled)\s+"
    r"(?:text|caption|lettering|sign|letters?|logo)|"
    r"(?:text|caption|lettering|sign|letters?|logo)[^.!?\n]{0,40}"
    r"(?:wrong|incorrect|misspelled|garbled)|"
    r"fix\s+(?:the\s+)?(?:text|caption|lettering|sign|letters?|logo|quality)|"
    r"bad\s+result|failed\s+(?:image|generation|result)"
    r")(?!\w)",
    re.IGNORECASE,
)

_EDIT_ACTION_RE = re.compile(
    r"(?<!\w)(?:"
    r"добав\w*|измени\w*|поменя\w*|замени\w*|убери\w*|сделай\w*|"
    r"доработ\w*|подправ\w*|улучши\w*|"
    r"change|add|remove|replace|make|modify|adjust|refine|update"
    r")(?!\w)",
    re.IGNORECASE,
)

_NEW_IMAGE_RE = re.compile(
    r"(?<!\w)(?:"
    r"с\s+нуля|нов\w*\s+(?:картинк\w*|изображен\w*)|"
    r"(?:create|generate|draw)\s+(?:a\s+)?new\s+(?:image|picture|illustration)"
    r")(?!\w)",
    re.IGNORECASE,
)

_AMBIGUOUS_TARGET_RE = re.compile(
    r"(?<!\w)(?:оба|обе|все\s+(?:картинк\w*|изображен\w*|результат\w*)|"
    r"нескольк\w*\s+(?:картинк\w*|изображен\w*|результат\w*)|"
    r"both\s+(?:images|pictures|results)|all\s+(?:images|pictures|results)|"
    r"multiple\s+(?:images|pictures|results))(?!\w)",
    re.IGNORECASE,
)

_IMAGE_INDEX_RE = re.compile(
    r"(?<!\w)(?:"
    r"(?:изображени\w*|картин\w*|фотографи\w*|фото|images?|pictures?|photos?)"
    r"\s*(?:(?:№|#|номер|number|no\.? )\s*)?|"
    r"(?:№|#)\s*"
    r")(\d+)(?!\w)",
    re.IGNORECASE,
)

_RETRY_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"передел\w*(?:\s+заново)?|перегенерир\w*|пересозда\w*|"
    r"повтор(?:и|ите|ить)?(?:\s+(?:еще|ещё)\s+раз)?|"
    r"попробу\w*\s+(?:снова|заново|(?:еще|ещё)\s+раз)|"
    r"(?:сделай|сделать|make)\s+(?:это\s+)?заново|"
    r"сгенериру\w*\s+заново|созда\w*\s+заново|"
    r"redo(?:\s+it)?|regenerate|try\s+again|start\s+over"
    r")[\s,.:;—-]*",
    re.IGNORECASE,
)


def has_paired_quotes(text: str | None) -> bool:
    """Проверяет непустые парные кавычки, не принимая английские апострофы."""

    if not isinstance(text, str) or not text:
        return False

    for opening, closing in _PAIRED_QUOTES:
        open_positions = [
            index
            for index, char in enumerate(text)
            if char == opening and not _is_in_word_apostrophe(text, index)
        ]
        close_positions = (
            open_positions if opening == closing else [
                index for index, char in enumerate(text) if char == closing
            ]
        )

        if opening == closing:
            for start, end in zip(open_positions[::2], open_positions[1::2]):
                if text[start + 1 : end].strip():
                    return True
            continue

        close_cursor = 0
        for start in open_positions:
            while close_cursor < len(close_positions) and close_positions[close_cursor] <= start:
                close_cursor += 1
            if close_cursor >= len(close_positions):
                break
            end = close_positions[close_cursor]
            close_cursor += 1
            if text[start + 1 : end].strip():
                return True

    return False


def _is_in_word_apostrophe(text: str, index: int) -> bool:
    """True для апострофа между буквами (например, ``don't`` или ``it’s``)."""

    return (
        index > 0
        and index + 1 < len(text)
        and text[index - 1].isalpha()
        and text[index + 1].isalpha()
    )


def has_image_text_marker(text: str | None) -> bool:
    """Ищет маркеры надписей как отдельные слова и устойчивые выражения."""

    return isinstance(text, str) and _TEXT_MARKER_RE.search(text) is not None


def _entries_by_index(
    image_entries: Sequence[Mapping[str, Any] | tuple[int, Mapping[str, Any]]]
    | None,
) -> dict[int, Mapping[str, Any]]:
    entries: dict[int, Mapping[str, Any]] = {}
    if image_entries is None:
        return entries

    for position, item in enumerate(image_entries, start=1):
        index = position
        image: Any = item
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], int)
            and not isinstance(item[0], bool)
        ):
            index, image = item
        if index < 1 or not isinstance(image, Mapping):
            continue
        entries[index] = image
    return entries


def _generation_data(image: Mapping[str, Any]) -> Mapping[str, Any]:
    generation = image.get("generation")
    return generation if isinstance(generation, Mapping) else {}


def _is_generated_image(image: Mapping[str, Any]) -> bool:
    if isinstance(image.get("generation"), Mapping):
        return True
    source_kind = image.get("source_kind")
    return isinstance(source_kind, str) and source_kind in {
        "generated",
        "text2image",
        "edit",
    }


def _positive_indices(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    result: list[int] = []
    seen: set[int] = set()
    for index in value:
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            continue
        if index not in seen:
            result.append(index)
            seen.add(index)
    return tuple(result)


def _target_fields(
    index: int | None,
    image: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if image is None:
        return {
            "target_found": False,
            "target_is_generated": False,
            "target_operation": None,
            "target_operation_id": None,
            "target_task_id": None,
            "target_model": None,
            "target_quality": None,
            "origin_known": False,
            "original_prompt": None,
            "source_image_indices": (),
        }

    generation = _generation_data(image)
    operation = generation.get("operation")
    if operation not in {"generate", "edit"}:
        operation = None
    operation_id = generation.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id.strip():
        operation_id = None
    task_id = generation.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        task_id = None
    model = generation.get("model")
    if not isinstance(model, str) or not model.strip():
        model = None
    quality = generation.get("quality")
    if quality not in {"standard", "pro"}:
        quality = None
    prompt = generation.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        prompt = None
    return {
        "target_found": index is not None,
        "target_is_generated": _is_generated_image(image),
        "target_operation": operation,
        "target_operation_id": operation_id,
        "target_task_id": task_id,
        "target_model": model,
        "target_quality": quality,
        "origin_known": quality is not None or model is not None,
        "original_prompt": prompt,
        "source_image_indices": _positive_indices(
            generation.get("source_image_indices")
        ),
    }


def analyze_image_retry(
    user_text: str | None,
    image_entries: Sequence[Mapping[str, Any] | tuple[int, Mapping[str, Any]]]
    | None,
) -> ImageRetryContext:
    """Определяет жалобу/повтор и адресованное изображение по текущему тексту.

    Анализ не использует историю сообщений: давняя жалоба сама по себе не
    влияет на новые запросы. Для неуказанного номера выбирается последний
    созданный результат, пропуская более поздние загруженные пользователем
    фотографии. Старые результаты с ``source_kind`` распознаются как цель, но
    их неизвестное происхождение не считается доказательством Klein.
    """

    text = user_text if isinstance(user_text, str) else ""
    complaint = _COMPLAINT_RE.search(text) is not None
    retry = complaint or _RETRY_RE.search(text) is not None
    entries = _entries_by_index(image_entries)
    explicit_indices = tuple(dict.fromkeys(int(value) for value in _IMAGE_INDEX_RE.findall(text)))
    edit_action = _EDIT_ACTION_RE.search(text) is not None
    is_improvement = (
        not retry
        and edit_action
        and _NEW_IMAGE_RE.search(text) is None
        and (bool(explicit_indices) or any(_is_generated_image(image) for image in entries.values()))
    )

    needs_target = retry or is_improvement
    ambiguous = needs_target and (
        len(explicit_indices) > 1 or _AMBIGUOUS_TARGET_RE.search(text) is not None
    )
    target_missing = False
    target_index: int | None = None

    if needs_target and not ambiguous:
        if explicit_indices:
            requested_index = explicit_indices[0]
            if requested_index in entries:
                target_index = requested_index
            else:
                target_missing = True
        else:
            generated_indices = [
                index for index, image in entries.items() if _is_generated_image(image)
            ]
            if generated_indices:
                target_index = max(generated_indices)
            elif is_improvement and len(entries) == 1:
                # Для обычной правки единственного загруженного фото цель ясна;
                # для жалобы на старый результат такого предположения нет.
                target_index = next(iter(entries))

    target = entries.get(target_index) if target_index is not None else None
    fields = _target_fields(target_index, target)
    return ImageRetryContext(
        is_retry=retry,
        is_complaint=complaint,
        is_improvement=is_improvement,
        target_index=target_index,
        target_missing=target_missing,
        ambiguous_target=ambiguous,
        **fields,
    )


def _retry_clarification(user_text: str | None) -> str | None:
    if not isinstance(user_text, str) or not user_text.strip():
        return None
    # Жалоба/сигнал повтора — не художественное требование. Удаляем их,
    # оставляя только сопровождающие уточнения вроде «сохрани красный шарф».
    text = user_text.strip()
    for _ in range(2):
        text = _COMPLAINT_RE.sub(" ", text)
        text = text.strip(" \t\r\n,.:;—-!")
        text = _RETRY_PREFIX_RE.sub("", text).strip()
    text = text.strip(" \t\r\n,.:;—-!")
    return text or None


def _compose_retry_prompt(
    original_prompt: str | None,
    user_text: str | None,
    current_prompt: str | None,
) -> str | None:
    parts: list[str] = []
    if isinstance(original_prompt, str) and original_prompt.strip():
        parts.append(original_prompt.strip())

    clarification = _retry_clarification(user_text)
    if clarification and clarification.casefold() not in {
        part.casefold() for part in parts
    }:
        parts.append(f"Актуальное уточнение пользователя: {clarification}")

    if isinstance(current_prompt, str) and current_prompt.strip():
        clean_current = current_prompt.strip()
        if clean_current.casefold() not in {part.casefold() for part in parts}:
            parts.append(f"Текущая полная инструкция: {clean_current}")

    return "\n\n".join(parts) or None


def resolve_image_retry_context(
    user_text: str | None,
    image_entries: Sequence[Mapping[str, Any] | tuple[int, Mapping[str, Any]]]
    | None,
    operation: str,
    *,
    current_prompt: str | None = None,
) -> ImageRetryContext:
    """Подготавливает инструкцию и точные исходники для связанной операции.

    Повтор ``generate`` восстанавливает исходный промпт и уточнения, но никогда
    не добавляет неудачный результат в референсы. Повтор ``edit`` восстанавливает
    исходные номера этой операции и сообщает об отсутствующих источниках вместо
    молчаливой подстановки всех текущих изображений.
    """

    if operation not in {"generate", "edit"}:
        raise ValueError("operation должен быть 'generate' или 'edit'")

    analysis = analyze_image_retry(user_text, image_entries)
    entries = _entries_by_index(image_entries)
    reference_indices: tuple[int, ...] = ()
    missing_sources: tuple[int, ...] = ()
    sources_available: bool | None = None
    use_result_as_reference = False

    if analysis.is_retry:
        resolved_prompt = _compose_retry_prompt(
            analysis.original_prompt,
            user_text,
            current_prompt,
        )
        if operation == "edit":
            sources = analysis.source_image_indices
            if sources:
                missing_sources = tuple(index for index in sources if index not in entries)
                sources_available = not missing_sources
                if sources_available:
                    reference_indices = sources
            else:
                # Нет сохранённого списка источников — это неизвестность, а не
                # разрешение использовать все изображения сессии.
                sources_available = False
    elif analysis.is_improvement:
        resolved_prompt = current_prompt.strip() if isinstance(current_prompt, str) and current_prompt.strip() else _retry_clarification(user_text)
        if operation == "edit" and analysis.target_index is not None:
            reference_indices = (analysis.target_index,)
            use_result_as_reference = analysis.target_is_generated
    else:
        resolved_prompt = current_prompt.strip() if isinstance(current_prompt, str) and current_prompt.strip() else None

    return replace(
        analysis,
        operation=operation,
        prompt=resolved_prompt,
        reference_image_indices=reference_indices,
        use_result_as_reference=use_result_as_reference,
        source_indices_available=sources_available,
        missing_source_indices=missing_sources,
    )


def _media_settings(settings: MediaSettings | Settings | Any) -> Any:
    media = getattr(settings, "media", None)
    return media if media is not None else settings


def _configured_model_quality(media: Any, operation: str | None, model: str) -> str | None:
    operations = (operation,) if operation in {"generate", "edit"} else ("generate", "edit")
    fast_models: set[str] = set()
    pro_models: set[str] = set()
    for item in operations:
        if item == "edit":
            fast_name, pro_name = "image_edit_fast_model", "image_edit_model"
        else:
            fast_name, pro_name = "image_generation_fast_model", "image_generation_model"
        fast_value = getattr(media, fast_name, None)
        pro_value = getattr(media, pro_name, None)
        if isinstance(fast_value, str) and fast_value:
            fast_models.add(fast_value)
        if isinstance(pro_value, str) and pro_value:
            pro_models.add(pro_value)

    if model in fast_models and model not in pro_models:
        return "standard"
    if model in pro_models and model not in fast_models:
        return "pro"
    return None


def _retry_target_values(retry_target: Any) -> tuple[bool, str | None, str | None, str | None, int | None]:
    """Возвращает retry flag, качество/модель/operation и номер цели."""

    if retry_target is None:
        return False, None, None, None, None

    container = retry_target
    nested_analysis = getattr(container, "analysis", None)
    if nested_analysis is not None:
        container = nested_analysis

    def value(name: str) -> Any:
        if isinstance(container, Mapping):
            return container.get(name)
        return getattr(container, name, None)

    retry_flag = value("is_retry") is True
    quality = value("target_quality")
    model = value("target_model")
    target_operation = value("target_operation")
    target_index = value("target_index")

    # Разрешаем передать как ImageRetryContext, так и саму запись сессии.
    generation = value("generation")
    target_record = value("target")
    if isinstance(target_record, Mapping):
        target_generation = target_record.get("generation")
        generation = target_generation if isinstance(target_generation, Mapping) else generation
    if isinstance(generation, Mapping):
        if quality is None:
            quality = generation.get("quality")
        if model is None:
            model = generation.get("model")
        if target_operation is None:
            target_operation = generation.get("operation")
    if quality is None:
        quality = value("quality")
    if model is None:
        model = value("model")
    if target_operation is None:
        target_operation = value("operation")

    if quality not in {"standard", "pro"}:
        quality = None
    if not isinstance(model, str) or not model.strip():
        model = None
    if target_operation not in {"generate", "edit"}:
        target_operation = None
    if isinstance(target_index, bool) or not isinstance(target_index, int) or target_index < 1:
        target_index = None
    return retry_flag, quality, model, target_operation, target_index


def resolve_image_route(
    settings: MediaSettings | Settings,
    operation: str,
    quality: Any = None,
    prompt: str = "",
    user_text: str | None = None,
    reference_count: int = 0,
    *,
    retry_target: ImageRetryContext | Mapping[str, Any] | None = None,
    is_retry: bool = False,
    generated_result_improvement: bool = False,
) -> ImageRoutingDecision:
    """Выбирает модель строго по приоритетам правил маршрутизации.

    Идентификатор модели никогда не берётся из аргументов инструмента:
    выбираются только настроенные основной/быстрый модели для операции.
    """

    if operation not in {"generate", "edit"}:
        raise ValueError("operation должен быть 'generate' или 'edit'")

    media = _media_settings(settings)
    if operation == "generate":
        pro_model = getattr(media, "image_generation_model", "")
        standard_model = getattr(media, "image_generation_fast_model", "")
    else:
        pro_model = getattr(media, "image_edit_model", "")
        standard_model = getattr(media, "image_edit_fast_model", "")
    if not isinstance(pro_model, str) or not pro_model:
        raise ValueError(f"В настройках не задана старшая модель для operation={operation!r}")
    if not isinstance(standard_model, str) or not standard_model:
        raise ValueError(f"В настройках не задана стандартная модель для operation={operation!r}")

    resolution_value = getattr(media, "image_default_resolution", None)
    resolution = resolution_value if isinstance(resolution_value, str) and resolution_value else None

    if getattr(media, "image_routing_mode", "pro_only") == "pro_only":
        selected_quality, reason = "pro", "pro_only_mode"
    elif quality == "pro" and isinstance(quality, str):
        selected_quality, reason = "pro", "explicit_pro"
    elif isinstance(user_text, str) and _EXPLICIT_PRO_RE.search(user_text):
        selected_quality, reason = "pro", "user_requested_pro"
    elif quality is not None and (not isinstance(quality, str) or quality not in {"standard", "pro"}):
        selected_quality, reason = "pro", "invalid_quality_hint"
    else:
        retry_flag, target_quality, target_model, target_operation, target_index = _retry_target_values(retry_target)
        retry_requested = is_retry is True or retry_flag
        inferred_target_quality = target_quality
        if inferred_target_quality is None and target_model is not None:
            inferred_target_quality = _configured_model_quality(
                media,
                target_operation,
                target_model,
            )

        if retry_requested and inferred_target_quality == "standard":
            selected_quality, reason = "pro", "retry_after_fast"
        elif retry_requested and inferred_target_quality == "pro":
            selected_quality, reason = "pro", "retry_preserve_pro"
        elif retry_requested:
            selected_quality, reason = "pro", "retry_unknown_origin"
        else:
            threshold = getattr(media, "image_multi_reference_pro_threshold", 2)
            if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
                threshold = 2
            count = (
                reference_count
                if isinstance(reference_count, int)
                and not isinstance(reference_count, bool)
                and reference_count >= 0
                else 0
            )
            if count > threshold:
                selected_quality, reason = "pro", "multi_reference_threshold"
            else:
                if has_paired_quotes(user_text) or has_paired_quotes(prompt):
                    selected_quality, reason = "pro", "quoted_text"
                elif has_image_text_marker(user_text) or has_image_text_marker(prompt):
                    selected_quality, reason = "pro", "text_marker"
                elif generated_result_improvement is True:
                    selected_quality, reason = "pro", "generated_result_improvement"
                else:
                    selected_quality, reason = "standard", "standard_default"

    if selected_quality == "pro":
        model = pro_model
    else:
        model = standard_model

    retry_requested = is_retry is True or _retry_target_values(retry_target)[0]
    target_index = _retry_target_values(retry_target)[4] if retry_requested else None
    return ImageRoutingDecision(
        model=model,
        quality=selected_quality,
        resolution=resolution,
        reason=reason,
        is_retry=retry_requested,
        retry_target_index=target_index,
    )
