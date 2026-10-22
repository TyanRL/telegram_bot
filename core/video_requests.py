"""Доменные параметры видео: активный профиль, разбор запроса и подготовка заявки.

Модуль не зависит от Telegram и от сетевого слоя. Здесь описаны:

- [`VideoModelProfile`](core/video_requests.py:78) — снимок активного видеопрофиля
  (модель, рабочее разрешение, допустимые длительности, варианты кнопок
  длительности, доступные пропорции, предел референсов);
- [`validate_duration()`](core/video_requests.py:163) — строгая проверка
  длительности по профилю без округления и подстановки значений;
- [`parse_explicit_duration()`](core/video_requests.py:196) — разбор однозначной
  длительности из исходного текста пользователя;
- [`parse_duration_answer()`](core/video_requests.py:281) — разбор ответа на
  вопрос длительности;
- [`parse_user_image_indices()`](core/video_requests.py:402) — разбор порядка
  номеров картинок, названного пользователем;
- [`resolve_video_references()`](core/video_requests.py:497) — разрешение
  референсов сессии и их реальных 1-based номеров в пользовательском порядке;
- [`VideoRequestDraft`](core/video_requests.py:577) — неизменяемый снимок
  пользовательской заявки, который допускает отсутствие длительности до
  ответа пользователя.

Готовые параметры запуска намеренно живут в
[`VideoGenerationOptions`](core/common_types.py:111): там длительность обязательна.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence
from uuid import uuid4

from core.common_types import GeneratedImage
from core.tool_helpers import (
    generated_image_from_session_dict,
    get_session_image_entries,
)

__all__ = [
    "AmbiguousDurationError",
    "AmbiguousImageOrderError",
    "ReferenceLimitExceededError",
    "UnsupportedDurationError",
    "VideoModelProfile",
    "VideoReferenceSelection",
    "VideoRequestDraft",
    "build_duration_validator",
    "build_video_request_draft",
    "parse_duration_answer",
    "parse_explicit_duration",
    "parse_user_image_indices",
    "resolve_video_references",
    "validate_duration",
]


class AmbiguousDurationError(ValueError):
    """Длительность в исходном запросе названа неоднозначно.

    Значение не угадывается: приложение задаёт вопрос о длительности.
    """


class UnsupportedDurationError(ValueError):
    """Пользователь явно попросил длительность вне диапазона профиля.

    Такой запрос отклоняется: молча подставлять ближайшее поддерживаемое
    значение нельзя.
    """


class AmbiguousImageOrderError(ValueError):
    """Порядок референсов в исходном запросе назван противоречиво.

    Значение не угадывается: приложение уточняет запрос у пользователя.
    """


class ReferenceLimitExceededError(ValueError):
    """Выбранный набор референсов больше предела активного видеопрофиля.

    Список не обрезается молча: пользователь должен явно уменьшить набор.
    """


# Единицы измерения длительности. Номера картинок длительностью не являются:
# требуется явная единица измерения рядом с числом.
_DURATION_UNIT = r"(?:секунд\w*|сек\.?|sec\w*|s\b)"
_DURATION_PATTERN = re.compile(
    r"(?<![\w.,])(\d{1,3})\s*" + _DURATION_UNIT,
    re.IGNORECASE,
)
_FRACTIONAL_DURATION_PATTERN = re.compile(
    r"(?<![\w.,])\d+[.,]\d+\s*" + _DURATION_UNIT,
    re.IGNORECASE,
)
_DURATION_RANGE_PATTERN = re.compile(
    r"(?<![\w.,])(\d+)\s*[-–—]\s*(\d+)\s*" + _DURATION_UNIT,
    re.IGNORECASE,
)
_DURATION_RANGE_FROM_TO_PATTERN = re.compile(
    r"(?<![\w.,])от\s*(\d+)\s*до\s*(\d+)\s*" + _DURATION_UNIT,
    re.IGNORECASE,
)
# Приблизительные и неопределённые указания длительности.
_APPROXIMATION_PATTERN = re.compile(
    r"(?<!\w)(?:около|примерно|приблизительно|пару|пара|несколько|"
    r"более|меньше|длиннее|короче|минимум|максимум|"
    r"about|around|approximately|roughly|a\s+few|several|longer|shorter)"
    r"(?!\w)",
    re.IGNORECASE,
)

# Слово-якорь, после которого пользователь перечисляет номера картинок.
_IMAGE_ANCHOR = (
    r"(?:изображени\w*|картин\w*|фотографи\w*|фото\w*|images?|pictures?|photos?)"
)
# Перечисление после одного слова: «картинки 3, 1, 2», «фото 3, 1 и 2».
_IMAGE_LIST_PATTERN = re.compile(
    r"(?<!\w)"
    + _IMAGE_ANCHOR
    + r"\s*(?:(?:№|#|номер|number|no\.?)\s*)?"
    + r"\d+(?:\s*(?:,|и|and|&)\s*\d+)*",
    re.IGNORECASE,
)
# Отдельные упоминания с явным знаком номера: «№3, №1, №2».
_IMAGE_MARKER_PATTERN = re.compile(r"(?<![\w])(?:№|#)\s*(\d+)(?!\w)")

_ORDINAL_WORDS: dict[str, int] = {
    "первый": 1, "первая": 1, "первое": 1, "первую": 1, "первого": 1,
    "первому": 1, "первой": 1, "первым": 1, "первыми": 1, "первых": 1,
    "первые": 1,
    "второй": 2, "вторая": 2, "второе": 2, "вторую": 2, "второго": 2,
    "второму": 2, "вторым": 2, "вторыми": 2, "вторых": 2, "вторые": 2,
    "третий": 3, "третья": 3, "третье": 3, "третью": 3, "третьего": 3,
    "третьему": 3, "третьей": 3, "третьим": 3, "третьими": 3, "третьих": 3,
    "третьи": 3,
    "четвёртый": 4, "четвертый": 4, "четвёртая": 4, "четвертая": 4,
    "четвёртое": 4, "четвертое": 4, "четвёртую": 4, "четвертую": 4,
    "четвёртого": 4, "четвертого": 4, "четвёртому": 4, "четвертому": 4,
    "четвёртой": 4, "четвертой": 4, "четвёртым": 4, "четвертым": 4,
    "четвёртыми": 4, "четвертыми": 4, "четвёртых": 4, "четвертых": 4,
    "четвёртые": 4, "четвертые": 4,
    "пятый": 5, "пятая": 5, "пятое": 5, "пятую": 5, "пятого": 5,
    "пятому": 5, "пятой": 5, "пятым": 5, "пятыми": 5, "пятых": 5,
    "пятые": 5,
    "шестой": 6, "шестая": 6, "шестое": 6, "шестую": 6, "шестого": 6,
    "шестому": 6, "шестой": 6, "шестым": 6, "шестыми": 6, "шестых": 6,
    "шестые": 6,
    "седьмой": 7, "седьмая": 7, "седьмое": 7, "седьмую": 7, "седьмого": 7,
    "седьмому": 7, "седьмой": 7, "седьмым": 7, "седьмыми": 7, "седьмых": 7,
    "седьмые": 7,
    "восьмой": 8, "восьмая": 8, "восьмое": 8, "восьмую": 8, "восьмого": 8,
    "восьмому": 8, "восьмой": 8, "восьмым": 8, "восьмыми": 8, "восьмых": 8,
    "восьмые": 8,
    "девятый": 9, "девятая": 9, "девятое": 9, "девятую": 9, "девятого": 9,
    "девятому": 9, "девятым": 9, "девятыми": 9, "девятых": 9, "девятые": 9,
    "десятый": 10, "десятая": 10, "десятое": 10, "десятую": 10,
    "десятого": 10, "десятому": 10, "десятым": 10, "десятыми": 10,
    "десятых": 10, "десятые": 10,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}

_ORDINAL_ALTERNATION = "|".join(
    sorted(_ORDINAL_WORDS, key=len, reverse=True)
)
_ORDINAL_PATTERN = re.compile(
    r"(?<![\w-])(?:" + _ORDINAL_ALTERNATION + r")(?![\w-])",
    re.IGNORECASE,
)
_ANCHOR_PRESENCE_PATTERN = re.compile(_IMAGE_ANCHOR, re.IGNORECASE)
# Между двумя порядковыми числительными может стоять только перечисление.
_ORDINAL_GROUP_GAP_PATTERN = re.compile(
    r"^[\s,;:()\-–—]*(?:(?:и|and|&)[\s,;:()\-–—]*)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class VideoModelProfile:
    """Неизменяемый снимок активного видеопрофиля.

    Профиль формируется из настроек приложения
    ([`Settings.video_profile`](core/config.py:536)) и передаётся дальше по
    слоям как готовый объект. Слои приложения не хранят собственные константы
    конкретной модели: смена модели означает смену профиля.
    """

    model: str
    resolution: str
    allowed_durations: tuple[int, ...]
    duration_presets: tuple[int, ...] = ()
    allowed_aspect_ratios: tuple[str, ...] = ()
    max_references: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("VideoModelProfile.model должен быть непустым идентификатором")
        if not isinstance(self.resolution, str) or not self.resolution.strip():
            raise ValueError("VideoModelProfile.resolution должен быть непустым")
        if not self.allowed_durations:
            raise ValueError(
                "VideoModelProfile.allowed_durations должен содержать хотя бы одно значение"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.allowed_durations
        ):
            raise ValueError(
                "VideoModelProfile.allowed_durations должен содержать только "
                "положительные целые числа"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.duration_presets
        ):
            raise ValueError(
                "VideoModelProfile.duration_presets должен содержать только "
                "положительные целые числа"
            )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.allowed_aspect_ratios
        ):
            raise ValueError(
                "VideoModelProfile.allowed_aspect_ratios должен содержать только "
                "непустые строки"
            )
        if (
            isinstance(self.max_references, bool)
            or not isinstance(self.max_references, int)
            or self.max_references < 1
        ):
            raise ValueError("VideoModelProfile.max_references должен быть положительным целым числом")

    @property
    def allowed_duration_values(self) -> frozenset[int]:
        """Допустимые длительности профиля как множество для быстрой проверки."""

        return frozenset(self.allowed_durations)

    def allows_duration(self, duration: int) -> bool:
        """Проверяет принадлежность длительности набору профиля."""

        if isinstance(duration, bool) or not isinstance(duration, int):
            return False
        return duration in self.allowed_duration_values

    def question_duration_presets(self) -> tuple[int, ...]:
        """Варианты длительности для кнопок вопроса.

        Пустой набор пресетов означает, что профиль их не задаёт: вопрос тогда
        строится из всех допустимых длительностей.
        """

        return tuple(self.duration_presets)


def validate_duration(duration: Any, profile: VideoModelProfile) -> int:
    """Проверяет длительность строго по активному профилю.

    Принимаются только целые числа (не ``bool``, не ``float``), входящие в
    ``profile.allowed_durations``. Значения не округляются, не подменяются и
    не нормализуются: любое несоответствие — ошибка валидации.
    """

    if isinstance(duration, bool) or not isinstance(duration, int):
        raise ValueError(
            "Длительность видео должна быть целым числом секунд, получено "
            f"{type(duration).__name__}"
        )
    if duration not in profile.allowed_duration_values:
        allowed = ", ".join(str(value) for value in profile.allowed_durations)
        raise ValueError(
            f"Длительность {duration} с не поддерживается активной моделью. "
            f"Допустимые значения: {allowed}"
        )
    return duration


def _duration_error_message(duration: int, profile: VideoModelProfile) -> str:
    allowed = ", ".join(str(value) for value in profile.allowed_durations)
    return (
        f"Длительность {duration} с не поддерживается активной моделью. "
        f"Допустимые значения: {allowed}"
    )


def parse_explicit_duration(text: Any, profile: VideoModelProfile) -> int | None:
    """Извлекает однозначную длительность из текста пользователя.

    Возвращает ``None``, если длительность в тексте не указана: тогда сервер
    задаёт вопрос о длительности. Указанная, но недопустимая длительность —
    ошибка [`UnsupportedDurationError`](core/video_requests.py:57): запрос
    отклоняется. Противоречивые, приблизительные и неоднозначные значения —
    ошибка [`AmbiguousDurationError`](core/video_requests.py:44): значение не
    угадывается, сервер задаёт вопрос.

    Значения не округляются и не подменяются ближайшим поддерживаемым.
    """

    if not isinstance(text, str) or not text.strip():
        return None

    range_matches = _DURATION_RANGE_PATTERN.findall(text)
    range_matches += _DURATION_RANGE_FROM_TO_PATTERN.findall(text)
    if range_matches:
        values = sorted({int(value) for pair in range_matches for value in pair})
        if values and all(
            not profile.allows_duration(value) for value in values
        ):
            raise UnsupportedDurationError(
                _duration_error_message(values[0], profile)
            )
        raise AmbiguousDurationError(
            "В запросе указан диапазон длительности: "
            + "-".join(str(value) for value in values)
            + " с. Уточните одну длительность."
        )

    if _FRACTIONAL_DURATION_PATTERN.search(text):
        raise AmbiguousDurationError(
            "В запросе указана дробная длительность. Уточните целое число секунд."
        )

    matches = {int(value) for value in _DURATION_PATTERN.findall(text)}
    if not matches:
        return None

    if _APPROXIMATION_PATTERN.search(text):
        raise AmbiguousDurationError(
            "Длительность в запросе указана приблизительно. Уточните точное "
            "значение в секундах."
        )

    if len(matches) > 1:
        raise AmbiguousDurationError(
            "В запросе указано несколько разных значений длительности: "
            + ", ".join(str(value) for value in sorted(matches))
            + ". Уточните одну длительность."
        )

    duration = matches.pop()
    if not profile.allows_duration(duration):
        raise UnsupportedDurationError(_duration_error_message(duration, profile))
    return duration


def parse_duration_answer(text: Any, profile: VideoModelProfile) -> int | None:
    """Извлекает длительность из свободного ответа пользователя.

    Ответ на вопрос длительности может быть записан как «7», «7 секунд» или
    «7 с». Значение проверяется по активному профилю: недопустимое значение
    не округляется и не подменяется ближайшим поддерживаемым, а отбрасывается
    целиком. Дробные значения, диапазоны и приблизительные указания также не
    принимаются.
    """

    if not isinstance(text, str) or not text.strip():
        return None

    cleaned = text.strip()
    if _APPROXIMATION_PATTERN.search(cleaned):
        return None

    numbers = re.findall(r"\d+(?:[.,]\d+)?", cleaned)
    if len(numbers) != 1:
        return None

    raw_value = numbers[0]
    if "." in raw_value or "," in raw_value:
        # Дробное значение не округляется: ответ не принимается целиком.
        return None

    try:
        return validate_duration(int(raw_value), profile)
    except ValueError:
        return None


def build_duration_validator(profile: VideoModelProfile) -> Callable[[str], str | None]:
    """Строит серверный валидатор ответа на вопрос длительности.

    Валидатор возвращает текст ошибки, если ответ нельзя применить, и
    ``None``, если ответ корректен.
    """

    allowed = ", ".join(str(value) for value in profile.allowed_durations)

    def validator(text: str) -> str | None:
        if parse_duration_answer(text, profile) is None:
            return (
                "Нужна длительность видео в секундах. "
                f"Допустимые значения: {allowed}."
            )
        return None

    return validator


def _collect_ordinal_groups(text: str) -> list[tuple[int, list[int]]]:
    """Группирует порядковые числительные в упоминания референсов.

    Порядковые числительные относятся к картинкам только если в тексте есть
    слово-якорь изображения. Группировка сохраняет порядок упоминаний и
    разделяет группы, между которыми стоят другие слова.
    """

    if _ANCHOR_PRESENCE_PATTERN.search(text) is None:
        return []

    matches = list(_ORDINAL_PATTERN.finditer(text))
    groups: list[tuple[int, list[int]]] = []
    for match in matches:
        word = match.group(0)
        number = _ORDINAL_WORDS.get(word.lower())
        if number is None:
            continue
        if groups:
            previous_end = groups[-1][0]
            gap = text[previous_end:match.start()]
            if _ORDINAL_GROUP_GAP_PATTERN.match(gap) is None:
                groups.append((match.start(), [number]))
                continue
            groups[-1][1].append(number)
            continue
        groups.append((match.start(), [number]))
    return groups


def parse_user_image_indices(text: Any) -> list[int]:
    """Извлекает номера картинок и их порядок из текста пользователя.

    Поддерживаются перечисления после одного слова («картинки 3, 1, 2»,
    «фото 3, 1 и 2»), отдельные номера со знаком («№3, №1, №2») и
    порядковые числительные («третью, первую и вторую картинку»).

    Длительность в секундах номером картинки не является: «на 10 секунд»
    разбирается отдельным парсером и сюда не попадает.

    Противоречивые перечисления считаются неоднозначными: вместо угадывания
    вызывающий код уточняет запрос у пользователя.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    mentions: list[tuple[int, list[int]]] = []

    for match in _IMAGE_LIST_PATTERN.finditer(text):
        numbers = [int(value) for value in re.findall(r"\d+", match.group(0))]
        if numbers:
            mentions.append((match.start(), numbers))

    for match in _IMAGE_MARKER_PATTERN.finditer(text):
        mentions.append((match.start(), [int(match.group(1))]))

    mentions.extend(_collect_ordinal_groups(text))

    if not mentions:
        return []

    mentions.sort(key=lambda item: item[0])

    ordered_mentions = [numbers for _, numbers in mentions]
    for first in ordered_mentions:
        if len(first) < 2:
            continue
        for second in ordered_mentions:
            if len(second) < 2 or second is first:
                continue
            if second != first:
                raise AmbiguousImageOrderError(
                    "Порядок картинок в запросе назван противоречиво: "
                    + ", ".join(str(value) for value in first)
                    + " и "
                    + ", ".join(str(value) for value in second)
                    + ". Уточните порядок референсов."
                )

    indices: list[int] = []
    seen: set[int] = set()
    for _, numbers in mentions:
        for number in numbers:
            if number < 1:
                continue
            if number in seen:
                # Номер используется один раз по первому вхождению.
                continue
            seen.add(number)
            indices.append(number)
    return indices


@dataclass(frozen=True, slots=True)
class VideoReferenceSelection:
    """Выбранные референсы и их реальные 1-based номера.

    Порядок совпадает с порядком пользователя: список не сортируется.
    """

    images: tuple[GeneratedImage, ...] = ()
    indices: tuple[int, ...] = ()


def resolve_video_references(
    session: dict[str, Any] | None,
    indices: Sequence[int] | None,
    *,
    profile: VideoModelProfile,
) -> VideoReferenceSelection:
    """Разрешает референсы видеозаявки и их реальные номера.

    Номера разрешаются через
    [`get_session_image_entries()`](core/tool_helpers.py:425): список не
    сортируется, а дубли отбрасываются по первому вхождению. При отсутствии
    или пустом списке номеров берутся все изображения сессии в её порядке;
    при отсутствии изображений заявка работает в текстовом режиме.

    Превышение предела референсов профиля, неверные номера и повреждённые
    картинки отклоняются до запуска генерации: список никогда не обрезается
    молча.
    """

    if session is None:
        if indices:
            raise ValueError(
                "В visual session нет изображений, но указаны номера референсов."
            )
        return VideoReferenceSelection()

    entries = get_session_image_entries(
        session,
        list(indices) if indices else None,
    )

    if len(entries) > profile.max_references:
        raise ReferenceLimitExceededError(
            f"Выбрано {len(entries)} референсов, а активная модель поддерживает "
            f"не более {profile.max_references}. Уменьшите набор явно: назовите "
            "нужные номера картинок."
        )

    images: list[GeneratedImage] = []
    real_indices: list[int] = []
    for number, image_dict in entries:
        images.append(generated_image_from_session_dict(image_dict))
        real_indices.append(number)

    return VideoReferenceSelection(
        images=tuple(images),
        indices=tuple(real_indices),
    )


@dataclass(frozen=True, slots=True)
class VideoRequestDraft:
    """Неизменяемый снимок пользовательской заявки на видео.

    Отличается от готовых параметров запуска тем, что длительность может быть
    ещё не выбрана: она указывается пользователем (явно в исходном запросе или
    ответом на вопрос) и только после проверки превращается в готовый запрос.

    Снимок содержит промпт, упорядоченные реальные номера референсов, байты
    выбранных изображений, снимок активного видеопрофиля и идентификатор
    операции. Новые загрузки изображений в уже созданную заявку не попадают.
    """

    prompt: str
    profile: VideoModelProfile
    reference_images: tuple[GeneratedImage, ...] = ()
    reference_indices: tuple[int, ...] = ()
    duration: int | None = None
    aspect_ratio: str | None = None
    operation_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("VideoRequestDraft.prompt не должен быть пустым")
        if not isinstance(self.profile, VideoModelProfile):
            raise ValueError("VideoRequestDraft.profile должен быть VideoModelProfile")
        if not isinstance(self.operation_id, str) or not self.operation_id.strip():
            raise ValueError("VideoRequestDraft.operation_id должен быть непустой строкой")
        object.__setattr__(self, "reference_images", tuple(self.reference_images))
        object.__setattr__(self, "reference_indices", tuple(self.reference_indices))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.duration is not None:
            validate_duration(self.duration, self.profile)
        if self.aspect_ratio is not None and self.profile.allowed_aspect_ratios:
            if self.aspect_ratio not in self.profile.allowed_aspect_ratios:
                allowed = ", ".join(self.profile.allowed_aspect_ratios)
                raise ValueError(
                    f"Пропорция {self.aspect_ratio} не поддерживается профилем. "
                    f"Допустимые значения: {allowed}"
                )
        if len(self.reference_images) > self.profile.max_references:
            raise ValueError(
                f"Выбрано {len(self.reference_images)} референсов, максимум "
                f"{self.profile.max_references}. Уменьшите набор явно."
            )
        if len(self.reference_indices) != len(self.reference_images):
            raise ValueError(
                "VideoRequestDraft.reference_indices должен соответствовать "
                "reference_images по длине"
            )
        for index in self.reference_indices:
            if isinstance(index, bool) or not isinstance(index, int) or index < 1:
                raise ValueError(
                    "VideoRequestDraft.reference_indices должен содержать "
                    "положительные целые номера картинок"
                )
        if len(set(self.reference_indices)) != len(self.reference_indices):
            raise ValueError(
                "VideoRequestDraft.reference_indices не должен содержать повторов"
            )

    @property
    def requires_duration(self) -> bool:
        """Длительность ещё не выбрана и должна быть уточнена у пользователя."""

        return self.duration is None

    def with_duration(self, duration: Any) -> "VideoRequestDraft":
        """Возвращает копию заявки с проверенной длительностью.

        Меняется только длительность: промпт, референсы, их порядок и снимок
        профиля остаются прежними.
        """

        return replace(self, duration=validate_duration(duration, self.profile))


def _snapshot_image(image: GeneratedImage) -> GeneratedImage:
    """Делает независимый снимок изображения для заявки."""

    return GeneratedImage(
        content=bytes(image.content),
        mime_type=image.mime_type,
        metadata=dict(image.metadata),
    )


def build_video_request_draft(
    prompt: str,
    profile: VideoModelProfile,
    *,
    reference_images: Sequence[GeneratedImage] = (),
    reference_indices: Sequence[int] = (),
    duration: Any = None,
    aspect_ratio: str | None = None,
    operation_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> VideoRequestDraft:
    """Готовит черновик заявки и сразу отклоняет заведомо недопустимые данные.

    Проверяется предел референсов профиля; лишние картинки не обрезаются
    молча, а длительность проверяется только если она уже задана.
    """

    return VideoRequestDraft(
        prompt=prompt,
        profile=profile,
        reference_images=tuple(_snapshot_image(image) for image in reference_images),
        reference_indices=tuple(reference_indices),
        duration=duration,
        aspect_ratio=aspect_ratio,
        operation_id=operation_id or uuid4().hex,
        metadata=dict(metadata or {}),
    )
