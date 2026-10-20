"""Проверки чистых правил маршрутизации изображений и контекста повторов."""

from dataclasses import FrozenInstanceError, replace

import pytest

from core.config import DEFAULT_SYSTEM_PROMPT, MediaSettings
from core.image_routing import (
    ImageRoutingDecision,
    analyze_image_retry,
    has_image_text_marker,
    has_paired_quotes,
    resolve_image_retry_context,
    resolve_image_route,
)
from core.tool_registry import TOOLS_SCHEMA


def _settings(**overrides: object) -> MediaSettings:
    defaults: dict[str, object] = {
        "image_generation_model": "test/generation-pro",
        "image_edit_model": "test/edit-pro",
        "image_generation_fast_model": "test/generation-fast",
        "image_edit_fast_model": "test/edit-fast",
        "image_routing_mode": "auto",
        "image_default_resolution": "1K",
        "image_multi_reference_pro_threshold": 2,
    }
    defaults.update(overrides)
    return MediaSettings(**defaults)  # type: ignore[arg-type]


def _image(
    *,
    operation: str = "generate",
    quality: str | None = "standard",
    model: str | None = "test/generation-fast",
    prompt: str = "A detailed red fox in a snowy forest",
    source_image_indices: list[int] | None = None,
    source_kind: str = "generated",
    generation: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] | None = generation
    if metadata is None and (quality is not None or model is not None):
        metadata = {
            "operation": operation,
            "quality": quality,
            "model": model,
            "prompt": prompt,
            "source_image_indices": source_image_indices or [],
        }
    image: dict[str, object] = {"source_kind": source_kind}
    image["generation"] = metadata
    return image


STAGE_8_QUALITY_FIXTURE = (
    pytest.param(
        "generate", "standard", "Нарисуй тихое озеро на рассвете",
        "A quiet lake at sunrise", 0, "standard", "standard_default",
        id="ordinary-scene",
    ),
    pytest.param(
        "generate", "standard", "Сделай книжную иллюстрацию с лисой",
        "A whimsical ink illustration of a red fox", 0, "standard", "standard_default",
        id="illustration",
    ),
    pytest.param(
        "generate", "standard", "Нарисуй латунный компас",
        "A polished brass compass on a neutral background", 0, "standard", "standard_default",
        id="ordinary-object",
    ),
    pytest.param(
        "edit", "standard", "Стилизуй фотографию под акварель",
        "Restyle the landscape as a loose watercolor", 1, "standard", "standard_default",
        id="stylization",
    ),
    pytest.param(
        "edit", "standard", "Поменяй цвет кружки с синего на красный",
        "Change only the mug color from blue to red", 2, "standard", "standard_default",
        id="simple-change",
    ),
    pytest.param(
        "generate", "standard", "Точная русская надпись «СЕВЕР» на упаковке",
        'A package with the exact Russian label «СЕВЕР»', 0, "pro", "quoted_text",
        id="exact-russian-text",
    ),
    pytest.param(
        "generate", "standard", 'Write the exact English text "OPEN 24/7" on the sign',
        'A storefront sign reading "OPEN 24/7"', 0, "pro", "quoted_text",
        id="exact-english-text",
    ),
    pytest.param(
        "generate", "pro", "Сохрани портретное сходство с человеком на фото",
        "A faithful portrait preserving the person's identity", 1, "pro", "explicit_pro",
        id="portrait-likeness",
    ),
    pytest.param(
        "generate", "pro", "Создай сложную многоплановую композицию",
        "A complex multi-layered editorial composition", 0, "pro", "explicit_pro",
        id="complex-composition",
    ),
    pytest.param(
        "edit", "pro", "Точно сохрани мелкие детали исходной гравировки",
        "Preserve every fine engraved detail in the source", 1, "pro", "explicit_pro",
        id="detail-preservation",
    ),
)


@pytest.mark.parametrize(
    (
        "operation",
        "quality_hint",
        "user_text",
        "prompt",
        "reference_count",
        "expected_quality",
        "expected_reason",
    ),
    STAGE_8_QUALITY_FIXTURE,
)
def test_stage_eight_fixed_offline_quality_fixture(
    operation: str,
    quality_hint: str,
    user_text: str,
    prompt: str,
    reference_count: int,
    expected_quality: str,
    expected_reason: str,
) -> None:
    """Проверяет ожидаемый маршрут фикстуры; качество live-вывода проверяется отдельно."""

    decision = resolve_image_route(
        _settings(),
        operation,
        quality=quality_hint,
        prompt=prompt,
        user_text=user_text,
        reference_count=reference_count,
    )

    expected_model = (
        "test/generation-pro"
        if operation == "generate" and expected_quality == "pro"
        else "test/generation-fast"
        if operation == "generate"
        else "test/edit-pro"
        if expected_quality == "pro"
        else "test/edit-fast"
    )
    assert decision.model == expected_model
    assert decision.quality == expected_quality
    assert decision.reason == expected_reason
    assert decision.resolution == "1K"


@pytest.mark.parametrize(
    ("routing_mode", "resolution", "expected_model", "expected_quality"),
    [
        pytest.param("pro_only", None, "test/generation-pro", "pro", id="legacy-pro-provider-resolution"),
        pytest.param("pro_only", "1K", "test/generation-pro", "pro", id="pro-only-1k"),
        pytest.param("auto", "1K", "test/generation-fast", "standard", id="hybrid-1k"),
    ],
)
def test_stage_eight_configuration_comparison_matrix(
    routing_mode: str,
    resolution: str | None,
    expected_model: str,
    expected_quality: str,
) -> None:
    decision = resolve_image_route(
        _settings(
            image_routing_mode=routing_mode,
            image_default_resolution=resolution,
        ),
        "generate",
        quality="standard",
        prompt="A simple landscape",
    )

    assert decision.model == expected_model
    assert decision.quality == expected_quality
    assert decision.resolution == resolution


@pytest.mark.parametrize(
    ("quality", "expected_quality", "expected_reason"),
    [
        ("standard", "standard", "standard_default"),
        ("pro", "pro", "explicit_pro"),
        (None, "standard", "standard_default"),
        ("ultra", "pro", "invalid_quality_hint"),
        (123, "pro", "invalid_quality_hint"),
        (True, "pro", "invalid_quality_hint"),
    ],
)
def test_quality_hints_select_expected_route(
    quality: object,
    expected_quality: str,
    expected_reason: str,
) -> None:
    decision = resolve_image_route(
        _settings(),
        "generate",
        quality=quality,
        prompt="A simple landscape",
    )

    assert decision.quality == expected_quality
    assert decision.reason == expected_reason
    assert decision.model == (
        "test/generation-pro" if expected_quality == "pro" else "test/generation-fast"
    )
    assert decision.resolution == "1K"


def test_route_decision_is_immutable() -> None:
    decision = ImageRoutingDecision(
        model="test/model",
        quality="standard",
        resolution="1K",
        reason="standard_default",
    )

    with pytest.raises(FrozenInstanceError):
        decision.quality = "pro"  # type: ignore[misc]


def test_pro_only_mode_is_highest_priority() -> None:
    decision = resolve_image_route(
        _settings(image_routing_mode="pro_only"),
        "edit",
        quality="standard",
        prompt='Put the word "HELLO" on the sign',
        reference_count=10,
        is_retry=True,
    )

    assert decision.quality == "pro"
    assert decision.model == "test/edit-pro"
    assert decision.reason == "pro_only_mode"


@pytest.mark.parametrize(
    "text",
    [
        "Сделай максимальное качество",
        "Нужно профессиональное качество",
        "Выбери наилучшее качество",
        "Use the highest quality",
        "Use best possible quality",
    ],
)
def test_explicit_user_request_for_maximum_quality(text: str) -> None:
    decision = resolve_image_route(
        _settings(),
        "generate",
        quality="standard",
        user_text=text,
    )

    assert decision.quality == "pro"
    assert decision.reason == "user_requested_pro"


def test_invalid_quality_is_checked_after_explicit_user_request() -> None:
    decision = resolve_image_route(
        _settings(),
        "generate",
        quality="ultra",
        user_text="Сделай максимальное качество",
    )

    assert decision.reason == "user_requested_pro"


@pytest.mark.parametrize(
    ("operation", "expected_fast", "expected_pro"),
    [
        ("generate", "test/generation-fast", "test/generation-pro"),
        ("edit", "test/edit-fast", "test/edit-pro"),
    ],
)
def test_model_ids_are_selected_from_operation_settings(
    operation: str,
    expected_fast: str,
    expected_pro: str,
) -> None:
    settings = _settings()

    standard = resolve_image_route(settings, operation, quality="standard")
    pro = resolve_image_route(settings, operation, quality="pro")

    assert standard.model == expected_fast
    assert pro.model == expected_pro


@pytest.mark.parametrize("operation", ["generate", "edit"])
def test_route_rejects_unknown_operation(operation: str) -> None:
    with pytest.raises(ValueError, match="operation"):
        resolve_image_route(_settings(), "video" if operation == "generate" else "invalid")


@pytest.mark.parametrize(
    ("text", "quote_type"),
    [
        ('A sign with "OPEN"', "straight double"),
        ("A sign with 'OPEN'", "straight single"),
        ("Вывеска «ОТКРЫТО»", "guillemets"),
        ("Вывеска “ОТКРЫТО”", "curly double"),
        ("Вывеска „ОТКРЫТО“", "low/high double"),
        ("Вывеска ‘ОТКРЫТО’", "curly single"),
    ],
)
def test_all_common_paired_quote_forms_force_pro(text: str, quote_type: str) -> None:
    assert has_paired_quotes(text), quote_type
    decision = resolve_image_route(_settings(), "generate", prompt=text)

    assert decision.quality == "pro"
    assert decision.reason == "quoted_text"


@pytest.mark.parametrize(
    "text",
    [
        "don't change the color",
        "the cat's hat is red",
        "don't touch the cat's hat",
        "it’s a nice illustration",
    ],
)
def test_english_word_apostrophes_are_not_paired_quotes(text: str) -> None:
    assert not has_paired_quotes(text)
    decision = resolve_image_route(_settings(), "generate", prompt=text)

    assert decision.quality == "standard"
    assert decision.reason == "standard_default"


def test_empty_or_unmatched_quotes_do_not_count() -> None:
    assert not has_paired_quotes('Draw an empty pair ""')
    assert not has_paired_quotes('Draw one unmatched " mark')
    assert not has_paired_quotes("don't")


@pytest.mark.parametrize(
    "text",
    [
        "Добавь надпись на фасад",
        "Нарисуй вывеску",
        "Напиши текст на чашке",
        "Добавь типографику",
        "Use bold typography",
        "Make a sign",
        "A sign saying OPEN",
        "Add text to the image",
        "Add lettering to the poster",
        "Put a caption under the photo",
        "A logo с текстом",
        "Add a label and a banner",
        "Write a title on the image",
        "Add an inscription",
        "Add text saying hello",
    ],
)
def test_image_text_markers_force_pro(text: str) -> None:
    assert has_image_text_marker(text)
    decision = resolve_image_route(_settings(), "generate", user_text=text)

    assert decision.quality == "pro"
    assert decision.reason == "text_marker"


@pytest.mark.parametrize(
    "text",
    [
        "текстура камня",
        "текстильная ткань",
        "контекст сцены",
        "архитектура здания",
        "modern design",
        "Textured fabric design",
    ],
)
def test_word_boundaries_avoid_false_text_marker_matches(text: str) -> None:
    assert not has_image_text_marker(text)
    decision = resolve_image_route(_settings(), "generate", prompt=text)

    assert decision.quality == "standard"
    assert decision.reason == "standard_default"


@pytest.mark.parametrize(
    ("reference_count", "expected_quality", "expected_reason"),
    [
        (0, "standard", "standard_default"),
        (1, "standard", "standard_default"),
        (2, "standard", "standard_default"),
        (3, "pro", "multi_reference_threshold"),
        (10, "pro", "multi_reference_threshold"),
    ],
)
def test_reference_threshold_is_strictly_exceeded(
    reference_count: int,
    expected_quality: str,
    expected_reason: str,
) -> None:
    decision = resolve_image_route(
        _settings(),
        "edit",
        quality="standard",
        reference_count=reference_count,
    )

    assert decision.quality == expected_quality
    assert decision.reason == expected_reason


@pytest.mark.parametrize(
    ("retry_target", "expected_reason"),
    [
        ({"is_retry": True, "target_quality": "standard"}, "retry_after_fast"),
        ({"is_retry": True, "target_quality": "pro"}, "retry_preserve_pro"),
        ({"is_retry": True}, "retry_unknown_origin"),
    ],
)
def test_retry_origin_selects_pro_and_is_not_downgraded_by_hints(
    retry_target: dict[str, object],
    expected_reason: str,
) -> None:
    decision = resolve_image_route(
        _settings(),
        "generate",
        quality="standard",
        prompt='Put "HELLO" on a sign',
        reference_count=10,
        retry_target=retry_target,
    )

    assert decision.quality == "pro"
    assert decision.reason == expected_reason


def test_retry_target_model_is_mapped_using_app_settings() -> None:
    decision = resolve_image_route(
        _settings(),
        "edit",
        quality="standard",
        retry_target={
            "is_retry": True,
            "target_model": "test/generation-fast",
            "target_operation": "generate",
        },
    )

    assert decision.quality == "pro"
    assert decision.reason == "retry_after_fast"


def test_priority_order_keeps_retry_above_reference_count_and_text_rules() -> None:
    fast_retry = resolve_image_route(
        _settings(),
        "generate",
        quality="standard",
        prompt='Put "HELLO" on a sign',
        reference_count=10,
        retry_target={"is_retry": True, "target_quality": "standard"},
    )
    unknown_retry = resolve_image_route(
        _settings(),
        "generate",
        quality="ultra",
        retry_target={"is_retry": True},
    )
    multi_reference = resolve_image_route(
        _settings(),
        "generate",
        quality="standard",
        prompt='Put "HELLO" on a sign',
        reference_count=3,
    )
    paired_quote = resolve_image_route(
        _settings(),
        "generate",
        quality="standard",
        prompt='A "quoted" sign',
        user_text="Добавь надпись",
    )

    assert fast_retry.reason == "retry_after_fast"
    assert unknown_retry.reason == "invalid_quality_hint"
    assert multi_reference.reason == "multi_reference_threshold"
    assert paired_quote.reason == "quoted_text"


def test_router_checks_user_text_and_prompt_independently() -> None:
    split_pair = resolve_image_route(
        _settings(),
        "generate",
        user_text='A scene starts "here',
        prompt='but this ends here"',
    )

    assert split_pair.quality == "standard"
    assert split_pair.reason == "standard_default"


def test_pro_only_is_default_for_settings_that_omit_routing_mode() -> None:
    settings = replace(_settings(), image_routing_mode="pro_only")
    decision = resolve_image_route(settings, "generate", quality="standard")

    assert decision.reason == "pro_only_mode"
    assert decision.model == settings.image_generation_model


def test_analysis_detects_current_complaint_but_does_not_reuse_old_history() -> None:
    generated = _image()
    complaint = analyze_image_retry("Плохо получилось, переделай заново", [generated])
    unrelated = analyze_image_retry("Нарисуй новый пейзаж", [generated])

    assert complaint.is_retry
    assert complaint.is_complaint
    assert not unrelated.is_retry
    assert not unrelated.is_complaint


@pytest.mark.parametrize(
    "text",
    [
        "Плохо получилось, сделай заново",
        "Качество плохое, переделай",
        "Надпись кривая, исправь",
        "Результат не похож на исходный портрет",
        "Картинка размытая, повтори",
        "Видны артефакты, попробуй еще раз",
        "The image is blurry, try again",
        "The sign text is incorrect; fix it",
    ],
)
def test_complaint_and_retry_phrases_are_recognized(text: str) -> None:
    analysis = analyze_image_retry(text, [_image()])

    assert analysis.is_retry


def test_retry_after_fast_raises_only_the_linked_operation() -> None:
    generated = _image(quality="standard", model="test/generation-fast")
    context = analyze_image_retry("Плохо получилось, переделай заново", [generated])
    retry = resolve_image_route(
        _settings(),
        "generate",
        quality="standard",
        user_text="Плохо получилось, переделай заново",
        retry_target=context,
    )
    new_task_context = analyze_image_retry("Нарисуй простой синий шар", [generated])
    new_task = resolve_image_route(
        _settings(),
        "generate",
        quality="standard",
        retry_target=new_task_context,
    )

    assert retry.quality == "pro"
    assert retry.reason == "retry_after_fast"
    assert retry.is_retry
    assert new_task.quality == "standard"
    assert new_task.reason == "standard_default"
    assert not new_task.is_retry


def test_ordinary_color_or_detail_change_is_not_a_complaint_or_retry() -> None:
    generated = _image()
    analysis = analyze_image_retry("Сделай фон синим и добавь шляпу", [generated])
    decision = resolve_image_route(
        _settings(),
        "edit",
        quality="standard",
        user_text="Сделай фон синим и добавь шляпу",
        retry_target=analysis,
    )

    assert not analysis.is_retry
    assert not analysis.is_complaint
    assert analysis.is_improvement
    assert decision.quality == "standard"
    assert decision.reason == "standard_default"


def test_explicit_improvement_of_generated_result_uses_pro_without_becoming_retry() -> None:
    decision = resolve_image_route(
        _settings(),
        "edit",
        quality="standard",
        reference_count=1,
        generated_result_improvement=True,
    )

    assert decision.quality == "pro"
    assert decision.reason == "generated_result_improvement"
    assert not decision.is_retry


def test_explicit_image_number_selects_that_retry_target() -> None:
    images = [
        _image(quality="standard", model="test/generation-fast"),
        _image(quality="pro", model="test/generation-pro"),
    ]
    analysis = analyze_image_retry("Переделай изображение №1, получилось плохо", images)

    assert analysis.is_retry
    assert analysis.target_index == 1
    assert analysis.target_quality == "standard"


def test_latest_generated_result_is_selected_not_later_uploaded_photo() -> None:
    images = [
        _image(quality="standard", model="test/generation-fast"),
        _image(
            quality=None,
            model=None,
            source_kind="uploaded",
            generation=None,
        ),
    ]
    analysis = analyze_image_retry("Плохо получилось, переделай заново", images)

    assert analysis.target_index == 1
    assert analysis.target_quality == "standard"


def test_legacy_generated_result_without_generation_metadata_keeps_unknown_origin() -> None:
    legacy = _image(quality=None, model=None, generation=None)
    analysis = analyze_image_retry("Неудачно получилось, повтори", [legacy])
    decision = resolve_image_route(
        _settings(),
        "generate",
        quality="standard",
        retry_target=analysis,
    )

    assert analysis.target_index == 1
    assert not analysis.origin_known
    assert decision.quality == "pro"
    assert decision.reason == "retry_unknown_origin"


def test_ambiguous_multiple_targets_are_not_arbitrarily_replaced() -> None:
    images = [_image(), _image(quality="standard", model="test/generation-fast")]
    analysis = analyze_image_retry(
        "Переделай изображения №1 и №2, результат плохой",
        images,
    )
    decision = resolve_image_route(
        _settings(),
        "edit",
        quality="standard",
        retry_target=analysis,
    )

    assert analysis.ambiguous_target
    assert analysis.target_index is None
    assert decision.quality == "pro"
    assert decision.reason == "retry_unknown_origin"


def test_retry_generation_restores_prompt_without_using_failed_result_as_reference() -> None:
    generated = _image(
        operation="generate",
        quality="standard",
        model="test/generation-fast",
        prompt="A red fox wearing a wool scarf in a snowy forest",
    )
    context = resolve_image_retry_context(
        "Плохо получилось, переделай заново, сохрани красный шарф",
        [generated],
        "generate",
        current_prompt="A red fox wearing a wool scarf in a snowy forest; keep the scarf red",
    )

    assert context.is_retry
    assert context.operation == "generate"
    assert context.reference_image_indices == ()
    assert not context.use_result_as_reference
    assert "A red fox wearing a wool scarf in a snowy forest" in (context.prompt or "")
    assert "сохрани красный шарф" in (context.prompt or "")


def test_failed_edit_retry_restores_original_available_sources() -> None:
    images = [
        {"source_kind": "uploaded", "generation": None},
        _image(
            operation="edit",
            quality="standard",
            model="test/edit-fast",
            prompt="Turn the uploaded portrait into a watercolor",
            source_image_indices=[1],
        ),
        {"source_kind": "uploaded", "generation": None},
    ]
    context = resolve_image_retry_context(
        "Качество плохое, переделай редактирование",
        images,
        "edit",
        current_prompt="Turn the portrait into a watercolor with softer edges",
    )

    assert context.target_index == 2
    assert context.target_operation == "edit"
    assert context.reference_image_indices == (1,)
    assert context.source_indices_available is True
    assert not context.use_result_as_reference
    assert "Turn the uploaded portrait into a watercolor" in (context.prompt or "")
    assert "softer edges" in (context.prompt or "")


def test_failed_edit_retry_does_not_silently_replace_unavailable_sources() -> None:
    failed_edit = _image(
        operation="edit",
        quality="standard",
        model="test/edit-fast",
        source_image_indices=[4, 7],
    )
    context = resolve_image_retry_context(
        "Плохой результат, переделай",
        [failed_edit, {"source_kind": "uploaded", "generation": None}],
        "edit",
    )

    assert context.reference_image_indices == ()
    assert context.source_indices_available is False
    assert context.missing_source_indices == (4, 7)


def test_edit_improvement_can_use_the_selected_generated_result_as_reference() -> None:
    generated = _image()
    context = resolve_image_retry_context(
        "Добавь шляпу и сделай фон синим",
        [generated],
        "edit",
        current_prompt="Add a hat and make the background blue",
    )

    assert not context.is_retry
    assert context.is_improvement
    assert context.use_result_as_reference
    assert context.target_is_generated
    assert context.reference_image_indices == (1,)
    assert context.prompt == "Add a hat and make the background blue"


def test_explicit_uploaded_photo_is_not_mislabeled_as_a_generated_result() -> None:
    uploaded = {"source_kind": "uploaded", "generation": None}
    context = resolve_image_retry_context(
        "Добавь шляпу на изображение №1",
        [uploaded],
        "edit",
    )

    assert context.is_improvement
    assert not context.target_is_generated
    assert not context.use_result_as_reference
    assert context.reference_image_indices == (1,)


def test_missing_explicit_target_does_not_pick_a_different_image() -> None:
    analysis = analyze_image_retry(
        "Исправь плохой результат на изображении №9",
        [_image()],
    )
    context = resolve_image_retry_context(
        "Исправь плохой результат на изображении №9",
        [_image()],
        "edit",
    )

    assert analysis.target_index is None
    assert analysis.target_missing
    assert context.reference_image_indices == ()
    assert context.target_missing


def test_edit_retry_without_source_metadata_reports_unavailable_sources() -> None:
    legacy_edit = {
        "source_kind": "edit",
        "generation": {"operation": "edit", "prompt": "original"},
    }
    context = resolve_image_retry_context(
        "Плохо получилось, переделай",
        [legacy_edit],
        "edit",
    )

    assert context.source_indices_available is False
    assert context.reference_image_indices == ()


def test_tool_schemas_expose_only_the_quality_choice_for_both_image_tools() -> None:
    image_tools = {
        tool["name"]: tool
        for tool in TOOLS_SCHEMA
        if tool.get("name") in {"generate_image", "generate_image_from_image"}
    }

    assert set(image_tools) == {"generate_image", "generate_image_from_image"}
    for tool in image_tools.values():
        parameters = tool["parameters"]
        assert parameters["required"] == ["prompt"]
        properties = parameters["properties"]
        assert properties["quality"]["type"] == "string"
        assert properties["quality"]["enum"] == ["standard", "pro"]
        assert "model" not in properties


def test_default_prompt_contains_routing_and_retry_instructions() -> None:
    assert 'quality="standard"' in DEFAULT_SYSTEM_PROMPT
    assert 'quality="pro"' in DEFAULT_SYSTEM_PROMPT
    assert "исходную полную инструкцию" in DEFAULT_SYSTEM_PROMPT
    assert "не превращай неудачный результат в референс" in DEFAULT_SYSTEM_PROMPT
    assert "исходные референсы той операции" in DEFAULT_SYSTEM_PROMPT
    assert "не обещать пользователю разрешение выше 1K" in DEFAULT_SYSTEM_PROMPT
