"""Регрессионные проверки ограниченного технического резерва для Images API."""

import asyncio
import base64
from dataclasses import replace
import json
import os
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:TEST_TOKEN")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import utils.openrouter_client as openrouter_client
from core.common_types import GeneratedImage, ImageEditOptions, ImageGenerationOptions
from core.config import settings
from core.generation_error_mapper import GenerationFailureReason, map_generation_error
from utils.openrouter_client import (
    OpenRouterBillingError,
    OpenRouterConfig,
    OpenRouterMediaError,
    OpenRouterResponseError,
    OpenRouterSdkUnavailableError,
    OpenRouterTemporaryError,
    OpenRouterTimeoutError,
    OpenRouterUnsupportedError,
    OpenRouterUnsupportedParameterError,
    OpenRouterAuthenticationError,
    OpenRouterRateLimitError,
    OpenRouterService,
)


GENERATION_PRO = settings.media.image_generation_model
GENERATION_FAST = settings.media.image_generation_fast_model
EDIT_PRO = settings.media.image_edit_model
EDIT_FAST = settings.media.image_edit_fast_model
PROMPT = "A detailed blue glass pavilion"


class _ProviderError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.body = json.dumps({"error": {"message": message}})


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (2, 2), color=(15, 25, 35)).save(output, format="PNG")
    return output.getvalue()


def _image_response(*, cost: float | None = 0.25) -> SimpleNamespace:
    encoded = base64.b64encode(_png_bytes()).decode("ascii")
    usage = SimpleNamespace(cost=cost) if cost is not None else None
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=encoded, media_type="image/png")],
        usage=usage,
    )


def _service(generate_async: AsyncMock, *, timeout_ms: int = 5_000) -> OpenRouterService:
    service = OpenRouterService(OpenRouterConfig(api_key="test-api-key", timeout_ms=timeout_ms))
    service._client = cast(
        Any,
        SimpleNamespace(images=SimpleNamespace(generate_async=generate_async)),
    )
    return service


def _settings_with(**media_overrides: Any) -> Any:
    media = replace(settings.media, **media_overrides)
    return replace(settings, media=media)


def _invoke_image(
    service: OpenRouterService,
    operation: str,
    options: ImageGenerationOptions | ImageEditOptions,
    references: list[GeneratedImage] | None = None,
) -> GeneratedImage:
    async def run() -> GeneratedImage:
        if operation == "generate":
            assert isinstance(options, ImageGenerationOptions)
            return await service.generate_image(PROMPT, options)
        assert isinstance(options, ImageEditOptions)
        return await service.edit_image(
            references or [_reference_image(b"source")],
            PROMPT,
            options,
        )

    return asyncio.run(run())


def _reference_image(content: bytes) -> GeneratedImage:
    return GeneratedImage(content=content, mime_type="image/png")


def _options(operation: str, model: str) -> ImageGenerationOptions | ImageEditOptions:
    if operation == "generate":
        return ImageGenerationOptions(
            aspect_ratio="16:9",
            resolution="2K",
            n=1,
            model=model,
        )
    return ImageEditOptions(
        aspect_ratio="3:2",
        resolution="1K",
        n=1,
        model=model,
    )


@pytest.mark.parametrize("operation", ["generate", "edit"])
@pytest.mark.parametrize(
    ("status_code", "message", "fallback_reason"),
    [
        (404, "Model flux-3-image-klein was not found", "unsupported_model"),
        (502, "Image generation failed", "generation_failed"),
    ],
)
def test_klein_falls_back_once_to_corresponding_pro_model(
    operation: str,
    status_code: int,
    message: str,
    fallback_reason: str,
) -> None:
    sdk_call = AsyncMock(
        side_effect=[_ProviderError(status_code, message), _image_response(cost=0.37)]
    )
    service = _service(sdk_call)
    options = _options(
        operation,
        GENERATION_FAST if operation == "generate" else EDIT_FAST,
    )

    with patch.object(openrouter_client, "settings", _settings_with(image_fallback_enabled=True)):
        image = _invoke_image(service, operation, options)

    pro_model = GENERATION_PRO if operation == "generate" else EDIT_PRO
    assert sdk_call.await_count == 2
    calls = [call.kwargs for call in sdk_call.await_args_list]
    assert [call["model"] for call in calls] == [options.model, pro_model]
    assert [call["prompt"] for call in calls] == [PROMPT, PROMPT]
    assert [call["aspect_ratio"] for call in calls] == [options.aspect_ratio] * 2
    assert [call["resolution"] for call in calls] == [options.resolution] * 2
    assert [call["n"] for call in calls] == [options.n, options.n]
    assert all(call["retries"] is None for call in calls)
    assert all(isinstance(call["timeout_ms"], int) and call["timeout_ms"] > 0 for call in calls)

    assert image.metadata["model"] == pro_model
    assert image.metadata["requested_model"] == options.model
    assert image.metadata["fallback_used"] is True
    assert image.metadata["fallback_reason"] == fallback_reason
    assert image.metadata["cost"] == 0.37
    assert image.metadata["attempts"] == [
        {
            "model": options.model,
            "status": status_code,
            "status_code": status_code,
            "duration_ms": image.metadata["attempts"][0]["duration_ms"],
            "cost": 0.0,
            "known_cost": 0.0,
            "billing_unknown": False,
            "error_reason": fallback_reason,
            "reason": fallback_reason,
        },
        {
            "model": pro_model,
            "status": 200,
            "status_code": 200,
            "duration_ms": image.metadata["attempts"][1]["duration_ms"],
            "cost": 0.37,
            "known_cost": 0.37,
            "billing_unknown": False,
            "error_reason": None,
            "reason": None,
        },
    ]
    assert image.metadata["total_known_cost"] == pytest.approx(0.37)
    assert image.metadata["billing_unknown"] is False


def test_first_attempt_success_reports_model_cost_and_complete_attempt_metadata() -> None:
    sdk_call = AsyncMock(return_value=_image_response(cost=0.08))
    service = _service(sdk_call)

    with patch.object(openrouter_client, "settings", _settings_with(image_fallback_enabled=True)):
        image = _invoke_image(
            service,
            "generate",
            _options("generate", GENERATION_FAST),
        )

    assert sdk_call.await_count == 1
    assert image.metadata["model"] == GENERATION_FAST
    assert image.metadata["requested_model"] == GENERATION_FAST
    assert image.metadata["fallback_used"] is False
    assert image.metadata["fallback_reason"] is None
    assert image.metadata["cost"] == 0.08
    assert image.metadata["total_known_cost"] == pytest.approx(0.08)
    assert image.metadata["billing_unknown"] is False
    assert len(image.metadata["attempts"]) == 1
    assert image.metadata["attempts"][0]["model"] == GENERATION_FAST
    assert image.metadata["attempts"][0]["status"] == 200
    assert image.metadata["attempts"][0]["cost"] == pytest.approx(0.08)


def test_edit_fallback_keeps_exact_prompt_options_and_reference_snapshot() -> None:
    external_references = [_reference_image(b"first-reference"), _reference_image(b"second-reference")]
    replacement_reference = _reference_image(b"concurrent-upload")
    invocation = 0

    async def fail_then_succeed(**kwargs: Any) -> SimpleNamespace:
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            external_references[:] = [replacement_reference]
            raise _ProviderError(404, "No available provider for model flux-3-image-klein")
        return _image_response(cost=0.31)

    sdk_call = AsyncMock(side_effect=fail_then_succeed)
    service = _service(sdk_call)
    options = ImageEditOptions(
        aspect_ratio="4:3",
        resolution="2K",
        n=1,
        model=EDIT_FAST,
    )

    with patch.object(openrouter_client, "settings", _settings_with(image_fallback_enabled=True)):
        image = _invoke_image(service, "edit", options, external_references)

    calls = [call.kwargs for call in sdk_call.await_args_list]
    expected_urls = [
        "data:image/png;base64," + base64.b64encode(b"first-reference").decode("ascii"),
        "data:image/png;base64," + base64.b64encode(b"second-reference").decode("ascii"),
    ]
    for call, model in zip(calls, [EDIT_FAST, EDIT_PRO], strict=True):
        assert call["model"] == model
        assert call["prompt"] == PROMPT
        assert call["aspect_ratio"] == options.aspect_ratio
        assert call["resolution"] == options.resolution
        assert call["n"] == options.n
        assert [item["image_url"]["url"] for item in call["input_references"]] == expected_urls
        assert call["retries"] is None
    assert image.metadata["model"] == EDIT_PRO
    assert image.metadata["requested_model"] == EDIT_FAST
    assert len(image.metadata["attempts"]) == 2
    assert external_references == [replacement_reference]


@pytest.mark.parametrize(
    ("status_code", "message", "expected_error", "expected_reason"),
    [
        (400, "invalid request parameters", OpenRouterUnsupportedParameterError, "unsupported_parameter"),
        (401, "unauthorized", OpenRouterAuthenticationError, "authentication"),
        (402, "insufficient credits", OpenRouterBillingError, "insufficient_credits"),
        (403, "forbidden", OpenRouterAuthenticationError, "authentication"),
        (413, "payload too large", OpenRouterUnsupportedParameterError, "unsupported_parameter"),
        (422, "unsupported aspect_ratio", OpenRouterUnsupportedParameterError, "unsupported_parameter"),
        (429, "rate limit exceeded", OpenRouterRateLimitError, "rate_limit"),
        (500, "internal provider error", OpenRouterTemporaryError, "temporary_provider_error"),
        (503, "provider unavailable", OpenRouterTemporaryError, "temporary_provider_error"),
        (403, "Content moderation blocked this prompt", OpenRouterUnsupportedError, "content_policy"),
        (404, "The requested route was not found", OpenRouterUnsupportedError, "unsupported_model"),
    ],
)
def test_noneligible_http_errors_never_switch_models(
    status_code: int,
    message: str,
    expected_error: type[OpenRouterMediaError],
    expected_reason: str,
) -> None:
    sdk_call = AsyncMock(side_effect=_ProviderError(status_code, message))
    service = _service(sdk_call)

    with (
        patch.object(openrouter_client, "settings", _settings_with(image_fallback_enabled=True)),
        pytest.raises(expected_error) as caught,
    ):
        _invoke_image(service, "generate", _options("generate", GENERATION_FAST))

    assert sdk_call.await_count == 1
    assert caught.value.reason == expected_reason
    assert caught.value.fallback_eligible is False
    assert len(caught.value.attempts) == 1
    assert caught.value.attempts[0]["model"] == GENERATION_FAST


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (502, "generation failed"),
        (404, "model flux-3-image was not found"),
    ],
)
def test_initially_selected_pro_model_is_never_downgraded_or_repeated(
    status_code: int,
    message: str,
) -> None:
    sdk_call = AsyncMock(side_effect=_ProviderError(status_code, message))
    service = _service(sdk_call)

    with patch.object(openrouter_client, "settings", _settings_with(image_fallback_enabled=True)):
        with pytest.raises(OpenRouterMediaError) as caught:
            _invoke_image(service, "generate", _options("generate", GENERATION_PRO))

    assert sdk_call.await_count == 1
    assert cast(Any, sdk_call.await_args).kwargs["model"] == GENERATION_PRO
    assert caught.value.requested_model == GENERATION_PRO
    assert caught.value.fallback_used is False
    assert len(caught.value.attempts) == 1


def test_fallback_is_disabled_by_configuration() -> None:
    sdk_call = AsyncMock(side_effect=_ProviderError(404, "Model not found"))
    service = _service(sdk_call)

    with patch.object(openrouter_client, "settings", _settings_with(image_fallback_enabled=False)):
        with pytest.raises(OpenRouterMediaError) as caught:
            _invoke_image(service, "generate", _options("generate", GENERATION_FAST))

    assert sdk_call.await_count == 1
    assert caught.value.fallback_used is False


def test_equal_fast_and_pro_model_ids_do_not_cause_duplicate_generation() -> None:
    sdk_call = AsyncMock(side_effect=_ProviderError(404, "Model flux-3-image was not found"))
    service = _service(sdk_call)
    same_model = "vendor/same-image-model"

    with patch.object(
        openrouter_client,
        "settings",
        _settings_with(
            image_generation_fast_model=same_model,
            image_generation_model=same_model,
            image_fallback_enabled=True,
        ),
    ):
        with pytest.raises(OpenRouterMediaError) as caught:
            _invoke_image(service, "generate", _options("generate", same_model))

    assert sdk_call.await_count == 1
    assert cast(Any, sdk_call.await_args).kwargs["model"] == same_model
    assert caught.value.fallback_used is False


def test_custom_initial_model_is_not_eligible_for_klein_fallback() -> None:
    sdk_call = AsyncMock(side_effect=_ProviderError(404, "Model custom/other was not found"))
    service = _service(sdk_call)

    with patch.object(openrouter_client, "settings", _settings_with(image_fallback_enabled=True)):
        with pytest.raises(OpenRouterMediaError):
            _invoke_image(service, "generate", _options("generate", "custom/other"))

    assert sdk_call.await_count == 1
    assert cast(Any, sdk_call.await_args).kwargs["model"] == "custom/other"


@pytest.mark.parametrize("corrupt_response", ["empty", "invalid_bytes"])
def test_empty_or_corrupt_image_response_is_not_retried_and_keeps_known_cost(
    corrupt_response: str,
) -> None:
    if corrupt_response == "empty":
        response = SimpleNamespace(data=[], usage=SimpleNamespace(cost=0.19))
    else:
        response = SimpleNamespace(
            data=[
                SimpleNamespace(
                    b64_json=base64.b64encode(b"not an image").decode("ascii"),
                    media_type="image/png",
                )
            ],
            usage=SimpleNamespace(cost=0.19),
        )
    sdk_call = AsyncMock(return_value=response)
    service = _service(sdk_call)

    with patch.object(openrouter_client, "settings", _settings_with(image_fallback_enabled=True)):
        with pytest.raises(OpenRouterResponseError) as caught:
            _invoke_image(service, "generate", _options("generate", GENERATION_FAST))

    assert sdk_call.await_count == 1
    assert caught.value.cost == 0.19
    assert caught.value.billing_unknown is False
    assert caught.value.attempts[0]["cost"] == 0.19
    assert caught.value.attempts[0]["status"] == 200
    assert caught.value.attempts[0]["billing_unknown"] is False


def test_unknown_or_missing_usage_does_not_trigger_paid_fallback() -> None:
    response = SimpleNamespace(data=[], usage=None)
    sdk_call = AsyncMock(return_value=response)
    service = _service(sdk_call)

    with patch.object(openrouter_client, "settings", _settings_with(image_fallback_enabled=True)):
        with pytest.raises(OpenRouterResponseError) as caught:
            _invoke_image(service, "generate", _options("generate", GENERATION_FAST))

    assert sdk_call.await_count == 1
    assert caught.value.billing_unknown is True
    assert caught.value.attempts[0]["billing_unknown"] is True


def test_unknown_sdk_failure_does_not_trigger_fallback() -> None:
    sdk_call = AsyncMock(side_effect=RuntimeError("unexpected SDK failure"))
    service = _service(sdk_call)

    with patch.object(openrouter_client, "settings", _settings_with(image_fallback_enabled=True)):
        with pytest.raises(OpenRouterSdkUnavailableError) as caught:
            _invoke_image(service, "generate", _options("generate", GENERATION_FAST))

    assert sdk_call.await_count == 1
    assert caught.value.fallback_eligible is False
    assert caught.value.billing_unknown is True
    assert caught.value.attempts[0]["error_reason"] == "sdk_unavailable"


def test_timeout_never_starts_a_second_generation_and_marks_billing_unknown() -> None:
    async def too_slow(**_: Any) -> SimpleNamespace:
        await asyncio.sleep(0.1)
        return _image_response()

    sdk_call = AsyncMock(side_effect=too_slow)
    service = _service(sdk_call, timeout_ms=20)

    with patch.object(openrouter_client, "settings", _settings_with(image_fallback_enabled=True)):
        with pytest.raises(OpenRouterTimeoutError) as caught:
            _invoke_image(service, "generate", _options("generate", GENERATION_FAST))

    assert sdk_call.await_count == 1
    assert caught.value.billing_unknown is True
    assert caught.value.attempts[0]["error_reason"] == "timeout"
    assert caught.value.attempts[0]["billing_unknown"] is True


def test_no_fallback_when_first_attempt_returns_after_the_shared_deadline() -> None:
    sdk_call = AsyncMock(side_effect=_ProviderError(404, "Model not found"))
    service = _service(sdk_call, timeout_ms=1_000)

    # deadline, remaining-before-first, start, first-attempt duration, remaining-before-fallback
    clock = [0.0, 0.0, 0.0, 0.0, 1.0]
    with (
        patch.object(openrouter_client, "settings", _settings_with(image_fallback_enabled=True)),
        patch.object(openrouter_client, "_monotonic", side_effect=clock),
        pytest.raises(OpenRouterMediaError) as caught,
    ):
        _invoke_image(service, "generate", _options("generate", GENERATION_FAST))

    assert sdk_call.await_count == 1
    assert caught.value.status_code == 404
    assert caught.value.fallback_used is False
    assert len(caught.value.attempts) == 1


def test_second_attempt_receives_only_the_remaining_shared_timeout() -> None:
    calls = 0

    async def delayed_502_then_success(**_: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0.02)
            raise _ProviderError(502, "generation failed")
        return _image_response()

    sdk_call = AsyncMock(side_effect=delayed_502_then_success)
    service = _service(sdk_call, timeout_ms=2_000)

    with patch.object(openrouter_client, "settings", _settings_with(image_fallback_enabled=True)):
        _invoke_image(service, "generate", _options("generate", GENERATION_FAST))

    timeouts = [call.kwargs["timeout_ms"] for call in sdk_call.await_args_list]
    assert sdk_call.await_count == 2
    assert timeouts[0] <= 2_000
    assert 0 < timeouts[1] < timeouts[0]


def test_cancellation_and_connection_break_never_trigger_fallback() -> None:
    cases: list[tuple[str, BaseException, type[BaseException]]] = [
        ("cancelled", asyncio.CancelledError(), asyncio.CancelledError),
        ("connection", ConnectionError("connection reset after request send"), OpenRouterMediaError),
    ]

    for name, failure, error_type in cases:
        sdk_call = AsyncMock(side_effect=failure)
        service = _service(sdk_call)
        with patch.object(openrouter_client, "settings", _settings_with(image_fallback_enabled=True)):
            with pytest.raises(error_type) as caught:
                _invoke_image(service, "generate", _options("generate", GENERATION_FAST))

        assert sdk_call.await_count == 1, name
        caught_error = cast(Any, caught.value)
        assert caught_error.billing_unknown is True
        assert len(caught_error.attempts) == 1
        assert caught_error.attempts[0]["billing_unknown"] is True


def test_failure_of_both_allowed_attempts_is_reported_without_a_third_call() -> None:
    sdk_call = AsyncMock(
        side_effect=[
            _ProviderError(404, "Model flux-3-image-klein was not found"),
            _ProviderError(500, "internal provider error"),
            _image_response(cost=0.99),
        ]
    )
    service = _service(sdk_call)

    with patch.object(openrouter_client, "settings", _settings_with(image_fallback_enabled=True)):
        with pytest.raises(OpenRouterTemporaryError) as caught:
            _invoke_image(service, "generate", _options("generate", GENERATION_FAST))

    error = caught.value
    assert sdk_call.await_count == 2
    assert [call.kwargs["model"] for call in sdk_call.await_args_list] == [
        GENERATION_FAST,
        GENERATION_PRO,
    ]
    assert len(error.attempts) == 2
    assert error.attempts[0]["status"] == 404
    assert error.attempts[0]["cost"] == 0.0
    assert error.attempts[0]["error_reason"] == "unsupported_model"
    assert error.attempts[1]["status"] == 500
    assert error.attempts[1]["error_reason"] == "temporary_provider_error"
    assert error.attempts[1]["billing_unknown"] is True
    assert error.requested_model == GENERATION_FAST
    assert error.model == GENERATION_PRO
    assert error.fallback_used is True
    assert error.fallback_reason == "unsupported_model"
    assert error.total_known_cost == 0.0
    assert error.billing_unknown is True


@pytest.mark.parametrize("failure", [
    _ProviderError(404, "Model not found"),
    _ProviderError(502, "generation failed"),
])
def test_fallback_never_retries_same_model_when_fast_and_pro_ids_are_equal(
    failure: _ProviderError,
) -> None:
    sdk_call = AsyncMock(side_effect=failure)
    service = _service(sdk_call)
    same_model = "provider/one-image-model"

    with patch.object(
        openrouter_client,
        "settings",
        _settings_with(
            image_generation_fast_model=same_model,
            image_generation_model=same_model,
            image_fallback_enabled=True,
        ),
    ):
        with pytest.raises(OpenRouterMediaError):
            _invoke_image(service, "generate", _options("generate", same_model))

    assert sdk_call.await_count == 1


def test_insufficient_credits_message_is_actionable_without_exposing_secrets() -> None:
    raw_error = _ProviderError(
        402,
        "Insufficient credits for request; api_key=sk-test-secret-must-not-leak",
    )
    error = OpenRouterService._translate_exception(raw_error, "image")
    assert isinstance(error, OpenRouterBillingError)

    mapped = map_generation_error(error)

    assert mapped.reason is GenerationFailureReason.BILLING_LIMIT
    assert "недостаточно средств" in mapped.user_message.lower()
    assert "лимит расходов" in mapped.user_message.lower()
    assert "sk-test-secret" not in mapped.user_message
    assert "api_key" not in mapped.user_message.lower()


def test_video_operation_does_not_use_image_fallback_or_sdk_retry_override() -> None:
    video_generate = AsyncMock(side_effect=_ProviderError(502, "generation failed"))
    video_client = SimpleNamespace(video_generation=SimpleNamespace(generate_async=video_generate))
    service = OpenRouterService(OpenRouterConfig(api_key="test-api-key", timeout_ms=5_000))
    service._client = cast(Any, video_client)

    with patch.object(openrouter_client, "settings", _settings_with(image_fallback_enabled=True)):
        with pytest.raises(OpenRouterTemporaryError) as caught:
            asyncio.run(service.generate_video("A short cinematic scene"))

    assert video_generate.await_count == 1
    assert "retries" not in cast(Any, video_generate.await_args).kwargs
    assert caught.value.fallback_eligible is False
