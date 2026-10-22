"""Проверки доменных параметров видео: профиль, длительность, порядок, снимок заявки."""

import base64
import unittest

from core.common_types import GeneratedImage
from core.config import settings
from core.video_requests import (
    AmbiguousDurationError,
    AmbiguousImageOrderError,
    ReferenceLimitExceededError,
    UnsupportedDurationError,
    VideoModelProfile,
    build_duration_validator,
    build_video_request_draft,
    parse_duration_answer,
    parse_explicit_duration,
    parse_user_image_indices,
    resolve_video_references,
    validate_duration,
)
from core.user_questions import build_duration_question_options


def _profile(
    *,
    model: str = "alibaba/wan-3.0-prime",
    resolution: str = "480p",
    allowed_durations: tuple[int, ...] = tuple(range(2, 31)),
    duration_presets: tuple[int, ...] = (5, 10, 15, 20, 30),
    allowed_aspect_ratios: tuple[str, ...] = ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9"),
    max_references: int = 10,
) -> VideoModelProfile:
    return VideoModelProfile(
        model=model,
        resolution=resolution,
        allowed_durations=allowed_durations,
        duration_presets=duration_presets,
        allowed_aspect_ratios=allowed_aspect_ratios,
        max_references=max_references,
    )


def _image(number: int) -> GeneratedImage:
    return GeneratedImage(content=f"image-{number}".encode("ascii"), mime_type="image/png")


def _session_image(number: int) -> dict[str, object]:
    return {
        "image_type": "image/png",
        "image": base64.b64encode(f"image-content-{number}".encode("ascii")).decode("ascii"),
        "width": 100,
        "height": 100,
        "aspect_ratio": 1.0,
    }


class VideoDurationValidationTests(unittest.TestCase):
    def test_active_profile_allows_documented_durations(self) -> None:
        profile = settings.video_profile

        for duration in (2, 7, 15, 30):
            with self.subTest(duration=duration):
                self.assertEqual(validate_duration(duration, profile), duration)

    def test_active_profile_rejects_out_of_range_durations(self) -> None:
        profile = settings.video_profile

        for duration in (1, 31, 0, -5):
            with self.subTest(duration=duration):
                with self.assertRaises(ValueError):
                    validate_duration(duration, profile)

    def test_fractional_and_boolean_durations_are_rejected(self) -> None:
        profile = settings.video_profile

        for duration in (7.5, 7.0, True, False):
            with self.subTest(duration=duration):
                with self.assertRaises(ValueError):
                    validate_duration(duration, profile)

    def test_duration_is_not_rounded_to_nearest_supported_value(self) -> None:
        profile = settings.video_profile

        with self.assertRaises(ValueError):
            validate_duration(31, profile)

    def test_replaced_profile_changes_validation(self) -> None:
        profile = _profile(allowed_durations=(4, 8), duration_presets=(4, 8))

        self.assertEqual(validate_duration(4, profile), 4)
        self.assertEqual(validate_duration(8, profile), 8)
        for duration in (2, 5, 7, 30):
            with self.subTest(duration=duration):
                with self.assertRaises(ValueError):
                    validate_duration(duration, profile)

    def test_compatible_profile_changes_model_resolution_buttons_and_reference_limit(self) -> None:
        profile = _profile(
            model="vendor/compatible-video-model",
            resolution="720p",
            allowed_durations=(4, 8),
            duration_presets=(4, 8),
            allowed_aspect_ratios=("16:9",),
            max_references=2,
        )

        self.assertEqual(profile.model, "vendor/compatible-video-model")
        self.assertEqual(profile.resolution, "720p")
        self.assertEqual(validate_duration(4, profile), 4)
        self.assertEqual(validate_duration(8, profile), 8)
        with self.assertRaises(ValueError):
            validate_duration(12, profile)

        question_options = build_duration_question_options(
            profile.question_duration_presets()
        )
        self.assertEqual(
            [(option.option_id, option.payload) for option in question_options],
            [("duration_4", 4), ("duration_8", 8)],
        )

        session = {"images": [_session_image(index) for index in range(1, 4)]}
        with self.assertRaises(ReferenceLimitExceededError):
            resolve_video_references(session, None, profile=profile)

    def test_profile_presets_come_from_profile(self) -> None:
        self.assertEqual(settings.video_profile.question_duration_presets(), (5, 10, 15, 20, 30))
        self.assertEqual(
            _profile(allowed_durations=(4, 8), duration_presets=(4, 8)).question_duration_presets(),
            (4, 8),
        )

    def test_profile_rejects_invalid_configuration(self) -> None:
        invalid_profiles = (
            {"model": ""},
            {"resolution": ""},
            {"allowed_durations": ()},
            {"allowed_durations": (5, 7.5)},
            {"allowed_durations": (5, True)},
            {"duration_presets": (5, 7.5)},
            {"allowed_aspect_ratios": ("",)},
            {"max_references": 0},
            {"max_references": True},
        )

        base = {
            "model": "alibaba/wan-3.0-prime",
            "resolution": "480p",
            "allowed_durations": (2, 30),
            "duration_presets": (5, 10),
            "allowed_aspect_ratios": ("16:9",),
            "max_references": 10,
        }
        for overrides in invalid_profiles:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    VideoModelProfile(**{**base, **overrides})


class ExplicitDurationParsingTests(unittest.TestCase):
    def test_duration_with_unit_is_recognized(self) -> None:
        profile = settings.video_profile

        self.assertEqual(parse_explicit_duration("сделай видео на 12 секунд", profile), 12)
        self.assertEqual(parse_explicit_duration("video 7 seconds", profile), 7)
        self.assertEqual(parse_explicit_duration("ролик 5 сек", profile), 5)

    def test_duration_without_unit_is_not_guessed(self) -> None:
        profile = settings.video_profile

        self.assertIsNone(parse_explicit_duration("сделай видео с картинками 3, 1, 2", profile))
        self.assertIsNone(parse_explicit_duration("сделай видео", profile))
        self.assertIsNone(parse_explicit_duration(None, profile))

    def test_ambiguous_duration_is_rejected(self) -> None:
        profile = settings.video_profile

        with self.assertRaises(AmbiguousDurationError):
            parse_explicit_duration("сначала 5 секунд, потом 10 секунд", profile)

    def test_contradictory_duration_is_rejected(self) -> None:
        profile = settings.video_profile

        with self.assertRaises(AmbiguousDurationError):
            parse_explicit_duration("на 5 секунд, хотя лучше на 10 секунд", profile)

    def test_approximate_duration_is_rejected(self) -> None:
        profile = settings.video_profile

        for text in ("около 10 секунд", "примерно 10 секунд"):
            with self.subTest(text=text):
                with self.assertRaises(AmbiguousDurationError):
                    parse_explicit_duration(text, profile)

    def test_duration_range_is_rejected(self) -> None:
        profile = settings.video_profile

        for text in ("5-10 секунд", "от 5 до 10 секунд"):
            with self.subTest(text=text):
                with self.assertRaises(AmbiguousDurationError):
                    parse_explicit_duration(text, profile)

    def test_fractional_duration_is_rejected(self) -> None:
        profile = settings.video_profile

        with self.assertRaises(AmbiguousDurationError):
            parse_explicit_duration("сделай видео на 7,5 секунд", profile)

    def test_out_of_range_duration_is_rejected(self) -> None:
        profile = settings.video_profile

        with self.assertRaises(UnsupportedDurationError):
            parse_explicit_duration("сделай видео на 45 секунд", profile)

    def test_replaced_profile_changes_accepted_durations(self) -> None:
        profile = _profile(allowed_durations=(4, 8), duration_presets=(4, 8))

        self.assertEqual(parse_explicit_duration("на 4 секунды", profile), 4)
        with self.assertRaises(UnsupportedDurationError):
            parse_explicit_duration("на 12 секунд", profile)


class DurationAnswerTests(unittest.TestCase):
    def test_answer_accepts_number_and_number_with_units(self) -> None:
        profile = settings.video_profile

        for text in ("7", "7 секунд", "7 с", "7 сек"):
            with self.subTest(text=text):
                self.assertEqual(parse_duration_answer(text, profile), 7)

    def test_answer_rejects_out_of_range_and_fractional_values(self) -> None:
        profile = settings.video_profile

        for text in ("1", "31", "45", "7.5", "около 7", "10-20"):
            with self.subTest(text=text):
                self.assertIsNone(parse_duration_answer(text, profile))

    def test_validator_uses_profile_allowed_durations(self) -> None:
        profile = _profile(allowed_durations=(4, 8), duration_presets=(4, 8))
        validator = build_duration_validator(profile)

        self.assertIsNone(validator("4"))
        self.assertIsNone(validator("8 секунд"))
        self.assertIsNotNone(validator("12"))
        self.assertIn("4, 8", validator("12") or "")

    def test_validator_of_replaced_profile_rejects_current_durations(self) -> None:
        profile = _profile(allowed_durations=(4, 8), duration_presets=(4, 8))
        validator = build_duration_validator(profile)

        self.assertIsNotNone(validator("30"))
        self.assertIsNone(validator("8"))


class UserImageOrderParsingTests(unittest.TestCase):
    def test_list_after_single_word_is_parsed(self) -> None:
        self.assertEqual(parse_user_image_indices("картинки 3, 1, 2"), [3, 1, 2])
        self.assertEqual(parse_user_image_indices("фото 3, 1 и 2"), [3, 1, 2])
        self.assertEqual(parse_user_image_indices("изображения 2, 1"), [2, 1])

    def test_number_sign_lists_are_parsed(self) -> None:
        self.assertEqual(parse_user_image_indices("№3, №1, №2"), [3, 1, 2])

    def test_ordinal_numerals_are_parsed(self) -> None:
        self.assertEqual(
            parse_user_image_indices("оживи третью, первую и вторую картинку"),
            [3, 1, 2],
        )

    def test_seconds_are_not_image_numbers(self) -> None:
        self.assertEqual(parse_user_image_indices("сделай видео на 12 секунд"), [])
        self.assertEqual(parse_user_image_indices("на 10 секунд"), [])

    def test_duplicates_use_first_occurrence(self) -> None:
        self.assertEqual(parse_user_image_indices("картинки 3, 1, 3, 2"), [3, 1, 2])

    def test_contradictory_lists_are_ambiguous(self) -> None:
        with self.assertRaises(AmbiguousImageOrderError):
            parse_user_image_indices("картинки 3, 1, 2 и фото 2, 1, 3")

    def test_empty_text_has_no_numbers(self) -> None:
        self.assertEqual(parse_user_image_indices(None), [])
        self.assertEqual(parse_user_image_indices("   "), [])


class VideoReferenceResolutionTests(unittest.TestCase):
    def test_user_order_is_preserved(self) -> None:
        profile = settings.video_profile
        session = {"images": [_session_image(index) for index in range(1, 4)]}

        selection = resolve_video_references(session, [3, 1, 2], profile=profile)

        self.assertEqual(selection.indices, (3, 1, 2))
        self.assertEqual(
            [image.content for image in selection.images],
            [
                base64.b64decode(_session_image(3)["image"]),
                base64.b64decode(_session_image(1)["image"]),
                base64.b64decode(_session_image(2)["image"]),
            ],
        )

    def test_duplicate_numbers_use_first_occurrence(self) -> None:
        profile = settings.video_profile
        session = {"images": [_session_image(index) for index in range(1, 4)]}

        selection = resolve_video_references(session, [3, 1, 3, 2], profile=profile)

        self.assertEqual(selection.indices, (3, 1, 2))

    def test_empty_selection_uses_all_session_images_in_order(self) -> None:
        profile = settings.video_profile
        session = {"images": [_session_image(index) for index in range(1, 4)]}

        selection = resolve_video_references(session, [], profile=profile)

        self.assertEqual(selection.indices, (1, 2, 3))

    def test_missing_session_switches_to_text_mode(self) -> None:
        profile = settings.video_profile

        selection = resolve_video_references(None, None, profile=profile)

        self.assertEqual(selection.indices, ())
        self.assertEqual(selection.images, ())

    def test_reference_limit_is_enforced_without_truncation(self) -> None:
        profile = _profile(max_references=2)
        session = {"images": [_session_image(index) for index in range(1, 4)]}

        with self.assertRaises(ReferenceLimitExceededError):
            resolve_video_references(session, [1, 2, 3], profile=profile)

        with self.assertRaises(ReferenceLimitExceededError):
            resolve_video_references(session, None, profile=profile)

    def test_invalid_reference_numbers_are_rejected(self) -> None:
        profile = settings.video_profile
        session = {"images": [_session_image(1)]}

        for indices in ([0], [-1], [2]):
            with self.subTest(indices=indices):
                with self.assertRaises(ValueError):
                    resolve_video_references(session, indices, profile=profile)

    def test_corrupted_image_is_rejected(self) -> None:
        profile = settings.video_profile
        session = {"images": [{"image_type": "image/png", "image": "not-base64!!"}]}

        with self.assertRaises(ValueError):
            resolve_video_references(session, None, profile=profile)


class VideoRequestDraftTests(unittest.TestCase):
    def test_draft_without_duration_requires_clarification(self) -> None:
        profile = settings.video_profile

        draft = build_video_request_draft("animate", profile)

        self.assertTrue(draft.requires_duration)
        self.assertIsNone(draft.duration)

    def test_draft_keeps_reference_order_and_numbers(self) -> None:
        profile = settings.video_profile
        images = [_image(3), _image(1), _image(2)]

        draft = build_video_request_draft(
            "animate",
            profile,
            reference_images=images,
            reference_indices=(3, 1, 2),
        )

        self.assertEqual(draft.reference_indices, (3, 1, 2))
        self.assertEqual(
            [image.content for image in draft.reference_images],
            [image.content for image in images],
        )

    def test_draft_rejects_reference_limit_without_truncation(self) -> None:
        profile = _profile(max_references=2)

        with self.assertRaises(ValueError):
            build_video_request_draft(
                "animate",
                profile,
                reference_images=[_image(1), _image(2), _image(3)],
                reference_indices=(1, 2, 3),
            )

    def test_draft_accepts_valid_duration(self) -> None:
        profile = settings.video_profile

        draft = build_video_request_draft("animate", profile, duration=10)

        self.assertFalse(draft.requires_duration)
        self.assertEqual(draft.duration, 10)

    def test_draft_rejects_invalid_duration(self) -> None:
        profile = settings.video_profile

        with self.assertRaises(ValueError):
            build_video_request_draft("animate", profile, duration=31)
        with self.assertRaises(ValueError):
            build_video_request_draft("animate", profile, duration=True)
        with self.assertRaises(ValueError):
            build_video_request_draft("animate", profile, duration=7.5)

    def test_draft_with_duration_replaces_only_duration(self) -> None:
        profile = settings.video_profile
        images = [_image(1)]

        draft = build_video_request_draft(
            "animate",
            profile,
            reference_images=images,
            reference_indices=(1,),
        )
        updated = draft.with_duration(15)

        self.assertEqual(updated.duration, 15)
        self.assertEqual(updated.prompt, draft.prompt)
        self.assertEqual(updated.profile, draft.profile)
        self.assertEqual(updated.reference_indices, draft.reference_indices)
        self.assertEqual(
            [image.content for image in updated.reference_images],
            [image.content for image in images],
        )
        self.assertEqual(updated.operation_id, draft.operation_id)
        self.assertIsNone(draft.duration)

    def test_draft_snapshot_is_independent_from_source_images(self) -> None:
        profile = settings.video_profile
        image = _image(1)

        draft = build_video_request_draft(
            "animate",
            profile,
            reference_images=[image],
            reference_indices=(1,),
        )

        self.assertEqual(draft.reference_images[0].content, image.content)
        self.assertIsNot(draft.reference_images[0], image)

    def test_draft_rejects_duplicated_reference_numbers(self) -> None:
        profile = settings.video_profile

        with self.assertRaises(ValueError):
            build_video_request_draft(
                "animate",
                profile,
                reference_images=[_image(1), _image(2)],
                reference_indices=(1, 1),
            )


if __name__ == "__main__":
    unittest.main()
