"""Ручная проверка генерации видео через доменный OpenRouter service.

Скрипт платный: он создаёт реальную задачу Video API. Длительность указывается
явно, как того требует актуальный контракт. Запуск выполняется отдельно от
автоматических тестов и только после подтверждения расходов.
"""

import asyncio
import logging

from core.common_types import GeneratedImage, VideoGenerationOptions
from core.config import settings
from utils.openrouter_client import OpenRouterService

logger = logging.getLogger(__name__)

# Длительность задаётся явно и проверяется по активному видеопрофилю.
DURATION_SECONDS = 5

# Управляемые входы: укажите файлы-референсы, чтобы отправить их провайдеру
# в указанном порядке. Пустой список означает text-to-video.
REFERENCE_FILES: tuple[str, ...] = ()


def _load_references() -> list[GeneratedImage]:
    """Загружает референсы из файлов в указанном порядке."""

    references: list[GeneratedImage] = []
    for name in REFERENCE_FILES:
        with open(name, "rb") as handle:
            content = handle.read()
        mime_type = "image/png" if name.lower().endswith(".png") else "image/jpeg"
        references.append(GeneratedImage(content=content, mime_type=mime_type))
    return references


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    service = OpenRouterService.from_env()
    async with service:
        video = await service.generate_video(
            "A serene mountain landscape at sunset with clouds drifting by",
            VideoGenerationOptions(
                duration=DURATION_SECONDS,
                model=settings.media.video_model,
                resolution=settings.media.video_resolution,
            ),
            reference_images=_load_references(),
        )
        try:
            print(f"Видео сохранено во временный файл: {video.path}")
            print(f"Идентификатор задачи: {video.job_id or 'неизвестен'}")
            print(
                "Известная стоимость: "
                f"{video.cost if video.cost is not None else 'неизвестна'}"
            )
        finally:
            # Временный файл удаляется в любом случае.
            video.cleanup()
            print("Временный файл удалён.")


if __name__ == "__main__":
    asyncio.run(main())
