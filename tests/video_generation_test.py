"""Ручная проверка генерации видео через доменный OpenRouter service."""

import asyncio

from utils.openrouter_client import OpenRouterService


async def main() -> None:
    service = OpenRouterService.from_env()
    async with service:
        video = await service.generate_video(
            "A serene mountain landscape at sunset with clouds drifting by"
        )
        try:
            print(f"Видео сохранено во временный файл: {video.path}")
        finally:
            video.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
