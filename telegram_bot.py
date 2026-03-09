from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

from history import create_bot_from_env

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


class SimpleHistoryResponder:
    def __init__(self) -> None:
        self.history_bot = None
        self.enabled = False

        try:
            self.history_bot = create_bot_from_env()
            self.enabled = True
            logger.info("GigaChat backend enabled.")
        except Exception as exc:
            logger.warning("GigaChat backend disabled: %s", exc)

    def answer(self, text: str) -> str:
        clean_text = text.strip()
        if not clean_text:
            return "Напиши что-нибудь, и я отвечу."

        if not self.enabled or self.history_bot is None:
            return (
                "Бот запущен, но ключи для GigaChat пока не настроены. "
                "Добавь их в .env, и тогда я начну отвечать как Пётр I."
            )

        try:
            response = self.history_bot.ask(clean_text)
        except Exception as exc:
            logger.exception("Failed to get response from history bot: %s", exc)
            return "Не удалось получить ответ от модели."

        if not response:
            return "Модель не вернула ответ. Проверь ключи и настройки в .env."

        return response

    def close(self) -> None:
        if self.history_bot is not None:
            try:
                self.history_bot.close()
            except Exception:
                logger.exception("Failed to close history bot cleanly.")


responder = SimpleHistoryResponder()
dp = Dispatcher()


@dp.message(CommandStart())
async def handle_start(message: Message) -> None:
    await message.answer(
        "Привет. Я простой Telegram-бот для проекта HistorAi.\n\n"
        "Пока я минимальный: отправь мне сообщение, и я либо отвечу через GigaChat, "
        "либо скажу, что ключи ещё не настроены."
    )


@dp.message(F.text)
async def handle_text(message: Message) -> None:
    reply = responder.answer(message.text or "")
    await message.answer(reply)


@dp.message()
async def handle_other(message: Message) -> None:
    await message.answer("Я пока понимаю только текстовые сообщения.")


async def main() -> None:
    token = get_env("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Create a .env file and add your bot token."
        )

    bot = Bot(token=token)
    try:
        await dp.start_polling(bot)
    finally:
        responder.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
