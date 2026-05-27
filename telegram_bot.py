from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

from history import SUPPORTED_MODELS, create_bot_from_env, normalize_model

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
        self.history_bots = {}
        self.model = normalize_model(os.getenv("DEFAULT_MODEL", "gigachat"))
        self.history: list[dict[str, str]] = []
        self.enabled = False

        try:
            self.history_bots[self.model] = create_bot_from_env(self.model)
            self.enabled = True
            logger.info("History backend enabled: %s.", self.model)
        except Exception as exc:
            logger.warning("History backend disabled: %s", exc)

    def set_model(self, model: str) -> str:
        selected = normalize_model(model)
        if selected not in self.history_bots:
            self.history_bots[selected] = create_bot_from_env(selected)
        self.model = selected
        self.history = []
        self.enabled = True
        return f"Модель переключена на {selected}. Диалог начат заново."

    def reset(self) -> str:
        self.history = []
        return "Диалог начат заново."

    def answer(self, text: str) -> str:
        clean_text = text.strip()
        if not clean_text:
            return "Напиши что-нибудь, и я отвечу."

        if not self.enabled or self.model not in self.history_bots:
            return (
                "Бот запущен, но ключи для модели пока не настроены. "
                "Добавь их в .env, и тогда я начну отвечать как Пётр I."
            )

        try:
            response = self.history_bots[self.model].ask(clean_text, self.history)
        except Exception as exc:
            logger.exception("Failed to get response from history bot: %s", exc)
            return "Не удалось получить ответ от модели."

        if not response:
            return "Модель не вернула ответ. Проверь ключи и настройки в .env."

        self.history.append({"role": "user", "content": clean_text})
        self.history.append({"role": "assistant", "content": response})
        self.history = self.history[-8:]
        return response

    def close(self) -> None:
        for bot in self.history_bots.values():
            try:
                bot.close()
            except Exception:
                logger.exception("Failed to close history bot cleanly.")


responder = SimpleHistoryResponder()
dp = Dispatcher()


@dp.message(CommandStart())
async def handle_start(message: Message) -> None:
    await message.answer(
        "Привет. Я Telegram-бот проекта HistorAi.\n\n"
        "Отправь вопрос Петру I. Команды:\n"
        "/model gigachat или /model deepseek — сменить модель и начать диалог заново\n"
        "/reset — очистить текущий диалог"
    )


@dp.message(Command("model"))
async def handle_model(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) == 1:
        await message.answer(
            "Укажи модель: /model gigachat или /model deepseek.\n"
            f"Сейчас выбрана: {responder.model}."
        )
        return

    try:
        reply = responder.set_model(parts[1])
    except Exception as exc:
        logger.exception("Failed to switch model: %s", exc)
        await message.answer(
            f"Не удалось переключить модель. Доступны: {', '.join(SUPPORTED_MODELS)}."
        )
        return

    await message.answer(reply)


@dp.message(Command("reset"))
async def handle_reset(message: Message) -> None:
    await message.answer(responder.reset())


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
