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


def model_title(model: str) -> str:
    titles = {
        "deepseek": "DeepSeek",
        "gigachat": "GigaChat",
    }
    return titles.get(model, model)


def help_text(current_model: str) -> str:
    return (
        "HistorAi: разговор с Петром I\n\n"
        "Задай вопрос о делах его времени: реформах, флоте, войне со шведами, "
        "Петербурге, дворе или государевом порядке.\n\n"
        f"Сейчас выбрана модель: {model_title(current_model)}.\n\n"
        "Команды:\n"
        "/model deepseek — перейти на DeepSeek и начать заново\n"
        "/model gigachat — перейти на GigaChat и начать заново\n"
        "/reset — очистить текущий диалог\n"
        "/help — показать эту справку"
    )


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
        return (
            f"Модель изменена: {model_title(selected)}.\n"
            "Диалог очищен, можно начать новый разговор с Петром I."
        )

    def reset(self) -> str:
        self.history = []
        return "Диалог очищен. Задай новый вопрос Петру I."

    def answer(self, text: str) -> str:
        clean_text = text.strip()
        if not clean_text:
            return "Сообщение пустое. Напиши вопрос одним текстовым сообщением."

        if not self.enabled or self.model not in self.history_bots:
            return (
                "Модель пока не готова к работе.\n\n"
                "Проверь ключи в .env и перезапусти бота. Нужны настройки для "
                f"{model_title(self.model)}."
            )

        try:
            response = self.history_bots[self.model].ask(clean_text, self.history)
        except Exception as exc:
            logger.exception("Failed to get response from history bot: %s", exc)
            return (
                "Не удалось получить ответ от модели.\n\n"
                "Обычно это значит, что не прошел запрос к API или неверно указан ключ."
            )

        if not response:
            return (
                "Модель не вернула текст ответа.\n\n"
                "Проверь ключи, выбранную модель и доступность API."
            )

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
    await message.answer(help_text(responder.model))


@dp.message(Command("help"))
async def handle_help(message: Message) -> None:
    await message.answer(help_text(responder.model))


@dp.message(Command("model"))
async def handle_model(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) == 1:
        await message.answer(
            "Укажи модель после команды.\n\n"
            "Примеры:\n"
            "/model deepseek\n"
            "/model gigachat\n\n"
            f"Сейчас выбрана: {model_title(responder.model)}."
        )
        return

    try:
        reply = responder.set_model(parts[1])
    except Exception as exc:
        logger.exception("Failed to switch model: %s", exc)
        await message.answer(
            "Не удалось переключить модель.\n\n"
            f"Доступные варианты: {', '.join(SUPPORTED_MODELS)}."
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
    await message.answer(
        "Я работаю только с текстовыми вопросами.\n\n"
        "Отправь вопрос сообщением или набери /help."
    )


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
