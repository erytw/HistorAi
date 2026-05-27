from __future__ import annotations

import base64
import asyncio
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import requests
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ("gigachat", "deepseek")
IGNORED_SOURCE_FILENAMES = {
    "requirements.txt",
}

ANACHRONISM_PATTERNS = (
    r"\b1[89]\d{2}\b",
    r"\b20\d{2}\b",
    r"\bxx\b",
    r"\bxxi\b",
    r"втор[а-яё]+\s+миров[а-яё]+",
    r"перв[а-яё]+\s+миров[а-яё]+",
    r"ссср",
    r"советск",
    r"ленин",
    r"сталин",
    r"гитлер",
    r"наци",
    r"фаши",
    r"интернет",
    r"компьютер",
    r"телефон",
    r"самолет",
    r"атомн",
    r"ядерн",
)


def _normalize_bool(value: str | bool | None, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def normalize_model(value: str | None) -> str:
    model = (value or _env("DEFAULT_MODEL", "gigachat")).strip().lower()
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{value}'. Supported models: {', '.join(SUPPORTED_MODELS)}."
        )
    return model


def is_obvious_anachronism(question: str) -> bool:
    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in ANACHRONISM_PATTERNS)


def get_gigachat_token(
    client_id: str,
    client_secret: str,
    *,
    verify_ssl_certs: bool = False,
    scope: str = "GIGACHAT_API_PERS",
) -> str:
    if not client_id or not client_secret:
        raise ValueError("GigaChat client_id and client_secret must be provided.")

    auth = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode(
        "utf-8"
    )
    headers = {
        "Authorization": f"Basic {auth}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
    }

    response = requests.post(
        "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        headers=headers,
        data=f"scope={scope}",
        verify=verify_ssl_certs,
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("GigaChat token response does not contain access_token.")

    return token


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    source: str = ""


class DocumentRetriever:
    def __init__(self, documents_path: str, chunk_size: int, top_k: int) -> None:
        self.documents_path = documents_path
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.chunks: List[RetrievedChunk] = []
        self.chunk_texts: List[str] = []
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True,
        )
        self.tfidf_matrix = None
        self._load_documents(documents_path)

    def _load_documents(self, path: str) -> None:
        base_path = Path(path)
        if not base_path.is_absolute():
            base_path = BASE_DIR / base_path

        if not base_path.exists():
            logger.warning("Documents path does not exist: %s", base_path)
            return

        if base_path.is_file() and base_path.suffix.lower() == ".txt":
            txt_files = [base_path]
        else:
            txt_files = sorted(base_path.glob("*.txt"))

        for file_path in txt_files:
            if file_path.name in IGNORED_SOURCE_FILENAMES:
                logger.info("Skipped non-source text file: %s", file_path.name)
                continue
            try:
                text = self._read_text_file(file_path)
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    self._create_chunks(text, file_path.name)
                    logger.info("Loaded document: %s", file_path.name)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", file_path, exc)

        if not self.chunks:
            logger.warning("No text documents were loaded.")
            return

        self.chunk_texts = [chunk.text for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)
        logger.info("Created %s chunks and built TF-IDF matrix.", len(self.chunks))

    def _read_text_file(self, file_path: Path) -> str:
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def _create_chunks(self, text: str, source: str) -> None:
        paragraphs = re.split(r"(?<=[.!?])\s+", text)
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_chunk) + len(paragraph) + 1 <= self.chunk_size:
                current_chunk = f"{current_chunk} {paragraph}".strip()
            else:
                if current_chunk:
                    self.chunks.append(
                        RetrievedChunk(text=current_chunk, score=0.0, source=source)
                    )
                current_chunk = paragraph

        if current_chunk:
            self.chunks.append(
                RetrievedChunk(text=current_chunk, score=0.0, source=source)
            )

    def search(self, query: str) -> List[RetrievedChunk]:
        if not self.chunks or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-self.top_k :][::-1]

        return [
            RetrievedChunk(
                text=self.chunks[i].text,
                score=float(similarities[i]),
                source=self.chunks[i].source,
            )
            for i in top_indices
            if similarities[i] > 0
        ]


def _ensure_thread_event_loop() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


class PeterTheGreatBotWithDocs:
    """
    RAG-сервис для ответов от лица Петра I через GigaChat или DeepSeek.
    """

    def __init__(
        self,
        model: str = "gigachat",
        documents_path: Optional[str] = None,
        chunk_size: int = 1200,
        top_k: int = 4,
        temperature: float = 0.7,
        max_tokens: int = 700,
        verify_ssl_certs: bool = False,
        logging_level: int = logging.INFO,
    ) -> None:
        logging.basicConfig(level=logging_level)

        self.model = normalize_model(model)
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verify_ssl_certs = verify_ssl_certs
        self.gigachat_client: GigaChat | None = None
        self._gigachat_lock = threading.Lock()

        self.system_prompt = (
            "Ты — Пётр Первый, русский царь и император. Отвечай от первого лица. "
            "Держи стиль властным, прямым, деятельным, но не превращай речь в пародию. "
            "Ты живешь в своем времени и не знаешь событий после смерти Петра I "
            "в 1725 году. Если вопрос касается будущего для тебя времени, поздних "
            "войн, технологий, людей, стран или понятий, отвечай из роли: скажи, "
            "что такого события, имени или дела тебе неведомо, и не объясняй его "
            "современными знаниями. Не называй и не угадывай даты после 1725 года. "
            "Сведения ниже — это твоя внутренняя память и опора для ответа, а не "
            "внешние документы для обсуждения с собеседником. Никогда не говори "
            "«в документах», «в бумагах», «в контексте», «в источниках», "
            "«в отрывках» и не раскрывай RAG. Если сведений из памяти не хватает, "
            "отвечай как Петр: «не ведаю», «мне о том не ведомо», «не слыхал о сем». "
            "Короткие цитаты можно приводить только как собственные слова или "
            "известную тебе речь, без упоминания источников."
        )

        self.retriever: DocumentRetriever | None = None
        if documents_path:
            self.retriever = DocumentRetriever(documents_path, chunk_size, top_k)

        self._init_model_client()

    def _init_model_client(self) -> None:
        if self.model == "gigachat":
            _ensure_thread_event_loop()
            access_token = _env("GIGACHAT_ACCESS_TOKEN")
            client_id = _env("GIGACHAT_CLIENT_ID")
            client_secret = _env("GIGACHAT_CLIENT_SECRET")

            if not access_token:
                if client_id and client_secret:
                    access_token = get_gigachat_token(
                        client_id=client_id,
                        client_secret=client_secret,
                        verify_ssl_certs=self.verify_ssl_certs,
                    )
                else:
                    raise RuntimeError(
                        "GigaChat credentials are not configured. Set "
                        "GIGACHAT_ACCESS_TOKEN or both GIGACHAT_CLIENT_ID and "
                        "GIGACHAT_CLIENT_SECRET in .env."
                    )

            self.gigachat_client = GigaChat(
                access_token=access_token,
                verify_ssl_certs=self.verify_ssl_certs,
            )
            logger.info("GigaChat client initialized.")
            return

        if self.model == "deepseek" and not _env("DEEPSEEK_API_KEY"):
            raise RuntimeError("DEEPSEEK_API_KEY is not configured in .env.")

    def _get_relevant_chunks(self, query: str) -> List[RetrievedChunk]:
        if self.retriever is None:
            return []
        return self.retriever.search(query)

    def build_context(self, question: str) -> tuple[str, List[RetrievedChunk]]:
        relevant = self._get_relevant_chunks(question)
        if not relevant:
            return (
                "Внутренняя память не дала точной опоры. Если вопрос вне времени "
                "или знаний Петра I, ответь, что тебе это неведомо. Не используй "
                "современные сведения и не упоминай внутреннюю память.",
                [],
            )

        parts = [
            "Внутренняя память персонажа. Используй эти сведения молча: не называй "
            "их документами, источниками, отрывками, контекстом или бумагами."
        ]
        for index, chunk in enumerate(relevant, start=1):
            parts.append(f"Память {index}:\n{chunk.text}")

        return "\n\n".join(parts), relevant

    def ask(
        self,
        question: str,
        history: Optional[Iterable[dict[str, str]]] = None,
    ) -> Optional[str]:
        question = question.strip()
        if not question:
            return None
        if is_obvious_anachronism(question):
            return self._anachronism_answer()

        context, _ = self.build_context(question)
        system_prompt = f"{self.system_prompt}\n\n{context}"
        messages = self._build_messages(system_prompt, question, history or [])

        if self.model == "gigachat":
            return self._ask_gigachat(messages)
        if self.model == "deepseek":
            return self._ask_deepseek(messages)

        raise RuntimeError(f"Unsupported model: {self.model}")

    def answer_with_sources(
        self,
        question: str,
        history: Optional[Iterable[dict[str, str]]] = None,
    ) -> dict[str, object]:
        if is_obvious_anachronism(question):
            return {
                "answer": self._anachronism_answer(),
                "model": self.model,
                "sources": [],
            }

        context, sources = self.build_context(question)
        system_prompt = f"{self.system_prompt}\n\n{context}"
        messages = self._build_messages(system_prompt, question, history or [])

        if self.model == "gigachat":
            answer = self._ask_gigachat(messages)
        else:
            answer = self._ask_deepseek(messages)

        return {
            "answer": answer,
            "model": self.model,
            "sources": [
                {
                    "text": source.text,
                    "score": round(source.score, 4),
                    "source": source.source,
                }
                for source in sources
            ],
        }

    def _anachronism_answer(self) -> str:
        return (
            "Не ведаю о сем. Для меня такого дела, войны или имени нет: "
            "я говорю о своем времени, о Русском государстве, войне со шведом, "
            "флоте, ремеслах и государевом порядке. О грядущих веках судить не стану."
        )

    def _build_messages(
        self,
        system_prompt: str,
        question: str,
        history: Iterable[dict[str, str]],
    ) -> List[dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        for item in list(history)[-8:]:
            role = item.get("role", "")
            content = item.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": question})
        return messages

    def _ask_gigachat(self, messages: List[dict[str, str]]) -> Optional[str]:
        if self.gigachat_client is None:
            raise RuntimeError("GigaChat client is not initialized.")

        _ensure_thread_event_loop()
        role_map = {
            "system": MessagesRole.SYSTEM,
            "user": MessagesRole.USER,
            "assistant": MessagesRole.ASSISTANT,
        }
        chat = Chat(
            messages=[
                Messages(role=role_map[item["role"]], content=item["content"])
                for item in messages
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        try:
            with self._gigachat_lock:
                _ensure_thread_event_loop()
                response = self.gigachat_client.chat(chat)
            return response.choices[0].message.content
        except Exception as exc:
            logger.exception("GigaChat request failed: %s", exc)
            return None

    def _ask_deepseek(self, messages: List[dict[str, str]]) -> Optional[str]:
        payload = {
            "model": _env("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {_env('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions"),
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.exception("DeepSeek request failed: %s", exc)
            return None

    def close(self) -> None:
        if self.gigachat_client is not None:
            try:
                _ensure_thread_event_loop()
                self.gigachat_client.close()
            except Exception:
                logger.debug("Failed to close GigaChat client cleanly.", exc_info=True)

    def __enter__(self) -> "PeterTheGreatBotWithDocs":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def create_bot_from_env(model: str | None = None) -> PeterTheGreatBotWithDocs:
    documents_path = _env(
        "DOCUMENTS_PATH",
        "sources",
    )
    top_k = int(_env("TOP_K", "4"))
    chunk_size = int(_env("CHUNK_SIZE", "1200"))
    temperature = float(_env("TEMPERATURE", "0.7"))
    max_tokens = int(_env("MAX_TOKENS", "700"))
    verify_ssl_certs = _normalize_bool(_env("VERIFY_SSL_CERTS"), default=False)

    return PeterTheGreatBotWithDocs(
        model=normalize_model(model),
        documents_path=documents_path,
        chunk_size=chunk_size,
        top_k=top_k,
        temperature=temperature,
        max_tokens=max_tokens,
        verify_ssl_certs=verify_ssl_certs,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        bot = create_bot_from_env()
    except Exception as exc:
        logger.error("Failed to initialize history service: %s", exc)
        raise SystemExit(1)

    try:
        question = "Кто ты?"
        answer = bot.ask(question)
        print(f"Вопрос: {question}")
        print(f"Ответ: {answer or 'Ответ не получен'}")
    finally:
        bot.close()
