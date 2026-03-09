from __future__ import annotations

import base64
import logging
import os
import re
import uuid
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger(__name__)


def _normalize_bool(value: str | bool | None, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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


class PeterTheGreatBotWithDocs:
    """
    Простой сервис для ответов от лица Петра I с использованием фрагментов
    исторических документов.
    """

    def __init__(
        self,
        token: str,
        documents_path: Optional[str] = None,
        chunk_size: int = 300,
        top_k: int = 3,
        model: str = "GigaChat",
        temperature: float = 0.7,
        max_tokens: int = 512,
        verify_ssl_certs: bool = False,
        logging_level: int = logging.INFO,
    ) -> None:
        logging.basicConfig(level=logging_level)

        if not token:
            raise ValueError("GigaChat access token must be provided.")

        self.top_k = top_k
        self.chunk_size = chunk_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.verify_ssl_certs = verify_ssl_certs

        self.chunks: List[str] = []
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True,
        )
        self.tfidf_matrix = None

        self.system_prompt = (
            "Ты — Пётр Первый, царь России. Отвечай на вопросы от его лица, "
            "используя исторически уместный стиль речи. "
            "Если в контексте ниже есть подлинные отрывки документов, опирайся на них. "
            "Говори кратко, уверенно и по делу."
        )

        if documents_path:
            self._load_documents(documents_path, chunk_size)

        self.client = GigaChat(
            access_token=token,
            verify_ssl_certs=verify_ssl_certs,
        )
        logger.info("GigaChat client initialized.")

    def _load_documents(self, path: str, chunk_size: int) -> None:
        base_path = Path(path)
        if not base_path.exists():
            logger.warning("Documents path does not exist: %s", base_path)
            return

        txt_files: List[Path]
        if base_path.is_file() and base_path.suffix.lower() == ".txt":
            txt_files = [base_path]
        else:
            txt_files = sorted(base_path.glob("*.txt"))

        all_texts: List[str] = []
        for file_path in txt_files:
            try:
                text = file_path.read_text(encoding="utf-8")
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    all_texts.append(text)
                    logger.info("Loaded document: %s", file_path.name)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", file_path, exc)

        if not all_texts:
            logger.warning("No text documents were loaded.")
            return

        full_text = " ".join(all_texts)
        sentences = re.split(r"(?<=[.!?])\s+", full_text)

        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                if current_chunk:
                    self.chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            self.chunks.append(current_chunk)

        if self.chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
            logger.info("Created %s chunks and built TF-IDF matrix.", len(self.chunks))
        else:
            logger.warning("No chunks were created from the loaded documents.")

    def _get_relevant_chunks(self, query: str) -> List[str]:
        if not self.chunks or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-self.top_k :][::-1]

        return [self.chunks[i] for i in top_indices if similarities[i] > 0]

    def build_context(self, question: str) -> str:
        relevant = self._get_relevant_chunks(question)
        if not relevant:
            return "Подходящих отрывков не найдено. Отвечай в общем историческом стиле Петра I."

        parts = [
            "Вот отрывки из исторических документов, которые могут помочь с ответом:"
        ]
        for index, chunk in enumerate(relevant, start=1):
            parts.append(f"Отрывок {index}:\n{chunk}")

        return "\n\n".join(parts)

    def ask(self, question: str) -> Optional[str]:
        question = question.strip()
        if not question:
            return None

        full_system_prompt = f"{self.system_prompt}\n\n{self.build_context(question)}"
        messages = [
            Messages(role=MessagesRole.SYSTEM, content=full_system_prompt),
            Messages(role=MessagesRole.USER, content=question),
        ]
        chat = Chat(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        try:
            response = self.client.chat(chat)
            return response.choices[0].message.content
        except Exception as exc:
            logger.exception("GigaChat request failed: %s", exc)
            return None

    def close(self) -> None:
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:
                logger.debug("Failed to close GigaChat client cleanly.", exc_info=True)

    def __enter__(self) -> "PeterTheGreatBotWithDocs":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def create_bot_from_env() -> PeterTheGreatBotWithDocs:
    access_token = os.getenv("GIGACHAT_ACCESS_TOKEN", "").strip()
    client_id = os.getenv("GIGACHAT_CLIENT_ID", "").strip()
    client_secret = os.getenv("GIGACHAT_CLIENT_SECRET", "").strip()
    documents_path = os.getenv("DOCUMENTS_PATH", "./")
    top_k = int(os.getenv("TOP_K", "3"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "300"))
    temperature = float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("MAX_TOKENS", "512"))
    verify_ssl_certs = _normalize_bool(os.getenv("VERIFY_SSL_CERTS"), default=False)

    if not access_token:
        if client_id and client_secret:
            access_token = get_gigachat_token(
                client_id=client_id,
                client_secret=client_secret,
                verify_ssl_certs=verify_ssl_certs,
            )
        else:
            raise RuntimeError(
                "GigaChat credentials are not configured. "
                "Set GIGACHAT_ACCESS_TOKEN or both GIGACHAT_CLIENT_ID and GIGACHAT_CLIENT_SECRET."
            )

    return PeterTheGreatBotWithDocs(
        token=access_token,
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
