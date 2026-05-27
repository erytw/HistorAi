from __future__ import annotations

import logging
from functools import lru_cache

from flask import Flask, jsonify, render_template, request

from history import SUPPORTED_MODELS, create_bot_from_env, normalize_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@lru_cache(maxsize=len(SUPPORTED_MODELS))
def get_bot(model: str):
    return create_bot_from_env(model)


@app.get("/")
def index():
    return render_template(
        "index.html",
        models=SUPPORTED_MODELS,
        default_model=normalize_model(None),
    )


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "models": SUPPORTED_MODELS})


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("message", "")).strip()
    history = payload.get("history") or []

    if not question:
        return jsonify({"error": "Сообщение пустое."}), 400

    try:
        model = normalize_model(str(payload.get("model", "")))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        result = get_bot(model).answer_with_sources(question, history)
    except Exception as exc:
        logger.exception("Failed to process chat request: %s", exc)
        return jsonify({"error": str(exc)}), 500

    if not result.get("answer"):
        return jsonify({"error": "Модель не вернула ответ. Проверьте ключи в .env."}), 502

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
