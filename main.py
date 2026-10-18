"""
Backend для ИИ-чатов (Telegram Mini App).
Установка: pip install flask requests
Запуск:    python main.py   (index.html должен лежать рядом)

Эндпоинт провайдера: <api_url>/chat/completions (OpenAI-совместимый).
API-ключ приходит из браузера с каждым запросом и на диск не пишется.
Для публичного деплоя добавь HTTPS, rate limit и проверку Telegram initData.
Если фронтенд на GitHub Pages, а бэкенд на другом хостинге — задай
переменную окружения ALLOWED_ORIGIN (например https://user.github.io).
"""
import base64
import json
import os
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
ROOT = Path(__file__).resolve().parent
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".txt", ".md",
    ".csv", ".json", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip",
}
IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
              ".gif": "image/gif", ".webp": "image/webp"}
TEXT_EXT = {".txt", ".md", ".csv", ".json"}
MAX_FILE_SIZE = 15 * 1024 * 1024
SYSTEM_PROMPT = "Ты полезный ИИ-помощник. Отвечай на языке пользователя."


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.route("/api/chat", methods=["OPTIONS"])
def preflight():
    return ("", 204)


def parse_history(raw):
    try:
        items = json.loads(raw or "[]")
    except ValueError:
        return []
    out = []
    for m in items[-20:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            out.append({"role": m["role"], "content": m["content"][:8000]})
    return out


@app.post("/api/chat")
def chat():
    message = (request.form.get("message") or "").strip()
    model = (request.form.get("model") or "").strip()
    api_key = (request.form.get("api_key") or "").strip()
    api_url = (request.form.get("api_url") or "").strip().rstrip("/")
    files = request.files.getlist("files")

    if not api_key or not model or not api_url:
        return jsonify({"error": "Не хватает названия модели, API-ключа или ссылки."}), 400
    if not api_url.startswith(("https://", "http://")):
        return jsonify({"error": "Некорректная ссылка API."}), 400
    if not message and not files:
        return jsonify({"error": "Напиши сообщение или прикрепи файл."}), 400
    if len(files) > 5:
        return jsonify({"error": "Можно отправить не более 5 файлов."}), 400

    text_parts = [message] if message else []
    images = []
    file_names = []
    for f in files:
        if not f.filename:
            continue
        name = secure_filename(f.filename) or "attachment"
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Тип файла не поддерживается: {name}"}), 400
        content = f.read(MAX_FILE_SIZE + 1)
        if len(content) > MAX_FILE_SIZE:
            return jsonify({"error": f"Файл {name} превышает лимит 15 МБ."}), 413
        file_names.append(name)
        if ext in TEXT_EXT:
            text_parts.append(f"[Файл {name}]\n{content.decode('utf-8', errors='replace')[:12000]}")
        elif ext in IMAGE_MIME:
            b64 = base64.b64encode(content).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{IMAGE_MIME[ext]};base64,{b64}"}})
        else:
            text_parts.append(f"[Прикреплён файл {name}: чтение этого формата не поддерживается]")

    text = "\n\n".join(text_parts) or "Посмотри вложения."
    user_content = [{"type": "text", "text": text}, *images] if images else text

    if api_url.endswith("/chat/completions"):
        endpoints = [api_url]
    elif api_url.endswith("/v1"):
        endpoints = [api_url + "/chat/completions"]
    else:
        endpoints = [api_url + "/v1/chat/completions", api_url + "/chat/completions"]

    payload = {"model": model, "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        *parse_history(request.form.get("history")),
        {"role": "user", "content": user_content},
    ]}

    try:
        response = None
        for endpoint in endpoints:
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload, timeout=90,
            )
            if response.status_code not in (404, 405):
                break
        if not response.ok:
            detail = response.text[:500].replace("\n", " ")
            return jsonify({"error": f"API вернул HTTP {response.status_code}. {detail}"}), 502
        try:
            data = response.json()
        except ValueError:
            return jsonify({"error": "API вернул не JSON. Проверь адрес API."}), 502
        answer = data["choices"][0]["message"]["content"]
        if isinstance(answer, list):
            answer = "\n".join(x.get("text", "") for x in answer if isinstance(x, dict))
        return jsonify({"reply": str(answer), "files": file_names})
    except requests.Timeout:
        return jsonify({"error": "API не ответил вовремя."}), 504
    except (requests.RequestException, KeyError, IndexError, TypeError) as exc:
        app.logger.exception("API request failed")
        return jsonify({"error": f"Не удалось обработать ответ API: {exc}"}), 502


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Слишком большой запрос. Общий лимит — 25 МБ."}), 413


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")), debug=False)
