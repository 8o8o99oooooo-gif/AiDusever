"""
Backend for AI chat manager.
Install: pip install flask requests
Run: python main.py

API-compatible endpoint expected: <api_url>/chat/completions
The API key is submitted from the browser for each request and is not stored on disk by this backend.
For a public deployment, add authentication, HTTPS, rate limits, and validate Telegram initData.
"""
import os
from pathlib import Path
import requests
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
ROOT = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".txt", ".md",
    ".csv", ".json", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip"
}
MAX_FILE_SIZE = 15 * 1024 * 1024

@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")

@app.get("/health")
def health():
    return jsonify({"ok": True})

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

    parts = [message] if message else []
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
        if ext in {".txt", ".md", ".csv", ".json"}:
            parts.append(f"\n\n[Содержимое файла {name}]\n{content.decode('utf-8', errors='replace')[:12000]}")
        elif ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            # Image bytes need provider-specific multimodal payload; not sent as pixels in this generic endpoint.
            parts.append(f"\nПользователь прикрепил изображение «{name}». Для анализа изображения настрой vision-формат провайдера.")
        else:
            parts.append(f"\nПользователь прикрепил файл «{name}». Извлечение текста из этого формата не настроено.")

    # Accept either a full chat-completions URL or a provider base URL.
    if api_url.endswith("/chat/completions"):
        endpoints = [api_url]
    elif api_url.endswith("/v1"):
        endpoints = [api_url + "/chat/completions"]
    else:
        endpoints = [api_url + "/v1/chat/completions", api_url + "/chat/completions"]

    try:
        payload = {"model": model, "messages": [
            {"role": "system", "content": "Ты полезный ИИ-помощник. Отвечай на языке пользователя."},
            {"role": "user", "content": "\\n".join(parts)}
        ]}
        response = None
        for endpoint in endpoints:
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=90,
            )
            if response.status_code not in (404, 405) or len(endpoints) == 1:
                break
        if response is None:
            return jsonify({"error": "Не удалось создать запрос к API."}), 502
        if not response.ok:
            detail = response.text[:500].replace("\\n", " ")
            return jsonify({"error": f"API вернул HTTP {response.status_code}. {detail}"}), 502
        try:
            data = response.json()
        except ValueError:
            return jsonify({"error": "API вернул не JSON. Проверь адрес API и совместимость провайдера."}), 502
        answer = data["choices"][0]["message"]["content"]
        if isinstance(answer, list):
            answer = "\n".join(x.get("text", "") for x in answer if isinstance(x, dict))
        return jsonify({"reply": str(answer), "files": file_names})
    except requests.Timeout:
        return jsonify({"error": "API не ответил вовремя."}), 504
    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        app.logger.exception("API request failed")
        return jsonify({"error": f"Не удалось обработать ответ API: {exc}"}), 502

@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Слишком большой запрос. Общий лимит — 25 МБ."}), 413

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
