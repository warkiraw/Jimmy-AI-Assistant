"""
Flask-приложение: RAG с локальной Ollama и ChromaDB.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory

from config import UPLOAD_FOLDER, GENERATED_REPORTS_FOLDER, GENERATED_PRESENTATIONS_FOLDER
from document_processor import process_document, ALLOWED_EXTENSIONS
from llm_service import (
    add_documents_to_db,
    delete_documents_by_source,
    get_sources_in_db,
    check_ollama_available,
    rag_query,
    get_report_suggestions,
    generate_report_text,
    get_presentation_suggestions,
    get_presentation_slides,
)
from report_generator import markdown_to_docx, generate_report_filename
from presentation_generator import build_pptx, generate_presentation_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["GENERATED_REPORTS_FOLDER"] = str(GENERATED_REPORTS_FOLDER)
app.config["GENERATED_PRESENTATIONS_FOLDER"] = str(GENERATED_PRESENTATIONS_FOLDER)

executor = ThreadPoolExecutor(max_workers=4)


def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Проверка доступности Ollama."""
    try:
        ok = check_ollama_available()
        return jsonify({"ollama_available": ok}), 200
    except Exception as e:
        return jsonify({"ollama_available": False, "error": str(e)}), 503


@app.route("/api/files", methods=["GET"])
def list_files():
    """Возвращает список файлов в базе (источников в ChromaDB)."""
    try:
        files = get_sources_in_db()
        return jsonify({"files": files}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/files/upload", methods=["POST"])
def upload_files():
    """Принимает файлы, парсит и добавляет в векторную БД."""
    if "files" not in request.files and "file" not in request.files:
        return jsonify({"error": "Файлы не переданы"}), 400

    files = request.files.getlist("files") or [request.files["file"]]
    uploaded = []
    errors = []

    for f in files:
        if not f or not f.filename:
            continue
        if not allowed_file(f.filename):
            errors.append(f"{f.filename}: неподдерживаемый формат")
            continue

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
        try:
            f.save(filepath)
            chunks = process_document(filepath)
            if chunks:
                add_documents_to_db(chunks)
                uploaded.append(f.filename)
            else:
                errors.append(f"{f.filename}: не удалось извлечь текст")
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            errors.append(f"{f.filename}: {str(e)}")

    return jsonify({
        "uploaded": uploaded,
        "errors": errors,
    }), 200 if uploaded else (400 if errors else 200)


@app.route("/api/files/<path:filename>", methods=["DELETE"])
def delete_file(filename):
    """Удаляет файл с диска и его векторы из ChromaDB."""
    if ".." in filename or "/" in filename or "\\" in filename:
        return jsonify({"error": "Недопустимое имя файла"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "Файл не найден"}), 404

    try:
        os.remove(filepath)
        delete_documents_by_source(filename)
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """RAG-чат: принимает вопрос, возвращает ответ на основе документов."""
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Параметр 'question' обязателен"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Вопрос не может быть пустым"}), 400

    selected_files = data.get("selected_files") or []
    ok, err = _require_selected_files(selected_files)
    if not ok:
        return jsonify({"error": err}), 400

    try:
        if not check_ollama_available():
            return jsonify({"error": "Сервер ИИ недоступен"}), 503

        def _query():
            return rag_query(question, selected_files=selected_files)

        future = executor.submit(_query)
        answer = future.result(timeout=120)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tools/report/suggestions", methods=["GET"])
def report_suggestions():
    """Возвращает 3 рекомендуемые темы для отчётов на основе загруженных документов."""
    selected = request.args.getlist("selected_files") or []
    ok, err = _require_selected_files(selected)
    if not ok:
        return jsonify({"error": err}), 400
    try:
        if not check_ollama_available():
            return jsonify({"error": "Сервер ИИ недоступен"}), 503
        suggestions = get_report_suggestions(selected_files=selected if selected else None)
        return jsonify({"suggestions": suggestions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tools/report/generate", methods=["POST"])
def report_generate():
    """Генерирует отчёт в DOCX по теме (prompt)."""
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Параметр 'prompt' обязателен"}), 400
    prompt = data["prompt"].strip()
    if not prompt:
        return jsonify({"error": "Тема отчёта не может быть пустой"}), 400
    selected_files = data.get("selected_files") or []
    ok, err = _require_selected_files(selected_files)
    if not ok:
        return jsonify({"error": err}), 400
    try:
        if not check_ollama_available():
            return jsonify({"error": "Сервер ИИ недоступен"}), 503

        def _generate():
            text = generate_report_text(prompt, selected_files=selected_files)
            filename = generate_report_filename()
            filepath = GENERATED_REPORTS_FOLDER / filename
            markdown_to_docx(text, prompt, filepath)
            return filename

        future = executor.submit(_generate)
        filename = future.result(timeout=300)  # 5 минут
        return jsonify({
            "download_url": f"/api/download/{filename}",
            "filename": filename,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tools/presentation/suggestions", methods=["GET"])
def presentation_suggestions():
    """3 рекомендуемые темы для презентаций."""
    selected = request.args.getlist("selected_files") or []
    ok, err = _require_selected_files(selected)
    if not ok:
        return jsonify({"error": err}), 400
    try:
        if not check_ollama_available():
            return jsonify({"error": "Сервер ИИ недоступен"}), 503
        suggestions = get_presentation_suggestions(selected_files=selected if selected else None)
        return jsonify({"suggestions": suggestions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tools/presentation/generate", methods=["POST"])
def presentation_generate():
    """Генерирует презентацию (PPTX) по теме. Agentic Pipeline: 9 запросов к LLM."""
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Параметр 'prompt' обязателен"}), 400
    prompt = data["prompt"].strip()
    if not prompt:
        return jsonify({"error": "Тема презентации не может быть пустой"}), 400
    selected_files = data.get("selected_files") or []
    ok, err = _require_selected_files(selected_files)
    if not ok:
        return jsonify({"error": err}), 400
    try:
        if not check_ollama_available():
            return jsonify({"error": "Сервер ИИ недоступен"}), 503

        def _generate():
            slides = get_presentation_slides(prompt, selected_files=selected_files)
            base = generate_presentation_filename()
            pptx_path = GENERATED_PRESENTATIONS_FOLDER / f"{base}.pptx"
            build_pptx(slides, pptx_path)
            return base

        future = executor.submit(_generate)
        base = future.result(timeout=300)  # 5 мин (1 план + 8 слайдов)
        filename = f"{base}.pptx"
        return jsonify({
            "pptx_url": f"/api/download/{filename}",
            "filename": filename,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _require_selected_files(selected_files: list) -> tuple[bool, str | None]:
    """Проверка: если selected_files пуст — возвращает (False, error_msg)."""
    if not selected_files:
        return False, "Файлы не выбраны"
    return True, None


@app.route("/api/tools/results", methods=["GET"])
def tools_results():
    """Список сгенерированных отчётов и презентаций, отсортированный по дате (новые сверху)."""
    results = []
    for folder, ftype in [
        (GENERATED_REPORTS_FOLDER, "report"),
        (GENERATED_PRESENTATIONS_FOLDER, "presentation"),
    ]:
        path = Path(folder)
        if not path.exists():
            continue
        for f in path.iterdir():
            if f.is_file():
                stat = f.stat()
                url = f"/api/download/{f.name}"
                results.append({
                    "filename": f.name,
                    "type": ftype,
                    "url": url,
                    "created_at": int(stat.st_mtime),
                })
    results.sort(key=lambda x: x["created_at"], reverse=True)
    return jsonify(results)


@app.route("/api/download/<path:filename>", methods=["GET"])
def download_file(filename):
    """Отдаёт сгенерированный файл (отчёт или презентация) для скачивания."""
    if ".." in filename or "/" in filename or "\\" in filename:
        return jsonify({"error": "Недопустимое имя файла"}), 400
    if filename.startswith("pres_"):
        folder = app.config["GENERATED_PRESENTATIONS_FOLDER"]
    else:
        folder = app.config["GENERATED_REPORTS_FOLDER"]
    return send_from_directory(
        folder, filename, as_attachment=True, download_name=filename
    )


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
