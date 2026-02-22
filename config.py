"""
Конфигурация приложения из переменных окружения.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
FLASK_APP = os.getenv("FLASK_APP", "app")
FLASK_ENV = os.getenv("FLASK_ENV", "development")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))

UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", "./uploads"))
VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", "./chroma_db"))
GENERATED_REPORTS_FOLDER = Path(os.getenv("GENERATED_REPORTS_FOLDER", "./generated_reports"))
GENERATED_PRESENTATIONS_FOLDER = Path(os.getenv("GENERATED_PRESENTATIONS_FOLDER", "./generated_presentations"))

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
GENERATED_REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)
GENERATED_PRESENTATIONS_FOLDER.mkdir(parents=True, exist_ok=True)
