# Jimmy - Local AI Assistant with RAG & Agentic Tools

Локальный AI-ассистент для анализа документов с поддержкой RAG (Retrieval-Augmented Generation) и инструментов генерации отчётов и презентаций.

## Описание

Jimmy позволяет загружать документы (PDF, Word, Excel), индексировать их в векторной базе ChromaDB и задавать вопросы по содержимому. Все вычисления выполняются локально через Ollama — без облачных API и утечки данных.

**Возможности:**
- Загрузка и векторизация PDF, DOCX, XLSX
- Чат с ИИ строго по загруженным документам (без галлюцинаций)
- Генерация Word-отчётов на основе документов
- Генерация PowerPoint-презентаций (Agentic Pipeline: план + 8 слайдов)
- Фильтрация по выбранным файлам (чекбоксы)

## Стек технологий

- **Backend:** Python 3.10+, Flask
- **Frontend:** HTML5, CSS3, Bootstrap 5, Vanilla JavaScript
- **AI:** Ollama (локальный LLM), LangChain
- **Векторная БД:** ChromaDB
- **Парсинг:** PyMuPDF, python-docx, pandas, openpyxl
- **Генерация:** python-docx, python-pptx

## Особенности (Features)

- **Строгий RAG:** ИИ отвечает только на основе контекста из документов. При отсутствии информации — явный отказ.
- **Agentic Pipeline для презентаций:** Двухэтапная генерация (план → 8 отдельных запросов) для максимального качества слайдов.
- **Локальность:** Все данные и вычисления на вашем компьютере.

## Запуск

### 1. Установка Ollama

Скачайте и установите [Ollama](https://ollama.ai/).

### 2. Загрузка моделей

```bash
ollama pull bge-m3:latest   # эмбеддинги
ollama pull gemma3:4b       # LLM для чата и генерации
```

### 3. Установка зависимостей Python

```bash
pip install -r requirements.txt
```

### 4. Конфигурация

Скопируйте `.env.example` в `.env` и при необходимости измените настройки:

```bash
copy .env.example .env
```

### 5. Запуск приложения

```bash
flask run
```

или

```bash
python app.py
```

Откройте в браузере: http://localhost:5000

## Структура проекта

```
├── app.py                 # Flask-приложение, API
├── config.py              # Конфигурация из .env
├── document_processor.py  # Парсинг PDF, Word, Excel
├── llm_service.py         # Ollama, ChromaDB, RAG
├── report_generator.py    # Генерация DOCX
├── presentation_generator.py  # Генерация PPTX
├── templates/
├── static/
├── uploads/               # Загруженные файлы
├── chroma_db/             # Векторная БД
├── generated_reports/     # Сгенерированные отчёты
└── generated_presentations/
```

## Лицензия

MIT
