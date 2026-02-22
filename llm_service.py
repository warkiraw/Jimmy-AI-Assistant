"""
Модуль работы с Ollama и ChromaDB: эмбеддинги, векторная БД, RAG-пайплайн.
"""
import json
import os
import re
from pathlib import Path

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")
TOP_K = int(os.getenv("TOP_K", "8"))

STRICT_SYSTEM_PROMPT = """Ты — профессиональный AI-ассистент по анализу документов. Твоя задача — отвечать на вопросы пользователя на основе предоставленного контекста. В контексте указаны источники (имена файлов) и фрагменты текста из них.

Правила:
1. Если пользователь просит общую информацию, суть или саммари, сделай краткий и логичный обзор на основе тех фрагментов, что тебе переданы.
2. Если информации для ответа на конкретный вопрос совершенно нет в контексте, ответь: 'В загруженных документах нет информации для ответа на этот вопрос.'
3. Не выдумывай факты, которых нет в тексте.
4. Отвечай всегда на русском языке.

Контекст:
{context}"""


def _get_embeddings() -> OllamaEmbeddings:
    """Создаёт экземпляр OllamaEmbeddings с настройками из .env."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def _get_vector_store() -> Chroma:
    """Возвращает персистентное хранилище ChromaDB."""
    Path(VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name="documents",
        embedding_function=_get_embeddings(),
        persist_directory=VECTOR_DB_PATH,
    )


def add_documents_to_db(chunks: list[tuple[str, str]]) -> None:
    """
    Добавляет чанки в ChromaDB с метаданными source.
    chunks: список кортежей (text, source_filename)
    """
    if not chunks:
        return
    texts = [c[0] for c in chunks]
    metadatas = [{"source": c[1]} for c in chunks]
    vector_store = _get_vector_store()
    vector_store.add_texts(texts=texts, metadatas=metadatas)


def delete_documents_by_source(filename: str) -> None:
    """Удаляет из ChromaDB все векторы с metadata.source == filename."""
    vector_store = _get_vector_store()
    collection = vector_store._collection
    # ChromaDB delete by metadata
    collection.delete(where={"source": filename})


def get_sources_in_db() -> list[str]:
    """Возвращает список уникальных имён файлов (source) в базе."""
    vector_store = _get_vector_store()
    collection = vector_store._collection
    result = collection.get(include=["metadatas"])
    if not result or not result["metadatas"]:
        return []
    sources = set(m["source"] for m in result["metadatas"] if m)
    return sorted(sources)


def check_ollama_available() -> bool:
    """Проверяет доступность Ollama-сервера."""
    try:
        embeddings = _get_embeddings()
        # Простой тест — эмбеддинг короткой строки
        embeddings.embed_query("test")
        return True
    except Exception:
        return False


def _retriever_kwargs(selected_files: list[str] | None = None) -> dict:
    """Формирует search_kwargs для retriever с опциональной фильтрацией по source."""
    kwargs = {"k": TOP_K}
    if selected_files:
        kwargs["filter"] = {"source": {"$in": selected_files}}
    return kwargs


def rag_query(question: str, selected_files: list[str] | None = None) -> str:
    """
    Выполняет RAG-запрос: поиск контекста в ChromaDB и генерация ответа через Ollama.
    selected_files: если задан и не пуст — поиск только по этим файлам.
    """
    vector_store = _get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs=_retriever_kwargs(selected_files))

    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    def format_docs(docs):
        parts = []
        for doc in docs:
            source = doc.metadata.get("source", "неизвестно")
            parts.append(f"Источник: [{source}]\nТекст: {doc.page_content}")
        return "\n---\n".join(parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", STRICT_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)


REPORT_SUGGESTIONS_PROMPT = """На основе следующих фрагментов документов предложи 3 темы для подробных аналитических отчетов, которые были бы полезны пользователю. Темы должны быть короткими (до 6 слов). Верни ответ СТРОГО в формате JSON-массива строк, без лишнего текста и форматирования. Пример: ["Анализ продаж за 2023 год", "Оценка рисков проекта", "Сводка по договорам"]. Контекст: {context}"""

REPORT_GENERATOR_PROMPT = """Ты — профессиональный аналитик. Напиши подробный, структурированный и красивый отчет на тему: '{prompt}'. Используй ТОЛЬКО предоставленный контекст. Отчет должен содержать введение, основную часть с подзаголовками и выводы.

Контекст:
{context}"""

DEFAULT_SUGGESTIONS = ["Общее резюме документов", "Ключевые тезисы", "Анализ данных"]


def get_report_suggestions(selected_files: list[str] | None = None) -> list[str]:
    """
    Возвращает 3 темы для отчётов на основе контекста из ChromaDB.
    selected_files: если задан и не пуст — поиск только по этим файлам.
    """
    vector_store = _get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs=_retriever_kwargs(selected_files))
    docs = retriever.invoke("общая тематика документов, ключевые темы, содержание")
    if not docs:
        return DEFAULT_SUGGESTIONS.copy()
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "неизвестно")
        parts.append(f"Источник: [{source}]\nТекст: {doc.page_content}")
    context = "\n---\n".join(parts)
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    prompt = REPORT_SUGGESTIONS_PROMPT.format(context=context)
    try:
        raw = llm.invoke(prompt)
        text = raw.content if hasattr(raw, "content") else str(raw)
        # Извлечь JSON-массив из ответа (модель может добавить текст вокруг)
        match = re.search(r"\[[\s\S]*?\]", text)
        if match:
            arr = json.loads(match.group())
            if isinstance(arr, list) and len(arr) >= 3:
                return [str(x) for x in arr[:3]]
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return DEFAULT_SUGGESTIONS.copy()


def generate_report_text(prompt: str, selected_files: list[str] | None = None) -> str:
    """
    Генерирует текст отчёта через RAG-пайплайн с промптом «Генератора отчётов».
    selected_files: если задан и не пуст — поиск только по этим файлам.
    """
    vector_store = _get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs=_retriever_kwargs(selected_files))
    docs = retriever.invoke(prompt)
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "неизвестно")
        parts.append(f"Источник: [{source}]\nТекст: {doc.page_content}")
    context = "\n---\n".join(parts) if parts else ""
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    system_prompt = REPORT_GENERATOR_PROMPT.format(prompt=prompt, context=context)
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Сгенерируй отчёт."),
    ])
    chain = chat_prompt | llm | StrOutputParser()
    return chain.invoke({})


PRESENTATION_SUGGESTIONS_PROMPT = """На основе следующих фрагментов документов предложи 3 короткие темы для презентаций, которые были бы полезны пользователю. Темы до 6 слов. Верни СТРОГО JSON-массив строк. Пример: ["Анализ продаж 2023", "Риски проекта", "Сводка договоров"]. Контекст: {context}"""

PRESENTATION_OUTLINE_PROMPT = """Ты эксперт по презентациям. Составь план презентации на тему: '{prompt}' на основе контекста. Верни СТРОГО JSON-массив из 8 объектов. Каждый объект: 'slide' (1-8), 'type' (строго: title, toc, intro, bullets, mindmap, two_columns, conclusion, outro), 'slide_topic' (о чём этот слайд). Никакого лишнего текста.

Контекст: {context}"""

TEXT_LIMIT = "ПРАВИЛО: Текст краткий! Текстовые блоки не более 3 предложений. Пункты не более 4 штук по 10 слов. Никакого markdown вне JSON."

SLIDE_PROMPTS = {
    "title": """Верни СТРОГО JSON-объект: {{"slide": {slide}, "type": "title", "title": "Заголовок", "subtitle": "Подзаголовок до 10 слов"}}""",
    "toc": """Верни СТРОГО JSON-объект: {{"slide": {slide}, "type": "toc", "title": "Содержание", "points": ["пункт 1", "пункт 2", "пункт 3", "пункт 4"]}}. Не более 4 пунктов по 10 слов.""",
    "intro": """Верни СТРОГО JSON-объект: {{"slide": {slide}, "type": "intro", "title": "Введение", "text": "Текст не более 3 предложений"}}""",
    "bullets": """Верни СТРОГО JSON-объект: {{"slide": {slide}, "type": "bullets", "title": "Заголовок", "points": ["тезис 1", "тезис 2", "тезис 3", "тезис 4"]}}. Не более 4 пунктов по 10 слов.""",
    "mindmap": """Верни СТРОГО JSON-объект: {{"slide": {slide}, "type": "mindmap", "title": "Структура", "core": "Центральная идея", "branches": ["ветвь 1", "ветвь 2", "ветвь 3"]}}. Не более 4 ветвей по 10 слов.""",
    "two_columns": """Верни СТРОГО JSON-объект: {{"slide": {slide}, "type": "two_columns", "title": "Заголовок", "col1_title": "Часть 1", "col1_text": "до 3 предложений", "col2_title": "Часть 2", "col2_text": "до 3 предложений"}}""",
    "conclusion": """Верни СТРОГО JSON-объект: {{"slide": {slide}, "type": "conclusion", "title": "Вывод", "text": "Текст не более 3 предложений"}}""",
    "outro": """Верни СТРОГО JSON-объект: {{"slide": {slide}, "type": "outro", "title": "Спасибо за внимание", "text": "Ваши вопросы?"}}""",
}


def get_presentation_suggestions(selected_files: list[str] | None = None) -> list[str]:
    """3 темы для презентаций (аналогично отчётам)."""
    vector_store = _get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs=_retriever_kwargs(selected_files))
    docs = retriever.invoke("общая тематика документов, ключевые темы")
    if not docs:
        return ["Обзор документов", "Ключевые выводы", "Анализ данных"]
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "неизвестно")
        parts.append(f"Источник: [{source}]\nТекст: {doc.page_content}")
    context = "\n---\n".join(parts)
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    prompt = PRESENTATION_SUGGESTIONS_PROMPT.format(context=context)
    try:
        raw = llm.invoke(prompt)
        text = raw.content if hasattr(raw, "content") else str(raw)
        match = re.search(r"\[[\s\S]*?\]", text)
        if match:
            arr = json.loads(match.group())
            if isinstance(arr, list) and len(arr) >= 3:
                return [str(x) for x in arr[:3]]
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return ["Обзор документов", "Ключевые выводы", "Анализ данных"]


def _get_context(retriever, query: str) -> str:
    """RAG-контекст по запросу."""
    docs = retriever.invoke(query)
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "неизвестно")
        parts.append(f"Источник: [{source}]\nТекст: {doc.page_content}")
    return "\n---\n".join(parts) if parts else ""


def get_presentation_outline(prompt: str, selected_files: list[str] | None) -> list[dict]:
    """Этап 1: генерация плана из 8 слайдов."""
    vector_store = _get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs=_retriever_kwargs(selected_files))
    context = _get_context(retriever, prompt)
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    full_prompt = PRESENTATION_OUTLINE_PROMPT.format(prompt=prompt, context=context)
    try:
        raw = llm.invoke(full_prompt)
        text = raw.content if hasattr(raw, "content") else str(raw)
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            arr = json.loads(match.group())
            if isinstance(arr, list) and len(arr) >= 8:
                return arr[:8]
            if isinstance(arr, list):
                return arr
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    # Fallback
    types = ["title", "toc", "intro", "bullets", "mindmap", "two_columns", "conclusion", "outro"]
    return [{"slide": i + 1, "type": types[i], "slide_topic": prompt} for i in range(8)]


def get_presentation_slide_content(
    outline_item: dict, prompt: str, context: str
) -> dict:
    """Этап 2: генерация контента одного слайда."""
    slide_num = outline_item.get("slide", 1)
    stype = outline_item.get("type", "intro")
    topic = outline_item.get("slide_topic", prompt)
    tmpl = SLIDE_PROMPTS.get(stype, SLIDE_PROMPTS["intro"])
    slide_prompt = tmpl.format(slide=slide_num)
    full_prompt = f"""Ты генерируешь ТОЛЬКО ОДИН слайд для презентации на тему '{prompt}'.
Текущий слайд: {slide_num}, Тип: {stype}, Тема слайда: {topic}.
{slide_prompt}
{TEXT_LIMIT}

Контекст:
{context}"""
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    try:
        raw = llm.invoke(full_prompt)
        text = raw.content if hasattr(raw, "content") else str(raw)
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                obj.setdefault("slide", slide_num)
                obj.setdefault("type", stype)
                return obj
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return {"slide": slide_num, "type": stype, "title": topic, "text": topic}


def get_presentation_slides(prompt: str, selected_files: list[str] | None = None) -> list[dict]:
    """
    Agentic Pipeline: 1) план (outline), 2) 8 отдельных запросов для каждого слайда.
    """
    vector_store = _get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs=_retriever_kwargs(selected_files))
    outline = get_presentation_outline(prompt, selected_files if selected_files else None)
    slides_data = []
    for s in outline:
        context = _get_context(retriever, s.get("slide_topic", prompt))
        slide_content = get_presentation_slide_content(s, prompt, context)
        slides_data.append(slide_content)
    return slides_data
