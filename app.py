"""
app.py — FastAPI REST-сервис для ответов на вопросы по документации (RAG).

Запуск:
    uvicorn app:app --reload

Swagger UI: http://127.0.0.1:8000/docs
"""

from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from documents import DOCUMENTS
from rag import rag_answer, search, _chunks

app = FastAPI(
    title="RAG Q&A Service",
    description="Сервис для ответов на вопросы по документации компании.",
    version="1.0.0",
)

class AskRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "example": {"question": "Как оформить возврат?"}
        }
    }


class SourceItem(BaseModel):
    doc_id: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]

@app.get("/health", summary="Проверка работоспособности")
def health():
    return {
        "status": "ok",
        "documents": len(DOCUMENTS),
        "chunks": len(_chunks),
    }


@app.get("/info", summary="Информация о сервисе")
def info():
    return {
        "service": "RAG Q&A Service",
        "version": "1.0.0",
        "documents": [{"id": d["id"], "title": d["title"]} for d in DOCUMENTS],
        "total_chunks": len(_chunks),
        "embedding_model": "all-MiniLM-L6-v2",
        "search_method": "cosine_similarity",
        "default_top_k": 3,
    }


@app.post("/ask", response_model=AskResponse, summary="Задать вопрос")
def ask(request: AskRequest):
    """
    Принимает вопрос на русском языке, ищет релевантные фрагменты
    в базе знаний и возвращает сформированный ответ с источниками.
    """
    if not request.question or len(request.question.strip()) < 2:
        raise HTTPException(status_code=400, detail="Вопрос слишком короткий.")

    result = rag_answer(request.question)

    sources = [
        SourceItem(
            doc_id=s["doc_id"],
            score=round(s["score"], 4),
            snippet=s["snippet"][:150] + "..." if len(s["snippet"]) > 150 else s["snippet"],
        )
        for s in result["sources"]
    ]

    return AskResponse(answer=result["answer"], sources=sources)
