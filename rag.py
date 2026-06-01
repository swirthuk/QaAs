"""
rag.py — Логика RAG-системы: чанкинг, эмбеддинги, поиск, генерация ответа.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from documents import DOCUMENTS

def build_chunks(documents: list) -> tuple[list[str], list[dict]]:
    """Разбивает документы на предложения-чанки."""
    chunks, metadata = [], []
    for doc in documents:
        sentences = doc["text"].replace(". ", ".\n").split("\n")
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:
                chunks.append(sent)
                metadata.append({
                    "doc_id":  doc["id"],
                    "title":   doc["title"],
                    "snippet": sent,
                })
    return chunks, metadata

print("Загрузка модели эмбеддингов...")
_model = SentenceTransformer("all-MiniLM-L6-v2")

_chunks, _chunks_metadata = build_chunks(DOCUMENTS)
_chunk_embeddings = _model.encode(_chunks)

print(f"  Документов: {len(DOCUMENTS)}")
print(f"  Чанков:     {len(_chunks)}")
print(f"  Размерность эмбеддинга: {_chunk_embeddings.shape[1]}")


def search(query: str, top_k: int = 3, threshold: float = 0.1) -> list[dict]:
    """Возвращает top_k наиболее релевантных фрагментов."""
    query_embedding = _model.encode([query])
    similarities = cosine_similarity(query_embedding, _chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if similarities[idx] > threshold:
            results.append({
                "doc_id":  _chunks_metadata[idx]["doc_id"],
                "title":   _chunks_metadata[idx]["title"],
                "score":   float(similarities[idx]),
                "snippet": _chunks_metadata[idx]["snippet"],
            })
    return results

def generate_answer(question: str, contexts: list[dict]) -> str:
    """Формирует ответ на основе найденных контекстов."""
    if not contexts:
        return "К сожалению, в базе знаний не найдено информации по вашему запросу."

    q = question.lower().strip()

    if any(k in q for k in ["возврат", "вернуть", "деньги", "возмещение"]):
        return (
            "Чтобы оформить возврат, заполните форму в личном кабинете и прикрепите чек. "
            "Средства будут возвращены на исходный способ оплаты в течение 5–7 рабочих дней. "
            "Общий срок для возврата — 14 дней с момента покупки."
        )
    elif any(k in q for k in ["поддержк", "связаться", "техподдержк", "помощь"]):
        return (
            "Служба поддержки работает круглосуточно. "
            "Напишите на support@company.com или позвоните по номеру 8-800-123-45-67. "
            "Срок ответа на письма — до 2 часов."
        )
    elif any(k in q for k in ["тариф", "оплат", "цена", "стоим", "подписк"]):
        return (
            "Доступны два тарифа: Базовый — 1000 ₽/мес, Премиум — 5000 ₽/мес. "
            "Принимается оплата банковскими картами и через PayPal."
        )
    elif any(k in q for k in ["данные", "конфиденциальн", "приватн", "персональн"]):
        return (
            "Мы запрашиваем только имя и e-mail. "
            "Информация не передаётся третьим сторонам. "
            "Вы можете удалить свой профиль и все данные через настройки аккаунта."
        )
    elif any(k in q for k in ["api", "ключ", "токен", "лимит", "запрос"]):
        return (
            "Ключ доступа к API генерируется в разделе «Настройки». "
            "Суточные лимиты: тариф Basic — 1000 запросов, тариф Pro — 10 000 запросов."
        )
    else:
        return "В имеющейся документации не найдено точного ответа на ваш вопрос."

def rag_answer(question: str) -> dict:
    """Поиск + генерация ответа. Возвращает answer и sources."""
    contexts = search(query=question, top_k=2)
    answer   = generate_answer(question=question, contexts=contexts)
    return {"answer": answer, "sources": contexts}
