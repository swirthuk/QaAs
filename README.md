# RAG Q&A Service

REST-API для ответов на вопросы по документации компании на основе RAG (Retrieval-Augmented Generation).

**Стек:** Python · sentence-transformers · scikit-learn · FastAPI

---

## Как это работает

1. Документы из `documents.py` разбиваются на предложения (чанки)
2. Каждый чанк кодируется в вектор с помощью модели `all-MiniLM-L6-v2`
3. При запросе система находит наиболее близкие по смыслу фрагменты через cosine similarity
4. На основе найденных фрагментов формируется ответ

---

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/swirthuk/rag-qa-service.git
cd rag-qa-service
```

### 2. Создать виртуальное окружение и установить зависимости

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

> При первом запуске модель `all-MiniLM-L6-v2` (~90 МБ) скачается автоматически.

### 3. Запустить сервер

```bash
uvicorn app:app --reload
```

Сервер доступен по адресу: **http://127.0.0.1:8000**  
Swagger UI: **http://127.0.0.1:8000/docs**

---

## API

### `GET /health`
```json
{
  "status": "ok",
  "documents": 5,
  "chunks": 15
}
```

### `GET /info`
Возвращает список документов, количество чанков и параметры поиска.

### `POST /ask`
Принимает вопрос, возвращает ответ и источники.

**Запрос:**
```json
{ "question": "Как оформить возврат?" }
```

**Ответ:**
```json
{
  "answer": "Чтобы оформить возврат, заполните форму в личном кабинете...",
  "sources": [
    {
      "doc_id": "doc_returns",
      "score": 0.8741,
      "snippet": "Клиент вправе оформить возврат покупки в двухнедельный срок..."
    }
  ]
}
```

---

## Структура проекта

```
rag-qa-service/
├── app.py           # FastAPI приложение (эндпоинты)
├── rag.py           # Логика RAG: чанкинг, эмбеддинги, поиск, генерация ответа
├── documents.py     # База знаний (документы компании)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Расширение базы знаний

Чтобы добавить новый документ, открой `documents.py` и добавь элемент в список `DOCUMENTS`:

```python
{
    "id": "doc_new",
    "title": "Название документа",
    "text": "Текст документа...",
},
```
