import os
import hashlib
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from rag_engine import RAGEngine

app = FastAPI(title="Local AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

rag = RAGEngine()

class ChatRequest(BaseModel):
    prompt: str
    context_docs: Optional[List[str]] = None

class TrainRequest(BaseModel):
    text: str
    document_name: str

# GET эндпоинты (работают через браузер)
@app.get("/")
def root():
    return {"message": "AI работает", "status": "active", "methods": "GET support added"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": rag.is_model_loaded()}

@app.get("/models/list")
def list_models():
    models_dir = "/app/models"
    if not os.path.exists(models_dir):
        return {"models": []}
    models = [f for f in os.listdir(models_dir) if f.endswith('.gguf')]
    return {"models": models}

@app.get("/models/load")
async def load_model_get(model_name: str):
    """Загрузить модель через GET запрос"""
    model_path = f"/app/models/{model_name}"
    if not os.path.exists(model_path):
        raise HTTPException(404, f"Модель {model_name} не найдена")
    success = rag.load_model(model_path)
    if success:
        return {"message": f"Модель {model_name} загружена", "model": model_name}
    raise HTTPException(500, "Ошибка загрузки")

@app.get("/chat")
async def chat_get(prompt: str, context: Optional[str] = None):
    """Отправить запрос через GET (пример: /chat?prompt=Привет)"""
    if not rag.is_model_loaded():
        return {"error": "Модель не загружена", "response": "Сначала загрузите модель через /models/load?model_name=...", "model_loaded": False}
    
    context_list = [context] if context else None
    rag_context = rag.search_relevant(prompt, top_k=3)
    response = rag.generate_response(prompt, rag_context)
    
    return {
        "response": response,
        "prompt": prompt,
        "used_context": len(rag_context) > 0,
        "model_loaded": True
    }

@app.get("/train/custom")
async def train_custom_get(text: str, document_name: str = "custom_doc"):
    """Обучить AI через GET запрос (/train/custom?text=текст&document_name=название)"""
    doc_id = rag.add_custom_knowledge(document_name, text)
    return {
        "message": f"Обучение завершено: {document_name}",
        "tokens": len(text),
        "id": doc_id,
        "document_name": document_name
    }

@app.get("/documents/upload")
async def upload_document_get(file_path: str):
    """Загрузить документ через GET (нужно указать полный путь к файлу)"""
    if not os.path.exists(file_path):
        raise HTTPException(404, f"Файл {file_path} не найден")
    
    doc_id = rag.add_document(file_path)
    return {
        "message": f"Документ загружен",
        "file_path": file_path,
        "doc_id": doc_id
    }

@app.get("/knowledge/stats")
def knowledge_stats():
    return rag.get_stats()

@app.get("/knowledge/clear")
def knowledge_clear():
    """Очистить базу знаний"""
    rag.knowledge_base.clear()
    rag.documents_index.clear()
    return {"message": "База знаний очищена"}

# POST эндпоинты (оставляем для обратной совместимости)
@app.post("/chat")
async def chat_post(request: ChatRequest):
    if not rag.is_model_loaded():
        raise HTTPException(503, "Модель не загружена")
    context = rag.search_relevant(request.prompt, top_k=3)
    response = rag.generate_response(request.prompt, context)
    return {"response": response, "used_context": len(context) > 0}

@app.post("/train/custom")
async def train_custom_post(request: TrainRequest):
    doc_id = rag.add_custom_knowledge(request.document_name, request.text)
    return {"message": "Обучение завершено", "id": doc_id}

@app.post("/documents/upload")
async def upload_document_post(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.txt', '.md')):
        raise HTTPException(400, "Только PDF, TXT или MD")
    file_path = f"/app/uploads/{uuid.uuid4()}_{file.filename}"
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    doc_id = rag.add_document(file_path)
    return {"message": f"Документ загружен", "doc_id": doc_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
