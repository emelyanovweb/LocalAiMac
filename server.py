import os
import hashlib
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from rag_engine import RAGEngine
import json

app = FastAPI(title="Local AI")

# Настройка CORS и UTF-8
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGEngine()

class ChatRequest(BaseModel):
    prompt: str
    context_docs: Optional[List[str]] = None

class TrainRequest(BaseModel):
    text: str
    document_name: str

# Middleware для UTF-8
@app.middleware("http")
async def add_utf8_header(request, call_next):
    response = await call_next(request)
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

@app.get("/")
def root():
    return JSONResponse(content={"message": "AI работает", "status": "active"}, headers={"Content-Type": "application/json; charset=utf-8"})

@app.get("/health")
def health():
    return JSONResponse(
        content={"status": "healthy", "model_loaded": rag.is_model_loaded()},
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

@app.get("/models/list")
def list_models():
    models_dir = "/app/models"
    if not os.path.exists(models_dir):
        return {"models": []}
    models = [f for f in os.listdir(models_dir) if f.endswith('.gguf')]
    return JSONResponse(content={"models": models}, headers={"Content-Type": "application/json; charset=utf-8"})

@app.get("/models/load")
def load_model_get(model_name: str):
    model_path = f"/app/models/{model_name}"
    if not os.path.exists(model_path):
        return JSONResponse(content={"error": f"Модель {model_name} не найдена"}, headers={"Content-Type": "application/json; charset=utf-8"})
    success = rag.load_model(model_path)
    if success:
        return JSONResponse(content={"message": f"Модель {model_name} загружена", "model": model_name}, headers={"Content-Type": "application/json; charset=utf-8"})
    return JSONResponse(content={"error": "Ошибка загрузки модели"}, headers={"Content-Type": "application/json; charset=utf-8"})

@app.get("/chat")
def chat_get(prompt: str):
    if not rag.is_model_loaded():
        return JSONResponse(content={"error": "Модель не загружена", "response": "Загрузите модель через /models/load?model_name=..."}, headers={"Content-Type": "application/json; charset=utf-8"})
    rag_context = rag.search_relevant(prompt, top_k=3)
    response = rag.generate_response(prompt, rag_context)
    return JSONResponse(
        content={"response": response, "prompt": prompt, "used_context": len(rag_context) > 0},
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

@app.get("/train/custom")
def train_custom_get(text: str, document_name: str = "custom_doc"):
    doc_id = rag.add_custom_knowledge(document_name, text)
    return JSONResponse(
        content={"message": f"Обучение завершено: {document_name}", "tokens": len(text), "id": doc_id},
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

@app.get("/knowledge/stats")
def knowledge_stats():
    return JSONResponse(content=rag.get_stats(), headers={"Content-Type": "application/json; charset=utf-8"})

@app.get("/knowledge/clear")
def knowledge_clear():
    rag.knowledge_base.clear()
    return JSONResponse(content={"message": "База знаний очищена"}, headers={"Content-Type": "application/json; charset=utf-8"})

@app.post("/chat")
async def chat_post(request: ChatRequest):
    if not rag.is_model_loaded():
        raise HTTPException(503, "Модель не загружена")
    context = rag.search_relevant(request.prompt, top_k=3)
    response = rag.generate_response(request.prompt, context)
    return JSONResponse(content={"response": response, "used_context": len(context) > 0}, headers={"Content-Type": "application/json; charset=utf-8"})

@app.post("/train/custom")
async def train_custom_post(request: TrainRequest):
    doc_id = rag.add_custom_knowledge(request.document_name, request.text)
    return JSONResponse(content={"message": "Обучение завершено", "id": doc_id}, headers={"Content-Type": "application/json; charset=utf-8"})

@app.post("/documents/upload")
async def upload_document_post(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.txt', '.md')):
        raise HTTPException(400, "Только PDF, TXT или MD")
    file_path = f"/app/uploads/{uuid.uuid4()}_{file.filename}"
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    doc_id = rag.add_document(file_path)
    return JSONResponse(content={"message": "Документ загружен", "doc_id": doc_id}, headers={"Content-Type": "application/json; charset=utf-8"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
