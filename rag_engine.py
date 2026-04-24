import os
import hashlib
from typing import Dict

class RAGEngine:
    def __init__(self):
        self.model = None
        self.knowledge_base: Dict[str, str] = {}
        self.model_loaded = False
        
    def load_model(self, model_path):
        try:
            from llama_cpp import Llama
            self.model = Llama(model_path=model_path, n_ctx=2048, n_threads=4)
            self.model_loaded = True
            return True
        except Exception as e:
            print(f"Ошибка: {e}")
            return False
    
    def is_model_loaded(self):
        return self.model_loaded
    
    def add_document(self, file_path):
        text = self._extract_text(file_path)
        doc_id = hashlib.md5(text.encode()).hexdigest()
        self.knowledge_base[doc_id] = text
        return doc_id
    
    def add_custom_knowledge(self, name, text):
        doc_id = hashlib.md5(text.encode()).hexdigest()
        self.knowledge_base[doc_id] = text
        return doc_id
    
    def _extract_text(self, file_path):
        if file_path.endswith(('.txt', '.md')):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif file_path.endswith('.pdf'):
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
            except:
                return "PDF parsing error"
        return ""
    
    def search_relevant(self, query, top_k=3):
        query_words = set(query.lower().split())
        scores = []
        for doc_id, text in self.knowledge_base.items():
            score = sum(1 for word in query_words if word in text.lower())
            if score > 0:
                scores.append((score, text[:500]))
        scores.sort(reverse=True, key=lambda x: x[0])
        relevant = [text for score, text in scores[:top_k]]
        return "\n---\n".join(relevant) if relevant else ""
    
    def generate_response(self, prompt, context):
        if not self.model_loaded:
            return "Модель не загружена. Сначала загрузите GGUF файл через API."
        if context:
            full_prompt = f"Используй контекст для ответа:\n{context}\n\nВопрос: {prompt}\n\nОтвет:"
        else:
            full_prompt = f"Вопрос: {prompt}\n\nОтвет:"
        try:
            output = self.model(full_prompt, max_tokens=512, temperature=0.7)
            return output['choices'][0]['text'].strip()
        except Exception as e:
            return f"Ошибка: {e}"
    
    def get_stats(self):
        return {
            "documents": len(self.knowledge_base),
            "total_chars": sum(len(t) for t in self.knowledge_base.values()),
            "model_loaded": self.model_loaded
        }
