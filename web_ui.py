import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Локальный ИИ", layout="wide")
st.title("🧠 Мой Локальный ИИ")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Управление")
    
    # Статус
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        if health.status_code == 200:
            st.success("✅ Сервер активен")
            data = health.json()
            if data.get('model_loaded'):
                st.info("🤖 Модель загружена")
            else:
                st.warning("⚠️ Модель не загружена")
    except:
        st.error("❌ Сервер не доступен")
    
    st.divider()
    
    # Загрузка модели
    st.subheader("📥 Загрузка GGUF модели")
    model_name = st.text_input("Имя файла .gguf", placeholder="tinyllama.gguf")
    if st.button("Загрузить модель", type="primary"):
        with st.spinner("Загрузка..."):
            resp = requests.post(f"{API_URL}/models/load", params={"model_name": model_name})
            if resp.status_code == 200:
                st.success("✅ Модель загружена!")
            else:
                st.error(f"Ошибка: {resp.text}")
    
    st.divider()
    
    # Загрузка документов
    st.subheader("📚 Обучение на документах")
    uploaded_file = st.file_uploader("Загрузить PDF/TXT", type=["pdf", "txt", "md"])
    if uploaded_file and st.button("📖 Обработать документ"):
        files = {"file": uploaded_file}
        with st.spinner("Индексация..."):
            resp = requests.post(f"{API_URL}/documents/upload", files=files)
            if resp.status_code == 200:
                st.success("✅ Документ добавлен в базу знаний!")
            else:
                st.error("Ошибка")
    
    st.subheader("✏️ Ручное обучение")
    doc_name = st.text_input("Название темы")
    custom_text = st.text_area("Текст для обучения", height=150)
    if st.button("🎓 Обучить на тексте") and custom_text:
        train_data = {"text": custom_text, "document_name": doc_name or "custom"}
        resp = requests.post(f"{API_URL}/train/custom", json=train_data)
        if resp.status_code == 200:
            st.success("✅ Знания добавлены!")
    
    if st.button("📊 Статистика БЗ"):
        resp = requests.get(f"{API_URL}/knowledge/stats")
        if resp.status_code == 200:
            st.json(resp.json())

# Main chat
st.header("💬 Диалог с ИИ")
prompt = st.text_area("Ваш вопрос:", height=100)
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🚀 Отправить", type="primary"):
        if prompt:
            with st.spinner("Генерация ответа..."):
                resp = requests.post(f"{API_URL}/chat", json={"prompt": prompt})
                if resp.status_code == 200:
                    result = resp.json()
                    st.subheader("Ответ:")
                    st.write(result["response"])
                    if result["used_context"]:
                        st.success("📚 Ответ основан на загруженных документах")
                else:
                    st.error(f"Ошибка: {resp.text}")
        else:
            st.warning("Введите вопрос")

st.markdown("---")
st.info("💡 **Как использовать:**\n1. Положите .gguf файл в папку `models`\n2. Загрузите модель через боковую панель\n3. Обучите на документах или тексте\n4. Задавайте вопросы!")
