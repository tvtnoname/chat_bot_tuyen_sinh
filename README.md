---
title: Chatbot Tuyen Sinh
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# Chatbot Tuyển Sinh - Trung Tâm Thăng Long

Đây là API Backend cho Chatbot tư vấn tuyển sinh, sử dụng kiến trúc RAG (Retrieval-Augmented Generation).

## Công nghệ
*   **FastAPI**: Web Framework.
*   **LangChain**: AI Framework.
*   **Google Gemini**: LLM & Embeddings.
*   **ChromaDB**: Vector Database.

## API Endpoints
*   `POST /api/chat`: Gửi câu hỏi và nhận câu trả lời.
    *   Input: `{"question": "..."}`
    *   Output: `{"answer": "..."}`
