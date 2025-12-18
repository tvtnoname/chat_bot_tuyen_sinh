# AI Agent Tư Vấn Tuyển Sinh (RAG + Agentic)

![GenAI](https://img.shields.io/badge/Generative%20AI-Agentic%20Workflow-purple)
![RAG](https://img.shields.io/badge/RAG-Hybrid%20Search-orange)
![LLM](https://img.shields.io/badge/LLM-Google%20Gemini-blue)

## Tổng quan

Hệ thống **AI Agent tự động hóa** quy trình tư vấn tuyển sinh. Khác với chatbot truyền thống, dự án sử dụng kiến trúc **RAG (Retrieval-Augmented Generation)** kết hợp với **Agentic Workflow** để tra cứu dữ liệu thời gian thực và trả lời chính xác, giải quyết triệt để vấn đề "ảo giác" (Hallucination) của LLM.

##  Điểm nhấn Công nghệ (AI Engineering)

1.  **Agentic Reasoning (Tư duy Agent)**:
    *   Sử dụng mô hình **ReAct (Reason + Act)** giúp AI tự động phân tích intent, trích xuất thực thể (NER) và quyết định gọi Tools.
    *   Tự động truy vấn API (Lịch học, học phí) hoặc tra cứu Knowledge Base (Quy chế, thông tin chung).

2.  **Advanced RAG Engine (Bộ nhớ)**:
    *   **Hybrid Search**: Kết hợp tìm kiếm từ khóa (**BM25**) và tìm kiếm ngữ nghĩa (**Vector/Embeddings**) để không bỏ sót thông tin.
    *   **Reranking**: Tái xếp hạng kết quả bằng **Cross-Encoder (FlashRank)**, đảm bảo độ chính xác cực cao trước khi gửi vào LLM.

3.  **Tối ưu Hiệu năng**:
    *   **Streaming Response**: Trả lời từng từ (Token streaming) qua SSE, mượt mà như ChatGPT.
    *   **Contextual Memory**: Ghi nhớ ngữ cảnh hội thoại đa lượt (Multi-turn conversation).

## 🛠️ Tech Stack

| Thành phần | Công nghệ | Vai trò |
| :--- | :--- | :--- |
| **LLM** | **Google Gemini 2.0 Flash** | Bộ não xử lý ngôn ngữ và suy luận |
| **Framework** | **LangChain** | Điều phối Agent và quản lý bộ nhớ |
| **Vector DB** | **ChromaDB** | Lưu trữ và tìm kiếm vector hiệu năng cao |
| **Backend** | **FastAPI (Python)** | Xử lý bất đồng bộ (Asyncio) và Streaming |
| **Cache** | **Redis** | Tăng tốc độ phản hồi và lưu session |
| **Deployment** | **Docker** | Đóng gói và triển khai dễ dàng |