import os
import sys
import logging
from typing import List
from dotenv import load_dotenv

# Tải biến môi trường từ file .env
load_dotenv()

# --- Sửa lỗi SQLite cho Render (Chỉ chạy khi có pysqlite3) ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- Cấu hình ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.warning("Biến môi trường GOOGLE_API_KEY chưa được thiết lập. Chức năng chat sẽ không hoạt động.")

# --- Khởi tạo Ứng dụng ---
app = FastAPI(title="Dịch vụ AI Chatbot (RAG)", version="1.0.0")

# --- Cấu hình CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn (origins) để đơn giản hóa. Hãy thay đổi trong môi trường production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Biến Toàn cục ---
vector_store = None
qa_chain = None

# --- Models ---
class ChatInput(BaseModel):
    question: str

class ChatOutput(BaseModel):
    answer: str

# --- Sự kiện Khởi động: Tải Knowledge Base & Khởi tạo RAG ---
@app.on_event("startup")
async def startup_event():
    global vector_store, qa_chain
    try:
        logging.info("Đang tải cơ sở tri thức (knowledge base)...")
        loader = TextLoader("knowledge_base.txt", encoding="utf-8")
        documents = loader.load()

        logging.info("Đang chia nhỏ văn bản (splitting text)...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        logging.info("Đang khởi tạo Embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        logging.info("Đang tạo Vector Store (ChromaDB)...")
        # Sử dụng in-memory ChromaDB để đơn giản hóa trên Render (hệ thống tệp tạm thời)
        # Để lưu trữ lâu dài, hãy gắn ổ đĩa hoặc sử dụng cloud vector DB.
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="knowledge_base"
        )

        logging.info("Đang khởi tạo LLM (Gemini 1.5 Flash)...")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)

        logging.info("Đang tạo chuỗi QA (QA Chain)...")
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        logging.info("Khởi động hoàn tất. Hệ thống đã sẵn sàng.")

    except Exception as e:
        logging.error(f"Lỗi trong quá trình khởi động: {e}")
        # Trong ứng dụng thực tế, bạn có thể muốn raise lỗi để dừng deployment nếu nghiêm trọng
        # raise e

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Dịch vụ AI Chatbot đang chạy. Sử dụng POST /api/chat để tương tác."}

@app.post("/api/chat", response_model=ChatOutput)
async def chat(input_data: ChatInput):
    global qa_chain
    if not qa_chain:
        raise HTTPException(status_code=503, detail="Hệ thống chưa được khởi tạo hoặc khởi động thất bại.")
    
    if not input_data.question:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    try:
        # Chạy chuỗi RAG
        response = qa_chain.invoke({"query": input_data.question})
        return {"answer": response["result"]}
    except Exception as e:
        logging.error(f"Lỗi khi xử lý yêu cầu chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
