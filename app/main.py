from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.chat import router as chat_router
from app.api.v1.history import router as history_router
from app.services.rag.engine import rag_service
import logging

# Cấu hình logging hệ thống
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Dịch vụ AI Chatbot (RAG)", version="1.0.0")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký router
app.include_router(chat_router, prefix="/api/v1")
app.include_router(history_router, prefix="/api/v1")

from app.core.database import init_db
# Import models để đăng ký bảng
from app.models import chat as chat_models

@app.on_event("startup")
async def startup_event():
    """Khởi tạo dịch vụ khi ứng dụng bắt đầu."""
    logging.info("Khởi tạo Database...")
    await init_db()
    await rag_service.initialize()

@app.get("/")
async def root():
    return {"message": "Dịch vụ AI Chatbot đang chạy. Sử dụng POST /api/chat để tương tác."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
