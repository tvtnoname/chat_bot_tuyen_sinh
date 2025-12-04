from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from app.services.rag import rag_service
import logging

# Cấu hình logging
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
app.include_router(api_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """Khởi tạo dịch vụ khi ứng dụng bắt đầu."""
    await rag_service.initialize()

@app.get("/")
async def root():
    return {"message": "Dịch vụ AI Chatbot đang chạy. Sử dụng POST /api/chat để tương tác."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
