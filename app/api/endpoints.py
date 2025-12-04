from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatInput, ChatOutput
from app.services.rag import rag_service
import logging

router = APIRouter()

@router.post("/chat", response_model=ChatOutput)
async def chat(input_data: ChatInput):
    """API endpoint để chat với bot."""
    if not input_data.question:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    try:
        answer = rag_service.get_answer(input_data.question)
        return {"answer": answer}
    except Exception as e:
        logging.error(f"Lỗi xử lý: {e}")
        if "Hệ thống chưa sẵn sàng" in str(e):
             raise HTTPException(status_code=503, detail="Hệ thống chưa sẵn sàng.")
        raise HTTPException(status_code=500, detail="Lỗi máy chủ nội bộ")
