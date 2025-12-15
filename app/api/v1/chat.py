from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatInput, ChatOutput
import logging

router = APIRouter()

@router.post("/chat", response_model=ChatOutput)
async def chat(input_data: ChatInput):
    """API endpoint để chat với bot."""
    if not input_data.question:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    try:
        # Sử dụng ChatOrchestrator
        from app.services.chat.orchestrator import chat_orchestrator
        
        answer, session_id = await chat_orchestrator.process_message(
            input_data.question, 
            input_data.session_id
        )
        
        return {"answer": answer, "session_id": session_id}
    except Exception as e:
        logging.error(f"Lỗi xử lý: {e}")
        # Có thể handle cụ thể các lỗi khác nếu cần
        raise HTTPException(status_code=500, detail="Lỗi máy chủ nội bộ")
