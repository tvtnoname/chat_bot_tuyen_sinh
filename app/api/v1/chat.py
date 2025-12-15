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
        
        answer, session_id, options = await chat_orchestrator.process_message(
            input_data.question, 
            input_data.session_id
        )
        
        return {"answer": answer, "session_id": session_id, "options": options}
    except Exception as e:
        logging.error(f"Lỗi xử lý: {e}")
        # Trả về thông báo lỗi thân thiện thay vì 500
        return {
            "answer": f"Hệ thống đang gặp sự cố kết nối hoặc lỗi nội bộ. Vui lòng thử lại sau. (Chi tiết kỹ thuật: {str(e)})", 
            "session_id": input_data.session_id or "error_session"
        }
