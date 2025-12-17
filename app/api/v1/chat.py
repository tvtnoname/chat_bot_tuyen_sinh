from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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
        
        answer, session_id, options, courses = await chat_orchestrator.process_message(
            input_data.question, 
            input_data.session_id,
            input_data.user_id
        )
        
        return {
            "answer": answer, 
            "session_id": session_id, 
            "options": options,
            "courses": courses
        }
    except Exception as e:
        logging.error(f"Lỗi xử lý: {e}")
        # Trả về thông báo lỗi thân thiện thay vì 500
        return {
            "answer": f"Hệ thống đang gặp sự cố kết nối hoặc lỗi nội bộ. Vui lòng thử lại sau. (Chi tiết kỹ thuật: {str(e)})", 
            "session_id": input_data.session_id or "error_session"
        }

@router.post("/stream")
async def chat_stream(input_data: ChatInput):
    """API endpoint để chat với bot (Streaming)."""
    if not input_data.question:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
        
    from app.services.chat.orchestrator import chat_orchestrator
    
    async def event_generator():
        try:
             async for chunk in chat_orchestrator.process_message_stream(input_data.question, input_data.session_id, input_data.user_id):
                 yield chunk
        except Exception as e:
             logging.error(f"Stream error: {e}")
             import json
             yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
