from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatInput
import logging

router = APIRouter()

@router.post("/stream")
async def chat_stream(input_data: ChatInput):
    """API endpoint để chat với bot (Streaming - ChatGPT style)."""
    if not input_data.question:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
        
    from app.services.chat.orchestrator import chat_orchestrator
    
    logging.info(f"DEBUG: Yêu cầu streaming cho user_id='{input_data.user_id}' session_id='{input_data.session_id}'")
    
    async def event_generator():
        try:
             async for chunk in chat_orchestrator.process_message_stream(input_data.question, input_data.session_id, input_data.user_id):
                 yield chunk
        except Exception as e:
             logging.error(f"Stream error: {e}")
             import json
             yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
