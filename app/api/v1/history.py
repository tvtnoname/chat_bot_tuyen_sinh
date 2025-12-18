from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from sqlalchemy import select, desc
from app.core.database import AsyncSessionLocal
from app.models.chat import ChatSession, ChatMessage
from pydantic import BaseModel
from datetime import datetime
import json

router = APIRouter()

class SessionDTO(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    
class MessageDTO(BaseModel):
    role: str
    content: str
    created_at: datetime
    options: Optional[List[str]] = None
    courses: Optional[List[dict]] = None

class UpdateTitleDTO(BaseModel):
    title: str

@router.get("/history/sessions", response_model=List[SessionDTO])
async def get_user_sessions(user_id: str, limit: int = 20):
    """Lấy danh sách các phiên chat của một user."""
    import logging
    logging.info(f"DEBUG: Đang lấy lịch sử cho user_id='{user_id}' limit={limit}")
    async with AsyncSessionLocal() as db:
        # Lấy sessions, sắp xếp mới nhất trước
        stmt = select(ChatSession).where(ChatSession.user_id == user_id).order_by(ChatSession.created_at.desc()).limit(limit)
        result = await db.execute(stmt)
        sessions = result.scalars().all()
        
        output = []
        for s in sessions:
            # Ưu tiên lấy title từ DB
            if s.title:
                title = s.title
            else:
                # Nếu không có (cũ), lấy tin nhắn đầu tiên làm title
                msg_stmt = select(ChatMessage).where(ChatMessage.session_id == s.id).order_by(ChatMessage.id.asc()).limit(1)
                msg_res = await db.execute(msg_stmt)
                first_msg = msg_res.scalar_one_or_none()
                
                title = "Cuộc trò chuyện mới"
                if first_msg:
                    # Cắt gọn content
                    title = (first_msg.content[:50] + '...') if len(first_msg.content) > 50 else first_msg.content
                
            output.append(SessionDTO(
                session_id=s.id,
                title=title,
                created_at=s.created_at
            ))
            
        return output

@router.patch("/history/{session_id}")
async def update_session_title(session_id: str, body: UpdateTitleDTO):
    """Đổi tên phiên chat."""
    from app.services.chat.memory import session_manager
    try:
        await session_manager.update_session_title(session_id, body.title)
        return {"message": "Updated successfully", "session_id": session_id, "new_title": body.title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/history/{session_id}")
async def delete_session(session_id: str):
    """Xóa phiên chat."""
    from app.services.chat.memory import session_manager
    try:
        await session_manager.delete_session(session_id)
        return {"message": "Deleted successfully", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{session_id}", response_model=List[MessageDTO])
async def get_session_history(session_id: str):
    """Lấy chi tiết lịch sử chat của một phiên."""
    async with AsyncSessionLocal() as db:
        stmt = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id.asc())
        result = await db.execute(stmt)
        msgs = result.scalars().all()
        
        output = []
        for m in msgs:
            opts = None
            if m.options:
                try:
                    opts = json.loads(m.options)
                except:
                    opts = None
            
            crs = None
            if m.courses:
                try:
                    crs = json.loads(m.courses)
                except:
                    crs = None
            
            output.append(MessageDTO(
                role=m.role, 
                content=m.content, 
                created_at=m.created_at,
                options=opts,
                courses=crs
            ))
        return output
