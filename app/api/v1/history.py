from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from sqlalchemy import select, desc
from app.core.database import AsyncSessionLocal
from app.models.chat import ChatSession, ChatMessage
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

class SessionDTO(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    
class MessageDTO(BaseModel):
    role: str
    content: str
    created_at: datetime

@router.get("/history/sessions", response_model=List[SessionDTO])
async def get_user_sessions(user_id: str, limit: int = 20):
    """Lấy danh sách các phiên chat của một user."""
    async with AsyncSessionLocal() as db:
        # Lấy sessions, sắp xếp mới nhất trước
        stmt = select(ChatSession).where(ChatSession.user_id == user_id).order_by(ChatSession.created_at.desc()).limit(limit)
        result = await db.execute(stmt)
        sessions = result.scalars().all()
        
        output = []
        for s in sessions:
            # Lấy tin nhắn đầu tiên làm title
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

@router.get("/history/{session_id}", response_model=List[MessageDTO])
async def get_session_history(session_id: str):
    """Lấy chi tiết lịch sử chat của một phiên."""
    async with AsyncSessionLocal() as db:
        stmt = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id.asc())
        result = await db.execute(stmt)
        msgs = result.scalars().all()
        
        return [MessageDTO(role=m.role, content=m.content, created_at=m.created_at) for m in msgs]
