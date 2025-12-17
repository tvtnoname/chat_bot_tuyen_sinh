import uuid
import logging
from typing import Dict, Optional, List
from sqlalchemy import select, update
from app.core.database import AsyncSessionLocal
from app.models.chat import ChatSession, ChatMessage

class SessionManager:
    async def create_session(self, user_id: str = None) -> str:
        """Tạo phiên mới trong DB và trả về session_id."""
        session_id = str(uuid.uuid4())
        async with AsyncSessionLocal() as db:
            new_session = ChatSession(id=session_id, user_id=user_id)
            db.add(new_session)
            await db.commit()
        return session_id

    async def get_session(self, session_id: str):
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
            return result.scalar_one_or_none()

    async def get_context(self, session_id: str) -> Dict[str, Optional[str]]:
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
            session = result.scalar_one_or_none()
            if session:
                return {
                    "branch": session.branch,
                    "grade": session.grade,
                    "subject": session.subject
                }
        return {"branch": None, "grade": None, "subject": None}

    async def update_context(self, session_id: str, branch: Optional[str] = None, grade: Optional[str] = None, subject: Optional[str] = None):
        async with AsyncSessionLocal() as db:
            # Check exist first to be safe, or direct update
            query = update(ChatSession).where(ChatSession.id == session_id)
            values = {}
            if branch: values["branch"] = branch
            if grade: values["grade"] = grade
            if subject: values["subject"] = subject
            
            if values:
                await db.execute(query.values(**values))
                await db.commit()

    async def add_message(self, session_id: str, role: str, content: str):
        async with AsyncSessionLocal() as db:
            msg = ChatMessage(session_id=session_id, role=role, content=content)
            db.add(msg)
            await db.commit()

    async def get_history(self, session_id: str) -> List[Dict]:
        async with AsyncSessionLocal() as db:
            # Lấy 10 tin nhắn gần nhất
            stmt = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id.desc()).limit(10)
            result = await db.execute(stmt)
            msgs = result.scalars().all()
            
            # Đảo ngược lại để đúng thứ tự thời gian (cũ -> mới) cho LLM học context
            history = [{"role": m.role, "content": m.content} for m in reversed(msgs)]
            return history

session_manager = SessionManager()
