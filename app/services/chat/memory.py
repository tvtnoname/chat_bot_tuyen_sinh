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
            # Kiểm tra tồn tại trước cho an toàn, hoặc update trực tiếp
            query = update(ChatSession).where(ChatSession.id == session_id)
            values = {}
            if branch: values["branch"] = branch
            if grade: values["grade"] = grade
            if subject: values["subject"] = subject
            
            if values:
                await db.execute(query.values(**values))
                await db.commit()

    async def add_message(self, session_id: str, role: str, content: str, options: list = None, courses: list = None):
        import json
        async with AsyncSessionLocal() as db:
            # 1. Lưu tin nhắn mới vào cơ sở dữ liệu
            options_json = json.dumps(options) if options else None
            courses_json = json.dumps(courses) if courses else None
            
            msg = ChatMessage(
                session_id=session_id, 
                role=role, 
                content=content, 
                options=options_json,
                courses=courses_json
            )
            db.add(msg)
            
            # 2. Nếu là tin nhắn người dùng và phiên chưa có tiêu đề -> Tự động tạo tiêu đề
            if role == "user":
                # Kiểm tra tiêu đề hiện tại
                session = await db.scalar(select(ChatSession).where(ChatSession.id == session_id))
                if session and not session.title:
                    # Tạo tiêu đề từ nội dung tin nhắn (tối đa 50 ký tự)
                    new_title = (content[:50] + "...") if len(content) > 50 else content
                    session.title = new_title
            
            await db.commit()

    async def get_history(self, session_id: str) -> List[Dict]:
        import json
        async with AsyncSessionLocal() as db:
            # Lấy 10 tin nhắn gần nhất từ lịch sử
            stmt = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id.desc()).limit(10)
            result = await db.execute(stmt)
            msgs = result.scalars().all()
            
            # Đảo ngược danh sách để đúng trình tự thời gian (Cũ -> Mới) cho LLM hiểu ngữ cảnh
            history = []
            for m in reversed(msgs):
                msg_dict = {"role": m.role, "content": m.content}
                if m.options:
                    try:
                        msg_dict["options"] = json.loads(m.options)
                    except:
                        pass
                if m.courses:
                    try:
                        msg_dict["courses"] = json.loads(m.courses)
                    except:
                        pass
                history.append(msg_dict)
            return history

    async def update_session_title(self, session_id: str, new_title: str):
        """Cập nhật tiêu đề phiên chat."""
        async with AsyncSessionLocal() as db:
            query = update(ChatSession).where(ChatSession.id == session_id).values(title=new_title)
            await db.execute(query)
            await db.commit()

    async def delete_session(self, session_id: str):
        """Xóa phiên chat và toàn bộ tin nhắn liên quan."""
        from sqlalchemy import delete
        async with AsyncSessionLocal() as db:
            # 1. Xóa tất cả tin nhắn trong phiên trước
            await db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
            # 2. Sau đó xóa phiên chat
            await db.execute(delete(ChatSession).where(ChatSession.id == session_id))
            await db.commit()

session_manager = SessionManager()
