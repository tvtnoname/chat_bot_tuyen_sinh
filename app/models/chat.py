from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from app.core.database import Base

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Ngữ cảnh (Lưu trữ thông tin trích xuất)
    branch = Column(String, nullable=True)
    grade = Column(String, nullable=True) 
    subject = Column(String, nullable=True)
    user_id = Column(String, nullable=True, index=True)
    title = Column(String, nullable=True)
    
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), index=True)
    role = Column(String)
    content = Column(Text)
    options = Column(Text, nullable=True)
    courses = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
