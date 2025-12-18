from pydantic import BaseModel

class ChatInput(BaseModel):
    """Mô hình dữ liệu đầu vào cho API Chat."""
    question: str
    session_id: str = None
    user_id: str = None

from typing import List, Optional

class Course(BaseModel):
    """Mô hình thông tin khóa học."""
    id: Optional[str] = None
    name: str
    schedule: Optional[str] = None
    location: Optional[str] = None
    price: Optional[str] = None
    status: Optional[str] = None
    endDate: Optional[str] = None

class ChatOutput(BaseModel):
    """Mô hình dữ liệu đầu ra cho API Chat (Non-streaming)."""
    answer: str
    session_id: str
    options: Optional[List[str]] = []
    courses: Optional[List[Course]] = []
