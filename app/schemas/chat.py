from pydantic import BaseModel

class ChatInput(BaseModel):
    question: str
    session_id: str = None

from typing import List, Optional

class Course(BaseModel):
    id: Optional[str] = None
    name: str
    schedule: Optional[str] = None
    location: Optional[str] = None
    price: Optional[str] = None
    status: Optional[str] = None
    endDate: Optional[str] = None

class ChatOutput(BaseModel):
    answer: str
    session_id: str
    options: Optional[List[str]] = []
    courses: Optional[List[Course]] = []
