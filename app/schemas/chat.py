from pydantic import BaseModel

class ChatInput(BaseModel):
    question: str
    session_id: str = None

from typing import List, Optional

class ChatOutput(BaseModel):
    answer: str
    session_id: str
    options: Optional[List[str]] = []
