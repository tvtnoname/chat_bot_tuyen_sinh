from pydantic import BaseModel

class ChatInput(BaseModel):
    question: str
    session_id: str = None

class ChatOutput(BaseModel):
    answer: str
    session_id: str
