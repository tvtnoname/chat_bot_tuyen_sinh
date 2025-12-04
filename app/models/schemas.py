from pydantic import BaseModel

class ChatInput(BaseModel):
    question: str

class ChatOutput(BaseModel):
    answer: str
