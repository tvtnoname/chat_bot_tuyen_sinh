import asyncio
import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.chat.orchestrator import chat_orchestrator
from app.core.database import init_db

async def test_streaming():
    print(">>> 1. Init DB...")
    await init_db()
    
    question = "Xin chào, bạn tên gì?"
    print(f">>> 2. Testing Stream with question: '{question}'")
    
    print("--- STREAM START ---")
    async for chunk in chat_orchestrator.process_message_stream(question):
        # Chunk is formatted as "data: ...\n\n"
        sys.stdout.write(chunk)
        sys.stdout.flush()
    print("\n--- STREAM END ---")

    question2 = "Tìm lớp toán 12 ở Hà Nội"
    print(f"\n>>> 3. Testing Stream with Tool Call: '{question2}'")
    async for chunk in chat_orchestrator.process_message_stream(question2):
        sys.stdout.write(chunk)
    print("\n--- STREAM END ---")
    
if __name__ == "__main__":
    asyncio.run(test_streaming())
