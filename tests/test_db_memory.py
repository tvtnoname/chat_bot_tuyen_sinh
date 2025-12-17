import asyncio
import sys
import os
import uuid

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.database import init_db
from app.services.chat.memory import session_manager

async def test_memory_persistence():
    print(">>> 1. Init Database...")
    await init_db()
    
    # Create Session
    print(">>> 2. Create Session...")
    session_id = await session_manager.create_session()
    print(f"    Session ID: {session_id}")
    
    # Add Message
    print(">>> 3. Add Messages...")
    await session_manager.add_message(session_id, "user", "Xin chào DB")
    await session_manager.add_message(session_id, "assistant", "Chào bạn, tôi đang sống trong Postgres")
    
    # Get History
    print(">>> 4. Get History (In-Process)...")
    history = await session_manager.get_history(session_id)
    print(f"    History len: {len(history)}")
    print(f"    Last msg: {history[-1]}")
    
    assert len(history) == 2
    assert history[-1]["content"] == "Chào bạn, tôi đang sống trong Postgres"
    
    print(">>> 5. Update Context...")
    await session_manager.update_context(session_id, branch="HVCT", grade="12")
    
    ctx = await session_manager.get_context(session_id)
    print(f"    Context: {ctx}")
    assert ctx["branch"] == "HVCT"
    
    print(">>> SUCCESS: Memory is working with Async DB!")

if __name__ == "__main__":
    asyncio.run(test_memory_persistence())
