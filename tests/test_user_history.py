import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app
import uuid

async def test_user_history_flow():
    user_id = f"user_{uuid.uuid4()}"
    question = "Tôi muốn tìm lớp học tiếng Anh"
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        print(f">>> 1. Chatting with user_id={user_id}")
        # 1. Chat to create session
        response = await client.post("/api/v1/chat", json={
            "question": question,
            "user_id": user_id
        })
        assert response.status_code == 200
        data = response.json()
        session_id = data["session_id"]
        print(f"    Created Session ID: {session_id}")
        
        # 2. Check History List
        print(f">>> 2. Get History Sessions for {user_id}")
        res_hist = await client.get(f"/api/v1/history/sessions?user_id={user_id}")
        assert res_hist.status_code == 200
        sessions = res_hist.json()
        print(f"    Sessions found: {len(sessions)}")
        assert len(sessions) >= 1
        assert sessions[0]["session_id"] == session_id
        print(f"    Title: {sessions[0]['title']}")
        
        # 3. Check Session Details
        print(f">>> 3. Get Details for {session_id}")
        res_det = await client.get(f"/api/v1/history/{session_id}")
        assert res_det.status_code == 200
        msgs = res_det.json()
        print(f"    Messages count: {len(msgs)}")
        assert len(msgs) >= 2 # User + Assistant
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == question
        
    print(">>> SUCCESS: User History flow verified!")

if __name__ == "__main__":
    asyncio.run(test_user_history_flow())
