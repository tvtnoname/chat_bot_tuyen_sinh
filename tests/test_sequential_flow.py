import asyncio
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Global Patches
patcher_auth = patch("google.auth.default", return_value=(MagicMock(), "test-project"))
patcher_llm = patch("langchain_google_genai.ChatGoogleGenerativeAI")
patcher_auth.start()
patcher_llm.start()

from app.schemas.chat import ChatInput
from app.api.v1.chat import chat
from app.services.chat.orchestrator import chat_orchestrator
from app.services.chat.memory import session_manager

async def test_sequential_flow():
    print("--- TESTING SEQUENTIAL FLOW ---")
    session_id = "sess_seq_001"
    
    # Ensure fresh session
    if session_id in session_manager._sessions:
        del session_manager._sessions[session_id]

    # 1. User asks question (Missing Branch & Grade)
    print("\n1. User: Học phí bao nhiêu?")
    
    # Use AsyncMock for async methods
    with patch("app.services.external.school_api.external_api_service.get_all_branches", new_callable=AsyncMock) as mock_branches:
        mock_branches.return_value = ["Số 1 ĐCV", "Số 2 ĐCV"]
        
        with patch("app.services.chat.intent.intent_classifier.classify", new_callable=AsyncMock) as mock_classify:
            mock_classify.return_value = "DATABASE_QUERY"
            
            with patch.object(chat_orchestrator, "_extract_entities", new_callable=AsyncMock) as mock_extract:
                mock_extract.return_value = (None, None)
                
                input_data = ChatInput(question="Học phí bao nhiêu", session_id=session_id)
                res1 = await chat(input_data)
                print(f"Bot Step 1: {res1['answer']} | Options: {res1.get('options')}")
                
                if "địa chỉ nào" in res1["answer"] and res1.get("options") == ["Số 1 ĐCV", "Số 2 ĐCV"]:
                    print("[PASS] Step 1")
                else:
                    print("[FAIL] Step 1")

    # 2. User provides Address "Số 1 ĐCV"
    print("\n2. User: Số 1 ĐCV")
    
    with patch("app.services.external.school_api.external_api_service.get_all_grades", new_callable=AsyncMock) as mock_grades:
        mock_grades.return_value = ["10", "11"]
        
        with patch("app.services.chat.intent.intent_classifier.classify", new_callable=AsyncMock) as mock_classify:
            mock_classify.return_value = "DATABASE_QUERY"
            
            with patch.object(chat_orchestrator, "_extract_entities", new_callable=AsyncMock) as mock_extract:
                mock_extract.return_value = ("Thăng Long A", None)
                 
                input_data = ChatInput(question="Số 1 ĐCV", session_id=session_id)
                res2 = await chat(input_data)
                print(f"Bot Step 2: {res2['answer']} | Options: {res2.get('options')}")
                 
                if "khối lớp nào" in res2["answer"] and res2.get("options") == ["10", "11"]:
                     print("[PASS] Step 2")
                else:
                     print("[FAIL] Step 2")

    # 3. User provides Grade "10"
    print("\n3. User: 10")
    with patch("app.services.chat.intent.intent_classifier.classify", new_callable=AsyncMock) as mock_classify:
        mock_classify.return_value = "DATABASE_QUERY"
        
        with patch.object(chat_orchestrator, "_extract_entities", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = (None, "10") # Branch preserved in session, grade extracted
             
            with patch.object(chat_orchestrator, "_generate_data_response", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = "Học phí là 10 triệu."
                
                with patch("app.services.external.school_api.external_api_service.get_filtered_data", new_callable=AsyncMock) as mock_data:
                    mock_data.return_value = {"some": "data"}
                    
                    input_data = ChatInput(question="10", session_id=session_id)
                    res3 = await chat(input_data)
                    print(f"Bot Step 3: {res3['answer']}")
                     
                    if "Học phí là" in res3["answer"]:
                         print("[PASS] Step 3")
                    else:
                         print("[FAIL] Step 3")

if __name__ == "__main__":
    try:
        asyncio.run(test_sequential_flow())
    finally:
        patcher_auth.stop()
        patcher_llm.stop()
