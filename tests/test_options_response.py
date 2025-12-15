import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Global Patches
patcher_auth = patch("google.auth.default", return_value=(MagicMock(), "test-project"))
patcher_llm = patch("langchain_google_genai.ChatGoogleGenerativeAI")
patcher_auth.start()
patcher_llm.start()

from app.schemas.chat import ChatInput
from app.api.v1.chat import chat

async def test_options_returned():
    print("--- TESTING OPTIONS FIELD IN RESPONSE ---")
    
    # Mock Orchestrator to simulate missing info response
    # We mock the method on the class or instance. 
    # Since chat_orchestrator is imported in api.chat, we need to patch where it's used.
    
    # But wait, app.api.v1.chat imports it inside the function: `from app.services.chat.orchestrator import chat_orchestrator`
    # So we must patch `app.services.chat.orchestrator.chat_orchestrator`
    
    with patch("app.services.chat.orchestrator.chat_orchestrator.process_message") as mock_process:
        # Scenario: Returning options
        mock_process.return_value = (
            "Vui lòng chọn chi nhánh:", 
            "sess_123", 
            ["Hà Nội", "Sài Gòn"]
        )
        
        input_data = ChatInput(question="Học phí bao nhiêu", session_id="sess_123")
        result = await chat(input_data)
        
        print("Result:", result)
        
        if result.get("options") == ["Hà Nội", "Sài Gòn"]:
            print("[PASS] Options field is present and correct.")
        else:
            print("[FAIL] Options field is missing or incorrect.")

if __name__ == "__main__":
    try:
        asyncio.run(test_options_returned())
    finally:
        patcher_auth.stop()
        patcher_llm.stop()
