import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Global patches setup
patcher_auth = patch("google.auth.default", return_value=(MagicMock(), "test-project"))
patcher_llm = patch("langchain_google_genai.ChatGoogleGenerativeAI")

def setup_mocks():
    patcher_auth.start()
    patcher_llm.start()

def teardown_mocks():
    patcher_auth.stop()
    patcher_llm.stop()

# Import after patches might help if checked at import time, but mostly it's instance creation.
# We will just start mocks immediately.
setup_mocks()

from app.schemas.chat import ChatInput
from app.api.v1.chat import chat
from app.services.rag.engine import RAGService

async def test_chat_exception_handling():
    print("--- TESTING CHAT API EXCEPTION HANDLING ---")
    
    # Mock input
    input_data = ChatInput(question="Test error", session_id="test_session")
    
    # Patch the orchestrator to raise an exception
    # Since orchestrator is imported inside API, we need to patch sys.modules or target where it ends up.
    # But since it is dynamically imported, we can mock the module in sys.modules before it gets imported?
    # Or rely on the fact that if we patch 'app.services.chat.orchestrator.chat_orchestrator.process_message', 
    # we need orchestrator to be loaded first.
    
    # Force load orchestrator so we can patch it
    try:
        from app.services.chat.orchestrator import chat_orchestrator
    except:
        pass # It is mocked by our global LLM patch so it should load fine
        
    with patch("app.services.chat.orchestrator.chat_orchestrator.process_message", side_effect=Exception("Simulated Connection Error")):
        result = await chat(input_data)
        
        print("\nResult received:")
        print(result)
        
        # Verify
        if isinstance(result, dict) and "answer" in result:
            answer = result["answer"]
            if "Hệ thống đang gặp sự cố" in answer and "Simulated Connection Error" in answer:
                print("\n[PASS] API returned friendly error message with details.")
            else:
                print("\n[FAIL] API did not return expected error message.")
        else:
            print("\n[FAIL] API did not return a dict or missing 'answer'.")

async def test_rag_service_resilience():
    print("\n--- TESTING RAG SERVICE RESILIENCE ---")
    
    # We create a NEW instance to test initialization independently of the global singleton
    with patch("langchain_community.document_loaders.TextLoader.load", side_effect=Exception("File not found")):
        rag = RAGService()
        await rag.initialize()
        
        if rag.ready is False:
             print("[PASS] RAGService handled initialization error gracefully (ready=False).")
        else:
             print(f"[FAIL] RAGService should be not ready after error. Got: {rag.ready}")

        # Test get_answer behavior
        response = rag.get_answer("test")
        if "chưa sẵn sàng" in response:
            print("[PASS] get_answer returned fallback message.")
        else:
            print(f"[FAIL] get_answer returned unexpected: {response}")

async def main():
    try:
        await test_chat_exception_handling()
        await test_rag_service_resilience()
    finally:
        teardown_mocks()


if __name__ == "__main__":
    asyncio.run(main())
