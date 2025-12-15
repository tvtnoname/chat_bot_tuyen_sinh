import asyncio
import logging
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mocking before importing services that use them
sys.modules["langchain_google_genai"] = MagicMock()
sys.modules["langchain_huggingface"] = MagicMock()
sys.modules["langchain_community.vectorstores"] = MagicMock()

# Mock settings
with patch("app.core.config.settings") as mock_settings:
    mock_settings.GOOGLE_API_KEY = "fake_key"
    mock_settings.KNOWLEDGE_BASE_PATH = "data/knowledge_base.txt"
    mock_settings.MODEL_NAME = "gemini-flash"
    mock_settings.EMBEDDING_MODEL = "fake-embedding"

    # Now import services
    from app.services.chat.orchestrator import chat_orchestrator
    from app.services.rag.engine import rag_service
    from app.services.external.school_api import external_api_service

    # Setup Mocks for LLM and Chains
    mock_llm = MagicMock()
    # Mock invoke to return something based on input
    async def mock_ainvoke(input_dict):
        text = input_dict.get("text", "") or input_dict.get("question", "")
        # Mock Intent Classifier
        if "DATABASE_QUERY" in str(chat_orchestrator.intent_classifier.prompt): 
            # This is hard to detect because we mock the class. 
            pass
        
        return MagicMock(content="MOCK CONTENT")

    # Patch Intent Classifier
    # Since chat_orchestrator module imports intent_classifier, we can just patch the classify method of that imported object
    # OR we can patch 'app.services.chat_orchestrator.intent_classifier'
    
    # Let's use `from app.services.chat_orchestrator import intent_classifier as orchestrator_intent_classifier`
    # But checking source code: `from app.services.intent_classifier import intent_classifier`
    # Both are the same object since it's a singleton instance in intent_classifier.py
    
    from unittest.mock import AsyncMock

    from app.services.chat.intent import intent_classifier
    intent_classifier.classify = AsyncMock()
    intent_classifier.classify.side_effect = lambda q: "DATABASE_QUERY" if any(x in q.lower() for x in ["lịch", "chi nhánh", "lớp", "học phí"]) else "GENERAL_CHAT"

    # Mock `extract_entities` logic inside orchestrator (it's a method)
    # Patching methods on an instance is fine.
    
    chat_orchestrator._extract_entities = AsyncMock()
    async def extract_impl(text):
        b, g = None, None
        if "hà nội" in text.lower(): b = "Thăng Long Hà Nội"
        if "đà nẵng" in text.lower(): b = "Thăng Long Đà Nẵng"
        if "lớp 10" in text.lower(): g = "10"
        if "lớp 11" in text.lower(): g = "11"
        return b, g
    chat_orchestrator._extract_entities.side_effect = extract_impl

    # Mock `_generate_data_response`
    chat_orchestrator._generate_data_response = AsyncMock(return_value="[MOCK ANSWER BASED ON DATA]")

    # Mock RAG
    rag_service.get_answer = MagicMock(return_value="[MOCK RAG ANSWER]")

    async def main():
        logging.basicConfig(level=logging.INFO)
        print("--- STARTING TEST ---")

        print("\n--- TEST CASE 1: General Chat ---")
        q1 = "Xin chào"
        print(f"User: {q1}")
        a1, sid1 = await chat_orchestrator.process_message(q1, session_id=None)
        print(f"Bot: {a1} (Session: {sid1})")

        print("\n--- TEST CASE 2: DB Query (Missing Info) ---")
        q2 = "Lịch học lớp 10 thế nào?"
        print(f"User: {q2}")
        a2, sid2 = await chat_orchestrator.process_message(q2, session_id=None)
        print(f"Bot: {a2} (Session: {sid2})")
        
        print("\n--- TEST CASE 3: Providing Branch ---")
        q3 = "Mình học ở Thăng Long Hà Nội"
        print(f"User: {q3}")
        a3, sid3 = await chat_orchestrator.process_message(q3, session_id=sid2) 
        print(f"Bot: {a3} (Session: {sid3})")

        print("\n--- TEST CASE 4: Full Info Query ---")
        q4 = "Cho mình biết lịch nghỉ lễ nhé"
        print(f"User: {q4}")
        a4, sid4 = await chat_orchestrator.process_message(q4, session_id=sid3)
        print(f"Bot: {a4} (Session: {sid4})")

    if __name__ == "__main__":
        asyncio.run(main())
