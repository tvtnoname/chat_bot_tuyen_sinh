import asyncio
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.chat.orchestrator import chat_orchestrator
from app.services.external.school_api import external_api_service

async def test_extraction():
    # Ensure data is loaded
    await external_api_service.fetch_all_data()
    
    query = "danh sách môn học lớp 12"
    print(f"Testing query: '{query}'")
    
    branch, grade, subject = await chat_orchestrator._extract_entities(query)
    
    print(f"Extracted Entities:")
    print(f"Branch: {branch}")
    print(f"Grade: {grade}")
    print(f"Subject: {subject}")
    
    if grade == "12":
        print("SUCCESS: Grade 12 extracted.")
    else:
        print(f"FAIL: Grade is '{grade}' (Expected '12')")

if __name__ == "__main__":
    asyncio.run(test_extraction())
