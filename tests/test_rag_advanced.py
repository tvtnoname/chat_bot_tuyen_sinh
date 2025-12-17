import asyncio
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.rag.engine import rag_service

async def main():
    print(">>> Starting RAG Verification...")
    try:
        await rag_service.initialize()
        if rag_service.ready:
            print(">>> RAG Initialized Successfully!")
            
            # Test Query
            query = "Trung tâm Thăng Long ở đâu?"
            print(f">>> Testing Query: {query}")
            answer = rag_service.get_answer(query)
            print(f">>> Answer: {answer}")
            
            # Test Specific Keyword (which Vector might miss)
            query2 = "XYZ-999"
            print(f">>> Testing Keyword Query: {query2}")
            answer2 = rag_service.get_answer(query2)
            print(f">>> Answer 2: {answer2}")
            
        else:
            print(">>> RAG Failed to Initialize.")
    except Exception as e:
        print(f">>> Error during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
