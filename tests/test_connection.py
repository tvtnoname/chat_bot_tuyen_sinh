from dotenv import load_dotenv
import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Suppress dotenv error if file not found
try:
    load_dotenv()
except Exception:
    pass

def test_connection():
    try:
        from app.core.config import settings
        api_key = settings.GOOGLE_API_KEY
        if not api_key:
            print("FAIL: No GOOGLE_API_KEY found.")
            return

        print(f"Testing Model: {settings.MODEL_NAME}")
        
        llm = ChatGoogleGenerativeAI(model=settings.MODEL_NAME, google_api_key=api_key)
        response = llm.invoke("Hello, say 'OK' if you can hear me.")
        print(f"Response: {response.content}")
        print("SUCCESS: Connection established.")
    except Exception as e:
        print(f"FAIL: {e}")

if __name__ == "__main__":
    test_connection()
