import sys
import os
sys.path.append(os.getcwd())

try:
    from app.services.chat.orchestrator import chat_orchestrator
    print("Import SUCCESS")
except Exception as e:
    print(f"Import FAILED: {e}")
except SyntaxError as e:
    print(f"SyntaxError: {e}")
