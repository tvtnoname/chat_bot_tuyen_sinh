from typing import Dict, Optional
import uuid

class SessionManager:
    def __init__(self):
        # Lưu trữ trong bộ nhớ: {session_id: {"branch": "...", "grade": "...", "history": []}}
        self._sessions: Dict[str, Dict] = {}

    def create_session(self) -> str:
        """Tạo phiên mới và trả về session_id."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "branch": None,
            "grade": None,
            "subject": None,
            "pending_query": None, # Câu hỏi đang chờ thông tin
            "history": [] 
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        return self._sessions.get(session_id)

    def update_context(self, session_id: str, branch: Optional[str] = None, grade: Optional[str] = None, subject: Optional[str] = None, pending_query: Optional[str] = None):
        if session_id in self._sessions:
            if branch:
                self._sessions[session_id]["branch"] = branch
            if grade:
                self._sessions[session_id]["grade"] = grade
            if subject:
                self._sessions[session_id]["subject"] = subject
            if pending_query is not None:
                 # Truyền chuỗi rỗng để xóa nếu cần
                 self._sessions[session_id]["pending_query"] = pending_query

    def get_context(self, session_id: str) -> Dict[str, Optional[str]]:
        if session_id in self._sessions:
            return {
                "branch": self._sessions[session_id]["branch"],
                "grade": self._sessions[session_id]["grade"],
                "subject": self._sessions[session_id].get("subject"),
                "pending_query": self._sessions[session_id].get("pending_query")
            }
        return {"branch": None, "grade": None, "subject": None, "pending_query": None}
    
    def clear_context(self, session_id: str):
         if session_id in self._sessions:
            self._sessions[session_id]["branch"] = None
            self._sessions[session_id]["grade"] = None
            self._sessions[session_id]["subject"] = None

session_manager = SessionManager()
