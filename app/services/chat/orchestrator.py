import logging
import json
from typing import Tuple, List, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.services.chat.memory import session_manager
from app.services.chat.tools import search_classes, search_general_info
from app.services.external.school_api import external_api_service

class ChatOrchestrator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=settings.MODEL_NAME, google_api_key=settings.GOOGLE_API_KEY, temperature=0.3)
        
        # Bind tools to LLM
        self.tools = [search_classes, search_general_info]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Prompt format response (keep existing logic for consistent UI)
        data_response_template = """
        Bạn là trợ lý ảo tuyển sinh.
        Dựa vào dữ liệu sau đây để trả lời câu hỏi của học sinh.
        Nếu dữ liệu không có thông tin, hãy nói rõ.
        
        Dữ liệu tra cứu được:
        {data}
        
        Câu hỏi: {question}
        
        YÊU CẦU ĐẦU RA (QUAN TRỌNG):
        Hãy trả về kết quả dưới dạng JSON object hợp lệ (không có markdown fence ```json).
        Cấu trúc JSON:
        {{
            "answer": "Câu trả lời ngắn gọn, thân thiện bằng lời (text)",
            "courses": [
                {{
                    "id": "Mã lớp",
                    "name": "Tên lớp/khoá học",
                    "schedule": "Lịch học",
                    "location": "Phòng học - Chi nhánh",
                    "price": "Học phí (VNĐ)",
                    "status": "Trạng thái",
                    "endDate": "Ngày kết thúc"
                }}
            ]
        }}
        
        Nếu không có khóa học nào trong dữ liệu, "courses" là danh sách rỗng [].
        Chỉ trả về JSON.
        """
        self.data_response_prompt = PromptTemplate.from_template(data_response_template)

    async def _generate_data_response(self, question: str, data: dict) -> Tuple[str, List[dict]]:
        """Sinh câu trả lời từ dữ liệu API dưới dạng text và courses list."""
        chain = self.data_response_prompt | self.llm
        try:
            result = await chain.ainvoke({"data": str(data), "question": question})
            content = result.content.strip()
            
            if content.startswith("```json"):
                content = content.replace("```json", "", 1)
            if content.startswith("```"):
                content = content.replace("```", "", 1)
            if content.endswith("```"):
                content = content.replace("```", "", 1)
                
            parsed_json = json.loads(content.strip())
            return parsed_json.get("answer", ""), parsed_json.get("courses", [])
        except Exception as e:
            logging.error(f"Error generating data response: {e}")
            return "Có lỗi khi xử lý dữ liệu.", []

    async def process_message(self, question: str, session_id: str = None) -> Tuple[str, str, list, list]:
        """
        Xử lý tin nhắn sử dụng Agentic Workflow (Tool Calling).
        """
        if not session_id:
            session_id = session_manager.create_session()
        
        # 1. Prepare Context & System Prompt
        # Fetch valid options to inject into System Prompt
        valid_branches = await external_api_service.get_all_branches()
        valid_grades = await external_api_service.get_all_grades()
        
        system_prompt = f"""Bạn là Trợ lý Tuyển sinh của Trung tâm Thăng Long.
Nhiệm vụ: Tư vấn khóa học, học phí và giải đáp thắc mắc.

CÁC CÔNG CỤ (TOOLS):
1. `search_classes(branch, grade, subject)`: Dùng tìm kiếm lớp học.
   - BẮT BUỘC phải có `branch` (Chi nhánh) và `grade` (Khối lớp) trước khi gọi.
   - Nếu thiếu, HÃY HỎI người dùng. Đừng tự bịa.
   - Branch hợp lệ: {valid_branches}
   - Grade hợp lệ: {valid_grades}
   
2. `search_general_info(query)`: Dùng cho câu hỏi chung, quy định, địa chỉ chung chung, hoặc chào hỏi.

QUY TRÌNH:
- Nếu người dùng chào hỏi -> Gọi `search_general_info` hoặc tự trả lời.
- Nếu người dùng hỏi lớp học -> Kiểm tra đã có Branch/Grade chưa?
  - Nếu chưa -> Hỏi người dùng (Trả về text).
  - Nếu đủ -> Gọi `search_classes`.
- Luôn ưu tiên thông tin mới nhất trong câu hỏi.

Lịch sử chat:
(Hệ thống tự quản lý memory, nhưng bạn hãy chú ý context thay đổi)
"""
        # Get history from session (Simple list of messages)
        # TODO: Implement persistent message history in SessionManager for robust chat.
        # For now, we rely on the LLM processing the curent query with the system prompt context. 
        # Ideally we pass previous messages too.
        
        messages = [SystemMessage(content=system_prompt)]
        
        # Add simple history if available (optional enhancement)
        context = session_manager.get_context(session_id)
        # Restore basic context if needed, but let's try pure Agent approach first.
        
        messages.append(HumanMessage(content=question))

        # 2. Invoke LLM with Tools
        response = await self.llm_with_tools.ainvoke(messages)

        # 3. Handle Response
        # Case A: Tool Call
        if response.tool_calls:
            tool_call = response.tool_calls[0] # Handle first tool call
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            logging.info(f"Agent chose tool: {tool_name} with args: {tool_args}")
            
            if tool_name == "search_classes":
                # Execute tool
                data = await search_classes.ainvoke(tool_args)
                # Format response for App
                answer, courses = await self._generate_data_response(question, data)
                
                # Save context for next turn (Explicitly save successful params)
                session_manager.update_context(session_id, **tool_args)
                return answer, session_id, [], courses
                
            elif tool_name == "search_general_info":
                # Execute tool
                answer_text = search_general_info.invoke(tool_args)
                return answer_text, session_id, [], []

        # Case B: No Tool Call (Agent asks clarifying question or chitchats)
        answer_text = response.content
        return answer_text, session_id, [], []

chat_orchestrator = ChatOrchestrator()
