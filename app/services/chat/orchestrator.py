import logging
import json
from typing import Tuple, List, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.services.chat.memory import session_manager
from app.services.chat.tools import search_classes, search_general_info, ask_for_branch, ask_for_grade, ask_for_subject
from app.services.external.school_api import external_api_service

class ChatOrchestrator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=settings.MODEL_NAME, google_api_key=settings.GOOGLE_API_KEY, temperature=0.3)
        
        # Bind tools to LLM
        self.tools = [search_classes, search_general_info, ask_for_branch, ask_for_grade, ask_for_subject]
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
        # Prompt để trích xuất thông tin (Entity Extraction)
        extraction_template = """
        Trích xuất thông tin "Chi nhánh" (Branch), "Khối" (Grade) và "Môn học" (Subject) từ câu nói của người dùng.
        Danh sách Chi nhánh hợp lệ: {valid_branches}
        Danh sách Khối hợp lệ: {valid_grades}
        Danh sách Môn hợp lệ: {valid_subjects}
        
        Nếu người dùng nhắc đến địa điểm hoặc địa chỉ (Hà Nội, Sài Gòn, Số 1 Đại Cồ Việt...), hãy map về Chi nhánh tương ứng.
        Nếu tìm thấy con số đại diện cho khối lớp (ví dụ: 10, 11, 12), hãy trích xuất nó vào cột Grade. Ưu tiên cập nhật Grade mới nếu người dùng nhắc đến nó.
        Nếu không tìm thấy, trả về "None".
        
        Câu nói: "{text}"
        
        Output format: Branch|Grade|Subject
        Ví dụ: 
        "Em học lớp 10 ở Hà Nội" -> Thăng Long Hà Nội|10|None
        "Mình ở số 1 đại cồ việt muốn học toán" -> Số 1 Đại Cồ Việt|None|Toán
        "Các môn học" -> None|None|None
        "Lớp Toán 10" -> None|10|Toán
        "Danh sách môn học lớp 9" -> None|9|None
        "Còn lớp 12 thì sao" -> None|12|None
        "học phí lớp 11" -> None|11|None
        """
        self.extraction_prompt = PromptTemplate.from_template(extraction_template)

    async def _extract_entities(self, text: str) -> Tuple[str, str, str]:
        """Trích xuất Branch, Grade và Subject từ text."""
        try:
            # Lấy danh sách hợp lệ từ API (cache)
            valid_branches = await external_api_service.get_all_branches()
            valid_grades = await external_api_service.get_all_grades()
            valid_subjects = await external_api_service.get_all_subjects()
            
            # Default fallback nếu API chưa có dữ liệu
            if not valid_branches:
                valid_branches = ["[Đang cập nhật]"]
            if not valid_grades:
                valid_grades = ["10", "11", "12"]

            chain = self.extraction_prompt | self.llm
            result = await chain.ainvoke({
                "valid_branches": str(valid_branches),
                "valid_grades": str(valid_grades),
                "valid_subjects": str(valid_subjects),
                "text": text
            })
            
            content = result.content.strip()
            if "|" in content:
                parts = content.split("|")
                if len(parts) >= 3:
                     branch = parts[0].strip() if parts[0].strip() != "None" else None
                     grade = parts[1].strip() if parts[1].strip() != "None" else None
                     subject = parts[2].strip() if parts[2].strip() != "None" else None
                     return branch, grade, subject
                elif len(parts) == 2:
                     # Fallback for old prompt format just in case
                     branch = parts[0].strip() if parts[0].strip() != "None" else None
                     grade = parts[1].strip() if parts[1].strip() != "None" else None
                     return branch, grade, None

            return None, None, None
        except Exception as e:
            logging.error(f"Lỗi extract entities: {e}")
            return None, None, None

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
        
        # 0. Pre-process: Extract entities to update state
        # Điều này cực kỳ quan trọng để Agent "thấy" được lựa chọn của user (Branch/Grade)
        extracted_branch, extracted_grade, extracted_subject = await self._extract_entities(question)
        if extracted_branch or extracted_grade or extracted_subject:
            session_manager.update_context(session_id, branch=extracted_branch, grade=extracted_grade, subject=extracted_subject)
        
        # 1. Prepare Context & System Prompt
        # Fetch valid options to inject into System Prompt
        valid_branches = await external_api_service.get_all_branches()
        valid_grades = await external_api_service.get_all_grades()
        
        system_prompt = f"""Bạn là Trợ lý Tuyển sinh của Trung tâm Thăng Long.
Nhiệm vụ: Tư vấn khóa học, học phí và giải đáp thắc mắc.

CÁC CÔNG CỤ (TOOLS):
1. `search_classes(branch, grade, subject)`: Dùng tìm kiếm lớp học.
   - ƯU TIÊN HÀNG ĐẦU.
   - BẮT BUỘC phải có `branch` (Chi nhánh) và `grade` (Khối lớp) trước khi gọi.
   
2. `ask_for_branch()`: Gọi công cụ này nếu thiếu thông tin Chi nhánh (Branch).
   - KHÔNG được hỏi bằng lời. BẮT BUỘC gọi tool này để hiển thị danh sách chọn.
   
3. `ask_for_grade()`: Gọi công cụ này nếu thiếu thông tin Khối lớp (Grade).
   - KHÔNG được hỏi bằng lời. BẮT BUỘC gọi tool này để hiển thị danh sách chọn.
   
4. `ask_for_subject()`: Gọi công cụ này nếu thiếu thông tin Môn học (Subject), hoặc người dùng muốn xem danh sách môn.
   - NÊN gọi nếu chưa biết môn, để kết quả tra cứu chính xác hơn.
   
5. `search_general_info(query)`: Dùng cho câu hỏi chung, quy định, địa chỉ chung chung, hoặc chào hỏi.

QUY TRÌNH:
- Nếu người dùng chào hỏi -> Gọi `search_general_info` hoặc tự trả lời.
- Nếu người dùng hỏi lớp học:
  - Kiểm tra `branch`? Chưa có -> Gọi `ask_for_branch()`.
  - Kiểm tra `grade`? Chưa có -> Gọi `ask_for_grade()`.
  - Kiểm tra `subject`? Chưa có -> NÊN Gọi `ask_for_subject()`.
  - Nếu đủ -> Gọi `search_classes`.
- Luôn ưu tiên thông tin mới nhất trong câu hỏi.

Lịch sử chat:
(Hệ thống tự quản lý memory)
"""
        messages = [SystemMessage(content=system_prompt)]
        context = session_manager.get_context(session_id)
        
        # Add simple history if available (optional enhancement)
        # Inject current known slots to help Agent decide
        current_slots_info = f"Current Knowledge: Branch={context.get('branch')}, Grade={context.get('grade')}, Subject={context.get('subject')}"
        messages.append(SystemMessage(content=current_slots_info))

        messages.append(HumanMessage(content=question))

        # 2. Invoke LLM with Tools
        response = await self.llm_with_tools.ainvoke(messages)

        # 3. Handle Response
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
                
                # Save context for next turn
                session_manager.update_context(session_id, **tool_args)
                return answer, session_id, [], courses
                
            elif tool_name == "ask_for_branch":
                options = await external_api_service.get_all_branches()
                return "Bạn vui lòng chọn chi nhánh để mình tư vấn chính xác nhé:", session_id, options, []
                
            elif tool_name == "ask_for_grade":
                options = await external_api_service.get_all_grades()
                return "Bạn đang quan tâm đến lớp mấy ạ?", session_id, options, []
                
            elif tool_name == "ask_for_subject":
                options = await external_api_service.get_all_subjects()
                return "Bạn muốn tìm lớp môn gì ạ?", session_id, options, []
                
            elif tool_name == "search_general_info":
                # Execute tool
                answer_text = search_general_info.invoke(tool_args)
                return answer_text, session_id, [], []

        # Case B: No Tool Call (Agent asks clarifying question or chitchats)
        answer_text = response.content
        return answer_text, session_id, [], []

chat_orchestrator = ChatOrchestrator()
