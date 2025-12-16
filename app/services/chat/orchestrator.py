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
        """
        self.data_response_prompt = PromptTemplate.from_template(data_response_template)

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
        
        # 0. Pre-process state update (KEEP EXISTING)
        extracted_branch, extracted_grade, extracted_subject = await self._extract_entities(question)
        if extracted_branch or extracted_grade or extracted_subject:
            session_manager.update_context(session_id, branch=extracted_branch, grade=extracted_grade, subject=extracted_subject)
        
        # 1. Prepare Context & System Prompt
        valid_branches = await external_api_service.get_all_branches()
        valid_grades = await external_api_service.get_all_grades()
        
        system_prompt = f"""Bạn là Trợ lý Tuyển sinh của Trung tâm Thăng Long.
Nhiệm vụ: Tư vấn khóa học, học phí và giải đáp thắc mắc.

CÁC CÔNG CỤ (TOOLS):
1. `search_classes(branch, grade, subject)`: Dùng tìm kiếm lớp học.
   - BẮT BUỘC phải có `branch` (Chi nhánh).
   - `grade` (Khối lớp) là BẮT BUỘC để có kết quả chính xác nhất. HÃY HỎI nếu thiếu.
   
2. `ask_for_branch()`: Gọi công cụ này nếu thiếu thông tin Chi nhánh (Branch).
   - BẮT BUỘC gọi tool này để hiển thị danh sách chọn.
   - KHÔNG ĐƯỢC hỏi bằng lời (text) dạng "Bạn muốn tìm ở chi nhánh nào?". HÃY GỌI TOOL.
   
3. `ask_for_grade()`: Gọi công cụ này nếu thiếu thông tin Khối lớp (Grade).
   - BẮT BUỘC gọi tool này để hiển thị danh sách chọn.
   - KHÔNG ĐƯỢC hỏi bằng lời (text) dạng "Bạn học khối mấy?". HÃY GỌI TOOL.
   
4. `ask_for_subject()`: Gọi công cụ này nếu thiếu dữ liệu Môn học, hoặc người dùng muốn xem danh sách môn.
   - NÊN gọi nếu chưa biết môn.
   
5. `search_general_info(query)`: Dùng cho câu hỏi chung, quy định, địa chỉ chung chung, hoặc chào hỏi.

QUY TẮC QUAN TRỌNG:
- NẾU THIẾU THÔNG TIN (Branch/Grade): ĐỪNG giải thích hay xin lỗi. GỌI TOOL NGAY LẬP TỨC.
- Ưu tiên gọi Tool hơn là trả lời Text.
- Tuyệt đối không yêu cầu người dùng "nhập" hoặc "cho biết". Hãy dùng từ "chọn".

Lịch sử chat:
"""
        messages = [SystemMessage(content=system_prompt)]
        
        # Inject History
        raw_history = session_manager.get_history(session_id)
        for msg in raw_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        # Inject Current Context
        context = session_manager.get_context(session_id)
        current_slots_info = f"SYSTEM_NOTE involved entities so far: Branch={context.get('branch')}, Grade={context.get('grade')}, Subject={context.get('subject')}"
        messages.append(SystemMessage(content=current_slots_info))

        # Add Current User Message
        messages.append(HumanMessage(content=question))
        # Save user msg to history
        session_manager.add_message(session_id, "user", question)

        # 2. Invoke LLM with Tools
        response = await self.llm_with_tools.ainvoke(messages)

        # 3. Handle Response
        final_answer_text = ""
        
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            logging.info(f"Agent chose tool: {tool_name} with args: {tool_args}")
            
            if tool_name == "search_classes":
                data = await search_classes.ainvoke(tool_args)
                answer, courses = await self._generate_data_response(question, data)
                session_manager.update_context(session_id, **tool_args)
                
                final_answer_text = answer
                session_manager.add_message(session_id, "assistant", final_answer_text)
                return final_answer_text, session_id, [], courses
                
            elif tool_name == "ask_for_branch":
                options = await external_api_service.get_all_branches()
                final_answer_text = "Bạn vui lòng chọn chi nhánh để mình tư vấn chính xác nhé:"
                session_manager.add_message(session_id, "assistant", final_answer_text)
                return final_answer_text, session_id, options, []
                
            elif tool_name == "ask_for_grade":
                options = await external_api_service.get_all_grades()
                final_answer_text = "Bạn vui lòng chọn khối lớp:"
                session_manager.add_message(session_id, "assistant", final_answer_text)
                return final_answer_text, session_id, options, []
                
            elif tool_name == "ask_for_subject":
                options = await external_api_service.get_all_subjects()
                final_answer_text = "Bạn muốn tìm lớp môn gì ạ?"
                session_manager.add_message(session_id, "assistant", final_answer_text)
                return final_answer_text, session_id, options, []
                
            elif tool_name == "search_general_info":
                answer_text = search_general_info.invoke(tool_args)
                final_answer_text = answer_text
                session_manager.add_message(session_id, "assistant", final_answer_text)
                return final_answer_text, session_id, [], []

        # Case B: No Tool Call
        final_answer_text = response.content
        session_manager.add_message(session_id, "assistant", final_answer_text)
        
        # HEURISTIC GUARDRAILS (Phòng hờ Agent quên gọi tool)
        # Nếu câu trả lời chứa từ khóa hỏi thông tin, tự động đính kèm options tương ứng.
        lower_answer = final_answer_text.lower()
        
        # Check Grade keywords
        if any(kw in lower_answer for kw in ["lớp mấy", "khối mấy", "khối nào", "lớp nào", "khối lớp", "chọn khối", "nhập khối"]):
            logging.info("Guardrail: Detected text asking for Grade. Attaching options.")
            options = await external_api_service.get_all_grades()
            return final_answer_text, session_id, options, []
            
        # Check Branch keywords
        if any(kw in lower_answer for kw in ["chi nhánh", "cơ sở", "địa chỉ", "ở đâu", "địa điểm", "chọn chi nhánh", "nhập chi nhánh"]):
             logging.info("Guardrail: Detected text asking for Branch. Attaching options.")
             options = await external_api_service.get_all_branches()
             return final_answer_text, session_id, options, []

        # Check Subject keywords
        if any(kw in lower_answer for kw in ["môn gì", "môn nào", "môn học"]):
             logging.info("Guardrail: Detected text asking for Subject. Attaching options.")
             options = await external_api_service.get_all_subjects()
             return final_answer_text, session_id, options, []

        return final_answer_text, session_id, [], []

chat_orchestrator = ChatOrchestrator()
