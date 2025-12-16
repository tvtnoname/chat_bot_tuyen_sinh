import logging
import json
import re
from typing import Tuple, List, Any
from app.services.rag.engine import rag_service
from app.services.external.school_api import external_api_service
from app.services.chat.memory import session_manager
from app.services.chat.intent import intent_classifier
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

class ChatOrchestrator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=settings.MODEL_NAME, google_api_key=settings.GOOGLE_API_KEY, temperature=0.3)
        
        # Prompt để tổng hợp câu trả lời từ dữ liệu
        # Prompt để tổng hợp câu trả lời từ dữ liệu
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
                    "id": "Mã lớp hoặc tạo ngẫu nhiên nếu không có",
                    "name": "Tên lớp/khoá học",
                    "schedule": "Lịch học (Thứ, Ca, Giờ)",
                    "location": "Phòng học - Chi nhánh",
                    "price": "Học phí (VNĐ)",
                    "status": "Trạng thái (Đang diễn ra/Sắp khai giảng)",
                    "endDate": "Ngày kết thúc"
                }}
            ]
        }}
        
        Nếu không có khóa học nào trong dữ liệu, "courses" là danh sách rỗng [].
        Chỉ trả về JSON, không thêm bất kỳ lời dẫn nào khác.
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

            # Format prompt với dữ liệu động
            formatted_prompt = self.extraction_prompt.format(
                valid_branches=str(valid_branches),
                valid_grades=str(valid_grades),
                valid_subjects=str(valid_subjects),
                text=text
            )
            
            # Vì self.extraction_prompt đã bind sẵn variables trong template?
            # Template hiện tại hardcode. Cần sửa template để có placeholder {valid_branches} và {valid_grades}.
            # Tuy nhiên LangChain PromptTemplate `|` LLM thì `chain.invoke` sẽ điền variables vào.
            # Nên ta cần sửa template ở __init__ trước.
            # Nhưng để đơn giản, ta format string trước khi đưa vào invoke hoặc tạo chain mới mỗi lần (hơi chậm).
            # Tốt nhất là update template trong `__init__` để nhận tham số.
            
            # Logic tạm thời: Tạo prompt string trực tiếp hoặc sửa template ở dưới.
            # Sẽ sửa template trong __init__ ở bước tiếp theo.
            # Ở đây ta giả định template đã có {valid_branches} và {valid_grades}.
             
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
        result = await chain.ainvoke({"data": str(data), "question": question})
        content = result.content.strip()
        
        # Clean up markdown code blocks if present
        if content.startswith("```json"):
            content = content.replace("```json", "", 1)
        if content.startswith("```"):
            content = content.replace("```", "", 1)
        if content.endswith("```"):
            content = content.replace("```", "", 1)
            
        try:
            parsed_json = json.loads(content.strip())
            answer = parsed_json.get("answer", "Xin lỗi, tôi không thể xử lý dữ liệu.")
            courses = parsed_json.get("courses", [])
            return answer, courses
        except json.JSONDecodeError:
            logging.error(f"Lỗi parse JSON từ LLM: {content}")
            return content, [] # Fallback: return raw content as answer

    async def process_message(self, question: str, session_id: str = None) -> Tuple[str, str, list, list]:
        """
        Xử lý tin nhắn chính.
        Trả về (câu trả lời, session_id, options, courses).
        """
        # 1. Init Session
        if not session_id:
            session_id = session_manager.create_session()
        
        # 2. Extract Entities & Update Context
        # Luôn cố gắng trích xuất thông tin dù ý định là gì
        extracted_branch, extracted_grade, extracted_subject = await self._extract_entities(question)
        if extracted_branch or extracted_grade or extracted_subject:
            session_manager.update_context(session_id, branch=extracted_branch, grade=extracted_grade, subject=extracted_subject)

        # Lấy context hiện tại
        context = session_manager.get_context(session_id)
        current_branch = context.get("branch")
        current_grade = context.get("grade")
        current_subject = context.get("subject")
        pending_query = context.get("pending_query")

        # 3. Check Pending Query
        # Nếu đang có câu hỏi treo và hiện tại đã đủ thông tin -> Trả lời câu hỏi treo
        if pending_query and current_branch and current_grade and current_subject:
            data = await external_api_service.get_filtered_data(branch=current_branch, grade=current_grade, subject=current_subject)
            # Trả lời câu hỏi treo
            # Trả lời câu hỏi treo
            # Trả lời câu hỏi treo
            answer, courses = await self._generate_data_response(pending_query, data)
            # Clear pending query
            session_manager.update_context(session_id, pending_query="") 
            return answer, session_id, [], courses

        # 4. Phân loại Intent cho câu hỏi hiện tại
        intent = await intent_classifier.classify(question)
        logging.info(f"Đã phân loại ý định: {intent}")

        if intent == "GENERAL_CHAT":
            # Nếu người dùng chỉ cung cấp thông tin (Vd: "Hà Nội") mà ý định lại là General Chat
            # có thể là do bộ phân loại không nhận ra đây là câu trả lời.
            # Tuy nhiên, nếu ở bước 3 chưa bắt được (do thiếu info), và ở đây là General Chat,
            # thì có thể người dùng đang nói chuyện phím.
            # Nhưng nếu pending_query vẫn còn và người dùng trả lời 1 thông tin nào đó (đã extract ở bước 2)
            # nhưng vẫn CHƯA ĐỦ info, thì ta nên nhắc lại về thông tin còn thiếu thay vì RAG.
            
            if pending_query and (extracted_branch or extracted_grade):
                # Người dùng có respond thông tin, nhưng vẫn chưa đủ
                pass # Chuyển xuống logic check missing info bên dưới (nhưng logic đó nằm trong DB_QUERY block)
                # Ta cần handle riêng hoặc coi đây là DB_QUERY tiếp diễn?
                
                # Simple hack: Nếu có extract được branch/grade, coi như user đang quan tâm DB.
                intent = "DATABASE_QUERY" # Force intent change
            else:
                answer = rag_service.get_answer(question)
                return answer, session_id, [], []
        
        if intent == "DATABASE_QUERY":
            # 5. Xử lý tra cứu DB
            
            # Kiểm tra thiếu info - SEQUENTIAL FLOW
            
            # Save pending query logic (restored)
            should_update_pending = True
            should_update_pending = True
            if pending_query and (extracted_branch or extracted_grade or extracted_subject):
                should_update_pending = False
            
            if should_update_pending:
                session_manager.update_context(session_id, pending_query=question)

            # 1. Check Branch (Address) First
            if not current_branch:
                # Fetch options (Addresses)
                options = await external_api_service.get_all_branches()
                return "Để tư vấn chính xác, bạn muốn được tư vấn tại địa chỉ nào?", session_id, options, []

            # 2. Check Grade Second (Only if Branch is present)
            if not current_grade:
                 # Fetch options (Grades)
                 options = await external_api_service.get_all_grades()
                 return "Bạn đang quan tâm đến khối lớp nào?", session_id, options, []

            # 3. Check Subject Third (Only if Branch & Grade are present)
            # This ensures we don't show cards immediately for "Các môn học" query or implicit logic
            if not current_subject:
                 subjects = await external_api_service.get_all_subjects()
                 return "Chúng tôi có các môn học sau, bạn quan tâm môn nào?", session_id, subjects, []
            
            # Đủ info
            # Nếu có pending query và user hỏi câu mới -> Ưu tiên câu mới
            # Hoặc thực hiện cả hai? Phức tạp.
            # Ưu tiên câu hỏi hiện tại.
            
            # Đủ info
            # Nếu có pending query và user hỏi câu mới -> Ưu tiên câu mới
            # Hoặc thực hiện cả hai? Phức tạp.
            # Ưu tiên câu hỏi hiện tại.
            
            data = await external_api_service.get_filtered_data(branch=current_branch, grade=current_grade, subject=current_subject)
            answer, courses = await self._generate_data_response(question, data)
            # Clear pending if any
            if pending_query:
                session_manager.update_context(session_id, pending_query="")
            return answer, session_id, [], courses
        
        return "Xin lỗi, hệ thống đang gặp sự cố không xác định được yêu cầu.", session_id, [], []

chat_orchestrator = ChatOrchestrator()
