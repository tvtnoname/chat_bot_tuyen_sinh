import logging
from typing import Tuple
from app.services.rag.engine import rag_service
from app.services.external.school_api import external_api_service
from app.services.chat.memory import session_manager
from app.services.chat.intent import intent_classifier
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

class ChatOrchestrator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=settings.MODEL_NAME, google_api_key=settings.GOOGLE_API_KEY, temperature=0.3)
        
        # Prompt để tổng hợp câu trả lời từ dữ liệu
        data_response_template = """
        Bạn là trợ lý ảo tuyển sinh.
        Dựa vào dữ liệu sau đây để trả lời câu hỏi của học sinh.
        Nếu dữ liệu không có thông tin, hãy nói rõ.
        
        Dữ liệu tra cứu được:
        {data}
        
        Câu hỏi: {question}
        
        Trả lời (ngắn gọn, đầy đủ, thân thiện):
        """
        self.data_response_prompt = PromptTemplate.from_template(data_response_template)

        # Prompt để trích xuất thông tin (Entity Extraction)
        extraction_template = """
        Trích xuất thông tin "Chi nhánh" (Branch) và "Khối" (Grade) từ câu nói của người dùng.
        Danh sách Chi nhánh hợp lệ: {valid_branches}
        Danh sách Khối hợp lệ: {valid_grades}
        
        Nếu người dùng nhắc đến địa điểm (Hà Nội, Sài Gòn...), hãy map về Chi nhánh tương ứng.
        Nếu không tìm thấy, trả về "None".
        
        Câu nói: "{text}"
        
        Output format: Branch|Grade
        Ví dụ: 
        "Em học lớp 10 ở Hà Nội" -> Thăng Long Hà Nội|10
        "Mình ở đà nẵng" -> Thăng Long Đà Nẵng|None
        "Không có gì" -> None|None
        """
        self.extraction_prompt = PromptTemplate.from_template(extraction_template)

    async def _extract_entities(self, text: str) -> Tuple[str, str]:
        """Trích xuất Branch và Grade từ text."""
        try:
            # Lấy danh sách hợp lệ từ API (cache)
            valid_branches = await external_api_service.get_all_branches()
            valid_grades = await external_api_service.get_all_grades()
            
            # Default fallback nếu API chưa có dữ liệu
            if not valid_branches:
                valid_branches = ["[Đang cập nhật]"]
            if not valid_grades:
                valid_grades = ["10", "11", "12"]

            # Format prompt với dữ liệu động
            formatted_prompt = self.extraction_prompt.format(
                valid_branches=str(valid_branches),
                valid_grades=str(valid_grades),
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
                "text": text
            })
            
            content = result.content.strip()
            if "|" in content:
                branch, grade = content.split("|")
                branch = branch.strip() if branch.strip() != "None" else None
                grade = grade.strip() if grade.strip() != "None" else None
                return branch, grade
            return None, None
        except Exception as e:
            logging.error(f"Lỗi extract entities: {e}")
            return None, None

    async def _generate_data_response(self, question: str, data: dict) -> str:
        """Sinh câu trả lời từ dữ liệu API."""
        chain = self.data_response_prompt | self.llm
        result = await chain.ainvoke({"data": str(data), "question": question})
        return result.content

    async def process_message(self, question: str, session_id: str = None) -> Tuple[str, str, list]:
        """
        Xử lý tin nhắn chính.
        Trả về (câu trả lời, session_id, options).
        """
        # 1. Init Session
        if not session_id:
            session_id = session_manager.create_session()
        
        # 2. Extract Entities & Update Context
        # Luôn cố gắng trích xuất thông tin dù ý định là gì
        extracted_branch, extracted_grade = await self._extract_entities(question)
        if extracted_branch or extracted_grade:
            session_manager.update_context(session_id, branch=extracted_branch, grade=extracted_grade)

        # Lấy context hiện tại
        context = session_manager.get_context(session_id)
        current_branch = context.get("branch")
        current_grade = context.get("grade")
        pending_query = context.get("pending_query")

        # 3. Check Pending Query
        # Nếu đang có câu hỏi treo và hiện tại đã đủ thông tin -> Trả lời câu hỏi treo
        if pending_query and current_branch and current_grade:
            data = await external_api_service.get_filtered_data(branch=current_branch, grade=current_grade)
            # Trả lời câu hỏi treo
            # Trả lời câu hỏi treo
            answer = await self._generate_data_response(pending_query, data)
            # Clear pending query
            session_manager.update_context(session_id, pending_query="") 
            return answer, session_id, []

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
                return answer, session_id, []
        
        if intent == "DATABASE_QUERY":
            # 5. Xử lý tra cứu DB
            
            # Kiểm tra thiếu info
            missing_info = []
            if not current_branch:
                missing_info.append("Chi nhánh")
            if not current_grade:
                missing_info.append("Khối")
            
            if missing_info:
                # Lưu câu hỏi hiện tại làm pending query (nếu chưa có hoặc ghi đè)
                # Nếu câu hỏi hiện tại chỉ là câu cung cấp thông tin (ngắn), 
                # thì nên giữ pending_query CŨ.
                # Logic: Nếu question dài > 10 chars hoặc có vẻ là câu hỏi mới thì replace.
                # Đơn giản: Nếu question KHÔNG chứa thông tin extract được (tức là câu hỏi mới) -> Replace.
                # Nếu question CHỨA thông tin extract được -> Giữ pending query cũ.
                
                should_update_pending = True
                if pending_query and (extracted_branch or extracted_grade):
                    should_update_pending = False
                
                if should_update_pending:
                    session_manager.update_context(session_id, pending_query=question)
                
                info_req = ", ".join(missing_info)
                
                # Fetch options for the user to choose
                options = []
                if "Chi nhánh" in info_req:
                    options.extend(await external_api_service.get_all_branches())
                if "Khối" in info_req:
                    # Nếu thiếu cả hai, có thể list cả hai hoặc ưu tiên list branch trước?
                    # Để đơn giản, nếu thiếu cả hai, ta chi list Branch trước để user chọn Branch, 
                    # sau đó user chọn Grade sau. Hoặc merge list.
                    # Nhu cầu thực tế: User chọn Branch -> Chọn Grade.
                    # Nếu thiếu Branch, ưu tiên show Branch options.
                    if not options: # Chỉ show grade nếu chưa có options (tức là đã có branch nhưng thiếu grade)
                         options.extend(await external_api_service.get_all_grades())
                
                return f"Để tư vấn chính xác, thầy/cô cần biết em đang quan tâm đến {info_req} nào? Em vui lòng chọn hoặc cung cấp thêm nhé.", session_id, options
            
            # Đủ info
            # Nếu có pending query và user hỏi câu mới -> Ưu tiên câu mới
            # Hoặc thực hiện cả hai? Phức tạp.
            # Ưu tiên câu hỏi hiện tại.
            
            data = await external_api_service.get_filtered_data(branch=current_branch, grade=current_grade)
            answer = await self._generate_data_response(question, data)
            # Clear pending if any
            if pending_query:
                session_manager.update_context(session_id, pending_query="")
            return answer, session_id, []

        return "Xin lỗi, hệ thống đang gặp sự cố không xác định được yêu cầu.", session_id, []

chat_orchestrator = ChatOrchestrator()
