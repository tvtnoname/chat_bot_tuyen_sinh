import logging
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

class IntentClassifier:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=settings.MODEL_NAME, google_api_key=settings.GOOGLE_API_KEY, temperature=0.0)
        
        template = """
        Bạn là một bộ phân loại ý định cho Chatbot tuyển sinh.
        Nhiệm vụ của bạn là xác định xem câu hỏi của người dùng thuộc loại nào trong 2 loại sau:
        
        1. "DATABASE_QUERY": Câu hỏi yêu cầu tra cứu dữ liệu cụ thể, động, thường xuyên thay đổi như:
           - Chi nhánh, địa chỉ cụ thể nào đó.
           - Khoá học có những gì.
           - Lịch nghỉ lễ, lịch học, ca học.
           - Danh sách giáo viên.
           - Thông tin về học kì (semester).
           Ví dụ: "Lịch học toán lớp 10 thế nào?", "Chi nhánh có những thầy cô nào?", "Mai có được nghỉ không?".

        2. "GENERAL_CHAT": Các câu hỏi chung chung, chào hỏi, hoặc kiến thức tĩnh có sẵn trong tài liệu tuyển sinh chung (quy chế, giới thiệu chung, học phí chung...).
           Ví dụ: "Xin chào", "Trung tâm thành lập năm nào?", "Học phí quy định chung ra sao?", "Em muốn đăng ký học".

        Câu hỏi: {question}
        
        Chỉ trả lời duy nhất một từ khoá: DATABASE_QUERY hoặc GENERAL_CHAT.
        """
        self.prompt = PromptTemplate.from_template(template)

    async def classify(self, question: str) -> str:
        try:
            chain = self.prompt | self.llm
            result = await chain.ainvoke({"question": question})
            content = result.content.strip().upper()
            
            # Fallback nếu LLM trả lời dài dòng
            if "DATABASE_QUERY" in content:
                return "DATABASE_QUERY"
            return "GENERAL_CHAT"
        except Exception as e:
            logging.error(f"Lỗi phân loại ý định: {e}")
            return "GENERAL_CHAT" # Mặc định an toàn

intent_classifier = IntentClassifier()
