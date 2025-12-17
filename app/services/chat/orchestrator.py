import logging
import json
import asyncio
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

    async def process_message(self, question: str, session_id: str = None, user_id: str = None) -> Tuple[str, str, list, list]:
        """
        Xử lý tin nhắn sử dụng Agentic Workflow (Tool Calling).
        """
        if not session_id:
            session_id = await session_manager.create_session(user_id=user_id)
        
        # 0. Pre-process state update (KEEP EXISTING)
        extracted_branch, extracted_grade, extracted_subject = await self._extract_entities(question)
        if extracted_branch or extracted_grade or extracted_subject:
            await session_manager.update_context(session_id, branch=extracted_branch, grade=extracted_grade, subject=extracted_subject)
        
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
        raw_history = await session_manager.get_history(session_id)
        for msg in raw_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        # Inject Current Context
        context = await session_manager.get_context(session_id)
        current_slots_info = f"SYSTEM_NOTE involved entities so far: Branch={context.get('branch')}, Grade={context.get('grade')}, Subject={context.get('subject')}"
        messages.append(SystemMessage(content=current_slots_info))

        # Add Current User Message
        messages.append(HumanMessage(content=question))
        # Save user msg to history
        await session_manager.add_message(session_id, "user", question)

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
                await session_manager.update_context(session_id, **tool_args)
                
                final_answer_text = answer
                await session_manager.add_message(session_id, "assistant", final_answer_text)
                return final_answer_text, session_id, [], courses
                
            elif tool_name == "ask_for_branch":
                options = await external_api_service.get_all_branches()
                final_answer_text = "Bạn vui lòng chọn chi nhánh để mình tư vấn chính xác nhé:"
                await session_manager.add_message(session_id, "assistant", final_answer_text)
                return final_answer_text, session_id, options, []
                
            elif tool_name == "ask_for_grade":
                options = await external_api_service.get_all_grades()
                final_answer_text = "Bạn vui lòng chọn khối lớp:"
                await session_manager.add_message(session_id, "assistant", final_answer_text)
                return final_answer_text, session_id, options, []
                
            elif tool_name == "ask_for_subject":
                options = await external_api_service.get_all_subjects()
                final_answer_text = "Bạn muốn tìm lớp môn gì ạ?"
                await session_manager.add_message(session_id, "assistant", final_answer_text)
                return final_answer_text, session_id, options, []
                
            elif tool_name == "search_general_info":
                answer_text = search_general_info.invoke(tool_args)
                final_answer_text = answer_text
                await session_manager.add_message(session_id, "assistant", final_answer_text)
                return final_answer_text, session_id, [], []

        # Case B: No Tool Call
        final_answer_text = response.content
        await session_manager.add_message(session_id, "assistant", final_answer_text)
        
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

    async def process_message_stream(self, question: str, session_id: str = None, user_id: str = None):
        """
        Streaming version of process_message.
        Yields chunks of text for the final answer.
        """
        if not session_id:
            session_id = await session_manager.create_session(user_id=user_id)
        
        # 1. State Update & Extract (Not streamed)
        extracted_branch, extracted_grade, extracted_subject = await self._extract_entities(question)
        if extracted_branch or extracted_grade or extracted_subject:
            await session_manager.update_context(session_id, branch=extracted_branch, grade=extracted_grade, subject=extracted_subject)
        
        # 2. Context & Messages
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
   
3. `ask_for_grade()`: Gọi công cụ này nếu thiếu thông tin Khối lớp (Grade).
   - BẮT BUỘC gọi tool này để hiển thị danh sách chọn.
   
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
        raw_history = await session_manager.get_history(session_id)
        for msg in raw_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        # Inject Current Context
        context = await session_manager.get_context(session_id)
        current_slots_info = f"SYSTEM_NOTE involved entities so far: Branch={context.get('branch')}, Grade={context.get('grade')}, Subject={context.get('subject')}"
        messages.append(SystemMessage(content=current_slots_info))

        # Add Current User Message
        messages.append(HumanMessage(content=question))
        await session_manager.add_message(session_id, "user", question)

        # 3. Invoke LLM (Check for tools first - Non-streaming step)
        response = await self.llm_with_tools.ainvoke(messages)
        
        final_answer_text = ""
        final_answer_chunk = ""

        # Case A: Tool Call
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            logging.info(f"Agent chose tool: {tool_name} with args: {tool_args}")
            
            if tool_name == "search_classes":
                data = await search_classes.ainvoke(tool_args)
                await session_manager.update_context(session_id, **tool_args)
                
                # Stream the data response generation
                # We need to construct the chain manually to stream it
                chain = self.data_response_prompt | self.llm
                async for chunk in chain.astream({"data": str(data), "question": question}):
                     text_chunk = chunk.content
                     final_answer_text += text_chunk
                     # Simple logic: avoid streaming raw JSON if possible, but strict JSON is prompted
                     # For streaming text, we might just yield the whole accumulation if it's JSON?
                     # Wait, prompt asks for JSON. Users can't read JSON stream easily.
                     # Compromise: Buffer the JSON, parse it, then yield the "answer" field.
                     # Streaming JSON is hard. Let's simplify: 
                     # Only stream if it's NOT a data tool that requires complex JSON UI.
                     # Actually, for data tools, user expects a UI card, not just text.
                     # So we should probably NOT stream the JSON generation, but just return standard response.
                     pass 
                
                # Fallback to standard wait for JSON tools to ensure valid UI structure
                answer, courses = await self._generate_data_response(question, data)
                final_answer_text = answer
                
                # Yield the text answer first
                yield f"data: {json.dumps({'text': answer, 'session_id': session_id})}\n\n"
                
                # Yield the complex data (courses)
                yield f"data: {json.dumps({'courses': courses})}\n\n"
                
                await session_manager.add_message(session_id, "assistant", final_answer_text)
                return

            elif tool_name in ["ask_for_branch", "ask_for_grade", "ask_for_subject"]:
                options = []
                if tool_name == "ask_for_branch":
                    options = await external_api_service.get_all_branches()
                    final_answer_text = "Bạn vui lòng chọn chi nhánh để mình tư vấn chính xác nhé:"
                elif tool_name == "ask_for_grade":
                    options = await external_api_service.get_all_grades()
                    final_answer_text = "Bạn vui lòng chọn khối lớp:"
                elif tool_name == "ask_for_subject":
                    options = await external_api_service.get_all_subjects()
                    final_answer_text = "Bạn muốn tìm lớp môn gì ạ?"
                
                yield f"data: {json.dumps({'text': final_answer_text, 'session_id': session_id, 'options': options})}\n\n"
                await session_manager.add_message(session_id, "assistant", final_answer_text)
                return
                
            elif tool_name == "search_general_info":
                # General info is usually text, we can stream this!
                # But search_general_info tool returns static text or RAG text.
                # If RAG is used inside the tool, it returns full text.
                # We can simulate streaming of the result.
                answer_text = search_general_info.invoke(tool_args)
                final_answer_text = answer_text
                
                # Simulate stream
                chunk_size = 10
                for i in range(0, len(answer_text), chunk_size):
                    chunk = answer_text[i:i+chunk_size]
                    yield f"data: {json.dumps({'text_chunk': chunk, 'session_id': session_id})}\n\n"
                    await asyncio.sleep(0.01)
                    
                await session_manager.add_message(session_id, "assistant", final_answer_text)
                return

        # Case B: No Tool Call -> Pure Conversational Stream (The Real Streaming)
        # We need to re-invoke the LLM in streaming mode because 'response' above was non-streaming (to check tool calls)
        # But we already got 'response'.
        # If response.content is populated, that's the answer.
        # To stream, we should have used stream=True initially?
        # Standard ReAct: Use streaming=True but capture chunks. If tool call chunk detected, stop yielding text?
        # Simpler: If we checked tools above and found none, we can just yield the content we already got (pseudo-stream)
        # OR re-generate. Re-generating adds cost/latency.
        # Since 'response' already has the full content:
        final_answer_text = response.content
        
        # Guardrails check first
        options = []
        lower = final_answer_text.lower()
        if any(kw in lower for kw in ["lớp mấy", "khối mấy", "khối lớp"]):
             options = await external_api_service.get_all_grades()
        elif any(kw in lower for kw in ["chi nhánh", "cơ sở", "địa chỉ"]):
             options = await external_api_service.get_all_branches()
        elif any(kw in lower for kw in ["môn gì", "môn nào"]):
             options = await external_api_service.get_all_subjects()

        # Stream the content we already have (Simulated streaming for consistency)
        # Or ideally, we should use llm.astream from start. But parsing tool_calls from a stream is complex.
        # For this phase, "Pseudo-streaming" the final text block is acceptable and safer.
        chunk_size = 5 # chars
        chunks = [final_answer_text[i:i+chunk_size] for i in range(0, len(final_answer_text), chunk_size)]
        
        for chunk in chunks:
             yield f"data: {json.dumps({'text_chunk': chunk, 'session_id': session_id})}\n\n"
             await asyncio.sleep(0.02) # Small delay for effect
             
        if options:
            yield f"data: {json.dumps({'options': options})}\n\n"

        await session_manager.add_message(session_id, "assistant", final_answer_text)

chat_orchestrator = ChatOrchestrator()
