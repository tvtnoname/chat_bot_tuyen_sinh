from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from app.services.external.school_api import external_api_service
from app.services.rag.engine import rag_service

@tool
async def search_classes(branch: str, grade: str, subject: Optional[str] = None) -> Dict[str, Any]:
    """
    Tra cứu danh sách lớp học dựa trên Chi nhánh (Branch), Khối lớp (Grade) và Môn học (Subject).
    Sử dụng công cụ này khi người dùng hỏi về khóa học, lớp học, học phí, lịch học.
    
    Args:
        branch: Tên chi nhánh hoặc địa chỉ (Ví dụ: "Thăng Long Hà Nội", "766 Cách Mạng Tháng 8").
        grade: Khối lớp (Ví dụ: "10", "11", "12").
        subject: Môn học (Ví dụ: "Toán", "Lý"). Có thể bỏ qua nếu người dùng muốn xem tất cả môn.
        
    Returns:
        JSON object chứa danh sách lớp học và thông tin liên quan.
    """
    # Gọi API
    data = await external_api_service.get_filtered_data(branch=branch, grade=grade, subject=subject)
    return data

@tool
def ask_for_branch() -> str:
    """
    Sử dụng công cụ này khi bạn cần người dùng chọn Chi nhánh (Branch/Location).
    Hệ thống sẽ hiển thị danh sách các chi nhánh để người dùng chọn.
    Không cần tham số.
    """
    return "DISPLAY_BRANCH_OPTIONS"

@tool
def ask_for_grade() -> str:
    """
    Sử dụng công cụ này khi bạn cần người dùng chọn Khối lớp (Grade).
    Hệ thống sẽ hiển thị danh sách khối lớp.
    Không cần tham số.
    """
    return "DISPLAY_GRADE_OPTIONS"

@tool
def ask_for_subject() -> str:
    """
    Sử dụng công cụ này khi bạn cần người dùng chọn Môn học (Subject).
    Hệ thống sẽ hiển thị danh sách môn học.
    Không cần tham số.
    """
    return "DISPLAY_SUBJECT_OPTIONS"

@tool
def search_general_info(query: str) -> str:
    """
    Tra cứu thông tin chung về trung tâm, quy định, chính sách, hoặc chào hỏi xã giao.
    Sử dụng công cụ này cho các câu hỏi không liên quan đến tìm kiếm lớp học cụ thể (như "Trung tâm ở đâu?", "Giới thiệu", "Xin chào").
    
    Args:
        query: Câu hỏi của người dùng.
        
    Returns:
        Câu trả lời dưới dạng văn bản.
    """
    return rag_service.get_answer(query)
