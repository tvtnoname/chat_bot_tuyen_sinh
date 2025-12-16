import logging
import urllib.request
import json
import asyncio
from typing import Dict, List, Optional, Any

from app.core.config import settings

class ExternalAPIService:
    def __init__(self):
        self.api_url = settings.SCHOOL_API_URL
        self.cached_data = None
    
    def _format_day(self, day_code: Any) -> str:
        try:
            val = int(day_code)
            # Rule: dayOfWeek = 1 -> Thứ 2, 2 -> Thứ 3... (val + 1)
            display_val = val + 1
            if display_val == 8:
                return "Chủ Nhật"
            return f"Thứ {display_val}"
        except (ValueError, TypeError):
            # Fallback for non-integer codes
            return f"Thứ {day_code}"

    def _sync_fetch(self):
        try:
            with urllib.request.urlopen(self.api_url) as url:
                return json.loads(url.read().decode())
        except Exception as e:
            raise e

    async def fetch_all_data(self) -> Dict[str, Any]:
        """Lấy toàn bộ dữ liệu từ API thật."""
        if self.cached_data:
            return self.cached_data

        try:
            logging.info(f"Đang lấy dữ liệu từ {self.api_url}...")
            # Run blocking call in a separate thread
            self.cached_data = await asyncio.to_thread(self._sync_fetch)
            logging.info("Lấy dữ liệu thành công.")
            return self.cached_data
        except Exception as e:
            logging.error(f"Lỗi khi gọi API tuyển sinh: {e}")
            return {}

    async def _ensure_data(self):
        if not self.cached_data:
            await self.fetch_all_data()

    async def check_valid_branch(self, branch_name: str) -> bool:
        """Kiểm tra tên chi nhánh có hợp lệ không."""
        await self._ensure_data()
        if not self.cached_data or "branches" not in self.cached_data:
            return False
        
        b_name_lower = branch_name.lower()
        for b in self.cached_data["branches"]:
            # Check name or address
            if (b_name_lower in b["name"].lower()) or \
               (b["name"].lower() in b_name_lower) or \
               (b_name_lower in b["address"].lower()) or \
               (b["address"].lower() in b_name_lower):
                return True
        return False

    async def check_valid_grade(self, grade_name: str) -> bool:
        """Kiểm tra khối học có hợp lệ không."""
        await self._ensure_data()
        if not self.cached_data or "grades" not in self.cached_data:
            return False

        # grade_name có thể là "10", "Lớp 10"...
        # API grades: "code": 10, "name": "Lớp 10"
        for g in self.cached_data["grades"]:
            val_code = str(g["code"])
            if val_code in grade_name or grade_name in val_code or grade_name in g["name"]:
                return True
        return False

    async def get_all_branches(self) -> List[str]:
        """Lấy danh sách tên tất cả chi nhánh."""
        await self._ensure_data()
        if not self.cached_data or "branches" not in self.cached_data:
            return []
        # Return addresses as requested by user
        return [b["address"] for b in self.cached_data["branches"]]

    async def get_all_grades(self) -> List[str]:
        """Lấy danh sách mã khối."""
        await self._ensure_data()
        if not self.cached_data or "grades" not in self.cached_data:
            return []
        # Trả về cả tên và code để prompt hiểu? Hoặc code. 
        # API trả "10", "11", "12" là OK.
        return [str(g["code"]) for g in self.cached_data["grades"]]

    async def get_all_subjects(self) -> List[str]:
        """Lấy danh sách các môn học có trong hệ thống."""
        await self._ensure_data()
        if not self.cached_data:
            return []
        
        subjects = set()
        # Collect subjects from classes
        for c in self.cached_data.get("classes", []):
            subj = c.get("subject")
            if subj:
                subj_name = subj.get("name")
                if subj_name:
                    subjects.add(subj_name)
        
        return list(subjects)

    async def get_filtered_data(self, branch: str, grade: str, subject: str = None) -> Dict[str, Any]:
        """Lấy dữ liệu đã lọc theo chi nhánh, khối và môn học (option)."""
        await self._ensure_data()
        if not self.cached_data:
            return {"message": "Không thể kết nối đến hệ thống."}

        logging.info(f"Đang lọc dữ liệu cho Chi nhánh: {branch}, Khối: {grade}")

        # 1. Resolve Branch ID
        branch_id = None
        branch_info = None
        for b in self.cached_data.get("branches", []):
            # Check name or address
            if (branch.lower() in b["name"].lower()) or \
               (b["name"].lower() in branch.lower()) or \
               (branch.lower() in b["address"].lower()) or \
               (b["address"].lower() in branch.lower()):
                branch_id = b["branchId"]
                
                # Update branch info name if user used address, 
                # but we still want to keep the canonical name in context if needed.
                branch_info = b
                break
        
        # 2. Resolve Grade ID
        grade_id = None
        grade_info = None
        for g in self.cached_data.get("grades", []):
            val_code = str(g["code"])
            if val_code in grade or grade in val_code or grade in g["name"]:
                grade_id = g["gradeId"]
                grade_info = g
                break

        if branch_id is None:
            return {"message": f"Không tìm thấy chi nhánh nào khớp với '{branch}'."}
        if grade_id is None:
            return {"message": f"Không tìm thấy khối nào khớp với '{grade}'."}

        # 3. Filter Classes
        filtered_classes = []
        relevant_class_ids = set()
        
        for c in self.cached_data.get("classes", []):
            # Cần check null safety
            if c.get("branchId") == branch_id and c.get("gradeId") == grade_id:
                # Filter by Subject if provided
                if subject:
                    c_subject = (c.get("subject") or {}).get("name", "")
                    # Loose matching for subject
                    if subject.lower() not in c_subject.lower() and c_subject.lower() not in subject.lower():
                        continue

                # Format schedule info nicely
                schedules = []
                for s in c.get("classSchedules", []):
                    slot = s.get("lessonSlot") or {}
                    room = s.get("room") or {}
                    schedules.append(f"{self._format_day(s.get('dayOfWeek'))} - {slot.get('name')} ({slot.get('startTime')}-{slot.get('endTime')}) tại {room.get('name')}")

                class_info = {
                    "id": c["classId"],
                    "name": c["name"],
                    "subject": (c.get("subject") or {}).get("name"),
                    "fee": c["fee"],
                    "schedules": schedules,
                    "startDate": c["startDate"],
                    "endDate": c["endDate"],
                    "status": c["status"]
                }
                filtered_classes.append(class_info)
                relevant_class_ids.add(c["classId"])

        # 4. Find Teachers for these classes
        relevant_teachers = []
        for t in self.cached_data.get("teachers", []):
            teach_assignments = t.get("teachingAssignments", [])
            matches = [assign for assign in teach_assignments if assign.get("classId") in relevant_class_ids]
            if matches:
                 relevant_teachers.append({
                     "name": t.get("user", {}).get("fullName"),
                     "qualification": t.get("qualification"),
                     "experience": t.get("experienceYears"),
                     "subjects": [(ts.get("subject") or {}).get("name") for ts in t.get("teacherSubjects", [])]
                 })

        # 5. Build Result
        result = {
            "query_context": {
                "branch": branch_info["name"],
                "address": branch_info["address"],
                "grade": grade_info["name"]
            },
            "classes_found": filtered_classes,
            "teachers": relevant_teachers,
            "holidays": [h["name"] + f" ({h['description']})" for h in self.cached_data.get("holidays", [])], # Global holidays
            "semesters": [s["name"] for s in self.cached_data.get("semesters", [])]
        }
        
        if not filtered_classes:
             result["message"] = "Hiện tại chưa có lớp học nào mở cho Khối và Chi nhánh này."

        return result

external_api_service = ExternalAPIService()
