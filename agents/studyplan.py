import os
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import httpx
from dotenv import load_dotenv
import asyncio
import traceback
import json
from typing import AsyncGenerator, Optional, List, Dict, Any


load_dotenv()

# GOOGLE_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=GOOGLE_API_KEY"
API_KEY = os.getenv("GOOGLE_API_KEY")

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(
        self,
        temperature: float = 0.7,
        model: str = "gemini-1.0-pro",
        streaming: bool = False,
        memory=None,
        tracing: bool = False
    ):
        self.temperature = temperature
        self.model = model
        self.streaming = streaming
        self.memory = memory
        self.tracing = tracing

    async def ask(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Gửi prompt đến Google Gemini API và nhận kết quả trả về dưới dạng stream các chunks.
        Nếu streaming=False, sẽ yield một chunk duy nhất chứa toàn bộ nội dung.
        """
        logger.info(f"[{self.__class__.__name__}] Prompt: {prompt[:300]}...") # Log một phần prompt
        async for chunk in self._call_gemini_api(prompt):
            yield chunk

    async def _call_gemini_api(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Gửi yêu cầu đến Gemini API và xử lý kết quả trả về.
        Hỗ trợ streaming nếu self.streaming là True, trả về AsyncGenerator các chunks.
        Nếu self.streaming là False, trả về AsyncGenerator chứa một chunk duy nhất (toàn bộ nội dung).
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        request_data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature
            }
        }
        
        if not API_KEY:
            logger.error("API_KEY for Google Gemini is not set.")
            yield "Lỗi: API key chưa được cấu hình."
            return
            
        api_url_base = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}"
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client: # Tăng timeout
                if self.streaming:
                    api_url = f"{api_url_base}:streamGenerateContent"
                    logger.info(f"[{self.__class__.__name__}] Gọi Gemini API (streaming): {api_url}")
                    async with client.stream("POST", api_url, json=request_data, params={"key": API_KEY}, headers=headers) as res:
                        res.raise_for_status()
                        async for line in res.aiter_lines():
                            if line.strip():
                                try:
                                    if line.startswith("data: "):
                                        line = line[len("data: "):]
                                    clean_line = line.strip()
                                    if clean_line.endswith(','):
                                        clean_line = clean_line[:-1]
                                    if not clean_line: continue
                                    
                                    chunk_data = json.loads(clean_line)
                                    if not (chunk_data.get("candidates") and
                                            chunk_data["candidates"] and
                                            isinstance(chunk_data["candidates"], list) and
                                            len(chunk_data["candidates"]) > 0 and
                                            chunk_data["candidates"][0].get("content") and
                                            chunk_data["candidates"][0]["content"].get("parts") and
                                            isinstance(chunk_data["candidates"][0]["content"]["parts"], list) and
                                            len(chunk_data["candidates"][0]["content"]["parts"]) > 0 and
                                            "text" in chunk_data["candidates"][0]["content"]["parts"][0]):
                                        logger.warning(f"Bỏ qua chunk với cấu trúc không mong đợi: {clean_line[:200]}")
                                        continue
                                    
                                    text_part = chunk_data["candidates"][0]["content"]["parts"][0]["text"]
                                    yield text_part
                                except json.JSONDecodeError as je:
                                    logger.error(f"Lỗi JSON Decode cho dòng: '{line}'. Lỗi: {je}")
                                except Exception as e_chunk:
                                    logger.error(f"Lỗi xử lý stream chunk: '{line}'. Lỗi: {e_chunk}")
                                    traceback.print_exc()
                else:
                    api_url = f"{api_url_base}:generateContent"
                    logger.info(f"[{self.__class__.__name__}] Gọi Gemini API (non-streaming): {api_url}")
                    res = await client.post(api_url, json=request_data, params={"key": API_KEY}, headers=headers)
                    res.raise_for_status()
                    data = res.json()
                    if not (data.get("candidates") and
                            data["candidates"] and
                            isinstance(data["candidates"], list) and
                            len(data["candidates"]) > 0 and
                            data["candidates"][0].get("content") and
                            data["candidates"][0]["content"].get("parts") and
                            isinstance(data["candidates"][0]["content"]["parts"], list) and
                            len(data["candidates"][0]["content"]["parts"]) > 0 and
                            "text" in data["candidates"][0]["content"]["parts"][0]):
                        logger.error(f"Cấu trúc response không mong đợi từ Gemini (non-streaming): {data}")
                        yield "Lỗi: Cấu trúc response không mong đợi từ Gemini."
                        return
                    
                    result_text = data["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info(f"[{self.__class__.__name__}] Gemini Response (non-streaming): {result_text[:200]}...")
                    yield result_text

        except httpx.RequestError as e:
            logger.error(f"Request Error: {e}")
            yield "Có lỗi xảy ra khi gọi API."
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Error: {e.response.status_code} - {e.request.url}")
            error_details = "Không có chi tiết lỗi."
            if e.response:
                try:
                    error_details = e.response.json()
                except json.JSONDecodeError:
                    error_details = e.response.text
            logger.error(f"Error response body: {error_details}")
            yield f"Có lỗi HTTP {e.response.status_code} xảy ra khi gọi API."
        except Exception as e:
            logger.error(f"General Error in _call_gemini_api: {e}")
            traceback.print_exc()
            yield "Đã có lỗi không xác định khi gọi API."

    @abstractmethod
    async def run(self, **kwargs) -> str:
        pass


class StudyPlanAgent(BaseAgent):
    def __init__(
        self, 
        temperature: float = 0.7,
        model: str = "gemini-1.0-pro",
        streaming: bool = False,
        memory=None,
        tracing: bool = False,
        # Thêm các tham số cho mục tiêu học tập
        goal: Optional[str] = None,
        time_frame: Optional[str] = None,
        current_level: Optional[str] = None,
        daily_study_hours: Optional[float] = None,
        focus_skills: Optional[List[str]] = None,
        preferred_activities: Optional[List[str]] = None
    ):
        super().__init__(temperature, model, streaming, memory, tracing)
        self.start_date = datetime.today()
        self.study_plan: Dict[str, Any] = {}  # Sẽ lưu trữ kế hoạch học tập chi tiết dưới dạng dictionary từ JSON
        self.progress: Dict[str, Any] = {}    # Lưu trữ tiến độ học tập của người dùng (có thể tích hợp sau)

        # Lưu trữ thông tin người dùng để tạo kế hoạch
        self.goal = goal if goal else "Cải thiện tiếng Anh toàn diện"
        self.time_frame = time_frame if time_frame else "1 tháng" # ví dụ: "1 tháng", "3 tuần"
        self.current_level = current_level if current_level else "intermediate" # ví dụ: "beginner", "intermediate", "advanced"
        self.daily_study_hours = daily_study_hours if daily_study_hours else 1.5 # Số giờ học trung bình mỗi ngày
        self.focus_skills = focus_skills if focus_skills else ["listening", "speaking", "vocabulary"] # Kỹ năng muốn tập trung
        self.preferred_activities = preferred_activities if preferred_activities else ["xem video", "luyện nói", "học từ mới qua ứng dụng"] # Hoạt động ưa thích

    def _generate_date_range(self, start_date: datetime, plan_duration_days: int) -> List[str]:
        return [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(plan_duration_days)]

    async def generate_study_plan(self) -> str: # Trả về string thông báo, kế hoạch lưu trong self.study_plan
        prompt = f"""Hãy tạo một lộ trình học tiếng Anh chi tiết dưới dạng JSON object.
Thông tin người học và yêu cầu:
- Mục tiêu: {self.goal}
- Khung thời gian: {self.time_frame}
- Trình độ hiện tại: {self.current_level}
- Thời gian học trung bình mỗi ngày: {self.daily_study_hours} giờ
- Kỹ năng muốn tập trung: {', '.join(self.focus_skills) if self.focus_skills else 'Toàn diện'}
- Các loại hoạt động ưa thích: {', '.join(self.preferred_activities) if self.preferred_activities else 'Đa dạng'}

JSON output cần có cấu trúc như sau:
{{
  "goal": "{self.goal}",
  "time_frame": "{self.time_frame}",
  "current_level": "{self.current_level}",
  "daily_study_hours_avg": {self.daily_study_hours},
  "overall_summary": "Một tóm tắt ngắn gọn về kế hoạch và lời khuyên chung cho toàn bộ lộ trình.",
  "plan": [ // Mảng chứa các đối tượng kế hoạch cho TỪNG NGÀY trong toàn bộ time_frame.
    // Ví dụ cho 2 ngày đầu tiên:
    {{
      "day_number": 1,
      "date": "{self.start_date.strftime('%Y-%m-%d')}", 
      "daily_focus": "Mục tiêu/chủ đề chính của ngày 1 (ví dụ: Ôn tập ngữ pháp cơ bản và từ vựng chủ đề gia đình)",
      "tasks": [
        {{
          "task_id": "day_1_task_1", 
          "name": "Học 10 từ vựng mới về chủ đề gia đình",
          "type": "vocabulary", 
          "duration_minutes": 30,
          "description": "Sử dụng flashcards, ứng dụng Quizlet hoặc ghi chép để học nghĩa, phát âm và ví dụ câu.",
          "resources": ["Từ điển Oxford Learner's Dictionaries", "Quizlet set: Family Vocabulary"],
          "completed": false 
        }},
        {{
          "task_id": "day_1_task_2", 
          "name": "Ôn tập thì Hiện tại đơn",
          "type": "grammar", 
          "duration_minutes": 45,
          "description": "Xem video bài giảng về thì Hiện tại đơn và làm bài tập áp dụng.",
          "resources": ["YouTube: English Grammar - Present Simple Tense", "Bài tập online: Present Simple exercises"],
          "completed": false 
        }}
      ],
      "estimated_total_duration_minutes": 75 // Tổng thời gian cho ngày 1
    }},
    {{
      "day_number": 2,
      "date": "{(self.start_date + timedelta(days=1)).strftime('%Y-%m-%d')}", // Ngày tiếp theo
      "daily_focus": "Luyện nghe cơ bản và thực hành phát âm các từ đã học",
      "tasks": [
        {{
          "task_id": "day_2_task_1", 
          "name": "Nghe đoạn hội thoại ngắn về gia đình",
          "type": "listening", 
          "duration_minutes": 25,
          "description": "Nghe và cố gắng hiểu nội dung chính, ghi lại các từ mới nghe được.",
          "resources": ["ESL Podcast: Daily English - Family Life"],
          "completed": false 
        }},
        {{
          "task_id": "day_2_task_2", 
          "name": "Luyện phát âm 10 từ vựng đã học",
          "type": "speaking", 
          "duration_minutes": 20,
          "description": "Sử dụng công cụ ghi âm để tự nghe lại và cải thiện phát âm.",
          "resources": ["Google Search: How to pronounce [word]"],
          "completed": false 
        }}
      ],
      "estimated_total_duration_minutes": 45 // Tổng thời gian cho ngày 2
    }}
    // ... và tiếp tục cho tất cả các ngày còn lại trong {self.time_frame}
  ]
}}
Lưu ý quan trọng:
- Tạo đủ số ngày cho toàn bộ khung thời gian ('{self.time_frame}'). Ví dụ '1 tháng' là 30 ngày, '2 tuần' là 14 ngày. PHẢI có đủ các ngày trong mảng 'plan'.
- 'date' cho mỗi ngày phải được tính toán chính xác, nối tiếp nhau, bắt đầu từ ngày hôm nay ({self.start_date.strftime('%Y-%m-%d')}).
- 'task_id' phải là duy nhất trong toàn bộ kế hoạch (ví dụ: day_X_task_Y).
- 'estimated_total_duration_minutes' cho mỗi ngày nên xấp xỉ 'daily_study_hours' đã cho (quy đổi ra phút).
- Cung cấp các gợi ý 'resources' hữu ích, cụ thể và đa dạng nếu có thể.
- Đảm bảo JSON trả về là một object JSON hợp lệ duy nhất, không có bất kỳ văn bản nào khác bao quanh.
"""
        
        logger.info(f"[{self.__class__.__name__}] Generating study plan with detailed prompt for JSON output.")
        full_response_parts = []
        async for chunk in self.ask(prompt): # BaseAgent.ask đã hỗ trợ streaming hoặc non-streaming
            full_response_parts.append(chunk)
        plan_json_str = "".join(full_response_parts)
        
        if plan_json_str.strip().startswith("```json"):
            plan_json_str = plan_json_str.strip()[7:]
        if plan_json_str.strip().endswith("```"):
            plan_json_str = plan_json_str.strip()[:-3]
        plan_json_str = plan_json_str.strip()

        try:
            logger.debug(f"Raw JSON string from LLM: {plan_json_str[:500]}...")
            parsed_plan = json.loads(plan_json_str)
            self.populate_study_plan(parsed_plan)
            logger.info(f"[{self.__class__.__name__}] Study plan generated and populated successfully.")
            return "Lộ trình học đã được tạo thành công và lưu trữ. Bạn có thể xem kế hoạch hôm nay hoặc toàn bộ kế hoạch."
        except json.JSONDecodeError as e:
            logger.error(f"[{self.__class__.__name__}] Failed to decode JSON from LLM: {e}. Response: {plan_json_str}")
            return f"Lỗi: Không thể phân tích lộ trình học từ AI. Chi tiết: {e}"
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Error processing study plan: {e}\n{traceback.format_exc()}")
            return f"Lỗi khi xử lý lộ trình học: {e}"

    def populate_study_plan(self, parsed_plan_data: Dict[str, Any]):
        """
        Điền dữ liệu kế hoạch học tập từ JSON đã được parse vào self.study_plan.
        parsed_plan_data là một dictionary đã được json.loads().
        """
        if not parsed_plan_data or "plan" not in parsed_plan_data or not isinstance(parsed_plan_data["plan"], list):
            logger.error("Dữ liệu kế hoạch không hợp lệ hoặc thiếu trường 'plan'.")
            self.study_plan = {}
            raise ValueError("Dữ liệu kế hoạch từ AI không hợp lệ (thiếu 'plan' hoặc 'plan' không phải list).")

        self.study_plan = {} 
        
        self.goal = parsed_plan_data.get("goal", self.goal)
        self.time_frame = parsed_plan_data.get("time_frame", self.time_frame)
        self.current_level = parsed_plan_data.get("current_level", self.current_level)

        for day_data in parsed_plan_data["plan"]:
            date_str = day_data.get("date")
            if not date_str:
                logger.warning(f"Bỏ qua ngày không có 'date': {day_data.get('day_number')}")
                continue
            
            self.study_plan[date_str] = {
                "day_number": day_data.get("day_number"),
                "daily_focus": day_data.get("daily_focus", "Không có mục tiêu cụ thể."),
                "tasks": day_data.get("tasks", []),
                "estimated_total_duration_minutes": day_data.get("estimated_total_duration_minutes", 0),
            }
            for task in self.study_plan[date_str]["tasks"]:
                if "completed" not in task:
                    task["completed"] = False
        
        logger.info(f"Đã điền {len(self.study_plan)} ngày vào kế hoạch học tập.")


    def calculate_days_in_plan(self) -> int: 
        time_frame_lower = self.time_frame.lower()
        if "tuần" in time_frame_lower:
            try:
                weeks = int(time_frame_lower.split()[0])
                return weeks * 7
            except: return 7 
        elif "tháng" in time_frame_lower:
            try:
                months = int(time_frame_lower.split()[0])
                return months * 30
            except: return 30
        elif "ngày" in time_frame_lower:
            try:
                days = int(time_frame_lower.split()[0])
                return days
            except: return 7 
        return 30 

    def generate_daily_study_plan_text(self) -> str: 
        """Tạo văn bản hiển thị kế hoạch học tập cho tất cả các ngày trong self.study_plan."""
        if not self.study_plan:
            return "Chưa có lộ trình học nào được tạo."

        plan_text_parts = []
        sorted_dates = sorted(self.study_plan.keys())

        for date_str in sorted_dates:
            day_info = self.study_plan[date_str]
            day_number = day_info.get("day_number", date_str) 
            daily_focus = day_info.get("daily_focus", "Không có mô tả.")
            
            day_plan_str = f"Ngày {day_number} ({date_str}) - Tập trung: {daily_focus}\n"
            
            tasks = day_info.get("tasks", [])
            if not tasks:
                day_plan_str += "  Không có nhiệm vụ nào cho ngày này.\n"
            else:
                for task_idx, task in enumerate(tasks, 1):
                    status = "✓" if task.get("completed", False) else "□"
                    task_name = task.get("name", "N/A")
                    duration = task.get("duration_minutes", 0)
                    description = task.get("description", "")
                    resources = ", ".join(task.get("resources", [])) if task.get("resources") else "Không có"

                    day_plan_str += (
                        f"  {task_idx}. {status} {task_name} ({duration} phút)\n"
                        f"      Loại: {task.get('type', 'N/A')}\n"
                        f"      Mô tả: {description}\n"
                        f"      Tài liệu: {resources}\n"
                    )
            plan_text_parts.append(day_plan_str)
        
        return "\n".join(plan_text_parts)

    def get_today_plan_text(self) -> str: 
        """Lấy và định dạng kế hoạch học tập cho ngày hôm nay."""
        today_str = datetime.today().strftime('%Y-%m-%d')
        
        if not self.study_plan or today_str not in self.study_plan:
            return "Không có kế hoạch học tập cho ngày hôm nay trong lộ trình hiện tại."
        
        day_info = self.study_plan[today_str]
        day_number = day_info.get("day_number", today_str)
        daily_focus = day_info.get("daily_focus", "Không có mô tả.")
        
        plan_str = f"Kế hoạch học tập ngày {day_number} ({today_str}) - Tập trung: {daily_focus}\n"
        
        tasks = day_info.get("tasks", [])
        if not tasks:
            plan_str += "  Không có nhiệm vụ nào cho ngày hôm nay.\n"
        else:
            for task_idx, task in enumerate(tasks, 1):
                status = "✓" if task.get("completed", False) else "□"
                task_name = task.get("name", "N/A")
                duration = task.get("duration_minutes", 0)
                plan_str += f"  {task_idx}. {status} {task_name} ({duration} phút) - ID: {task.get('task_id', 'N/A')}\n"
        return plan_str

    def generate_missing_today_plan(self) -> str: 
        """
        Tạo một kế hoạch mặc định cho ngày hôm nay nếu chưa có trong self.study_plan.
        """
        today_str = datetime.today().strftime('%Y-%m-%d')
        if today_str not in self.study_plan:
            logger.info(f"Không tìm thấy kế hoạch cho ngày {today_str}. Tạo kế hoạch mặc định.")
            default_task_types = self.focus_skills if self.focus_skills else ["listening", "vocabulary", "speaking"]
            tasks = []
            base_duration = int((self.daily_study_hours * 60) / len(default_task_types)) if self.daily_study_hours and len(default_task_types) > 0 else 30
            
            for i, skill in enumerate(default_task_types):
                tasks.append({
                    "task_id": f"default_today_task_{i+1}",
                    "name": f"Luyện tập kỹ năng {skill}",
                    "type": skill,
                    "duration_minutes": base_duration,
                    "description": f"Bài tập cơ bản cho kỹ năng {skill}.",
                    "resources": ["Tự tìm tài liệu phù hợp"],
                    "completed": False
                })

            self.study_plan[today_str] = {
                "day_number": "Hôm nay (Mặc định)",
                "date": today_str,
                "daily_focus": "Tập trung vào các kỹ năng chính.",
                "tasks": tasks,
                "estimated_total_duration_minutes": sum(t["duration_minutes"] for t in tasks)
            }
            return "Đã tạo kế hoạch học tập mặc định cho ngày hôm nay. " + self.get_today_plan_text()
        return self.get_today_plan_text()


    def update_progress(self, task_id_to_update: str, completed: bool = True) -> str:
        """
        Cập nhật trạng thái hoàn thành của một nhiệm vụ dựa trên task_id.
        """
        today_str = datetime.today().strftime('%Y-%m-%d') 
        
        if today_str not in self.study_plan or not self.study_plan[today_str].get("tasks"):
            return "Không có kế hoạch hoặc nhiệm vụ nào cho ngày hôm nay để cập nhật."
        
        task_found = False
        for task in self.study_plan[today_str]["tasks"]:
            if task.get("task_id") == task_id_to_update:
                task["completed"] = completed
                task_found = True
                task_name = task.get("name", task_id_to_update)
                status_str = "hoàn thành" if completed else "chưa hoàn thành"
                logger.info(f"Nhiệm vụ '{task_name}' (ID: {task_id_to_update}) trong ngày {today_str} đã được cập nhật thành {status_str}.")
                return f"Đã cập nhật nhiệm vụ '{task_name}' thành {status_str}."
        
        if not task_found:
            return f"Không tìm thấy nhiệm vụ với ID '{task_id_to_update}' trong kế hoạch ngày hôm nay."
        return "Có lỗi xảy ra khi cập nhật tiến độ." 

    async def run(self, user_input: Optional[str] = None, **kwargs) -> str:
        """
        Xử lý input của người dùng và điều phối các hành động của StudyPlanAgent.
        """
        if "goal" in kwargs: self.goal = kwargs["goal"]
        if "time_frame" in kwargs: self.time_frame = kwargs["time_frame"]
        if "current_level" in kwargs: self.current_level = kwargs["current_level"]
        if "daily_study_hours" in kwargs: self.daily_study_hours = float(kwargs["daily_study_hours"])
        if "focus_skills" in kwargs and isinstance(kwargs["focus_skills"], list): self.focus_skills = kwargs["focus_skills"]
        if "preferred_activities" in kwargs and isinstance(kwargs["preferred_activities"], list): self.preferred_activities = kwargs["preferred_activities"]

        if user_input:
            user_input_lower = user_input.lower()
            if "tạo lộ trình" in user_input_lower or "generate plan" in user_input_lower or not self.study_plan:
                logger.info("Yêu cầu tạo lộ trình học mới.")
                return await self.generate_study_plan()
            
            elif "hoàn thành" in user_input_lower or "mark complete" in user_input_lower:
                parts = user_input.split()
                task_id_candidate = None
                if len(parts) > 1: 
                     if "_" in parts[-1] and ("day" in parts[-1] or "task" in parts[-1]):
                        task_id_candidate = parts[-1]

                if task_id_candidate:
                    return self.update_progress(task_id_candidate, True)
                else: 
                    return "Vui lòng cung cấp ID của nhiệm vụ bạn muốn đánh dấu hoàn thành (ví dụ: 'hoàn thành day_1_task_2')."

            elif "xem kế hoạch hôm nay" in user_input_lower or "today's plan" in user_input_lower:
                return self.get_today_plan_text()
            
            elif "xem toàn bộ kế hoạch" in user_input_lower or "full plan" in user_input_lower:
                return self.generate_daily_study_plan_text()

            elif "generate hôm nay" in user_input_lower: 
                 return self.generate_missing_today_plan()
            else:
                return (
                    "Các lệnh hợp lệ: 'tạo lộ trình', 'xem kế hoạch hôm nay', "
                    "'xem toàn bộ kế hoạch', 'hoàn thành [task_id]', 'generate hôm nay'.\n"
                    "Bạn cũng có thể cung cấp chi tiết hơn khi tạo lộ trình."
                )
        else: 
            if self.study_plan:
                return self.get_today_plan_text()
            else: 
                return "Vui lòng yêu cầu 'tạo lộ trình' hoặc cung cấp thông tin chi tiết hơn."


# async def test_study_plan_agent():
#     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
#     custom_agent = StudyPlanAgent(
#         goal="Đạt IELTS Band 7.0",
#         time_frame="3 ngày", # Ngắn để test nhanh
#         current_level="upper-intermediate",
#         daily_study_hours=1,
#         focus_skills=["writing", "speaking"],
#         preferred_activities=["viết luận", "luyện nói với AI"],
#         model="gemini-1.5-flash-latest" # Dùng flash cho nhanh
#     )
#     print("--- Test với thông số tùy chỉnh ---")
#     response = await custom_agent.run(user_input="tạo lộ trình") 
#     print(f"Thông báo từ agent: {response}")
    
#     if custom_agent.study_plan:
#         print("\n--- Toàn bộ kế hoạch (chi tiết): ---")
#         print(custom_agent.generate_daily_study_plan_text())
        
#         print("\n--- Kế hoạch hôm nay: ---")
#         print(custom_agent.get_today_plan_text())
        
#         today_date_str = datetime.today().strftime('%Y-%m-%d')
#         if today_date_str in custom_agent.study_plan and custom_agent.study_plan[today_date_str]["tasks"]:
#             first_task_today = custom_agent.study_plan[today_date_str]["tasks"][0]
#             first_task_id = first_task_today.get("task_id")
#             if first_task_id:
#                 print(f"\n--- Test cập nhật tiến độ cho task ID: {first_task_id} ---")
#                 update_msg = custom_agent.update_progress(first_task_id, True)
#                 print(update_msg)
#                 print("\n--- Kế hoạch hôm nay (sau cập nhật): ---")
#                 print(custom_agent.get_today_plan_text())
#             else:
#                 print("\nKhông tìm thấy task_id để test cập nhật.")
#         else:
#             print("\nKhông có task nào cho hôm nay để test cập nhật.")
#     else:
#         print("Không thể test các chức năng khác vì lộ trình chưa được tạo.")

#     print("\n--- Test generate hôm nay khi chưa có plan ---")
#     no_plan_agent = StudyPlanAgent(daily_study_hours=0.5, focus_skills=["grammar"])
#     today_plan_default = no_plan_agent.generate_missing_today_plan()
#     print(today_plan_default)


# if __name__ == '__main__':
#     asyncio.run(test_study_plan_agent())
