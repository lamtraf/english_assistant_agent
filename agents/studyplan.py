import os
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import httpx
from dotenv import load_dotenv
import asyncio


load_dotenv()

GOOGLE_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=GOOGLE_API_KEY"
API_KEY = os.getenv("GOOGLE_API_KEY")

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(
        self,
        temperature: float = 0.7,
        model: str = "gemini-2.0-flash",  # Model Gemini mặc định
        streaming: bool = False,
        memory=None,
        tracing: bool = False
    ):
        self.temperature = temperature
        self.model = model
        self.streaming = streaming
        self.memory = memory
        self.tracing = tracing

    async def ask(self, prompt: str) -> str:
        """
        Gửi prompt đến Google Gemini API và nhận kết quả trả về.
        """
        logger.info(f"[{self.__class__.__name__}] Prompt: {prompt}")
        response = await self._call_gemini_api(prompt)
        return response

    async def _call_gemini_api(self, prompt: str) -> str:
        """
        Gửi yêu cầu đến Gemini API và xử lý kết quả trả về.
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        # Định dạng prompt
        request_data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        try:
            # Gửi yêu cầu tới Gemini API
            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.post(
                    GOOGLE_GEMINI_URL,
                    json=request_data,
                    params={"key": API_KEY},
                    headers=headers
                )

                res.raise_for_status() 
                data = res.json()
                
                result_text = data["candidates"][0]["content"]["parts"][0]["text"]
                logger.info(f"[{self.__class__.__name__}] Gemini Response: {result_text}")
                return result_text 

        except httpx.RequestError as e:
            logger.error(f"Request Error: {e}")
            return "Có lỗi xảy ra khi gọi API."

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Error: {e}")
            return "Có lỗi HTTP xảy ra khi gọi API."

        except Exception as e:
            logger.error(f"General Error: {e}")
            traceback.print_exc()
            return "Đã có lỗi không xác định khi gọi API."

    @abstractmethod
    async def run(self, **kwargs) -> str:
        pass


class StudyPlanAgent(BaseAgent):
    def __init__(
        self, 
        temperature: float = 0.7,
        model: str = "gemini-2.0-flash",
        streaming: bool = False,
        memory=None,
        tracing: bool = False
    ):
        super().__init__(temperature, model, streaming, memory, tracing)
        self.start_date = datetime.today()
        self.study_plan = {}  # Lưu trữ lộ trình chi tiết theo ngày
        self.progress = {}    # Lưu trữ tiến độ học tập của người dùng

    def generate_daily_study_plan(self) -> str:
        days_in_plan = self.calculate_days_in_plan()
        plan_text = []
        
        for day in range(days_in_plan):
            date = self.start_date + timedelta(days=day)
            date_str = date.strftime('%Y-%m-%d')
            
            if date_str not in self.study_plan:
                self.study_plan[date_str] = {
                    "tasks": [
                        {"name": "Luyện tập viết bài luận IELTS Task 2", "duration": 60, "completed": False},
                        {"name": "Học từ vựng chủ đề môi trường", "duration": 30, "completed": False},
                        {"name": "Luyện nghe đề thi IELTS", "duration": 45, "completed": False}
                    ],
                    "total_duration": 135,
                    "completed_duration": 0
                }
            
            day_plan = f"Ngày {date_str}:"
            for task in self.study_plan[date_str]["tasks"]:
                status = "✓" if task["completed"] else "□"
                day_plan += f"\n  {status} {task['name']} ({task['duration']} phút)"
            
            plan_text.append(day_plan)
        
        return "\n\n".join(plan_text)

    def calculate_days_in_plan(self) -> int:
        if "tuần" in self.time_frame.lower():
            weeks = int(self.time_frame.split()[0])
            return weeks * 7
        elif "tháng" in self.time_frame.lower():
            months = int(self.time_frame.split()[0])
            return months * 30
        return 30

    async def generate_study_plan(self) -> str:
        prompt = (
            f"Tạo một lộ trình học tiếng Anh chi tiết cho mục tiêu {self.goal} trong thời gian {self.time_frame}. "
            f"Trình độ hiện tại của người học là {self.current_level}. "
            f"Lộ trình học cần bao gồm:\n"
            f"1. Các kỹ năng cần cải thiện\n"
            f"2. Các bài tập cụ thể cho từng ngày\n"
            f"3. Phân bổ thời gian học hợp lý (phút cho mỗi hoạt động)\n"
            f"4. Tài liệu học tập khuyến nghị\n"
            f"Định dạng đầu ra là danh sách các hoạt động theo ngày với thời lượng cụ thể."
        )
        
        plan_json_str = await self.ask(prompt)
        
        try:
            self.populate_study_plan(plan_json_str)
        except Exception as e:
            print(f"Lỗi khi xử lý lộ trình học: {e}")
        
        return plan_json_str

    def populate_study_plan(self, plan_text: str):
        days_in_plan = self.calculate_days_in_plan()
        
        for day in range(days_in_plan):
            date = self.start_date + timedelta(days=day)
            date_str = date.strftime('%Y-%m-%d')
            
            if day % 7 == 0:  # Ngày đầu tuần
                self.study_plan[date_str] = {
                    "tasks": [{"name": "Học từ vựng mới", "duration": 45, "completed": False}]
                }
            else:
                self.study_plan[date_str] = {
                    "tasks": [{"name": "Luyện nghe", "duration": 60, "completed": False}]
                }

    def get_today_plan(self) -> str:
        today_str = datetime.today().strftime('%Y-%m-%d')
        
        if today_str not in self.study_plan:
            return "Không có kế hoạch học tập cho ngày hôm nay."
        
        plan = f"Kế hoạch học tập ngày {today_str}:\n"
        for idx, task in enumerate(self.study_plan[today_str]["tasks"], 1):
            status = "✓" if task["completed"] else "□"
            plan += f"{idx}. {status} {task['name']} ({task['duration']} phút)\n"
        
        return plan

    def generate_today_plan(self) -> str:
        today_str = datetime.today().strftime('%Y-%m-%d')

        if today_str not in self.study_plan:
            # Nếu không có kế hoạch cho hôm nay, tạo một kế hoạch cơ bản cho ngày hôm nay
            self.study_plan[today_str] = {
                "tasks": [
                    {"name": "Luyện nghe IELTS", "duration": 45, "completed": False},
                    {"name": "Luyện viết IELTS Task 2", "duration": 60, "completed": False},
                    {"name": "Học từ vựng tiếng Anh", "duration": 30, "completed": False}
                ],
                "total_duration": 135,
                "completed_duration": 0
            }
        
        plan = f"Kế hoạch học tập cho ngày hôm nay ({today_str}):\n"
        for idx, task in enumerate(self.study_plan[today_str]["tasks"], 1):
            status = "✓" if task["completed"] else "□"
            plan += f"{idx}. {status} {task['name']} ({task['duration']} phút)\n"

        return plan

    def update_progress(self, task_index: int, completed: bool = True) -> str:
        today_str = datetime.today().strftime('%Y-%m-%d')
        
        if today_str not in self.study_plan:
            return "Không có kế hoạch học tập cho ngày hôm nay."
        
        if 0 <= task_index < len(self.study_plan[today_str]["tasks"]):
            task = self.study_plan[today_str]["tasks"][task_index]
            task["completed"] = completed
            return f"Đã cập nhật trạng thái nhiệm vụ '{task['name']}' thành {'hoàn thành' if completed else 'chưa hoàn thành'}."
        return "Không tìm thấy nhiệm vụ với chỉ số này."

    async def run(self, user_input: str = None, **kwargs) -> str:
        if not self.study_plan:
            # Nếu không có kế hoạch học, tạo kế hoạch học
            if user_input and "lộ trình" in user_input.lower():
                self.goal = "Cải thiện kỹ năng tiếng Anh"
                self.time_frame = "2 tháng"
                study_plan = await self.generate_study_plan()
                daily_plan = self.generate_daily_study_plan()
                return f"{study_plan}\n\nLộ trình học theo ngày:\n{daily_plan}"
            else:
                return "Vui lòng nhập yêu cầu tạo lộ trình học."
        
        elif "hoàn thành" in user_input.lower():
            task_name = user_input.lower().replace("hoàn thành", "").strip()
            for idx, task in enumerate(self.study_plan.get(datetime.today().strftime('%Y-%m-%d'), {}).get("tasks", [])):
                if task_name in task["name"].lower():
                    return self.update_progress(idx, True)
            return "Không tìm thấy nhiệm vụ cần hoàn thành."
        
        elif "xem kế hoạch" in user_input.lower():
            return self.get_today_plan()
        
        elif "generate hôm nay" in user_input.lower():
            return self.generate_today_plan()

        else:
            return "Vui lòng nhập yêu cầu hợp lệ như: 'Tạo lộ trình học', 'Hoàn thành nhiệm vụ', 'Xem kế hoạch học hôm nay', 'Generate hôm nay'."

# # Chạy kiểm tra
# async def test_study_plan_agent():
#     study_plan_agent = StudyPlanAgent()
#     print(await study_plan_agent.run("Tạo lộ trình học cho IELTS trong 2 tháng"))
#     print(await study_plan_agent.run("Hoàn thành luyện nghe hôm nay"))
#     print(await study_plan_agent.run("Xem kế hoạch học hôm nay"))
#     print(await study_plan_agent.run("Generate hôm nay"))

# asyncio.run(test_study_plan_agent())
