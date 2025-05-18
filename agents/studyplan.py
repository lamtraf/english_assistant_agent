import os
from datetime import datetime, timedelta
import logging
import json
import re
from typing import Optional, List, Dict, Any
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class StudyPlanAgent(BaseAgent):
    def __init__(
        self, 
        streaming: bool = False
    ):
        super().__init__(streaming=streaming)
        self.conversation_history: List[Dict[str, str]] = []
        self.start_date = datetime.today()

    def _clean_content(self, content: str) -> str:
        """Xử lý nội dung từ Gemini API để loại bỏ markdown và định dạng không mong muốn"""
        # Loại bỏ các ký tự markdown
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Loại bỏ **
        content = re.sub(r'\*(.*?)\*', r'\1', content)      # Loại bỏ *
        content = re.sub(r'_(.*?)_', r'\1', content)        # Loại bỏ _
        content = re.sub(r'`(.*?)`', r'\1', content)        # Loại bỏ `
        
        # Thay thế các ký tự xuống dòng
        content = content.replace('\\n', '\n')
        
        # Loại bỏ khoảng trắng thừa
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = content.strip()
        
        return content

    async def run(self, user_input: str) -> Dict[str, Any]:
        """Tạo lộ trình học dựa trên yêu cầu của người dùng"""
        try:
            if not user_input or not user_input.strip():
                return {
                    "error": "Yêu cầu không được để trống",
                    "details": "Vui lòng nhập yêu cầu của bạn"
                }

            # Tạo prompt chi tiết
            prompt = f"""Bạn là một gia sư tiếng Anh thân thiện và nhiệt tình. Hãy tạo một lộ trình học tiếng Anh dựa trên yêu cầu của học viên.

Yêu cầu của học viên:
{user_input}

Hãy tạo một lộ trình học chi tiết với các phần sau:
1. Lời chào và giới thiệu
2. Phân tích yêu cầu của học viên
3. Lộ trình học chi tiết theo tuần
4. Tài nguyên học tập
5. Phương pháp đánh giá
6. Lời khuyên học tập
7. Câu hỏi tương tác

Yêu cầu:
1. Sử dụng ngôn ngữ đơn giản, dễ hiểu
2. Nội dung phải phù hợp với trình độ và sở thích của học viên
3. Tập trung vào các chủ đề thực tế và hữu ích
4. KHÔNG sử dụng markdown hoặc định dạng đặc biệt
5. Sử dụng dấu xuống dòng để phân tách các phần
6. Thêm các câu hỏi để tương tác với học viên"""

            logger.info(f"[StudyPlanAgent] Generating study plan with detailed prompt.")
            logger.info(f"[StudyPlanAgent] Prompt: {prompt}")

            # Gọi Gemini API thông qua BaseAgent
            full_response = ""
            async for chunk in self.ask(prompt):
                if chunk:
                    full_response += chunk

            if not full_response:
                return {
                    "error": "Không nhận được phản hồi từ AI",
                    "details": "Vui lòng thử lại sau"
                }

            # Xử lý response
            try:
                # Xử lý nội dung
                content = self._clean_content(full_response)
                
                # Trả về dạng text đơn giản
                return {
                    "content": content,
                    "type": "text"
                }
                
            except Exception as e:
                logger.error(f"[StudyPlanAgent] Failed to process response: {e}. Response: {full_response}")
                return {
                    "error": "Không thể xử lý phản hồi từ AI",
                    "details": str(e)
                }

        except Exception as e:
            logger.error(f"[StudyPlanAgent] Error generating study plan: {e}")
            return {
                "error": "Lỗi khi tạo lộ trình học",
                "details": str(e)
            } 