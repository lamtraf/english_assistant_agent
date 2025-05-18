import asyncio
import logging
from typing import List, Dict, Any, Optional
import json
import traceback
import re

from .studyplan import BaseAgent # Sử dụng BaseAgent đã có

logger = logging.getLogger(__name__)

class ReadingComprehensionAgent(BaseAgent):
    def __init__(
        self,
        streaming: bool = False
    ):
        super().__init__(streaming=streaming)

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
        """Tạo bài đọc hiểu dựa trên yêu cầu của người dùng"""
        try:
            if not user_input or not user_input.strip():
                return {
                    "error": "Yêu cầu không được để trống",
                    "details": "Vui lòng nhập yêu cầu của bạn"
                }

            # Tạo prompt chi tiết
            prompt = f"""Bạn là một giáo viên tiếng Anh chuyên về đọc hiểu. Hãy tạo một bài đọc hiểu dựa trên yêu cầu của học viên.

Yêu cầu của học viên:
{user_input}

Hãy tạo một bài đọc hiểu với các phần sau:
1. Lời chào và giới thiệu
2. Bài đọc (khoảng 300-500 từ)
3. Từ vựng mới và giải thích
4. Câu hỏi đọc hiểu (5-7 câu)
5. Đáp án và giải thích
6. Bài tập luyện tập
7. Lời khuyên học đọc hiểu

Yêu cầu:
1. Sử dụng ngôn ngữ đơn giản, dễ hiểu
2. Nội dung phải phù hợp với trình độ của học viên
3. Tập trung vào các chủ đề thực tế và hữu ích
4. KHÔNG sử dụng markdown hoặc định dạng đặc biệt
5. Sử dụng dấu xuống dòng để phân tách các phần
6. Thêm các câu hỏi để tương tác với học viên"""

            logger.info(f"[ReadingComprehensionAgent] Generating reading comprehension with detailed prompt.")
            logger.info(f"[ReadingComprehensionAgent] Prompt: {prompt}")

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
                logger.error(f"[ReadingComprehensionAgent] Failed to process response: {e}. Response: {full_response}")
                return {
                    "error": "Không thể xử lý phản hồi từ AI",
                    "details": str(e)
                }

        except Exception as e:
            logger.error(f"[ReadingComprehensionAgent] Error generating reading comprehension: {e}")
            return {
                "error": "Lỗi khi tạo bài đọc hiểu",
                "details": str(e)
            }

# --- Ví dụ sử dụng (để test) ---
async def test_reading_agent():
    agent = ReadingComprehensionAgent(streaming=False) # Non-streaming để dễ xem JSON

    print("--- Test: Generate Passage ---")
    passage_data = await agent.run("the future of remote work")
    print(f"Generated Passage (first 100 chars): {passage_data.get('content', '')[:100]}...")
    # print(passage_data) # In toàn bộ để xem
    passage_text = passage_data.get("content")

    if passage_text:
        print("\n--- Test: Generate Questions --- (sử dụng passage ở trên)")
        questions_data = await agent.run("What is the main idea of the passage?")
        print(json.dumps(questions_data, indent=2, ensure_ascii=False))

        print("\n--- Test: Summarize Passage --- (sử dụng passage ở trên)")
        summary_data = await agent.run("Summarize the passage in a single, concise sentence.")
        print(summary_data)
    else:
        print("\nSkipping question generation and summarization as passage generation failed or returned no text.")

    print("\n--- Test: Unknown command ---")
    unknown_res = await agent.run("translate_passage")
    print(unknown_res)

    print("\n--- Test Generate Questions (JSON Error Simulation) ---")
    original_ask = agent.ask
    async def mock_ask_non_json(prompt: str):
        yield "This is not a valid JSON for questions."
    agent.ask = mock_ask_non_json
    questions_error = await agent.run("Some short passage.")
    print(json.dumps(questions_error, indent=2, ensure_ascii=False))
    agent.ask = original_ask # Khôi phục

    print("\n--- Test Generate Questions (Malformed JSON Simulation) ---")
    async def mock_ask_malformed_json(prompt: str):
        yield '{"wrong_key": []}' # Thiếu key "questions"
    agent.ask = mock_ask_malformed_json
    questions_malformed = await agent.run("Another passage.")
    print(json.dumps(questions_malformed, indent=2, ensure_ascii=False))
    agent.ask = original_ask

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s')
    asyncio.run(test_reading_agent()) 