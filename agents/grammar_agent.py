import asyncio
import json
from typing import List, Dict, Any, Optional

from agents.studyplan import BaseAgent # Sử dụng BaseAgent từ studyplan.py
import logging
import re

logger = logging.getLogger(__name__)

class GrammarAgent(BaseAgent):
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
        """Kiểm tra và giải thích ngữ pháp"""
        try:
            if not user_input or not user_input.strip():
                return {
                    "error": "Yêu cầu không được để trống",
                    "details": "Vui lòng nhập yêu cầu của bạn"
                }

            # Tạo prompt chi tiết
            prompt = f"""Bạn là một giáo viên tiếng Anh chuyên về ngữ pháp. Hãy kiểm tra và giải thích ngữ pháp dựa trên yêu cầu của học viên.

Yêu cầu của học viên:
{user_input}

Hãy phân tích và giải thích với các phần sau:
1. Lời chào và giới thiệu
2. Phân tích câu/đoạn văn
3. Chỉ ra lỗi ngữ pháp (nếu có)
4. Giải thích quy tắc ngữ pháp
5. Đưa ra ví dụ minh họa
6. Bài tập luyện tập
7. Lời khuyên học ngữ pháp

Yêu cầu:
1. Sử dụng ngôn ngữ đơn giản, dễ hiểu
2. Giải thích chi tiết và rõ ràng
3. Đưa ra nhiều ví dụ thực tế
4. KHÔNG sử dụng markdown hoặc định dạng đặc biệt
5. Sử dụng dấu xuống dòng để phân tách các phần
6. Thêm các câu hỏi để tương tác với học viên"""

            logger.info(f"[GrammarAgent] Generating grammar analysis with detailed prompt.")
            logger.info(f"[GrammarAgent] Prompt: {prompt}")

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
                logger.error(f"[GrammarAgent] Failed to process response: {e}. Response: {full_response}")
                return {
                    "error": "Không thể xử lý phản hồi từ AI",
                    "details": str(e)
                }

        except Exception as e:
            logger.error(f"[GrammarAgent] Error analyzing grammar: {e}")
            return {
                "error": "Lỗi khi phân tích ngữ pháp",
                "details": str(e)
            }

# Test thử agent
async def main():
    agent = GrammarAgent()

    # Test giải thích quy tắc
    print("\n--- Test Explain Grammar Rule ---")
    explanation = await agent.run(command="explain_rule", rule_name="Present Perfect Tense", level="beginner")
    print(explanation)

    # Test sửa văn bản (có giải thích)
    print("\n--- Test Correct Text (with explanations) ---")
    text_to_correct = "She go to school yesterday. He dont like apples. I has a cat."
    correction_result = await agent.run(command="correct_text", text=text_to_correct, explain_errors=True)
    print(json.dumps(correction_result, indent=2, ensure_ascii=False))

    # Test sửa văn bản (không giải thích)
    print("\n--- Test Correct Text (without explanations) ---")
    text_simple_correct = "i love learn english very much"
    simple_correction = await agent.run(command="correct_text", text=text_simple_correct, explain_errors=False)
    print(json.dumps(simple_correction, indent=2, ensure_ascii=False))

    # Test cung cấp ví dụ
    print("\n--- Test Provide Examples ---")
    examples = await agent.run(command="provide_examples", grammar_point="phrasal verb 'look up'", count=2, context="formal writing")
    for ex in examples:
        print(ex)
        
    examples_no_context = await agent.run(command="provide_examples", grammar_point="idiom 'break a leg'")
    for ex in examples_no_context:
        print(ex)

if __name__ == "__main__":
    asyncio.run(main()) 