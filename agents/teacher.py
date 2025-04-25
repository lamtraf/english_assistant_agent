import re
import nltk
from typing import List, Tuple
from nltk.corpus import stopwords
import os
import logging
import traceback
import httpx
from dotenv import load_dotenv
from abc import ABC, abstractmethod


# Tải stopwords từ NLTK
nltk.download('stopwords')

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


class EnglishTeacherAgent(BaseAgent):
    def __init__(self, temperature: float = 0.7, model: str = "gemini-2.0-flash", **kwargs):
        super().__init__(temperature=temperature, model=model, **kwargs)
    
    def list_difficult_vocabulary(self, text: str) -> List[str]:
        """Liệt kê các từ vựng trong văn bản, bỏ qua stop words."""
        # Lấy danh sách stop words từ NLTK
        stop_words = set(stopwords.words('english'))
        
        # Tách các từ trong văn bản và chuyển thành chữ thường
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Lọc ra các từ không phải là stop words
        difficult_vocabulary = [word for word in words if word not in stop_words]
        return list(set(difficult_vocabulary))  # Trả về các từ duy nhất

    async def translate_to_vietnamese(self, word: str) -> str:
        """Dịch nghĩa của từ vựng sang tiếng Việt sử dụng mô hình Gemini."""
        prompt = f"Please translate the word '{word}' into Vietnamese."
        
        # Gọi mô hình Gemini để dịch từ vựng sang tiếng Việt
        response = await self.ask(prompt)
        translated_word = response.strip()
        return translated_word

    async def correct_grammar(self, text: str) -> str:
        """Sửa ngữ pháp thông minh bằng cách sử dụng LLM (Gemini)."""
        prompt = f"Please correct the following English text and improve its grammar:\n\n{text}"
        
        # Gọi mô hình Gemini để sửa ngữ pháp và cải thiện bài viết
        response = await self.ask(prompt)
        corrected_text = response.strip()
        return corrected_text

    async def explain_correction(self, text: str) -> str:
        """Giải thích các sửa chữa ngữ pháp đã thực hiện bằng cách sử dụng LLM."""
        prompt = f"Explain the grammar mistakes and corrections made in the following text:\n\n{text}"
        
        # Gọi mô hình Gemini để giải thích sửa chữa ngữ pháp
        response = await self.ask(prompt)
        explanation = response.strip()
        return explanation

    async def improve_writing(self, text: str) -> str:
        """Cải thiện bài viết bằng cách sửa ngữ pháp và phong cách viết."""
        prompt = f"Please improve the following English text to make it more natural and fluent:\n\n{text}"
        
        # Gọi mô hình Gemini để cải thiện bài viết
        response = await self.ask(prompt)
        improved_text = response.strip()
        return improved_text

    async def analyze_text(self, text: str) -> Tuple[List[str], str, str]:
        """Phân tích văn bản, liệt kê từ vựng và sửa ngữ pháp, giải thích sửa chữa."""
        difficult_vocabulary = self.list_difficult_vocabulary(text)
        
        # Dịch nghĩa các từ vựng khó sang tiếng Việt
        translated_vocabulary = {}
        for word in difficult_vocabulary:
            translated_vocabulary[word] = await self.translate_to_vietnamese(word)
        
        corrected_text = await self.correct_grammar(text)
        explanation = await self.explain_correction(text)
        
        return translated_vocabulary, corrected_text, explanation

    async def run(self, user_input: str) -> str:
        """Xử lý yêu cầu từ người dùng."""
        if user_input:
            # Liệt kê từ vựng khó
            translated_vocabulary, corrected_text, explanation = await self.analyze_text(user_input)

            # Hiển thị kết quả
            result = "### Từ vựng và nghĩa tiếng Việt:\n"
            for word, translation in translated_vocabulary.items():
                result += f"- {word}: {translation}\n"
            
            result += f"\n### Bài viết đã được cải thiện:\n{corrected_text}\n\n"
            result += f"### Giải thích ngữ pháp:\n{explanation}"

            return result
        else:
            return "Vui lòng nhập văn bản để tôi có thể giúp bạn cải thiện."

# Kiểm tra EnglishTeacherAgent
# import asyncio

# async def test_english_teacher_agent():
#     agent = EnglishTeacherAgent()
    
#     # Ví dụ văn bản người dùng
#     user_input = "I have an ephemeral idea about the benevolent nature of the man. The juxtaposition of facts was obscure."
    
#     # Xử lý văn bản
#     result = await agent.run(user_input)
#     print(result)

# # Chạy kiểm tra
# asyncio.run(test_english_teacher_agent())
