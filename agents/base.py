import httpx
import logging
import traceback
import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta

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


if __name__ == "__main__":
    import asyncio

    # Khởi tạo BaseAgent
    class TestBaseAgent(BaseAgent):
        async def run(self, **kwargs):
            prompt = kwargs.get("prompt", "default test prompt")
            return await self.ask(prompt)

    # Test trực tiếp BaseAgent
    async def test_base_agent():
        agent = TestBaseAgent()

        # Gọi phương thức ask với một prompt
        prompt = "Lộ trình học tập tiếng Anh cho tôi, trình độ trung bình và cần đạt trình độ B2 trong 6 tháng."
        print(f"Prompt: {prompt}")
        response = await agent.ask(prompt)

        # Hiển thị kết quả trả về
        print(f"Response: {response}")

    # Test SpeakingAgent (Lớp con của BaseAgent)
    async def test_speaking_agent():
        class SpeakingAgent(BaseAgent):
            async def run(self, **kwargs):
                prompt = kwargs.get("prompt", "Travel topic for speaking practice")
                return await self.ask(prompt)

        agent = SpeakingAgent()
        prompt = "Tell me about the topic of travel."
        print(f"Prompt: {prompt}")
        response = await agent.run(prompt)

        # Hiển thị kết quả trả về
        print(f"Response: {response}")

    # Chạy test
    asyncio.run(test_base_agent())