import logging
import json
from typing import AsyncGenerator, Optional, Dict, Any
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(
        self,
        temperature: float = 0.7,
        model: str = "gemini-1.5-flash-latest",
        streaming: bool = False,
        memory=None,
        tracing: bool = False
    ):
        self.temperature = temperature
        self.model_name = model
        self.streaming = streaming
        self.memory = memory
        self.tracing = tracing
        self.model = None
        self.initialize()

    def initialize(self):
        """Khởi tạo model Gemini"""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY không được tìm thấy trong biến môi trường")
                
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest"),
                generation_config={
                    "temperature": self.temperature,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                }
            )
            logger.info(f"[{self.__class__.__name__}] Đã khởi tạo model {os.getenv('GEMINI_MODEL', 'gemini-1.5-flash-latest')}")
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Lỗi khởi tạo model: {e}")
            raise

    async def ask(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Gửi prompt đến Google Gemini API và nhận kết quả trả về dưới dạng stream các chunks.
        Nếu streaming=False, sẽ yield một chunk duy nhất chứa toàn bộ nội dung.
        """
        if not self.model:
            error_msg = f"[{self.__class__.__name__}] Model chưa được khởi tạo"
            logger.error(error_msg)
            yield error_msg
            return

        logger.info(f"[{self.__class__.__name__}] Prompt: {prompt[:300]}...") # Log một phần prompt
        
        try:
            if self.streaming:
                logger.info(f"[{self.__class__.__name__}] Gọi Gemini API (streaming)")
                response = await self.model.generate_content_async(prompt, stream=True)
                async for chunk in response:
                    if chunk.text:
                        yield chunk.text
            else:
                logger.info(f"[{self.__class__.__name__}] Gọi Gemini API (non-streaming)")
                response = await self.model.generate_content_async(prompt)
                if response and response.text:
                    yield response.text
                else:
                    yield "Không nhận được phản hồi từ AI"

        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Error calling Gemini API: {e}")
            yield f"Lỗi khi gọi API: {str(e)}"

    async def run(self, user_input: str) -> Dict[str, Any]:
        """
        Xử lý input từ người dùng và trả về kết quả dạng JSON
        """
        raise NotImplementedError("Các agent con phải implement phương thức này") 