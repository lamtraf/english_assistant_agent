import asyncio
import logging
from typing import List, Dict, Any
import json
import traceback
import re

from .studyplan import BaseAgent # Sử dụng BaseAgent đã có

logger = logging.getLogger(__name__)

class VocabularyAgent(BaseAgent):
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
        """Tạo bài học từ vựng dựa trên yêu cầu của người dùng"""
        try:
            if not user_input or not user_input.strip():
                return {
                    "error": "Yêu cầu không được để trống",
                    "details": "Vui lòng nhập yêu cầu của bạn"
                }

            # Tạo prompt chi tiết
            prompt = f"""Bạn là một giáo viên tiếng Anh chuyên về từ vựng. Hãy tạo một bài học từ vựng dựa trên yêu cầu của học viên.

Yêu cầu của học viên:
{user_input}

Hãy tạo một bài học từ vựng với các phần sau:
1. Lời chào và giới thiệu
2. Danh sách từ vựng (5-10 từ)
3. Giải thích nghĩa và cách dùng
4. Ví dụ minh họa
5. Bài tập luyện tập
6. Đáp án và giải thích
7. Lời khuyên học từ vựng

Yêu cầu:
1. Sử dụng ngôn ngữ đơn giản, dễ hiểu
2. Nội dung phải phù hợp với trình độ của học viên
3. Tập trung vào các từ vựng thực tế và hữu ích
4. KHÔNG sử dụng markdown hoặc định dạng đặc biệt
5. Sử dụng dấu xuống dòng để phân tách các phần
6. Thêm các câu hỏi để tương tác với học viên"""

            logger.info(f"[VocabularyAgent] Generating vocabulary lesson with detailed prompt.")
            logger.info(f"[VocabularyAgent] Prompt: {prompt}")

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
                logger.error(f"[VocabularyAgent] Failed to process response: {e}. Response: {full_response}")
                return {
                    "error": "Không thể xử lý phản hồi từ AI",
                    "details": str(e)
                }

        except Exception as e:
            logger.error(f"[VocabularyAgent] Error generating vocabulary lesson: {e}")
            return {
                "error": "Lỗi khi tạo bài học từ vựng",
                "details": str(e)
            }

    async def explain_word(self, word: str, level: str = "intermediate") -> Dict[str, Any]:
        """
        Giải thích một từ, bao gồm nghĩa, loại từ, câu ví dụ, từ đồng nghĩa/trái nghĩa.
        Trả về JSON chứa thông tin chi tiết hoặc dictionary chứa lỗi.
        """
        prompt = (
            f"Explain the English word '{word}' in detail. Provide the following information in English: "
            f"1. Part of Speech (e.g., noun, verb, adjective)."
            f"2. Primary meaning(s)."
            f"3. At least 2 example sentences showing its usage."
            f"4. Common synonyms (if any)."
            f"5. Common antonyms (if any)."
            f"6. If the word has multiple distinct meanings, briefly list them or focus on the most common one based on general usage."
            f"Format the response as a JSON object with keys: \"word\", \"part_of_speech\", \"meanings\" (list of strings), \"examples\" (list of strings), \"synonyms\" (list of strings), \"antonyms\" (list of strings). If synonyms/antonyms are not applicable or common, provide an empty list."
        )
        if level == "simple":
            prompt = (
                f"Explain the English word '{word}' simply. Provide:"
                f"1. Part of Speech."
                f"2. Main meaning."
                f"3. One example sentence."
                f"Format as JSON: \"word\", \"part_of_speech\", \"meaning\", \"example\"'."
            )
        
        response_str_parts = []
        async for chunk in self.ask(prompt):
            response_str_parts.append(chunk)
        response_str = "".join(response_str_parts)

        logger.debug(f"Raw Gemini response for explaining '{word}': {response_str}")
        
        if response_str.strip().endswith("```"):
            response_str = response_str.strip()[:-3]
        response_str = response_str.strip()

        try:
            explanation_data = json.loads(response_str)
            # Kiểm tra các trường cơ bản - tùy chọn, tùy thuộc vào mức độ nghiêm ngặt bạn muốn
            # if not all(k in explanation_data for k in ["word", "meaning", "part_of_speech", "examples"]):
            #     logger.warning(f"JSON response for word explanation is missing some expected keys. Word: {word}")
            #     # Vẫn trả về data nếu parse được, client có thể xử lý thiếu sót
            logger.info(f"[{self.__class__.__name__}] Word '{word}' explained successfully with JSON output.")
            return explanation_data
        except json.JSONDecodeError as e:
            logger.error(f"[{self.__class__.__name__}] Failed to decode JSON for word '{word}': {e}. Response: {response_str}")
            return {
                "error": "Không thể phân tích giải thích từ vựng từ AI.",
                "word": word,
                "details": f"Lỗi JSON: {e}. Dữ liệu nhận được: {response_str[:200]}..."
            }
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Unexpected error explaining word '{word}': {e}\n{traceback.format_exc()}")
            return {
                "error": "Lỗi không mong muốn khi giải thích từ vựng.",
                "word": word,
                "details": str(e)
            }

    async def get_words_by_topic(self, topic: str, difficulty: str = "intermediate", count: int = 10) -> Dict[str, Any]:
        """
        Cung cấp danh sách các từ theo chủ đề và độ khó.
        Trả về JSON chứa danh sách từ hoặc dictionary chứa lỗi.
        """
        prompt = (
            f"Provide a list of {count} English vocabulary words related to the topic '{topic}' suitable for an {difficulty} level learner. "
            f"For each word, include its primary meaning in English."
            f"Format the response as a JSON object with a single key '{topic}_words', which is a list of objects. Each object should have \"word\" and \"meaning\" keys."
            f"Example for topic 'technology' and count 2: "
            f"{{ \"technology_words\": [ {{ \"word\": \"algorithm\", \"meaning\": \"A set of rules to be followed in calculations or other problem-solving operations.\" }}, {{ \"word\": \"encryption\", \"meaning\": \"The process of converting information or data into a code, especially to prevent unauthorized access.\" }} ] }}"
        )
        
        response_str_parts = []
        async for chunk in self.ask(prompt):
            response_str_parts.append(chunk)
        response_str = "".join(response_str_parts)
        logger.debug(f"Raw Gemini response for topic '{topic}': {response_str}")

        if response_str.strip().endswith("```"):
            response_str = response_str.strip()[:-3]
        response_str = response_str.strip()

        try:
            word_list_data = json.loads(response_str)
            if "words" not in word_list_data or not isinstance(word_list_data["words"], list):
                logger.error(f"[{self.__class__.__name__}] Invalid JSON structure for word list (topic: {topic}). Missing 'words' list. Response: {response_str}")
                return {
                    "error": "Cấu trúc JSON không hợp lệ cho danh sách từ vựng.",
                    "topic": topic,
                    "details": f"Thiếu key 'words' hoặc 'words' không phải là list. Dữ liệu nhận được: {response_str[:200]}..."
                }
            logger.info(f"[{self.__class__.__name__}] Word list for topic '{topic}' retrieved successfully.")
            return word_list_data # Trả về toàn bộ dict {"words": [...]}
        except json.JSONDecodeError as e:
            logger.error(f"[{self.__class__.__name__}] Failed to decode JSON for word list (topic: {topic}): {e}. Response: {response_str}")
            return {
                "error": "Không thể phân tích danh sách từ vựng từ AI.",
                "topic": topic,
                "details": f"Lỗi JSON: {e}. Dữ liệu nhận được: {response_str[:200]}..."
            }
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Unexpected error getting words for topic '{topic}': {e}\n{traceback.format_exc()}")
            return {
                "error": "Lỗi không mong muốn khi lấy danh sách từ vựng.",
                "topic": topic,
                "details": str(e)
            }

# Test thử agent
async def main():
    agent = VocabularyAgent(streaming=False) # Test với non-streaming để dễ xem JSON đầy đủ

    print("--- Test: Explain Word (normal) ---")
    explanation1 = await agent.run(user_input="ubiquitous")
    print(json.dumps(explanation1, indent=2, ensure_ascii=False))

    print("\n--- Test: Explain Word (simple) ---")
    explanation2 = await agent.run(user_input="ephemeral")
    print(json.dumps(explanation2, indent=2, ensure_ascii=False))

    print("\n--- Test: Get Words by Topic ---")
    word_list = await agent.run(user_input="sustainability")
    print(json.dumps(word_list, indent=2, ensure_ascii=False))
    
    print("\n--- Test: Get Words by Topic (parse error simulation - Gemini might not return perfect JSON) ---")
    # Để mô phỏng lỗi parse, chúng ta có thể sửa prompt để nó không yêu cầu JSON
    # Hoặc chúng ta có thể test với một topic mà Gemini có thể trả về format lạ.
    # Trong trường hợp này, chúng ta chỉ chạy và xem log nếu có lỗi parse.
    topic_words_tricky = await agent.run(user_input="অলিক") # non-English topic name
    print(json.dumps(topic_words_tricky, indent=2, ensure_ascii=False))

    print("\n--- Test: Unknown command ---")
    unknown_res = await agent.run(user_input="make_sentence")
    print(json.dumps(unknown_res, indent=2, ensure_ascii=False))

    print("\n--- Test Get Words By Topic ---")
    word_list = await agent.run(user_input="technology")
    print(json.dumps(word_list, indent=2, ensure_ascii=False))

    print("\n--- Test Explain Word (JSON Error Simulation) ---")
    # Giả lập Gemini trả về non-JSON
    original_ask = agent.ask
    async def mock_ask_non_json(prompt: str):
        yield "This is not a valid JSON."
    agent.ask = mock_ask_non_json
    explanation_error = await agent.run(user_input="ubiquitous")
    print(json.dumps(explanation_error, indent=2, ensure_ascii=False))
    agent.ask = original_ask # Khôi phục hàm ask

    print("\n--- Test Get Words (JSON Error Simulation) ---")
    async def mock_ask_malformed_json(prompt: str):
        yield '{"wrong_key": ["word1", "word2"]}' # Thiếu key "words"
    agent.ask = mock_ask_malformed_json
    words_error = await agent.run(user_input="nature")
    print(json.dumps(words_error, indent=2, ensure_ascii=False))
    agent.ask = original_ask

if __name__ == '__main__':
    # Cấu hình logging để xem output debug/info
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s')
    asyncio.run(main()) 