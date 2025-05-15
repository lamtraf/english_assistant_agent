import asyncio
import logging
from typing import List, Dict, Any, Optional
import json
import traceback

from .studyplan import BaseAgent # Sử dụng BaseAgent đã có

logger = logging.getLogger(__name__)

class ReadingComprehensionAgent(BaseAgent):
    def __init__(
        self,
        temperature: float = 0.7, # Cho phép một chút sáng tạo khi tạo bài đọc/câu hỏi
        gemini_model: str = "gemini-1.5-flash-latest",
        gemini_streaming: bool = True, 
        memory=None,
        tracing: bool = False
    ):
        super().__init__(temperature, gemini_model, gemini_streaming, memory, tracing)
        logger.info(f"ReadingComprehensionAgent khởi tạo với model: {gemini_model}, streaming: {gemini_streaming}")

    async def generate_reading_passage(self, topic: str, difficulty: str = "intermediate", length: str = "short") -> Dict[str, str]:
        """Tạo một đoạn văn đọc hiểu theo chủ đề, độ khó và độ dài."""
        prompt = (
            f"Generate a {length} English reading passage on the topic of '{topic}' suitable for an {difficulty} level learner. "
            f"The passage should be engaging and well-structured. "
            f"After the passage, clearly indicate the end of the passage with a line containing only '---END OF PASSAGE---'."
            f"Do not ask any questions after the passage in this step."
        )
        
        response_str_parts = []
        logger.info(f"Generating reading passage for topic: {topic}, difficulty: {difficulty}, length: {length}")
        async for chunk in self.ask(prompt):
            response_str_parts.append(chunk)
        full_response = "".join(response_str_parts)

        # Tách đoạn văn
        passage_end_marker = "---END OF PASSAGE---"
        if passage_end_marker in full_response:
            passage = full_response.split(passage_end_marker)[0].strip()
        else:
            # Nếu không có marker, lấy toàn bộ phản hồi làm đoạn văn, nhưng cảnh báo
            logger.warning("Passage end marker not found in Gemini response. Using full response as passage.")
            passage = full_response.strip()
        
        logger.debug(f"Generated passage for '{topic}':\n{passage[:300]}...")
        return {"topic": topic, "difficulty": difficulty, "length": length, "passage": passage}

    async def generate_comprehension_questions(self, passage: str, num_questions: int = 5, question_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Tạo câu hỏi hiểu bài cho một đoạn văn nhất định.
        Trả về JSON chứa câu hỏi hoặc dictionary lỗi.
        """
        if question_types is None:
            question_types = ["MCQ", "True/False", "Short Answer"]
        
        types_str = ", ".join(question_types)
        prompt = (
            f"Based on the following English reading passage, generate {num_questions} comprehension questions. "
            f"Try to include a mix of question types, such as: {types_str}. "
            f"For each question, also provide the correct answer (or an expected answer for open-ended questions)."
            f"Passage:\n\"\"\"{passage}\"\"\"\n\n"
            f"Format the response as a JSON object with a single key 'questions', which is a list of objects. "
            f"Each object in the list should have the keys: \"question_text\", \"question_type\" (e.g., 'main_idea', 'detail'), and \"answer_text\"."
        )
        
        response_str_parts = []
        logger.info(f"Generating {num_questions} questions for the provided passage.")
        async for chunk in self.ask(prompt):
            response_str_parts.append(chunk)
        response_str = "".join(response_str_parts)
        logger.debug(f"Raw Gemini response for questions: {response_str}")

        if raw_response.strip().endswith("```"):
            raw_response = raw_response.strip()[:-3]
        raw_response = raw_response.strip()

        try:
            questions_data = json.loads(raw_response)
            if "questions" not in questions_data or not isinstance(questions_data["questions"], list):
                logger.error(f"[{self.__class__.__name__}] Invalid JSON structure for comprehension questions. Missing 'questions' list. Response: {raw_response}")
                return {
                    "error": "Cấu trúc JSON không hợp lệ cho câu hỏi hiểu bài.",
                    "details": f"Thiếu key 'questions' hoặc 'questions' không phải là list. Dữ liệu nhận được: {raw_response[:200]}..."
                }
            
            # Kiểm tra sâu hơn cấu trúc của từng câu hỏi (tùy chọn)
            # for q_idx, question in enumerate(questions_data["questions"]):
            #     if not all(k in question for k in ["question_text", "question_type", "answer"]):
            #         logger.warning(f"Câu hỏi thứ {q_idx+1} thiếu các trường bắt buộc.")
            #         # Có thể quyết định trả về lỗi hoặc chấp nhận một phần

            logger.info(f"[{self.__class__.__name__}] Comprehension questions generated successfully.")
            return questions_data # Trả về toàn bộ dict {"questions": [...]}
        except json.JSONDecodeError as e:
            logger.error(f"[{self.__class__.__name__}] Failed to decode JSON for comprehension questions: {e}. Response: {raw_response}")
            return {
                "error": "Không thể phân tích câu hỏi hiểu bài từ AI.",
                "details": f"Lỗi JSON: {e}. Dữ liệu nhận được: {raw_response[:200]}..."
            }
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Unexpected error generating comprehension questions: {e}\n{traceback.format_exc()}")
            return {
                "error": "Lỗi không mong muốn khi tạo câu hỏi hiểu bài.",
                "details": str(e)
            }

    async def summarize_passage(self, passage: str, length: str = "medium") -> str: # Giữ nguyên trả về string cho summary
        """Tóm tắt một đoạn văn."""
        length_instruction = "in a single, concise sentence" if length == "one_sentence" else "in a short paragraph (2-3 sentences)"

        prompt = (
            f"Summarize the following English reading passage {length_instruction}.\n"
            f"Passage:\n\"\"\"{passage}\"\"\"\n\n"
            f"Format the response as a JSON object with a single key 'summary_text'."
        )
        response_str_parts = []
        logger.info(f"Summarizing passage ({length}).")
        async for chunk in self.ask(prompt):
            response_str_parts.append(chunk)
        response_str = "".join(response_str_parts)
        logger.debug(f"Raw Gemini response for summary: {response_str}")
        
        return response_str

    async def run(self, command: str, **kwargs) -> Any:
        logger.info(f"[{self.__class__.__name__}] Running command '{command}' with kwargs: {kwargs}")
        if command == "generate_reading_passage":
            topic = kwargs.get("topic")
            if not topic:
                return {"error": "Lỗi: 'topic' là bắt buộc cho lệnh 'generate_reading_passage'."} # Trả về dict lỗi
            difficulty = kwargs.get("difficulty", "intermediate")
            passage = await self.generate_reading_passage(topic=topic, difficulty=difficulty)
            # Nếu generate_reading_passage có thể trả lỗi, cần xử lý ở đây hoặc để nó trả về string lỗi
            return passage # Hiện tại generate_reading_passage trả về string
        
        elif command == "generate_comprehension_questions":
            passage_text = kwargs.get("passage")
            if not passage_text:
                return {"error": "Lỗi: 'passage' là bắt buộc cho lệnh 'generate_comprehension_questions'."} # Trả về dict lỗi
            num_q = kwargs.get("num_questions", 5)
            q_types = kwargs.get("question_types")
            return await self.generate_comprehension_questions(passage=passage_text, num_questions=num_q, question_types=q_types)
            
        elif command == "summarize_passage":
            passage_text = kwargs.get("passage")
            if not passage_text:
                return {"error": "Lỗi: 'passage' là bắt buộc cho lệnh 'summarize_passage'."} # Trả về dict lỗi
            length = kwargs.get("length", "medium")
            summary = await self.summarize_passage(passage=passage_text, length=length)
            return summary # Hiện tại summarize_passage trả về string
            
        else:
            logger.warning(f"[{self.__class__.__name__}] Unknown command: {command}")
            return {"error": f"Lệnh không xác định: {command}. Các lệnh hợp lệ: 'generate_reading_passage', 'generate_comprehension_questions', 'summarize_passage'."}

# --- Ví dụ sử dụng (để test) ---
async def test_reading_agent():
    agent = ReadingComprehensionAgent(gemini_streaming=False) # Non-streaming để dễ xem JSON

    print("--- Test: Generate Passage ---")
    passage_data = await agent.run("generate_reading_passage", {"topic": "the future of remote work", "difficulty": "intermediate", "length": "medium"})
    print(f"Generated Passage (first 100 chars): {passage_data.get('passage', '')[:100]}...")
    # print(passage_data) # In toàn bộ để xem
    passage_text = passage_data.get("passage")

    if passage_text:
        print("\n--- Test: Generate Questions --- (sử dụng passage ở trên)")
        questions_data = await agent.run("generate_comprehension_questions", {"passage": passage_text, "num_questions": 2})
        print(json.dumps(questions_data, indent=2, ensure_ascii=False))

        print("\n--- Test: Summarize Passage --- (sử dụng passage ở trên)")
        summary_data = await agent.run("summarize_passage", {"passage": passage_text, "length": "short_paragraph"})
        print(summary_data)
    else:
        print("\nSkipping question generation and summarization as passage generation failed or returned no text.")

    print("\n--- Test: Unknown command ---")
    unknown_res = await agent.run("translate_passage", {"passage": "test"})
    print(unknown_res)

    print("\n--- Test Generate Questions (JSON Error Simulation) ---")
    original_ask = agent.ask
    async def mock_ask_non_json(prompt: str):
        yield "This is not a valid JSON for questions."
    agent.ask = mock_ask_non_json
    questions_error = await agent.run(command="generate_comprehension_questions", passage="Some short passage.")
    print(json.dumps(questions_error, indent=2, ensure_ascii=False))
    agent.ask = original_ask # Khôi phục

    print("\n--- Test Generate Questions (Malformed JSON Simulation) ---")
    async def mock_ask_malformed_json(prompt: str):
        yield '{"wrong_key": []}' # Thiếu key "questions"
    agent.ask = mock_ask_malformed_json
    questions_malformed = await agent.run(command="generate_comprehension_questions", passage="Another passage.")
    print(json.dumps(questions_malformed, indent=2, ensure_ascii=False))
    agent.ask = original_ask

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s')
    asyncio.run(test_reading_agent()) 