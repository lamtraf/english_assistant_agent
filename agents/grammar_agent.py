import asyncio
import json
from typing import List, Dict, Any, Optional

from agents.studyplan import BaseAgent # Sử dụng BaseAgent từ studyplan.py
import logging

logger = logging.getLogger(__name__)

class GrammarAgent(BaseAgent):
    def __init__(
        self,
        model: str = "gemini-1.0-pro", # Cân nhắc dùng model mạnh hơn nếu cần giải thích sâu
        streaming: bool = False, # Thường không cần streaming cho giải thích / sửa lỗi
        **kwargs
    ):
        super().__init__(model=model, streaming=streaming, **kwargs)

    async def explain_grammar_rule(self, rule_name: str, level: str = "intermediate") -> str:
        """
        Giải thích một quy tắc ngữ pháp.
        """
        prompt = f"""Hãy giải thích quy tắc ngữ pháp sau: '{rule_name}'.
Giải thích một cách rõ ràng, dễ hiểu, phù hợp với người học tiếng Anh ở trình độ {level}.
Bao gồm các điểm chính, cách sử dụng và ít nhất 2 ví dụ minh họa cho mỗi điểm chính.
Nếu quy tắc có các trường hợp ngoại lệ phổ biến, hãy đề cập đến chúng.
Trình bày câu trả lời một cách có cấu trúc, sử dụng markdown nếu cần thiết (ví dụ: tiêu đề, danh sách)."""
        
        logger.info(f"[{self.__class__.__name__}] Explaining grammar rule: {rule_name} for level {level}")
        
        response_parts = []
        async for chunk in self.ask(prompt):
            response_parts.append(chunk)
        explanation = "".join(response_parts)
        
        logger.info(f"[{self.__class__.__name__}] Explanation for '{rule_name}': {explanation[:200]}...")
        return explanation

    async def correct_text(self, text: str, explain_errors: bool = True) -> Dict[str, Any]:
        """
        Sửa lỗi ngữ pháp trong văn bản và giải thích các lỗi (nếu được yêu cầu).
        Trả về một dictionary chứa văn bản gốc, văn bản đã sửa, và danh sách các giải thích.
        """
        if explain_errors:
            prompt = f"""Vui lòng sửa lỗi ngữ pháp, chính tả và cách dùng từ trong đoạn văn bản sau.
Đồng thời, hãy cung cấp một danh sách các lỗi đã được sửa, giải thích ngắn gọn cho từng lỗi và gợi ý cách sửa.
Định dạng output mong muốn là một JSON object với các key sau:
- "original_text": (string) văn bản gốc.
- "corrected_text": (string) văn bản đã được sửa hoàn chỉnh.
- "corrections": (array of objects) mỗi object chứa:
    - "error_type": (string) loại lỗi (ví dụ: "verb tense", "preposition", "spelling", "word choice").
    - "original_phrase": (string) cụm từ/từ bị lỗi trong văn bản gốc.
    - "corrected_phrase": (string) cụm từ/từ đã được sửa.
    - "explanation": (string) giải thích ngắn gọn về lỗi và tại sao lại sửa như vậy.

Văn bản cần sửa:
---
{text}
---
Chỉ trả về JSON object, không có bất kỳ văn bản nào khác trước hoặc sau JSON.
"""
        else:
            prompt = f"""Vui lòng sửa lỗi ngữ pháp, chính tả và cách dùng từ trong đoạn văn bản sau.
Chỉ trả về văn bản đã được sửa hoàn chỉnh, không có giải thích hay bất kỳ định dạng nào khác.

Văn bản cần sửa:
---
{text}
---
"""
        logger.info(f"[{self.__class__.__name__}] Correcting text (explain_errors={explain_errors}): {text[:100]}...")
        
        response_parts = []
        async for chunk in self.ask(prompt):
            response_parts.append(chunk)
        raw_response = "".join(response_parts)

        if explain_errors:
            try:
                # Cố gắng loại bỏ ```json và ``` nếu Gemini trả về
                if raw_response.strip().startswith("```json"):
                    raw_response = raw_response.strip()[7:]
                if raw_response.strip().endswith("```"):
                    raw_response = raw_response.strip()[:-3]
                
                result = json.loads(raw_response)
                # Đảm bảo các trường bắt buộc có mặt
                if not all(k in result for k in ["original_text", "corrected_text", "corrections"]):
                    logger.error(f"JSON response from LLM is missing required keys. Response: {raw_response}")
                    # Fallback nếu JSON không đúng cấu trúc mong đợi
                    return {
                        "original_text": text,
                        "corrected_text": "Error: Could not parse corrections from LLM. Raw response: " + raw_response,
                        "corrections": []
                    }
                logger.info(f"[{self.__class__.__name__}] Text corrected successfully with explanations.")
                return result
            except json.JSONDecodeError:
                logger.error(f"[{self.__class__.__name__}] Failed to decode JSON response for corrections: {raw_response}")
                # Fallback nếu không parse được JSON
                return {
                    "original_text": text,
                    "corrected_text": "Error: LLM did not return valid JSON. Raw response: " + raw_response,
                    "corrections": []
                }
        else:
            logger.info(f"[{self.__class__.__name__}] Text corrected (no explanations).")
            return {
                "original_text": text,
                "corrected_text": raw_response.strip(),
                "corrections": []
            }

    async def provide_examples(self, grammar_point: str, count: int = 3, context: Optional[str] = None) -> List[str]:
        """
        Cung cấp các ví dụ sử dụng một điểm ngữ pháp cụ thể.
        """
        prompt_context = f" trong ngữ cảnh {context}" if context else ""
        prompt = f"""Cung cấp {count} câu ví dụ minh họa cách sử dụng điểm ngữ pháp sau: '{grammar_point}'{prompt_context}.
Các ví dụ nên rõ ràng, đa dạng và dễ hiểu.
Chỉ trả về danh sách các câu ví dụ, mỗi câu trên một dòng mới. Không cần giải thích thêm.
Ví dụ:
Ví dụ 1...
Ví dụ 2...
Ví dụ 3...
"""
        logger.info(f"[{self.__class__.__name__}] Providing {count} examples for '{grammar_point}' (context: {context})")
        
        response_parts = []
        async for chunk in self.ask(prompt):
            response_parts.append(chunk)
        full_response = "".join(response_parts)
        
        # Tách các ví dụ dựa trên xuống dòng
        examples = [ex.strip() for ex in full_response.split('\n') if ex.strip()]
        
        logger.info(f"[{self.__class__.__name__}] Provided examples: {examples}")
        return examples

    async def run(self, command: str, **kwargs) -> Any:
        """
        Chạy một lệnh cụ thể của GrammarAgent.
        - command: "explain_rule", "correct_text", "provide_examples"
        - kwargs: phụ thuộc vào lệnh
            - explain_rule: rule_name (str), level (str, optional)
            - correct_text: text (str), explain_errors (bool, optional, default True)
            - provide_examples: grammar_point (str), count (int, optional, default 3), context (str, optional)
        """
        logger.info(f"[{self.__class__.__name__}] Running command '{command}' with kwargs: {kwargs}")
        if command == "explain_rule":
            rule_name = kwargs.get("rule_name")
            if not rule_name:
                return "Lỗi: 'rule_name' là bắt buộc cho lệnh 'explain_rule'."
            level = kwargs.get("level", "intermediate")
            return await self.explain_grammar_rule(rule_name=rule_name, level=level)
        
        elif command == "correct_text":
            text_to_correct = kwargs.get("text")
            if not text_to_correct:
                return {"error": "Lỗi: 'text' là bắt buộc cho lệnh 'correct_text'."}
            explain = kwargs.get("explain_errors", True)
            return await self.correct_text(text=text_to_correct, explain_errors=explain)
            
        elif command == "provide_examples":
            grammar_point = kwargs.get("grammar_point")
            if not grammar_point:
                return "Lỗi: 'grammar_point' là bắt buộc cho lệnh 'provide_examples'."
            count = kwargs.get("count", 3)
            context = kwargs.get("context")
            return await self.provide_examples(grammar_point=grammar_point, count=count, context=context)
            
        else:
            logger.warning(f"[{self.__class__.__name__}] Unknown command: {command}")
            return f"Lệnh không xác định: {command}. Các lệnh hợp lệ: 'explain_rule', 'correct_text', 'provide_examples'."

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