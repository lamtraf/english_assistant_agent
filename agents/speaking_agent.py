import asyncio
import logging
import os
import tempfile
import traceback
from typing import Tuple, Optional, List, Dict, Any

from .studyplan import BaseAgent # Giả sử BaseAgent nằm trong studyplan.py cùng thư mục agents

try:
    import whisper
except ImportError:
    print("Thư viện openai-whisper chưa được cài đặt. Vui lòng chạy: pip install openai-whisper")
    whisper = None

# Thay thế Piper TTS bằng pyttsx3
try:
    import pyttsx3
    pyttsx3_available = True
except ImportError:
    print("Thư viện pyttsx3 chưa được cài đặt. Vui lòng chạy: pip install pyttsx3")
    pyttsx3 = None 
    pyttsx3_available = False

logger = logging.getLogger(__name__)

class SpeakingPracticeAgent(BaseAgent):
    def __init__(
        self,
        whisper_model_name: str = "tiny.en", # Các model Whisper: tiny, base, small, medium, large
        gemini_model: str = "gemini-1.5-flash-latest",
        gemini_temperature: float = 0.7,
        gemini_streaming: bool = True,
        initial_greeting: Optional[str] = "Hello! What would you like to talk about today?",
        language_code: str = "en" # Mã ngôn ngữ cho Whisper và pyttsx3
    ):
        super().__init__(temperature=gemini_temperature, model=gemini_model, streaming=gemini_streaming)
        self.whisper_model_name = whisper_model_name
        self.whisper_model = None
        self.tts_engine = None
        self.language_code = language_code
        self.initial_greeting = initial_greeting

        # Lịch sử hội thoại được cấu trúc lại
        self.conversation_history: List[Dict[str, str]] = []
        if self.initial_greeting:
            self.conversation_history.append({"role": "model", "content": self.initial_greeting})

        self._load_whisper_model()
        self._initialize_tts()

    def _load_whisper_model(self):
        try:
            logger.info(f"Đang tải model Whisper: {self.whisper_model_name}...")
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            logger.info(f"Model Whisper '{self.whisper_model_name}' đã được tải thành công.")
        except Exception as e:
            logger.error(f"Lỗi khi tải model Whisper ({self.whisper_model_name}): {e}")
            logger.error("Chi tiết lỗi Whisper: " + traceback.format_exc())
            self.whisper_model = None
            # Gợi ý người dùng cài đặt ffmpeg nếu lỗi liên quan
            if "ffmpeg" in str(e).lower():
                logger.error("Lỗi này có thể do ffmpeg chưa được cài đặt hoặc không tìm thấy trong PATH.")
                logger.error("Vui lòng cài đặt ffmpeg: https://ffmpeg.org/download.html và đảm bảo nó có trong PATH.")

    def _initialize_tts(self):
        try:
            logger.info("Đang khởi tạo engine pyttsx3...")
            self.tts_engine = pyttsx3.init()
            
            # Cố gắng chọn giọng nói tiếng Anh
            voices = self.tts_engine.getProperty('voices')
            english_voice = None
            for voice in voices:
                if voice.languages:
                    # voice.languages là list, ví dụ [b'\x05en-us']
                    lang_str = voice.languages[0].decode('utf-8', errors='ignore') if isinstance(voice.languages[0], bytes) else str(voice.languages[0])
                    if self.language_code in lang_str.lower(): # So sánh với language_code (ví dụ 'en')
                        english_voice = voice
                        break
            
            if english_voice:
                logger.info(f"Đã tìm thấy giọng nói tiếng Anh: {english_voice.id}")
                self.tts_engine.setProperty('voice', english_voice.id)
            else:
                logger.warning(f"Không tìm thấy giọng nói tiếng Anh ({self.language_code}) cụ thể. Sử dụng giọng mặc định.")

            logger.info("Engine pyttsx3 đã khởi tạo thành công.")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo pyttsx3: {e}")
            logger.error("Chi tiết lỗi pyttsx3: " + traceback.format_exc())
            self.tts_engine = None

    async def speech_to_text(self, audio_file_path: str) -> Optional[str]:
        if not self.whisper_model:
            logger.error("Model Whisper chưa được tải. Không thể thực hiện Speech-to-Text.")
            return None
        try:
            logger.info(f"Đang chuyển đổi giọng nói thành văn bản từ file: {audio_file_path}")
            # result = self.whisper_model.transcribe(audio_file_path, language=self.language_code)
            # Whisper tự động phát hiện ngôn ngữ khá tốt, không cần chỉ định language code trừ khi muốn ép buộc
            result = await asyncio.to_thread(self.whisper_model.transcribe, audio_file_path, fp16=False) # fp16=False nếu chạy trên CPU
            text = result["text"]
            logger.info(f"Whisper - Văn bản nhận dạng được: {text}")
            return text.strip()
        except Exception as e:
            logger.error(f"Lỗi khi thực hiện Speech-to-Text với Whisper: {e}")
            logger.error("Chi tiết lỗi STT: " + traceback.format_exc())
            return None

    async def text_to_speech(self, text: str) -> Optional[str]:
        if not self.tts_engine:
            logger.error("Engine TTS (pyttsx3) chưa được khởi tạo. Không thể thực hiện Text-to-Speech.")
            return None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
                output_path = tmp_audio_file.name
            
            logger.info(f"Đang tổng hợp giọng nói cho văn bản: '{text[:50]}...' -> {output_path}")
            
            # pyttsx3.save_to_file là blocking, chạy trong executor
            await asyncio.to_thread(self.tts_engine.save_to_file, text, output_path)
            await asyncio.to_thread(self.tts_engine.runAndWait) # Cần runAndWait để đảm bảo file được ghi xong
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"TTS thành công. File audio đã được lưu tại: {output_path}")
                return output_path
            else:
                logger.error(f"TTS không thành công hoặc file audio trống: {output_path}")
                if os.path.exists(output_path): os.remove(output_path) # Xóa file trống nếu có
                return None
        except Exception as e:
            logger.error(f"Lỗi khi thực hiện Text-to-Speech với pyttsx3: {e}")
            logger.error("Chi tiết lỗi TTS: " + traceback.format_exc())
            if 'output_path' in locals() and os.path.exists(output_path):
                 try: os.remove(output_path) 
                 except: pass
            return None

    def _format_conversation_history_for_prompt(self) -> str:
        """Định dạng lịch sử hội thoại thành một chuỗi cho prompt của Gemini."""
        prompt_parts = []
        for message in self.conversation_history:
            role = "User" if message["role"] == "user" else "AI Assistant"
            prompt_parts.append(f"{role}: {message['content']}")
        return "\n".join(prompt_parts)

    async def run(self, user_audio_path: Optional[str] = None, user_text_input: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Xử lý đầu vào âm thanh hoặc văn bản của người dùng, tương tác với Gemini và trả về phản hồi.
        Trả về: (văn bản phản hồi từ AI, đường dẫn đến file audio TTS của AI nếu thành công)
        """
        if not self.whisper_model and user_audio_path:
            logger.error("Run: Model Whisper chưa sẵn sàng cho audio input.")
            return "Lỗi: Dịch vụ nhận dạng giọng nói chưa sẵn sàng.", None
        if not self.tts_engine:
            logger.warning("Run: Engine TTS chưa sẵn sàng. Sẽ chỉ trả về text.")

        user_input_text: Optional[str] = None

        if user_audio_path:
            user_input_text = await self.speech_to_text(user_audio_path)
            if not user_input_text:
                logger.error("Run: Không thể nhận dạng giọng nói từ file audio.")
                # Trả về lời chào ban đầu nếu STT thất bại và chưa có hội thoại
                if not any(msg['role'] == 'user' for msg in self.conversation_history) and self.initial_greeting:
                    ai_response_text = self.initial_greeting
                else:
                    ai_response_text = "Sorry, I couldn't understand what you said. Could you please try again?"
                
                # Thêm lỗi STT vào lịch sử như một lượt nói của AI để thông báo cho người dùng
                self.conversation_history.append({"role": "model", "content": ai_response_text})
                ai_audio_output_path = await self.text_to_speech(ai_response_text)
                return ai_response_text, ai_audio_output_path
        elif user_text_input:
            user_input_text = user_text_input.strip()
            logger.info(f"Run: Nhận đầu vào văn bản trực tiếp: {user_input_text}")
        
        if not user_input_text and not self.conversation_history:
             # Trường hợp gọi run() lần đầu mà không có input nào (ví dụ: để lấy lời chào đầu tiên)
            if self.initial_greeting:
                logger.info("Run: Không có input, trả về lời chào ban đầu.")
                ai_audio_output_path = await self.text_to_speech(self.initial_greeting)
                return self.initial_greeting, ai_audio_output_path
            else:
                return "Hello! How can I help you practice speaking today?", None # Fallback greeting
        elif not user_input_text and self.conversation_history:
            # Nếu có lịch sử nhưng không có input mới (ví dụ: lỗi STT ở trên đã xử lý)
            # thì không cần làm gì ở đây, vì lỗi đã được log và phản hồi.
            # Hoặc có thể là một lượt gọi trống không mong muốn.
            last_ai_message = self.conversation_history[-1]["content"] if self.conversation_history and self.conversation_history[-1]["role"] == "model" else "How can I assist you?"
            logger.warning("Run: Được gọi không có input mới nhưng có lịch sử. Trả về tin nhắn AI cuối cùng hoặc mặc định.")
            ai_audio_output_path = await self.text_to_speech(last_ai_message)
            return last_ai_message, ai_audio_output_path

        # Thêm lượt nói của người dùng vào lịch sử (nếu có user_input_text)
        if user_input_text:
            self.conversation_history.append({"role": "user", "content": user_input_text})

        # Tạo prompt cho Gemini dựa trên lịch sử hội thoại
        # System prompt để hướng dẫn Gemini cách hành xử
        system_prompt = ( # TODO: Tinh chỉnh system_prompt này
            "You are a friendly and patient English speaking practice partner. "
            "Your goal is to help the user practice speaking English by having a natural conversation. "
            "Listen carefully to what the user says. Respond naturally and try to keep the conversation flowing. "
            "If appropriate, ask follow-up questions. You can also suggest topics if the conversation lulls or if the user asks. "
            "Keep your responses relatively concise and suitable for a spoken conversation. Avoid very long paragraphs. "
            "If the user makes small grammar mistakes, don't correct them too strictly unless they ask or it severely hinders understanding. Focus on fluency and conversation."
            "If the user says something like 'I don\'t know' or seems stuck, gently encourage them or offer a new direction/topic."
        )
        
        # Lấy một phần lịch sử gần đây để tránh prompt quá dài (ví dụ: 10 lượt cuối)
        # Kết hợp system prompt với lịch sử hội thoại
        MAX_HISTORY_TURNS = 10 # Số lượt hội thoại (user + model) tối đa để đưa vào prompt
        recent_history = self.conversation_history[-(MAX_HISTORY_TURNS * 2):] # Mỗi lượt có 2 phần (user, model)
        
        # Formatted_history không bao gồm lời chào ban đầu nếu nó đã được xử lý
        # hoặc nếu system_prompt đã đủ để định hướng vai trò.
        # Hiện tại, _format_conversation_history_for_prompt sẽ bao gồm cả lời chào nếu nó có trong history.
        formatted_history = self._format_conversation_history_for_prompt() # Sẽ dùng toàn bộ history
                                                                       # Cân nhắc dùng recent_history nếu prompt quá dài

        # Prompt cuối cùng cho Gemini
        # Chúng ta muốn Gemini tiếp tục từ lượt nói của AI, nên prompt kết thúc bằng "AI Assistant:"
        # Hoặc, nếu để Gemini tự do hơn, chỉ cần đưa lịch sử vào.
        # Hiện tại, BaseAgent.ask sẽ tự động thêm prompt này vào cấu trúc JSON cho Gemini.
        # Nên chúng ta chỉ cần truyền nội dung cần Gemini hoàn thiện.
        
        # Tạo prompt hoàn chỉnh: system prompt + lịch sử + yêu cầu AI tiếp tục
        # Cấu trúc này giống với cách các model chat thường nhận input.
        # Tuy nhiên, Gemini API (model gemini-pro) có thể không xử lý "roles" một cách tường minh như model chat.
        # Nó sẽ coi toàn bộ là một chuỗi prompt lớn. Ta cần đảm bảo nó hiểu được luồng.
        
        # Phiên bản 1: Prompt đơn giản hơn, ghép lịch sử
        # final_prompt_to_gemini = f"{system_prompt}\n\nConversation History:\n{formatted_history}\n\nAI Assistant:"

        # Phiên bản 2: Sử dụng cấu trúc gần với messages của API chat (cần BaseAgent hỗ trợ hoặc điều chỉnh)
        # Hiện tại BaseAgent chỉ nhận 1 string prompt. Nên ta phải ghép lại.
        # Ta sẽ thử nghiệm với prompt ghép.

        # Tạo một context từ lịch sử để Gemini biết phải nói gì tiếp theo
        # Cần đảm bảo prompt này không quá dài và Gemini hiểu được vai trò của nó.
        if len(self.conversation_history) == 1 and self.conversation_history[0]["role"] == "model" and self.initial_greeting:
            # Đây là trường hợp user chưa nói gì, chỉ có lời chào ban đầu của AI.
            # Gemini không cần làm gì, vì lời chào đã được xử lý.
            # Tuy nhiên, nếu logic ở trên (không có user_input_text) đã xử lý, thì không vào đây.
            # Khối này có thể không cần thiết nếu logic đầu vào đã tốt.
            ai_response_text = self.initial_greeting
            logger.info("Run: Lượt đầu, chỉ có lời chào của AI. Không gọi Gemini.")
        else:
            # Prompt hướng dẫn Gemini tiếp tục cuộc hội thoại
            # formatted_history đã chứa lượt nói cuối của user
            prompt_for_gemini = (
                f"{system_prompt}\n\nHere is the conversation so far:\n"
                f"{formatted_history}\n\nNow, it's your turn to respond as the AI Assistant. Please continue the conversation naturally."
                f"AI Assistant:"
            )

            logger.info(f"Run: Gọi Gemini với prompt (đã bao gồm system prompt và history). Kích thước prompt: {len(prompt_for_gemini)} chars.")
            # logger.debug(f"Prompt to Gemini:\n{prompt_for_gemini}") # Log toàn bộ prompt nếu cần debug

            ai_response_parts = []
            async for chunk in self.ask(prompt_for_gemini): # self.ask từ BaseAgent
                ai_response_parts.append(chunk)
            ai_response_text = "".join(ai_response_parts).strip()

        if not ai_response_text:
            logger.warning("Run: Gemini không trả về nội dung nào. Sử dụng phản hồi mặc định.")
            ai_response_text = "I'm sorry, I didn't quite catch that. Could you say it again?"
        
        # Thêm phản hồi của AI vào lịch sử
        self.conversation_history.append({"role": "model", "content": ai_response_text})
        logger.info(f"Run: Phản hồi từ Gemini: {ai_response_text}")

        # TTS cho phản hồi của AI
        ai_audio_output_path = await self.text_to_speech(ai_response_text)

        return ai_response_text, ai_audio_output_path


# Test thử agent (cần có file audio mẫu)
async def main_test_speaking_agent():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Tạo một agent với lời chào tùy chỉnh
    agent = SpeakingPracticeAgent(
        whisper_model_name="tiny.en", 
        gemini_model="gemini-1.5-flash-latest",
        initial_greeting="Hi there! I am your English practice buddy. What shall we discuss?"
    )

    if not agent.whisper_model or not agent.tts_engine:
        logger.error("Không thể chạy test: Whisper model hoặc TTS engine chưa được khởi tạo.")
        return

    # 1. Lấy lời chào đầu tiên (không có input audio)
    print("\n--- Test 1: Lấy lời chào đầu tiên ---")
    ai_text, ai_audio = await agent.run()
    print(f"AI Chào: {ai_text}")
    if ai_audio:
        print(f"AI Audio Chào: {ai_audio} (Hãy nghe thử file này)")
        # os.remove(ai_audio) # Xóa file tạm sau khi nghe
    else:
        print("AI Audio Chào: Không có file audio được tạo.")
    print(f"Lịch sử hội thoại: {agent.conversation_history}")

    # 2. Người dùng nói (giả lập bằng file audio)
    print("\n--- Test 2: Người dùng nói lần đầu ---")
    # Tạo một file audio WAV giả để test STT
    # Bạn cần một file audio thực tế hoặc dùng thư viện để tạo file test động
    dummy_audio_content = b"RIFF..." # Đây chỉ là placeholder, bạn cần file WAV thật
    temp_user_audio_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
            # Ghi dữ liệu WAV hợp lệ vào đây nếu bạn có cách tạo động
            # Ví dụ đơn giản: copy từ một file mẫu nhỏ
            # Ở đây, chúng ta sẽ giả định file không hợp lệ để test nhánh lỗi STT trước
            # Hoặc bạn có thể tạo một file text và yêu cầu người dùng nhập text thay vì audio
            # tmp_f.write(dummy_audio_content) # Ghi nội dung WAV giả
            temp_user_audio_file = tmp_f.name
        
        # Ghi một file text "Hello AI" vào file đó để Whisper có thể đọc (cần file thực)
        # Nếu không có file audio thật, Whisper sẽ báo lỗi.
        # Thay vào đó, hãy chuẩn bị một file audio ngắn, ví dụ "test_audio.wav"
        test_audio_file_path = "test_input/test_audio.wav" # Đặt file audio mẫu vào đây
        
        if not os.path.exists(test_audio_file_path):
            logger.warning(f"File audio test '{test_audio_file_path}' không tồn tại. Bỏ qua test 2, 3.")
            # Test với text input thay thế
            print("\n--- Test 2.1: Người dùng nhập text lần đầu ---")
            user_text = "Hello there, I want to talk about my hobbies."
            ai_text, ai_audio = await agent.run(user_text_input=user_text)
            print(f"User nói (text): {user_text}")
            print(f"AI phản hồi: {ai_text}")
            if ai_audio: print(f"AI Audio: {ai_audio}") # else: print("AI Audio: Không có")
            print(f"Lịch sử hội thoại: {agent.conversation_history}")

        else:
            ai_text, ai_audio = await agent.run(user_audio_path=test_audio_file_path)
            # STT sẽ đọc từ file test_audio_file_path
            print(f"AI phản hồi: {ai_text}")
            if ai_audio: print(f"AI Audio: {ai_audio}") # else: print("AI Audio: Không có")
            print(f"Lịch sử hội thoại: {agent.conversation_history}")

            # 3. Người dùng nói tiếp
            print("\n--- Test 3: Người dùng nói tiếp (cần file audio khác hoặc cùng file) ---")
            # Giả sử người dùng nói "That sounds interesting. Tell me more about your hobbies."
            # Bạn cần một file audio khác "test_audio_2.wav" cho nội dung này
            test_audio_2_file_path = "test_input/test_audio_2.wav"
            if not os.path.exists(test_audio_2_file_path):
                logger.warning(f"File audio test '{test_audio_2_file_path}' không tồn tại. Bỏ qua test 3.")
                # Test với text input thay thế
                print("\n--- Test 3.1: Người dùng nhập text tiếp theo ---")
                user_text_2 = "That sounds fun. What kind of movies do you like?"
                ai_text, ai_audio = await agent.run(user_text_input=user_text_2)
                print(f"User nói (text): {user_text_2}")
                print(f"AI phản hồi: {ai_text}")
                if ai_audio: print(f"AI Audio: {ai_audio}")
                print(f"Lịch sử hội thoại: {agent.conversation_history}")
            else:
                ai_text, ai_audio = await agent.run(user_audio_path=test_audio_2_file_path)
                print(f"AI phản hồi: {ai_text}")
                if ai_audio: print(f"AI Audio: {ai_audio}")
                print(f"Lịch sử hội thoại: {agent.conversation_history}")

    except Exception as e:
        logger.error(f"Lỗi trong quá trình test: {e}")
        traceback.print_exc()
    finally:
        if temp_user_audio_file and os.path.exists(temp_user_audio_file):
            # os.remove(temp_user_audio_file) # Xóa file audio giả nếu được tạo
            pass # Để lại file để kiểm tra nếu cần
        # Dọn dẹp các file audio do AI tạo ra nếu chúng vẫn còn (thường là không vì FileResponse xử lý)
        # Hoặc nếu test trực tiếp không qua API.
        # for item in agent.conversation_history:
        #     if item["role"] == "model" and item.get("audio_path") and os.path.exists(item["audio_path"]):
        #         try: os.remove(item["audio_path"])
        #         except: pass
        pass # Dọn dẹp thủ công nếu cần khi test.

if __name__ == '__main__':
    # Để test, bạn cần tạo thư mục `test_input` và đặt 2 file audio mẫu vào đó:
    # - test_input/test_audio.wav (ví dụ: người dùng nói "Hello AI, I want to talk about movies")
    # - test_input/test_audio_2.wav (ví dụ: người dùng nói "That sounds interesting. What kind of movies do you recommend?")
    # Và đảm bảo whisper, ffmpeg, pyttsx3 hoạt động.
    os.makedirs("test_input", exist_ok=True)
    asyncio.run(main_test_speaking_agent()) 