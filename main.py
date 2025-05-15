from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from pydantic import BaseModel
import asyncio
from typing import List, Annotated
from agents.studyplan import StudyPlanAgent  
from agents.teacher import EnglishTeacherAgent  
import logging
import os
import tempfile
import shutil
from typing import Optional
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from agents.speaking_agent import SpeakingPracticeAgent
from agents.vocabulary_agent import VocabularyAgent
from agents.reading_agent import ReadingComprehensionAgent
from agents.grammar_agent import GrammarAgent
import database_manager
from datetime import datetime
import traceback

# Khởi tạo FastAPI
app = FastAPI(
    title="Speaking Practice API",
    description="API để luyện nói tiếng Anh với AI Agent.",
    version="0.1.0"
)

class StudyPlanRequest(BaseModel):
    text: str

class EnglishCorrectionRequest(BaseModel):
    text: str

class TextAnalysisResponse(BaseModel):
    vocabulary: List[str]
    corrected_text: str
    explanation: str

# Pydantic models cho GrammarAgent
class GrammarExplainRequest(BaseModel):
    rule_name: str
    level: Optional[str] = "intermediate"

class GrammarCorrectRequest(BaseModel):
    text: str
    explain_errors: Optional[bool] = True

class GrammarExamplesRequest(BaseModel):
    grammar_point: str
    count: Optional[int] = 3
    context: Optional[str] = None

# Cấu hình logging cơ bản
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo SpeakingPracticeAgent
# Bạn có thể muốn cấu hình các tham số này từ biến môi trường hoặc file config
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "tiny.en")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")

# Khởi tạo agent một lần khi ứng dụng khởi động
try:
    speaking_agent = SpeakingPracticeAgent(
        whisper_model_name=WHISPER_MODEL_NAME,
        gemini_model=GEMINI_MODEL,
        gemini_streaming=True # Nên để True để có phản hồi nhanh từ Gemini
    )
    logger.info(f"SpeakingPracticeAgent đã khởi tạo thành công với Whisper model: {WHISPER_MODEL_NAME} và Gemini model: {GEMINI_MODEL}")
    if not speaking_agent.whisper_model:
        logger.warning("Model Whisper không được tải. STT có thể không hoạt động.")
    if not speaking_agent.tts_engine:
        logger.warning("Engine pyttsx3 không được khởi tạo. TTS có thể không hoạt động.")
except Exception as e:
    logger.exception("LỖI NGHIÊM TRỌNG: Không thể khởi tạo SpeakingPracticeAgent!")
    speaking_agent = None # Đặt là None để kiểm tra trong endpoint

study_plan_agent = StudyPlanAgent(model=GEMINI_MODEL, streaming=False) # Giả sử study plan không cần streaming
english_teacher_agent = EnglishTeacherAgent(model=GEMINI_MODEL, streaming=False) # Tương tự teacher agent
vocabulary_agent = VocabularyAgent(model=GEMINI_MODEL, streaming=False)
reading_agent = ReadingComprehensionAgent(model=GEMINI_MODEL, streaming=False)
grammar_agent = GrammarAgent(model=GEMINI_MODEL, streaming=False) # Khởi tạo GrammarAgent

logger.info("Tất cả các agent đã được khởi tạo.")

@app.post("/generate_study_plan")
async def generate_study_plan(request: StudyPlanRequest):
    user_input = request.text
    study_plan = await study_plan_agent.run(user_input)   
    
    return {"study_plan": study_plan}

@app.post("/correct_english")
async def correct_english(request: EnglishCorrectionRequest):
    """
    Endpoint sửa ngữ pháp và cải thiện văn bản tiếng Anh.
    """
    user_input = request.text
    result = await english_teacher_agent.run(user_input)
    
    return {"result": result}

# Hàm tiện ích để dọn dẹp file
def cleanup_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Đã xóa file tạm: {file_path}")
    except Exception as e:
        logger.error(f"Lỗi khi xóa file tạm {file_path}: {e}")

@app.post("/speak/", 
          summary="Luyện nói với AI",
          description="Upload một file audio (ví dụ WAV, MP3) chứa giọng nói của bạn. AI sẽ phản hồi bằng văn bản và một file audio (nếu TTS thành công).",
          responses={
              200: {
                  "description": "Phản hồi audio từ AI. Văn bản phản hồi nằm trong header 'X-AI-Response-Text'.",
                  "content": {"audio/mpeg": {}, "audio/wav": {}} # Có thể là mp3 hoặc wav tùy pyttsx3
              },
              400: {"description": "Lỗi đầu vào (ví dụ: không có file audio, agent chưa sẵn sàng)"},
              500: {"description": "Lỗi xử lý phía server"}
          }
)
async def practice_speaking(audio_file: UploadFile = File(...) ):
    if not speaking_agent:
        logger.error("Endpoint /speak/: SpeakingPracticeAgent chưa được khởi tạo.")
        raise HTTPException(status_code=503, detail="Dịch vụ AI hiện không sẵn sàng. Vui lòng thử lại sau.")
    if not speaking_agent.whisper_model or not speaking_agent.tts_engine:
        logger.error("Endpoint /speak/: Model Whisper hoặc engine TTS chưa được tải/khởi tạo đầy đủ.")
        raise HTTPException(status_code=503, detail="Dịch vụ AI chưa sẵn sàng hoàn toàn (STT/TTS). Vui lòng thử lại sau.")

    if not audio_file:
        raise HTTPException(status_code=400, detail="Không có file audio nào được upload.")

    logger.info(f"Endpoint /speak/: Đã nhận file: {audio_file.filename}, content_type: {audio_file.content_type}")

    # Tạo file tạm để lưu audio upload
    # Whisper có thể xử lý nhiều định dạng, nhưng WAV là an toàn nhất.
    # Đuôi file không quá quan trọng nếu Whisper có thể tự nhận dạng.
    # Chúng ta sẽ dùng NamedTemporaryFile để dễ quản lý.
    input_temp_file = None
    output_audio_path: Optional[str] = None

    try:
        # Lưu file upload vào file tạm
        # suffix giữ lại đuôi file gốc nếu có, hoặc đặt một đuôi mặc định
        original_suffix = os.path.splitext(audio_file.filename or ".wav")[1] if audio_file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix, prefix="user_audio_") as tmp_in:
            shutil.copyfileobj(audio_file.file, tmp_in)
            input_temp_file_path = tmp_in.name
        logger.info(f"File audio của người dùng đã được lưu tạm tại: {input_temp_file_path}")

        # Gọi agent để xử lý
        # Lưu ý: speaking_agent.run bây giờ là async
        ai_response_text, output_audio_path = await speaking_agent.run(input_temp_file_path)
        logger.info(f"Agent đã xử lý: text='{ai_response_text}', audio_path='{output_audio_path}'")

        if output_audio_path and os.path.exists(output_audio_path):
            # Trả về file audio, text nằm trong header
            # Client cần kiểm tra header 'X-AI-Response-Text'
            # Xác định content_type cho file audio trả về (pyttsx3 có thể ra mp3 hoặc wav)
            media_type = "audio/mpeg" if output_audio_path.lower().endswith(".mp3") else "audio/wav"
            
            # Tạo background task để xóa file output_audio_path sau khi gửi
            cleanup_task = BackgroundTask(cleanup_file, output_audio_path)
            
            # FileResponse sẽ tự đọc file và stream. Output_audio_path là file tạm do agent tạo.
            return FileResponse(
                path=output_audio_path, 
                media_type=media_type, 
                filename=os.path.basename(output_audio_path), # Gợi ý tên file cho client
                headers={"X-AI-Response-Text": ai_response_text.replace("\n", " ")}, # Header không nên có newline
                background=cleanup_task
            )
        else:
            # TTS không thành công hoặc không tạo ra file
            logger.warning("TTS không thành công hoặc không tạo ra file audio. Trả về JSON.")
            return JSONResponse(
                status_code=200, # Vẫn là 200 OK vì STT và Gemini vẫn có thể thành công
                content={
                    "ai_text_response": ai_response_text,
                    "audio_response_status": "TTS failed or no audio generated.",
                    "detail": "AI processed the request, but audio generation was unsuccessful."
                }
            )

    except HTTPException: # Re-raise HTTPException để FastAPI xử lý
        raise
    except Exception as e:
        logger.exception("Endpoint /speak/: Đã có lỗi xảy ra trong quá trình xử lý.")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ: {str(e)}")
    finally:
        # Dọn dẹp file audio input tạm
        if input_temp_file_path and os.path.exists(input_temp_file_path):
            cleanup_file(input_temp_file_path)
        # File output audio (nếu được tạo và không được gửi qua FileResponse với background task)
        # đã được xử lý bởi BackgroundTask của FileResponse hoặc logic bên trên nếu TTS fail.

@app.get("/", summary="Endpoint gốc", description="Endpoint cơ bản để kiểm tra API có hoạt động không.")
async def read_root():
    return {"message": "Chào mừng bạn đến với API Luyện Nói Tiếng Anh! Hãy thử POST file audio đến /speak/"}

# Để chạy ứng dụng này:
# 1. Mở terminal.
# 2. cd đến thư mục chứa file main.py này.
# 3. Chạy lệnh: uvicorn main:app --reload
#    --reload sẽ tự động tải lại server khi bạn thay đổi code.
# 4. Truy cập API tại http://127.0.0.1:8000 (hoặc http://localhost:8000)
#    Docs tự động sẽ có tại http://127.0.0.1:8000/docs

if __name__ == "__main__":
    # Phần này thường không được dùng khi chạy với uvicorn, nhưng hữu ích để debug.
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("Để chạy ứng dụng FastAPI, sử dụng lệnh: uvicorn main:app --reload")
    logger.info("API docs sẽ có tại: http://localhost:8000/docs")

# Chạy FastAPI server bằng Uvicorn (Chạy trong terminal)
# uvicorn main:app --reload

@app.post("/english_teacher/correct", response_model=TextAnalysisResponse)
async def correct_english_text(
    request: EnglishCorrectionRequest,
    user_id: Annotated[str, Header(convert_underscores=True)] = "default_user_001"
):
    start_time = datetime.now().timestamp()
    try:
        correction_result = await english_teacher_agent.run(
            command="correct_text",
            text=request.text,
            explain_errors=True
        )
        end_time = datetime.now().timestamp()
        database_manager.log_interaction(
            user_id=user_id,
            agent_name="EnglishTeacherAgent",
            user_input_type="text",
            user_input_content=request.text,
            ai_response_type="json_result",
            ai_response_content=correction_result,
            duration_ms=int((end_time - start_time) * 1000),
            metadata={"model": english_teacher_agent.model}
        )
        if isinstance(correction_result, dict) and correction_result.get("error"):
            raise HTTPException(status_code=400, detail=correction_result.get("error"))
        if isinstance(correction_result, dict) and "Error:" in correction_result.get("corrected_text", ""):
            # Lỗi từ LLM không parse được JSON hoặc không đúng cấu trúc
            raise HTTPException(status_code=502, detail=correction_result.get("corrected_text"))
        return correction_result
    except Exception as e:
        logger.error(f"Lỗi endpoint /english_teacher/correct: {e}\\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ khi sửa văn bản: {str(e)}")

# Endpoint cho VocabularyAgent
class VocabExplainRequest(BaseModel):
    word: str

class VocabByTopicRequest(BaseModel):
    topic: str
    difficulty: Optional[str] = "intermediate"
    count: Optional[int] = 3

@app.post("/vocabulary/explain", summary="Giải thích từ vựng")
async def vocabulary_explain_word(
    request: VocabExplainRequest,
    user_id: Annotated[str, Header(convert_underscores=True)] = "default_user_001"
):
    start_time = datetime.now().timestamp()
    try:
        explanation = await vocabulary_agent.run(
            command="explain_word",
            word=request.word
        )
        end_time = datetime.now().timestamp()
        database_manager.log_interaction(
            user_id=user_id,
            agent_name="VocabularyAgent",
            user_input_type="json_params",
            user_input_content=request.model_dump(),
            ai_response_type="text",
            ai_response_content=explanation,
            duration_ms=int((end_time - start_time) * 1000),
            metadata={"model": vocabulary_agent.model, "command": "explain_word"}
        )
        if isinstance(explanation, str) and explanation.startswith("Lỗi:"):
            raise HTTPException(status_code=400, detail=explanation)
        return {"word": request.word, "explanation": explanation}
    except Exception as e:
        logger.error(f"Lỗi endpoint /vocabulary/explain: {e}\\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ khi giải thích từ vựng: {str(e)}")

@app.post("/vocabulary/get_words_by_topic", summary="Lấy danh sách từ vựng theo chủ đề")
async def vocabulary_get_words_by_topic(
    request: VocabByTopicRequest,
    user_id: Annotated[str, Header(convert_underscores=True)] = "default_user_001"
):
    start_time = datetime.now().timestamp()
    try:
        result = await vocabulary_agent.run(
            command="get_words_by_topic",
            topic=request.topic,
            difficulty=request.difficulty,
            count=request.count
        )
        end_time = datetime.now().timestamp()
        database_manager.log_interaction(
            user_id=user_id,
            agent_name="VocabularyAgent",
            user_input_type="json_params",
            user_input_content=request.model_dump(),
            ai_response_type="json_result",
            ai_response_content=result,
            duration_ms=int((end_time - start_time) * 1000),
            metadata={"model": vocabulary_agent.model, "command": "get_words_by_topic"}
        )
        if isinstance(result, str) and result.startswith("Lỗi:"):
            raise HTTPException(status_code=400, detail=result)
        return {"topic": request.topic, "words": result}
    except Exception as e:
        logger.error(f"Lỗi endpoint /vocabulary/get_words_by_topic: {e}\\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ khi lấy danh sách từ vựng: {str(e)}")

# Endpoints cho ReadingComprehensionAgent

class ReadingPassageRequest(BaseModel):
    topic: str
    difficulty: Optional[str] = "intermediate"

class ReadingQuestionsRequest(BaseModel):
    topic: str
    difficulty: Optional[str] = "intermediate"

@app.post("/reading/passage", summary="Lấy đoạn văn bản đọc được")
async def reading_get_passage(
    request: ReadingPassageRequest,
    user_id: Annotated[str, Header(convert_underscores=True)] = "default_user_001"
):
    start_time = datetime.now().timestamp()
    try:
        passage = await reading_agent.run(
            command="generate_reading_passage",
            topic=request.topic,
            difficulty=request.difficulty
        )
        end_time = datetime.now().timestamp()
        database_manager.log_interaction(
            user_id=user_id,
            agent_name="ReadingComprehensionAgent",
            user_input_type="json_params",
            user_input_content=request.model_dump(),
            ai_response_type="text",
            ai_response_content=passage,
            duration_ms=int((end_time - start_time) * 1000),
            metadata={"model": reading_agent.model, "command": "generate_reading_passage"}
        )
        if isinstance(passage, str) and passage.startswith("Lỗi:"):
            raise HTTPException(status_code=400, detail=passage)
        return {"passage": passage}
    except Exception as e:
        logger.error(f"Lỗi endpoint /reading/passage: {e}\\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ khi lấy đoạn văn bản: {str(e)}")

@app.post("/reading/questions", summary="Lấy câu hỏi để đánh giá đọc được")
async def reading_get_questions(
    request: ReadingQuestionsRequest,
    user_id: Annotated[str, Header(convert_underscores=True)] = "default_user_001"
):
    start_time = datetime.now().timestamp()
    try:
        questions = await reading_agent.run(
            command="generate_comprehension_questions",
            topic=request.topic,
            difficulty=request.difficulty
        )
        end_time = datetime.now().timestamp()
        database_manager.log_interaction(
            user_id=user_id,
            agent_name="ReadingComprehensionAgent",
            user_input_type="json_params",
            user_input_content=request.model_dump(),
            ai_response_type="json_result",
            ai_response_content=questions,
            duration_ms=int((end_time - start_time) * 1000),
            metadata={"model": reading_agent.model, "command": "generate_comprehension_questions"}
        )
        if isinstance(questions, str) and questions.startswith("Lỗi:"):
            raise HTTPException(status_code=400, detail=questions)
        return {"questions": questions}
    except Exception as e:
        logger.error(f"Lỗi endpoint /reading/questions: {e}\\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ khi lấy câu hỏi: {str(e)}")

@app.post("/reading/summary", summary="Tóm tắt đoạn văn bản")
async def reading_summarize_passage(
    request: ReadingPassageRequest,
    user_id: Annotated[str, Header(convert_underscores=True)] = "default_user_001"
):
    start_time = datetime.now().timestamp()
    try:
        summary = await reading_agent.run(
            command="summarize_passage",
            topic=request.topic,
            difficulty=request.difficulty
        )
        end_time = datetime.now().timestamp()
        database_manager.log_interaction(
            user_id=user_id,
            agent_name="ReadingComprehensionAgent",
            user_input_type="json_params",
            user_input_content=request.model_dump(),
            ai_response_type="text",
            ai_response_content=summary,
            duration_ms=int((end_time - start_time) * 1000),
            metadata={"model": reading_agent.model, "command": "summarize_passage"}
        )
        if isinstance(summary, str) and summary.startswith("Lỗi:"):
            raise HTTPException(status_code=400, detail=summary)
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Lỗi endpoint /reading/summary: {e}\\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ khi tóm tắt đoạn văn bản: {str(e)}")

# --- Endpoints cho GrammarAgent ---
@app.post("/grammar/explain", summary="Giải thích quy tắc ngữ pháp")
async def grammar_explain_rule(
    request: GrammarExplainRequest,
    user_id: Annotated[str, Header(convert_underscores=True)] = "default_user_001"
):
    start_time = datetime.now().timestamp()
    try:
        explanation = await grammar_agent.run(
            command="explain_rule",
            rule_name=request.rule_name,
            level=request.level
        )
        end_time = datetime.now().timestamp()
        database_manager.log_interaction(
            user_id=user_id,
            agent_name="GrammarAgent",
            user_input_type="json_params",
            user_input_content=request.model_dump(),
            ai_response_type="text",
            ai_response_content=explanation,
            duration_ms=int((end_time - start_time) * 1000),
            metadata={"model": grammar_agent.model, "command": "explain_rule"}
        )
        if isinstance(explanation, str) and explanation.startswith("Lỗi:"):
             raise HTTPException(status_code=400, detail=explanation)
        return {"rule_name": request.rule_name, "explanation": explanation}
    except Exception as e:
        logger.error(f"Lỗi endpoint /grammar/explain: {e}\\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ khi giải thích ngữ pháp: {str(e)}")

@app.post("/grammar/correct", summary="Sửa lỗi ngữ pháp trong văn bản")
async def grammar_correct_text(
    request: GrammarCorrectRequest,
    user_id: Annotated[str, Header(convert_underscores=True)] = "default_user_001"
):
    start_time = datetime.now().timestamp()
    try:
        correction_result = await grammar_agent.run(
            command="correct_text",
            text=request.text,
            explain_errors=request.explain_errors
        )
        end_time = datetime.now().timestamp()
        database_manager.log_interaction(
            user_id=user_id,
            agent_name="GrammarAgent",
            user_input_type="json_params",
            user_input_content=request.model_dump(),
            ai_response_type="json_result", # Agent trả về dict
            ai_response_content=correction_result,
            duration_ms=int((end_time - start_time) * 1000),
            metadata={"model": grammar_agent.model, "command": "correct_text"}
        )
        if isinstance(correction_result, dict) and correction_result.get("error"):\
            raise HTTPException(status_code=400, detail=correction_result.get("error"))
        if isinstance(correction_result, dict) and "Error:" in correction_result.get("corrected_text", ""):
             # Lỗi từ LLM không parse được JSON hoặc không đúng cấu trúc
            raise HTTPException(status_code=502, detail=correction_result.get("corrected_text"))
        return correction_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi endpoint /grammar/correct: {e}\\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ khi sửa văn bản: {str(e)}")

@app.post("/grammar/examples", summary="Cung cấp ví dụ ngữ pháp")
async def grammar_provide_examples(
    request: GrammarExamplesRequest,
    user_id: Annotated[str, Header(convert_underscores=True)] = "default_user_001"
):
    start_time = datetime.now().timestamp()
    try:
        examples = await grammar_agent.run(
            command="provide_examples",
            grammar_point=request.grammar_point,
            count=request.count,
            context=request.context
        )
        end_time = datetime.now().timestamp()
        database_manager.log_interaction(
            user_id=user_id,
            agent_name="GrammarAgent",
            user_input_type="json_params",
            user_input_content=request.model_dump(),
            ai_response_type="list_of_strings", # Agent trả về list
            ai_response_content=examples,
            duration_ms=int((end_time - start_time) * 1000),
            metadata={"model": grammar_agent.model, "command": "provide_examples"}
        )
        if isinstance(examples, str) and examples.startswith("Lỗi:"):
            raise HTTPException(status_code=400, detail=examples)
        return {"grammar_point": request.grammar_point, "examples": examples}
    except Exception as e:
        logger.error(f"Lỗi endpoint /grammar/examples: {e}\\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ khi cung cấp ví dụ: {str(e)}")

# --- Kết thúc Endpoints cho GrammarAgent ---

@app.get("/history/{target_user_id}", summary="Lấy lịch sử tương tác của người dùng")
async def get_user_interaction_history(
    target_user_id: str,
    user_id: Annotated[str, Header(convert_underscores=True)] = "default_user_001"
):
    # Implementation of the endpoint
    # This is a placeholder and should be replaced with the actual implementation
    # using database_manager.get_user_interaction_history(target_user_id)
    return {"message": "This endpoint is not implemented yet."}
