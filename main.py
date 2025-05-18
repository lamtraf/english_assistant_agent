from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from pydantic import BaseModel
import asyncio
from typing import List, Annotated, Optional, Any, Dict
import logging
import os
import tempfile
import shutil
from datetime import datetime
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import agents
from agents.studyplan import StudyPlanAgent  
from agents.teacher import EnglishTeacherAgent  
from agents.speaking_agent import SpeakingPracticeAgent
from agents.vocabulary_agent import VocabularyAgent
from agents.reading_agent import ReadingComprehensionAgent
from agents.grammar_agent import GrammarAgent

# Import database
import database_manager

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI
app = FastAPI(
    title="English Learning Assistant API",
    description="API hỗ trợ học tiếng Anh với AI",
    version="1.0.0"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các origin trong môi trường development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo các agent
agents = {}

# Models
class BaseRequest(BaseModel):
    user_id: Annotated[str, Header(convert_underscores=True)] = "default_user_001"

class StudyPlanRequest(BaseRequest):
    goal: Optional[str] = None
    time_frame: Optional[str] = None
    current_level: Optional[str] = None
    daily_study_hours: Optional[float] = None
    focus_skills: Optional[List[str]] = None
    preferred_activities: Optional[List[str]] = None

class SpeakingPracticeRequest(BaseModel):
    current_level: Optional[str] = None
    preferred_topics: Optional[List[str]] = None
    practice_duration: Optional[int] = None
    focus_areas: Optional[List[str]] = None

class EnglishCorrectionRequest(BaseRequest):
    text: str

class TextAnalysisResponse(BaseModel):
    vocabulary: List[str]
    corrected_text: str
    explanation: str

class GrammarExplainRequest(BaseRequest):
    rule_name: str
    level: Optional[str] = "intermediate"

class GrammarCorrectRequest(BaseRequest):
    text: str
    explain_errors: Optional[bool] = True

class GrammarExamplesRequest(BaseRequest):
    grammar_point: str
    count: Optional[int] = 3
    context: Optional[str] = None

class VocabExplainRequest(BaseRequest):
    word: str

class VocabByTopicRequest(BaseRequest):
    topic: str
    difficulty: Optional[str] = "intermediate"
    count: Optional[int] = 3

class ReadingPassageRequest(BaseRequest):
    topic: str
    difficulty: Optional[str] = "intermediate"

class ReadingQuestionsRequest(BaseRequest):
    topic: str
    difficulty: Optional[str] = "intermediate"

class TextRequest(BaseModel):
    text: str

# Agent Factory
class AgentFactory:
    _instances = {}
    
    @classmethod
    def get_agent(cls, agent_type: str):
        if agent_type not in cls._instances:
            if agent_type == "speaking":
                cls._instances[agent_type] = SpeakingPracticeAgent(
                    whisper_model_name=os.getenv("WHISPER_MODEL_NAME", "tiny.en"),
                    gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest"),
                    gemini_streaming=True
                )
            elif agent_type == "study_plan":
                cls._instances[agent_type] = StudyPlanAgent(True)
            elif agent_type == "teacher":
                cls._instances[agent_type] = EnglishTeacherAgent(True)
            elif agent_type == "vocabulary":
                cls._instances[agent_type] = VocabularyAgent(True)
            elif agent_type == "reading":
                cls._instances[agent_type] = ReadingComprehensionAgent(True)
            elif agent_type == "grammar":
                cls._instances[agent_type] = GrammarAgent(True)
        return cls._instances[agent_type]

# Dependencies
async def get_agent(agent_type: str):
    agent = AgentFactory.get_agent(agent_type)
    if hasattr(agent, 'initialize'):
        await agent.initialize()
    return agent

async def get_study_plan_agent():
    return await get_agent("study_plan")

async def get_speaking_agent():
    return await get_agent("speaking")

async def get_vocabulary_agent():
    return await get_agent("vocabulary")

async def get_reading_agent():
    return await get_agent("reading")

async def get_grammar_agent():
    return await get_agent("grammar")

# Utility functions
def cleanup_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Đã xóa file tạm: {file_path}")
    except Exception as e:
        logger.error(f"Lỗi khi xóa file tạm {file_path}: {e}")

async def log_interaction(
    user_id: str,
    agent_name: str,
    user_input: Any,
    ai_response: Any,
    duration_ms: int,
    metadata: Optional[dict] = None
):
    await database_manager.log_interaction(
        user_id=user_id,
        agent_name=agent_name,
        user_input_type="text",
        user_input_content=user_input,
        ai_response_type="json_result",
        ai_response_content=ai_response,
        duration_ms=duration_ms,
        metadata=metadata
    )

def determine_agent_type(text: str) -> str:
    """
    Tự động xác định loại agent phù hợp dựa trên nội dung input
    """
    text = text.lower()
    
    # Các từ khóa để xác định loại agent
    study_plan_keywords = ["lộ trình", "kế hoạch", "học", "luyện", "trình độ", "thời gian", "giờ", "ngày", "tuần", "tháng"]
    grammar_keywords = ["ngữ pháp", "grammar", "cấu trúc", "câu", "thì", "tense", "sai", "lỗi", "sửa"]
    reading_keywords = ["đọc", "reading", "bài đọc", "đoạn văn", "passage", "comprehension"]
    vocabulary_keywords = ["từ vựng", "vocabulary", "từ", "word", "nghĩa", "meaning", "giải thích"]
    
    # Đếm số từ khóa xuất hiện cho mỗi loại
    study_plan_count = sum(1 for keyword in study_plan_keywords if keyword in text)
    grammar_count = sum(1 for keyword in grammar_keywords if keyword in text)
    reading_count = sum(1 for keyword in reading_keywords if keyword in text)
    vocabulary_count = sum(1 for keyword in vocabulary_keywords if keyword in text)
    
    # Tìm loại có nhiều từ khóa nhất
    counts = {
        "study_plan": study_plan_count,
        "grammar": grammar_count,
        "reading": reading_count,
        "vocabulary": vocabulary_count
    }
    
    # Nếu không có từ khóa nào phù hợp, mặc định là study_plan
    if all(count == 0 for count in counts.values()):
        return "study_plan"
        
    return max(counts.items(), key=lambda x: x[1])[0]

@app.on_event("startup")
async def startup_event():
    try:
        agents["study_plan"] = StudyPlanAgent(streaming=False)
        agents["grammar"] = GrammarAgent(streaming=False)
        agents["reading"] = ReadingComprehensionAgent(streaming=False)
        agents["vocabulary"] = VocabularyAgent(streaming=False)
        logger.info("Đã khởi tạo các agent thành công")
    except Exception as e:
        logger.error(f"Lỗi khởi tạo agent: {e}")
        raise

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to English Learning Assistant API"}

@app.post("/api/process")
async def process_text(request: TextRequest):
    """
    Xử lý văn bản với agent phù hợp nhất
    """
    try:
        # Xác định loại agent phù hợp
        agent_type = determine_agent_type(request.text)
        logger.info(f"Đã xác định loại agent: {agent_type}")
        
        if agent_type not in agents:
            raise HTTPException(
                status_code=503, 
                detail="Agent chưa được khởi tạo"
            )
            
        agent = agents[agent_type]
        result = await agent.run(request.text)
        
        # Thêm thông tin về loại agent đã sử dụng
        if isinstance(result, dict):
            result["agent_type"] = agent_type
            
        return result
    except Exception as e:
        logger.error(f"Error processing text with {agent_type} agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speak")
async def practice_speaking(
    audio_file: UploadFile = File(...),
    agent: SpeakingPracticeAgent = Depends(get_speaking_agent)
):
    if not agent or not agent.whisper_model or not agent.tts_engine:
        raise HTTPException(status_code=503, detail="Dịch vụ AI chưa sẵn sàng")

    input_temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename or ".wav")[1]) as tmp_in:
            shutil.copyfileobj(audio_file.file, tmp_in)
            input_temp_file_path = tmp_in.name

        ai_response_text, output_audio_path = await agent.run(input_temp_file_path)

        if output_audio_path and os.path.exists(output_audio_path):
            media_type = "audio/mpeg" if output_audio_path.lower().endswith(".mp3") else "audio/wav"
            cleanup_task = BackgroundTask(cleanup_file, output_audio_path)
            
            return FileResponse(
                path=output_audio_path,
                media_type=media_type,
                filename=os.path.basename(output_audio_path),
                headers={"X-AI-Response-Text": ai_response_text.replace("\n", " ")},
                background=cleanup_task
            )
        else:
            return JSONResponse(
                status_code=200,
                content={
                    "ai_text_response": ai_response_text,
                    "audio_response_status": "TTS failed or no audio generated"
                }
            )
    except Exception as e:
        logger.exception("Lỗi xử lý audio")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if input_temp_file_path:
            cleanup_file(input_temp_file_path)

@app.post("/vocabulary/explain")
async def vocabulary_explain_word(
    request: VocabExplainRequest,
    agent: VocabularyAgent = Depends(get_vocabulary_agent)
):
    start_time = datetime.now().timestamp()
    result = await agent.run(
        command="explain_word",
        word=request.word
    )
    end_time = datetime.now().timestamp()
    
    await log_interaction(
        request.user_id,
        "VocabularyAgent",
        request.word,
        result,
        int((end_time - start_time) * 1000)
    )
    
    return result

@app.post("/vocabulary/get_words_by_topic")
async def vocabulary_get_words_by_topic(
    request: VocabByTopicRequest,
    agent: VocabularyAgent = Depends(get_vocabulary_agent)
):
    start_time = datetime.now().timestamp()
    result = await agent.run(
        command="get_words_by_topic",
        topic=request.topic,
        difficulty=request.difficulty,
        count=request.count
    )
    end_time = datetime.now().timestamp()
    
    await log_interaction(
        request.user_id,
        "VocabularyAgent",
        request.dict(),
        result,
        int((end_time - start_time) * 1000)
    )
    
    return result

@app.post("/reading/passage")
async def reading_get_passage(
    request: ReadingPassageRequest,
    agent: ReadingComprehensionAgent = Depends(get_reading_agent)
):
    start_time = datetime.now().timestamp()
    result = await agent.run(
        command="generate_passage",
        topic=request.topic,
        difficulty=request.difficulty
    )
    end_time = datetime.now().timestamp()
    
    await log_interaction(
        request.user_id,
        "ReadingComprehensionAgent",
        request.dict(),
        result,
        int((end_time - start_time) * 1000)
    )
    
    return result

@app.post("/reading/questions")
async def reading_get_questions(
    request: ReadingQuestionsRequest,
    agent: ReadingComprehensionAgent = Depends(get_reading_agent)
):
    start_time = datetime.now().timestamp()
    result = await agent.run(
        command="generate_questions",
        topic=request.topic,
        difficulty=request.difficulty
    )
    end_time = datetime.now().timestamp()
    
    await log_interaction(
        request.user_id,
        "ReadingComprehensionAgent",
        request.dict(),
        result,
        int((end_time - start_time) * 1000)
    )
    
    return result

@app.post("/grammar/explain")
async def grammar_explain_rule(
    request: GrammarExplainRequest,
    agent: GrammarAgent = Depends(get_grammar_agent)
):
    start_time = datetime.now().timestamp()
    result = await agent.run(
        command="explain_rule",
        rule_name=request.rule_name,
        level=request.level
    )
    end_time = datetime.now().timestamp()
    
    await log_interaction(
        request.user_id,
        "GrammarAgent",
        request.dict(),
        result,
        int((end_time - start_time) * 1000)
    )
    
    return result

@app.post("/grammar/correct")
async def grammar_correct_text(
    request: GrammarCorrectRequest,
    agent: GrammarAgent = Depends(get_grammar_agent)
):
    start_time = datetime.now().timestamp()
    result = await agent.run(
        command="correct_text",
        text=request.text,
        explain_errors=request.explain_errors
    )
    end_time = datetime.now().timestamp()
    
    await log_interaction(
        request.user_id,
        "GrammarAgent",
        request.dict(),
        result,
        int((end_time - start_time) * 1000)
    )
    
    return result

@app.post("/grammar/examples")
async def grammar_provide_examples(
    request: GrammarExamplesRequest,
    agent: GrammarAgent = Depends(get_grammar_agent)
):
    start_time = datetime.now().timestamp()
    result = await agent.run(
        command="provide_examples",
        grammar_point=request.grammar_point,
        count=request.count,
        context=request.context
    )
    end_time = datetime.now().timestamp()
    
    await log_interaction(
        request.user_id,
        "GrammarAgent",
        request.dict(),
        result,
        int((end_time - start_time) * 1000)
    )
    
    return result

@app.get("/history/{user_id}")
async def get_user_history(user_id: str, limit: int = 20):
    return await database_manager.get_user_history(user_id, limit)

if __name__ == "__main__":
    logger.info("Để chạy ứng dụng FastAPI, sử dụng lệnh: uvicorn main:app --reload")
    logger.info("API docs sẽ có tại: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
