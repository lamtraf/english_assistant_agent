from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from typing import List
from agents.studyplan import StudyPlanAgent  
from agents.teacher import EnglishTeacherAgent  

# Khởi tạo FastAPI
app = FastAPI()

class StudyPlanRequest(BaseModel):
    text: str

class EnglishCorrectionRequest(BaseModel):
    text: str

class TextAnalysisResponse(BaseModel):
    vocabulary: List[str]
    corrected_text: str
    explanation: str

# Khởi tạo các agent
study_plan_agent = StudyPlanAgent(temperature=0.7, model="gemini-2.0-flash")
english_teacher_agent = EnglishTeacherAgent(temperature=0.7, model="gemini-2.0-flash")

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

# Chạy FastAPI server bằng Uvicorn (Chạy trong terminal)
# uvicorn main:app --reload
