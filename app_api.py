from fastapi import FastAPI, File, UploadFile
from speech_to_text import transcribe_audio_vosk
from agents import handle_speaking

app = FastAPI()

@app.post("/speaking-full/")
async def full_speaking_workflow(file: UploadFile = File(...)):
    try:
        # 1. Chuyển giọng nói → văn bản
        transcript = transcribe_audio_vosk(file)

        # 2. Gửi đến agent feedback
        feedback = handle_speaking(transcript)

        return {
            "transcript": transcript,
            "feedback": feedback
        }

    except Exception as e:
        return {"error": str(e)}
