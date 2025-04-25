import wave
import json
from vosk import Model, KaldiRecognizer
from tempfile import NamedTemporaryFile
import shutil

vosk_model = Model("models/vosk-model-small-en-us-0.15")

def transcribe_audio_vosk(file) -> str:
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    wf = wave.open(tmp_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        raise ValueError("Audio phải là mono PCM WAV")

    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    result = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            part = json.loads(rec.Result()).get("text", "")
            result += part + " "

    final = json.loads(rec.FinalResult()).get("text", "")
    result += final
    return result.strip()
