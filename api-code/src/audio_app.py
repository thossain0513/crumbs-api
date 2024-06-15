from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import shutil
import os

app = FastAPI()

# Load your Whisper model (ensure the model is loaded outside the request scope to save resources)
model = whisper.load_model("base")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file part")

    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    # Save the audio file temporarily
    audio_path = "/tmp/audio_to_transcribe.wav"
    try:
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Use Whisper to transcribe the audio
        result = model.transcribe(audio_path)
        transcript = result['text']

    finally:
        # Ensure the file is deleted after processing
        if os.path.exists(audio_path):
            os.remove(audio_path)

    # Return the transcription
    return JSONResponse(content={'transcript': transcript})