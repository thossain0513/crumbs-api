from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
import datetime
from src.audio_app import app as audio_app
from src.helpers import extract_and_clean_text
from src.recipe import app as recipe_app
from src.database_app import app as database_app
from dotenv import load_dotenv


image_model = "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"

app = FastAPI()
app.mount("/audio", audio_app)
app.mount("/recipe", recipe_app)
app.mount("/database", database_app)
model_name = "meta/meta-llama-3-70b-instruct"


@app.get("/health")
async def health_check():
    current_time = datetime.now().isoformat()
    return {"time": current_time}
        
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
