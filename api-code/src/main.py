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

image_model = "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"
os.environ["REPLICATE_API_TOKEN"] = "r8_dzkkqN96nqOQC7YhTNJ2gUam62Cu38z4aMu9S"
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI()


app = FastAPI()
app.mount("/audio", audio_app)
app.mount("/recipe", recipe_app)


@app.get("/health")
async def health_check():
    current_time = datetime.now().isoformat()
    return {"time": current_time}


class PhotoInput(BaseModel):
    file: str  # Assuming this will be a base64 string or a URL to the image

@app.post("/analyze")
async def analyze(photo: PhotoInput):
    analyzer_prompt = """
    You identify food ingredients in the provided images. You list them in a python string list format so it can be easily parsed. 
    If there are no food ingredients in the provided image, return an empty list.
    """


    # Define the input parameters for the API call
    input_params = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": analyzer_prompt},
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": "What food ingredients are in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": photo.file
                    }
                }
            ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    # Make OpenAI API call for image analysis using GPT-4V
    response = client.chat.completions.create(**input_params)
    return {"ingredients": extract_and_clean_text(response.choices[0].message.content)}
        
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
