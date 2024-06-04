from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
import whisper
import os
import replicate
from typing import List, Dict, Any
import json
from langchain_community.llms import Replicate
import shutil
from pydantic import BaseModel
import aiohttp
import asyncio
import io
import uvicorn
from openai import OpenAI
from PIL import Image
import re
import base64

model_name = "meta/meta-llama-3-70b-instruct"
image_model = "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"
os.environ["REPLICATE_API_TOKEN"] = "r8_dzkkqN96nqOQC7YhTNJ2gUam62Cu38z4aMu9S"
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI()
app = FastAPI()

# Load your Whisper model (ensure the model is loaded outside the request scope to save resources)
model = whisper.load_model("base")

system_prompt = """
        You are a creative chef that can generate recipes in a JSON format. 
        Make it with the following structure exactly so that we can parse the string properly. 
        Make sure to only include ingredients that we use. Make sure to include quantities.
        You don't need to include all the ingredients to generate the recipe if you don't need to.
        Make sure to pay attention to correctly label the dietary identification of the recipe based on the ingredients used.
        You can only use raw food ingredients that are provided to you in the prompt. Assume freedom with the spices.
        You will only recommend recipes strictly in this following format with no inconsistencies so that a python function can properly parse it as a dictionary:
        
        { "name": "Name of the Recipe" (Python String),
         "ingredients": ["ingredient", "ingredient"] (a list of strings with their quantities),  
        "description": "A brief description, incorporating the listed ingredients with their quantities" (a string), 
        "instructions": ["Step 1: Description with Quantity of Used Ingredient 1", "Step 2: Description with Quantity of Used Ingredient 2"] (a list of strings), 
        "cuisine": "Appropriate Cuisine Type" (a string), 
        "prepTime": "prep time (in minutes)" (a string),
        "servings": servings (integer),
        "is_vegetarian": "Yes" if there are no animal-based ingredients/ "No" if there are animal-based ingredients (a string),
        "is_vegan": "Yes" if there are only vegan-compliant ingredients/"No" if there are one or more non-vegan ingredients (a string)
        }
"""

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

#The following two functions are helper functions to help us parse our text data and turn into JSON format

async def extract_jsons(string, counter=0):
    jsons = []
    while '{' in string and '}' in string:
        start_index = string.index('{')
        end_index = string.index('}') + 1
        json_string = string[start_index:end_index].replace('\n', "").replace("\t", "")
        try:
            jsons.append(json.loads(json_string))
        except ValueError:
            # Handle invalid JSON
            print(f"Invalid JSON: {json_string}")  # Use fixer LLM agent to deal with invalid inputs, THIS IS KEY
            if counter < 1:  # Ensure we only attempt fixing once more
                fixed_json = await fixer_agent(json_string, counter + 1)
                jsons.append(fixed_json)
        string = string[end_index:]
    return jsons

async def extract_json(string, counter=0):
    if '{' in string and '}' in string:
        start_index = string.index('{')
        end_index = string.index('}') + 1
        json_string = string[start_index:end_index].replace('\n', "").replace("\t", "")
        try:
            return json.loads(json_string)
        except ValueError:
            # Handle invalid JSON
            print(f"Invalid JSON: {json_string}")  # Use fixer LLM agent to deal with invalid inputs, THIS IS KEY
            if counter < 3:  # Ensure we only attempt fixing once more
                fixed_json = await fixer_agent(FixRequest(json_string, counter + 1))
                return fixed_json
    return {}  # Return empty dictionary if no valid JSON found

    
class RecipeRequest(BaseModel):
    ingredients: str
    is_vegetarian: bool = False
    is_vegan: bool = False

@app.post("/generate_recipe")
async def generate_recipe(request: RecipeRequest):
    ingredients = request.ingredients
    is_vegan = request.is_vegan
    is_vegetarian = request.is_vegetarian

    another = False  # If we want more responses, we show the previous recipes so it generates different ones
    temperature = 0.5
    previous_response = []
    num_recipes = 3
    top_p = 1
    max_tokens = 1500

    if another:
        prompt = f"""Here are my ingredients: {ingredients}.
            Here are the previous {num_recipes} recipes you generated: {[recipe['name'] for recipe in previous_response]}.
            Generate another {num_recipes} recipes for me based on the previous conversation."""
    else:
        prompt = f"""Make {num_recipes} recipes for me with these ingredients: {ingredients}"""

    if is_vegan and is_vegetarian:
        prompt = "I am vegetarian and vegan. " + prompt
    elif is_vegetarian:
        prompt = "I am vegetarian. " + prompt
    elif is_vegan:
        prompt = "I am vegan. " + prompt
    input = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    try:
        response = client.chat.completions.create(**input)
        fin_string = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    recipes = await extract_jsons(fin_string)
    print(recipes)
    return recipes

class FixRequest(BaseModel):
    fix: str
    counter: int = 0

class FixResponse(BaseModel):
    result: Any
    counter: int

@app.post("/fix_recipe", response_model=FixResponse)
async def fixer_agent(request: FixRequest):
    if request.counter > 2:
        return FixResponse(result={}, counter=request.counter)

    temperature = 0.1
    top_p = 1
    max_tokens = 1500

    fixer_prompt = f"""
    You are a fixer LLM Agent. I will give you strings that can't be parsed into JSON purely from their string format,
    I want only the JSON format with no commented out parts and perfectly parsable using json.loads. I want them in a JSON format strictly in the following
    structure so that I can use Python to parse it into a JSON format the following way:

    {{ "name": "Name of the Recipe" (Python String),
         "ingredients": ["ingredient", "ingredient"] (a list of strings),  
        "description": "A brief description, incorporating the listed ingredients with their quantities" (a string), 
        "instructions": ["Step 1: Description with Quantity of Used Ingredient 1", "Step 2: Description with Quantity of Used Ingredient 2"] (a list of strings), 
        "cuisine": "Appropriate Cuisine Type" (a string), 
        "prepTime": "prep time (in minutes)" (a string),
        "servings": servings (integer),
        "is_vegetarian": "Yes" if there are no animal-based ingredients/ "No" if there are animal-based ingredients (a string),
        "is_vegan": "Yes" if there are only vegan-compliant ingredients/"No" if there are one or more non-vegan ingredients (a string)
    }}
    """

    user_prompt = f"""Here is the string that I want fixed:
    {request.fix}"""

    input = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "system", "content": fixer_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    try:
        response = client.chat.completions.create(**input)
        fin_string = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = await extract_json(fin_string, request.counter + 1)
    return FixResponse(result=result, counter=request.counter + 1)

class ImageRequest(BaseModel):
    recipeName: str
    recipeDescription: str

@app.post("/generate_image")
def generate_image(request: ImageRequest):
    prompt = f"{request.recipeName}: {request.recipeDescription}"
    input = {
    "prompt": prompt,
    "scheduler": "K_EULER"
    }

    output = replicate.run(
        image_model,
        input=input
    )
    return output


def extract_contents_from_brackets(response_text):
    pattern = re.compile(r'\[(.*?)\]')
    contents = pattern.findall(response_text)
    cleaned_contents = [','.join(re.findall(r'[a-zA-Z ]+', content)) for content in contents]
    cleaned_contents = [content for content in cleaned_contents if content and re.search(r'[a-zA-Z]', content)]

    return cleaned_contents

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
                    "text": "What ingredients are in this image?"
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

    # Extract the contents from the response within square brackets
    ingredients_list = extract_contents_from_brackets(response.choices[0].message.content)

    return {"ingredients": ingredients_list}
        
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
