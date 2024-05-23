from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import os
import replicate
from typing import List, Dict, Any
import json
from langchain_community.llms import Replicate
import shutil
from pydantic import BaseModel


app = FastAPI()
model_name = "meta/meta-llama-3-70b-instruct"
os.environ["REPLICATE_API_TOKEN"] = "r8_dzkkqN96nqOQC7YhTNJ2gUam62Cu38z4aMu9S"

# Load your Whisper model (ensure the model is loaded outside the request scope to save resources)
model = whisper.load_model("base")

system_prompt = """
        You are a creative chef that can generate recipes in a JSON format. 
        Make it with the following structure exactly so that we can parse the string properly. 
        Make sure to only include ingredients that we use. 
        You don't need to include all the ingredients to generate the recipe if you don't need to.
        Make sure to pay attention to correctly label the dietary identification of the recipe based on the ingredients used.
        You can only use raw food ingredients that are provided to you in the prompt. Assume freedom with the spices.
        You will only recommend recipes strictly in this following format with no inconsistencies so that a python function can properly parse it as a dictionary:
        
        { "name": "Name of the Recipe" (Python String),
         "ingredients": ["ingredient", "ingredient"] (a list of strings),  
        "description": "A brief description, incorporating the listed ingredients with their quantities" (a string), 
        "instructions": ["Step 1: Description with Quantity of Used Ingredient 1", "Step 2: Description with Quantity of Used Ingredient 2"] (a list of strings), 
        "cuisine": "Appropriate Cuisine Type" (a string), 
        "prepTime": "prep time (in minutes)" (a string),
        "servings": servings (integer),
        "is_vegetarian": "Yes" if there are no animal-based ingredients/ "No" if there are animal-based ingredients (a string),
        "is_vegan": "Yes" if there are only vegan-compliant ingredients/"No" if there are one or more non-vegan ingredients (a string)
        }
"""
class RecipeRequest(BaseModel):
    ingredients: str

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
                fixed_json = await fixer_agent(json_string, counter + 1)
                return fixed_json
    return {}  # Return empty dictionary if no valid JSON found

    
@app.post("/generate_recipe")
async def generate_recipe(request: RecipeRequest):
    ingredients = request.ingredients

    another = False  # If we want more responses, we show the previous recipes so it generates different ones
    temperature = 0.5
    previous_response = []
    num_recipes = 3
    is_vegetarian = False
    is_vegan = False
    top_p = 1
    max_tokens = 1500

    if another:
        prompt = f"""Here are my ingredients. {ingredients}.
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
        "top_p": top_p,
        "prompt": system_prompt + prompt,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "frequency_penalty": 0.5,
        "repetition_penalty": 1,
        "max_tokens": max_tokens,
    }

    fin_string = ""
    for event in replicate.stream(model_name, input=input):
        fin_string += str(event)

    recipes = await extract_jsons(fin_string)
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
    
    Here is the string that I want fixed:
    {request.fix}
    """

    input = {
        "top_p": top_p,
        "prompt": fixer_prompt,
        "temperature": temperature,
        "system_prompt": fixer_prompt,
        "frequency_penalty": 0.5,
        "repetition_penalty": 1,
        "max_tokens": max_tokens,
    }

    fin_string = ""
    async for event in replicate.stream(model_name, input=input):
        fin_string += str(event)

    result = await extract_json(fin_string, request.counter + 1)
    return FixResponse(result=result, counter=request.counter + 1)

