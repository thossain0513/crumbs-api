from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
import whisper
import os
import replicate
from typing import List, Dict, Any, Optional
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
import datetime
from src.audio_app import app as audio_app

model_name = "meta/meta-llama-3-70b-instruct"
image_model = "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"
os.environ["REPLICATE_API_TOKEN"] = "r8_dzkkqN96nqOQC7YhTNJ2gUam62Cu38z4aMu9S"
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI()


app = FastAPI()
app.mount("/audio", audio_app)


system_prompt = """
        You are a creative chef that can generate recipes in a JSON format. 
        Make it with the following structure exactly so that we can parse the string properly. 
        Make sure to only include ingredients that we use. Make sure to include quantities.
        You don't need to include all the ingredients to generate the recipe if you don't need to.
        Make sure to pay attention to correctly label the dietary identification of the recipe based on the ingredients used. Assume freedom with the spices.
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

@app.get("/health")
async def health_check():
    current_time = datetime.now().isoformat()
    return {"time": current_time}

#The following two functions are helper functions to help us parse our text data and turn into JSON format

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

class RecipeStorage(BaseModel): #Pydantic model to store recipe information
    name: str
    ingredients: List[str]
    description: str
    instructions: List[str]
    cuisine: str
    prepTime: str
    servings: int
    is_vegetarian: str
    is_vegan: str
    image: Optional[str] = None

@app.post("/generate_single_recipe", response_model=RecipeStorage)
async def generate_single_recipe(request: RecipeRequest, index: int):
    ingredients = request.ingredients
    is_vegan = request.is_vegan
    is_vegetarian = request.is_vegetarian

    temperature = 0.5
    top_p = 1
    max_tokens = 500

    prompt = f"Make recipe {index+1} for me with these ingredients: {ingredients}"
    if is_vegan:
        prompt = "I am vegan. " + prompt
    elif is_vegetarian:
        prompt = "I am vegetarian. " + prompt

    input = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
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

    recipe = await extract_json(fin_string)
    image = ImageRequest(recipeName = recipe['name'], recipeDescription = recipe['description'])
    recipe['image'] = generate_image(image)
    return RecipeStorage(**recipe)

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
    return output[0]


def extract_and_clean_text(input_string):
    # Regular expression pattern to find text between square brackets
    pattern = r'\[(.*?)\]'
    match = re.search(pattern, input_string)
    if match:
        extracted_text = match.group(1)  # Get the text between brackets
        cleaned_text = extracted_text.replace('"', '')  # Remove all double quotes
        return cleaned_text
    else:
        return " "  # Return None if no match is found

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
