from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import os
import replicate
from typing import List, Dict, Any
import json
from pydantic import BaseModel
import shutil
from openai import OpenAI
import re
import uvicorn
import asyncio

system_prompt = """
        You are a creative chef that can generate recipes in a JSON format. 
        Make it with the following structure exactly so that we can parse the string properly. 
        Make sure to only include ingredients that we use. Make sure to include quantities.
        You don't need to include all the ingredients to generate the recipe if you don't need to.
        Make sure to pay attention to correctly label the dietary identification of the recipe based on the ingredients used.
        You can only use raw food ingredients that are provided to you in the prompt. Assume freedom with the spices.
        Make sure the recipes are reasonable to make and are edible.
        You will only recommend recipes strictly in this following format with no inconsistencies so that a python function 
        can properly parse it as a dictionary:
        
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

fixer_prompt = """
    You are a fixer LLM Agent. I will give you strings that can't be parsed into JSON purely from their string format,
    I want only the JSON format with no commented out parts and perfectly parsable using json.loads. I want them in a JSON format strictly in the following
    structure so that I can use Python to parse it into a JSON format the following way:

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


app = FastAPI()


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
    image: str





if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)