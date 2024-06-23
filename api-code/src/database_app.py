from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import httpx
import os
from supabase import create_client, Client
import uvicorn
from pydantic import BaseModel
from typing import List
import logging
import requests
from src.helpers import transform_string
from io import BytesIO
import uuid
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(message)s')  # Format to include only the message

# Remove existing handlers to avoid duplicate logs in certain environments
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add a new StreamHandler with the specified format
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)


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
    image: str = None

class UploadImageRequest(BaseModel):
    image_url: str

# this function checks if the selected table at the selected column has the specified value
def check_cols(table_name, col_name, value):
    data, count = supabase.table(table_name).select(col_name).execute()
    return value in [item[col_name] for item in data[1]]

def make_bucket(name):
    if name not in [item.name for item in supabase.storage.list_buckets()]:
        supabase.storage.create_bucket(name = name, id = name)

def change_bucket(name, difference):
    if name not in [item.name for item in supabase.storage.list_buckets()]:
        supabase.storage.create_bucket(name = name, id = name)
    else:
        supabase.storage.update_bucket(name, difference)


def insert_recipe(request: RecipeStorage):
    recipe = {
        "name": request.name,
        "ingredients": request.ingredients,
        "description": request.description,
        "instructions": request.instructions,
        "cuisine": request.cuisine,
        "prepTime": request.prepTime,
        "servings": request.servings,
        "is_vegetarian": True if request.is_vegetarian.lower() == "yes" else False,
        "is_vegan": True if request.is_vegan.lower() == "yes" else False,
        "image": request.image
    }

    data, count = supabase.table('recipes').select("name").execute()
    logger.info(data[1])
    if check_cols('recipes', 'name', recipe['name']):
        return JSONResponse(content= {"message": "recipe already in database"})
    data, count = supabase.table('recipes') \
    .insert(recipe) \
    .execute()

    #Return the recipe id of the inserted recipe
    return data[1][0]['id']

class LikeRequest(BaseModel):
    recipe_id: str
    user_id: str

def curr_like(request: LikeRequest):
    pass

def add_like(request: LikeRequest):
     like_info = {
        "recipe_id": request.recipe_id,
        "user_id": request.user_id
     }
     data, count = supabase.table('user_liked_recipes') \
        .insert(like_info) \
        .execute()

async def upload_image(request: RecipeStorage):
    # Fetch the image from the URL
    image_url = request.image
    make_bucket("recipe_images")
    change_bucket("recipe_images", {"public": True})

    try:
        response = requests.get(image_url).content
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")

    file_path = f"public/{transform_string(request.name)}.png"
    image = BytesIO(response).read()

    # Upload the image to Supabase storage
    try:
        res = supabase.storage.from_("recipe_images").upload(file_path, image, file_options={"content-type": 'image/png'})
        logger.info("passed the storage input")
        if res.status_code != 200:
            raise HTTPException(status_code=400, detail="Error uploading image to Supabase")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    return supabase.storage.from_('recipe_images').get_public_url(file_path)

@app.post("/like_recipe")
async def like_recipe(request: RecipeStorage, user_id: str):
    """
    first, check if recipe exists in database. If not, insert it.
    Then, add liked recipe to curr_liked_recipes
    Then add to liked_recipes
    """
    recipe_id = ""
    if check_cols("recipes", "name", request.name):
        #This means that the recipe is already in the database
        data, count = supabase.table('recipes') \
        .select("id") \
        .eq('name', request.name) \
        .execute()
        #we need to set the recipe id to that existing id
        recipe_id = data[1][0]['id']
    else:
        recipe_id = insert_recipe(request)
    
    data, count = supabase.table('user_liked_recipes') \
    .insert({"user_id": user_id, "recipe_id": recipe_id}) \
    .execute()

    return JSONResponse(content={"message": "successful"})
    

    
    







    



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)