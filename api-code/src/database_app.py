from fastapi import FastAPI, HTTPException
import httpx
from io import BytesIO
from PIL import Image
import os
from supabase import create_client, Client
import uvicorn
from pydantic import BaseModel
from typing import List
import logging
import requests
from src.helpers import transform_string
from io import BytesIO


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

url = "https://bwnhavvhadbgqhldazvs.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ3bmhhdnZoYWRiZ3FobGRhenZzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcxNjc1NTk1NSwiZXhwIjoyMDMyMzMxOTU1fQ.Y25tTkfBdQgFhkbIJhdK9QApXUP3EZhAL6Xka0p_7xM"
supabase: Client = create_client(url, key)
os.environ["REPLICATE_API_TOKEN"] = "r8_dzkkqN96nqOQC7YhTNJ2gUam62Cu38z4aMu9S"


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

class UploadImageRequest(BaseModel):
    image_url: str

def make_bucket(name):
    logger.info(list(supabase.storage.list_buckets()))
    if name not in [item.name for item in supabase.storage.list_buckets()]:
        supabase.storage.create_bucket(name = name, id = name)
    logger.info(f"buckets: {supabase.storage.list_buckets()}")

def change_bucket(name, difference):
    logger.info(list(supabase.storage.list_buckets()))
    if name not in [item.name for item in supabase.storage.list_buckets()]:
        supabase.storage.create_bucket(name = name, id = name)
        supabase.storage.update_bucket(name, difference)
    logger.info(f"buckets: {supabase.storage.list_buckets()}")

@app.post("/upload_image")
async def upload_image(request: RecipeStorage):
    # Fetch the image from the URL
    image_url = request.image
    make_bucket("recipe_images")
    change_bucket("recipe_images", {"public": True})

    logger.info(f"image url: {image_url}")
    try:
        response = requests.get(image_url).content
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")

   
    file_path = f"public/{transform_string(request.name)}.png"

    # Upload the image to Supabase storage
    try:
        res = supabase.storage.from_("recipe_images").upload(file_path, BytesIO(response).read(), file_options={"content-type": 'image/png'})
        logger.info("passed the storage input")
        if res.status_code != 200:
            raise HTTPException(status_code=400, detail="Error uploading image to Supabase")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    return {"message": "Image uploaded successfully", "file_path": file_path}




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)