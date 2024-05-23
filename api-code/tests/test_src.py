from fastapi.testclient import TestClient
import pytest
from unittest.mock import patch, MagicMock

from src.main import app

client = TestClient(app)

# Define the mock for the replicate.stream method
async def mock_replicate_stream(model_name, input):
    for response in mock_response:
        yield response

def test_generate_recipe():
    # Define the ingredients as a string
    ingredients = "feta cheese,olives,cucumber,tomato,oregano,lamb,pita bread,noodles,soy sauce,sesame oil,bell pepper,carrot,spring onion,ginger,garlic,chicken"
    
    # Send a POST request to the /generate_recipe endpoint
    response = client.post("/generate_recipe", json={"ingredients": ingredients})
    
    # Check if the status code is 200
    assert response.status_code == 200
    
    # Check if the response is a list of recipes
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3
    
    # Check if each recipe has the required keys
    for recipe in data:
        assert "name" in recipe
        assert isinstance(recipe['name'], str)

        assert "ingredients" in recipe
        assert isinstance(recipe['ingredients'], list)
        

        assert "description" in recipe
        assert isinstance(recipe['description'], str)
        

        assert "instructions" in recipe
        assert isinstance(recipe['instructions'], list)
        

        assert "cuisine" in recipe
        assert isinstance(recipe['cuisine'], str)
        

        assert "prepTime" in recipe
        assert isinstance(recipe['prepTime'], str)
        

        assert "servings" in recipe
        assert isinstance(recipe['servings'], int)
        

        assert "is_vegetarian" in recipe
        assert isinstance(recipe['is_vegetarian'], str)
        

        assert "is_vegan" in recipe
        assert isinstance(recipe['is_vegan'], str)

    

# Mock data to simulate the response from replicate.stream
mock_response = [
    MagicMock(text='{"name": "Greek Salad", "ingredients": ["feta cheese", "olives", "cucumber", "tomato"], "description": "A refreshing salad with feta and olives.", "instructions": ["Step 1: Chop vegetables", "Step 2: Mix ingredients"], "cuisine": "Greek", "prepTime": "10 minutes", "servings": 2, "is_vegetarian": "Yes", "is_vegan": "No"}')
]

# Test function for the fixer_agent
@pytest.mark.asyncio
@patch('src.main.replicate.stream', side_effect=mock_replicate_stream)  # Correct patch target
async def test_fixer_agent(mock_stream):
    fix_input = 'Invalid JSON string'
    response = client.post("/fix_recipe", json={"fix": fix_input, "counter": 0})
    
    expected_result = [
        {
            "name": "Greek Salad",
            "ingredients": ["feta cheese", "olives", "cucumber", "tomato"],
            "description": "A refreshing salad with feta and olives.",
            "instructions": ["Step 1: Chop vegetables", "Step 2: Mix ingredients"],
            "cuisine": "Greek",
            "prepTime": "10 minutes",
            "servings": 2,
            "is_vegetarian": "Yes",
            "is_vegan": "No"
        }
    ]
    
    assert response.status_code == 200
    data = response.json()
    assert data['counter'] == 1
    assert isinstance(data['result'], dict)
    mock_stream.assert_called_once()

@pytest.mark.asyncio
@patch('src.main.replicate.stream', side_effect=mock_replicate_stream)
async def test_fixer_agent_counter_exceed(mock_stream):
    fix_input = 'Invalid JSON string'
    response = client.post("/fix_recipe", json={"fix": fix_input, "counter": 3})
    
    assert response.status_code == 200
    data = response.json()
    assert data['result'] == {}
    assert data['counter'] == 3
    mock_stream.assert_not_called()
