# crumbs-api

This is the FastAPI for the AI-generated portion of the Crumbs app. To run this on your local machine, create a .env file with a REPLICATE_API_TOKEN and an OPENAI_API_KEY.
Run this api on your local machine with the following call on your terminal:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload


