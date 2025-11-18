# backend.py
import requests
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DB, PG_SCHEMA

app = FastAPI()

LLM_URL = "http://localhost:8000/v1/chat/completions"  # Docker runs locally on EC2

class QueryInput(BaseModel):
    user_message: str

def get_postgres_data(query):
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB
    )
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result

@app.post("/chat")
def chat_with_llm(input_data: QueryInput):
    try:
        payload = {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "messages": [
                {"role": "user", "content": input_data.user_message}
            ]
        }
        response = requests.post(LLM_URL, json=payload)
        response.raise_for_status()
        reply = response.json()['choices'][0]['message']['content']
        return {"llm_reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
