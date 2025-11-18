import requests
import psycopg2
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DB, PG_SCHEMA

app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

LLM_URL = "http://localhost:8000/v1/chat/completions"  # LLM API endpoint

class QueryInput(BaseModel):
    user_message: str

def run_sql_query(query):
    """Run SQL query on Postgres and return results as list of tuples."""
    logging.info(f"Executing SQL Query:\n{query}")
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            dbname=PG_DB
        )
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        logging.error(f"SQL Execution failed: {e}")
        return None

def generate_sql(prompt):
    """
    Use LLM to translate natural language prompt into an SQL query.
    """
    sql_prompt = (
        f"You are a helpful assistant that translates natural language requests into "
        f"PostgreSQL queries. The schema name is '{PG_SCHEMA}'.\n\n"
        f"User question: \"{prompt}\"\n\n"
        f"Generate a safe, accurate, and optimized SQL query for this. Return only SQL."
    )
    llm_payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": sql_prompt}]
    }
    response = requests.post(LLM_URL, json=llm_payload)
    response.raise_for_status()
    sql_query = response.json()['choices'][0]['message']['content']
    return sql_query.strip("` \n")

def summarize_data_for_llm(data):
    """
    Format data summary string for LLM prompt.
    """
    if not data:
        return "No data found."
    # Show first few rows (limit=5)
    head_rows = data[:5]
    summary_lines = [str(row) for row in head_rows]
    return "\n".join(summary_lines)

@app.post("/chat")
def chat(input_data: QueryInput):
    try:
        user_question = input_data.user_message
        logging.info(f"User Question: {user_question}")

        # Step 1: Translate natural language question to SQL using LLM
        sql_query = generate_sql(user_question)
        logging.info(f"Generated SQL:\n{sql_query}")

        # Step 2: Execute generated SQL on Postgres
        data = run_sql_query(sql_query)

        # Step 3: Summarize data for LLM context
        data_summary = summarize_data_for_llm(data)

        # Compose final prompt combining user question and data summary for LLM answer generation
        llm_answer_prompt = (
            f"User question: {user_question}\n"
            f"Database query result summary:\n{data_summary}\n\n"
            f"Provide a concise, accurate answer based on data above."
        )
        llm_payload = {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "messages": [{"role": "user", "content": llm_answer_prompt}]
        }
        llm_response = requests.post(LLM_URL, json=llm_payload)
        llm_response.raise_for_status()
        answer = llm_response.json()['choices'][0]['message']['content']

        # Prepare data for frontend visualization
        columns = []
        if data:
            # Fetch columns names from Postgres info schema for the executed query's first table
            conn = psycopg2.connect(
                host=PG_HOST,
                port=PG_PORT,
                user=PG_USER,
                password=PG_PASSWORD,
                dbname=PG_DB
            )
            cur = conn.cursor()
            cur.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = '{PG_SCHEMA}'
                ORDER BY ordinal_position
                LIMIT {len(data[0])};
            """)
            columns = [row[0] for row in cur.fetchall()]
            cur.close()
            conn.close()

        # Return answer and data table for frontend
        return {
            "answer": answer,
            "table_data": {
                "columns": columns,
                "rows": data
            },
            "recommended_questions": [
                "Show top 5 revenues as table",
                "Display sales trend chart",
                "What is total profit last month?",
                "List top customers by sales"
            ]
        }

    except Exception as e:
        logging.error(f"Error in /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
