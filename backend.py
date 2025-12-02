"""
Solis AI - STRICT MODE (no fuzzy corrections; deterministic validation)
- Loads schema directly from DB (no CSVs)
- Strict validation: reject any SQL that references unknown tables or columns
- Deterministically qualifies bare tables to DEFAULT_SCHEMA if the table exists
- Preserves all Flask routes from previous version
"""

import csv
import os
import re
import io
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
import requests
from config import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD, PG_SCHEMA as DEFAULT_SCHEMA
from pglast import parse_sql
import pglast.ast as ast  # For isinstance checks
import psycopg2
from psycopg2 import pool
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import Flask, render_template, request, jsonify, send_file
import difflib
from sentence_transformers import SentenceTransformer, util
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger("solis_ai")
logger.setLevel(logging.INFO)
file_handler = TimedRotatingFileHandler('app.log', when='midnight', backupCount=30)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
file_handler.setFormatter(file_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ---------------------------
# Flask app + limiter
# ---------------------------
app = Flask(__name__)
limiter = Limiter(key_func=get_remote_address, app=app, default_limits=["1000 per hour"], storage_uri="memory://")

# ---------------------------
# Postgres connection pool
# ---------------------------
pg_pool = None
try:
    pg_pool = psycopg2.pool.ThreadedConnectionPool(
        1, 20,
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
    )
    logger.info("PostgreSQL connection pool created.")
except Exception as e:
    logger.exception("Failed to create PostgreSQL pool: %s", e)
    raise

# LLM endpoint (Ollama) — updated for Mixtral-8x7B-Instruct
LLM_API_URL = "http://ollama:11434/v1/completions"
LLM_MODEL = "mixtral:8x7b"
MAX_LLM_TOKENS = 1024  # Increased further to handle potential large prompts
LLM_TIMEOUT_SECONDS = 120  # Increased for model loading/large prompts

# Overview / batching
DEFAULT_OVERVIEW_CHAR_LIMIT = 20000  # start size, will shrink on 400s

# Flask limiter
RATE_LIMIT = "20 per minute"

# ---------------------------
# Schema cache structures
# ---------------------------
schema_cache = {}        # table_name -> {'columns': list of (col_name, data_type), 'samples': list of rows}
schema_table_list = []   # sorted table names
SCHEMA_NAME = DEFAULT_SCHEMA

def load_schema_from_db():
    """Load schema information directly from Postgres information_schema (dynamic, no CSVs). Skip empty tables."""
    global schema_cache, schema_table_list
    schema_cache = {}
    parsed_tables = set()
    conn = None
    try:
        conn = pg_pool.getconn()
        with conn.cursor() as cur:
            logger.info("Fetching tables from information_schema for schema '%s'", SCHEMA_NAME)
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s AND table_type = 'BASE TABLE';
            """, (SCHEMA_NAME,))
            tables = cur.fetchall()
            schema_table_list = sorted([t[0] for t in tables])
            logger.info("Fetched %d tables total", len(schema_table_list))

            non_empty_count = 0
            for table in schema_table_list:
                # Check if table has data
                cur.execute(f"SELECT COUNT(*) FROM \"{SCHEMA_NAME}\".\"{table}\";")
                count = cur.fetchone()[0]
                if count == 0:
                    continue  # Skip without logging

                # Fetch columns
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position;
                """, (SCHEMA_NAME, table))
                columns = cur.fetchall()
                cols = [(row[0], row[1]) for row in columns]

                # Fetch 3 sample rows
                cur.execute(f"SELECT * FROM \"{SCHEMA_NAME}\".\"{table}\" LIMIT 3;")
                samples = cur.fetchall()

                parsed_tables.add(table)
                schema_cache[table] = {'columns': cols, 'samples': samples}
                non_empty_count += 1
            logger.info("Loaded metadata for %d non-empty tables", non_empty_count)
    except Exception as e:
        logger.exception("Failed to load schema from DB: %s", e)
        raise
    finally:
        if conn:
            pg_pool.putconn(conn)

load_schema_from_db()

# ---------------------------
# Helper functions
# ---------------------------
def filter_schema_for_query(query, max_tables=3):
    """Filter schema to relevant tables based on query keywords to reduce prompt length (improved with column matching)."""
    query_words = set(re.findall(r'\w+', query.lower()))
    table_scores = {}
    for table in schema_table_list:
        table_score = sum(1 for word in query_words if word in table.lower())
        col_score = sum(1 for word in query_words for col in schema_cache.get(table, {}).get('columns', []) if word in col[0].lower())
        score = table_score + col_score
        if score > 0:
            table_scores[table] = score
    sorted_tables = sorted(table_scores, key=table_scores.get, reverse=True)[:max_tables]
    filtered_schema = {t: schema_cache[t] for t in sorted_tables}
    logger.info("Filtered to %d relevant tables for query '%s': %s", len(filtered_schema), query[:50], ', '.join(filtered_schema))
    return filtered_schema

# Alternate: Embedding-based filter (uncomment after pip install sentence_transformers)
def filter_schema_for_query_embedding(query, max_tables=3):
    query_emb = embedder.encode(query.lower())
    table_embs = {t: embedder.encode(t.lower() + ' ' + ' '.join([c[0].lower() for c in schema_cache[t]['columns']])) for t in schema_table_list}
    scores = {t: util.cos_sim(query_emb, emb)[0][0].item() for t, emb in table_embs.items()}
    sorted_tables = sorted(scores, key=scores.get, reverse=True)[:max_tables]
    filtered_schema = {t: schema_cache[t] for t in sorted_tables}
    logger.info("Embedding-filtered to %d relevant tables for query '%s': %s", len(filtered_schema), query[:50], ', '.join(filtered_schema))
    return filtered_schema

def build_schema_str(schema_dict=None):
    """Build CREATE TABLE string for prompt from schema_dict (full or filtered), with samples."""
    if schema_dict is None:
        schema_dict = schema_cache
    schema_str = ""
    for table, info in sorted(schema_dict.items()):
        cols = info['columns']
        samples = info['samples']
        col_str = ", ".join([f'"{c[0]}" {c[1]}' for c in cols]) or "/* no columns defined */"
        sample_str = f"Sample data (3 rows): {json.dumps(samples, default=str)}" if samples else "No samples (table empty)"
        schema_str += f'CREATE TABLE {SCHEMA_NAME}."{table}" ({col_str});\n{sample_str}\n\n'
    logger.info("Built schema_str length: %d characters (with samples)", len(schema_str))
    return schema_str.strip()

def ask_llm(prompt, max_tokens=MAX_LLM_TOKENS, overview_budget=None):
    """Call LLM API (Ollama - completions endpoint for Mixtral)."""
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,  # Deterministic
            "top_p": 1.0,
        }
        logger.info("Sending prompt to Ollama (length: %d chars)", len(prompt))
        response = requests.post(LLM_API_URL, headers=headers, json=data, timeout=LLM_TIMEOUT_SECONDS)
        response.raise_for_status()
        resp_json = response.json()
        logger.info("Ollama response keys: %s", list(resp_json.keys()))
        completion = resp_json.get("choices", [{}])[0].get("text", "").strip()
        logger.info("Ollama raw output (truncated): %s", completion[:120])
        return completion
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            logger.error("Ollama 400 error - possible prompt too long or config issue: %s", e)
            return None
        logger.exception("Ollama call failed: %s", e)
        return None
    except Exception as e:
        logger.exception("Ollama call failed: %s", e)
        return None

def generate_sql(user_message):
    """Generate SQL from user message using LLM with improved prompt (metadata only, no examples)."""
    # Filter schema to reduce length
    filtered_schema = filter_schema_for_query_embedding(user_message)  # Use embedding for better accuracy
    schema_str = build_schema_str(filtered_schema)

    # Prompt with metadata only – no examples, for reliance on schema
    prompt = f"""Generate a PostgreSQL query to answer the question: {user_message}

Instructions (STRICT - FOLLOW EXACTLY):
- Use ONLY EXACT table and column names from the schema below, with exact capitalization and no alterations. Do not guess or change case (e.g., "Productfamily" NOT "ProductFamily").
- Always qualify tables with the schema '{SCHEMA_NAME}.' (e.g., {SCHEMA_NAME}."table_name").
- Use double quotes for ALL identifiers to preserve case (e.g., "column_name").
- Use ONLY tables and columns that EXIST in the schema – do not invent any.
- For aggregations (e.g., total, sum), use EXACT column names matching the intent (e.g., 'ChargeAmount' for charge-related).
- If the query is aggregate (e.g., SUM, COUNT), do NOT add LIMIT unless specified.
- For non-aggregate queries, add LIMIT 100 if no limit specified.
- Use sample data to understand formats/values, but do not hardcode samples in query.
- Output ONLY the SQL query – no explanations, no extra text, no tags.

Database schema (with samples for each table):
{schema_str}"""

    sql_output = ask_llm(prompt)
    logger.info("Generated raw SQL by model: %s", sql_output)  # Log raw for testing
    if not sql_output:
        return None

    # Clean SQL (remove any non-SQL text)
    sql = re.sub(r'[^SELECTFROMWHEREGROUPBYORDERBYLIMIT;*\"0-9a-zA-Z_\.,=\(\) ]', '', sql_output).strip()  # Basic clean
    if sql.endswith(';'):
        sql = sql[:-1].strip()
    logger.info("Generated cleaned SQL by model: %s", sql)  # Log cleaned for testing

    return sql

def validate_sql(sql):
    """Strict validation using pglast AST (no fuzzy). Qualify bare tables."""
    unknown_tables = []
    unknown_columns = []
    try:
        parsed = parse_sql(sql)
        for raw_stmt in parsed:
            stmt = raw_stmt.stmt
            if not stmt:
                continue

            # Check relations (tables)
            for node in stmt.fromClause or []:
                if isinstance(node, ast.RangeVar):
                    schema_name = node.schemaname or ''
                    rel_name = node.relname if node.relname else ''
                    if schema_name and schema_name.lower() != SCHEMA_NAME.lower():
                        unknown_tables.append(f"{schema_name}.{rel_name}")
                        continue
                    table_name = rel_name
                    if table_name not in schema_cache:
                        close_matches = difflib.get_close_matches(table_name, schema_table_list, n=3, cutoff=0.6)
                        unknown_tables.append(f"{table_name} (Did you mean: {', '.join(close_matches)}?)")
                    else:
                        # Qualify bare tables if no schema
                        if not node.schemaname:
                            sql = re.sub(r'\b' + re.escape(rel_name) + r'\b', f'"{SCHEMA_NAME}"."{rel_name}"', sql)

            # Check columns in targetList
            for target in stmt.targetList or []:
                if isinstance(target, ast.ResTarget):
                    val = target.val
                    if isinstance(val, ast.ColumnRef):
                        fields = val.fields  # list of ast.A_Star or ast.String
                        if len(fields) == 1:
                            continue  # Bare column
                        elif len(fields) >= 2:
                            table_ref_node = fields[-2]
                            col_node = fields[-1]
                            if isinstance(col_node, ast.String):
                                col_name = col_node.sval
                            else:
                                continue
                            if isinstance(table_ref_node, ast.String):
                                table_ref = table_ref_node.sval
                                if table_ref not in schema_cache:
                                    continue
                                cols = {c[0] for c in schema_cache[table_ref]['columns']}
                                if col_name not in cols:
                                    close_col_matches = difflib.get_close_matches(col_name, list(cols), n=3, cutoff=0.6)
                                    unknown_columns.append(f"{col_name} in {table_ref} (Did you mean: {', '.join(close_col_matches)}?)")

    except Exception as e:
        logger.exception("SQL parse failed: %s", e)
        return None, f"Parse error: {str(e)}"

    if unknown_tables or unknown_columns:
        error_msg = "Validation failed: Unknown table(s) referenced. Unknown tables: " + ", ".join(unknown_tables)
        if unknown_columns:
            error_msg += ". Unknown columns: " + ", ".join(unknown_columns)
        return None, error_msg
    return sql, None

def execute_sql(sql, params=None):
    """Execute SQL on Postgres (safe, read-only assumed)."""
    conn = None
    try:
        conn = pg_pool.getconn()
        with conn.cursor() as cur:
            logger.info("Executing query: %s; (params: %s)", sql, params)
            cur.execute(sql, params)
            if cur.description:
                cols = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                logger.info("Query returned %d rows. First row: %s", len(rows), rows[0] if rows else 'None')
                return cols, rows
            else:
                conn.commit()
                logger.info("Non-select query executed successfully.")
                return None, None
    except Exception as e:
        logger.exception("Query execution failed: %s", e)
        raise
    finally:
        if conn:
            pg_pool.putconn(conn)

def generate_csv(rows, cols):
    """Generate CSV from rows/cols."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(cols)
    writer.writerows(rows)
    return output

def generate_smart_suggestions(user_message, rows, cols, error):
    """Generate suggestions (stub)."""
    return ["Show table list", "Show columns of <table>", "Refresh schema"]

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_initial_suggestions", methods=["GET"])
@limiter.limit(RATE_LIMIT)
def get_initial_suggestions():
    """Initial suggestions for empty state."""
    suggestions = [
        {"icon": "📊", "text": "What is the total charge amount from invoice items?"},
        {"icon": "📈", "text": "Show me the top 5 products by revenue"},
        {"icon": "🔍", "text": "List all tables in the schema"},
        {"icon": "📋", "text": "Describe the zuora_QA_invoiceitem table"},
    ]
    return jsonify(suggestions)

@app.route("/show_table_list", methods=["GET"])
@limiter.limit(RATE_LIMIT)
def show_table_list():
    return jsonify({"tables": schema_table_list})

@app.route("/show_columns/<table>", methods=["GET"])
@limiter.limit(RATE_LIMIT)
def show_columns(table):
    if table not in schema_cache:
        return jsonify({"error": "Unknown table"}), 400
    cols = [{"name": c[0], "type": c[1]} for c in schema_cache[table]['columns']]
    return jsonify({"columns": cols})

@app.route("/chat", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def chat():
    data = request.json or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Handle special commands
    if user_message.lower() == "show table list":
        return jsonify({"type": "tables", "tables": schema_table_list})
    if user_message.lower().startswith("show columns of "):
        table = user_message.split("of ", 1)[-1].strip()
        if table not in schema_cache:
            return jsonify({"error": "Unknown table"}), 400
        cols = [{"name": c[0], "type": c[1]} for c in schema_cache[table]['columns']]
        return jsonify({"type": "columns", "table": table, "columns": cols})
    if user_message.lower() == "refresh schema":
        try:
            load_schema_from_db()
            return jsonify({"type": "text", "summary": f"Schema refreshed successfully from DB. Loaded {len(schema_table_list)} tables."})
        except Exception as e:
            return jsonify({"error": f"Failed to refresh schema from DB: {str(e)}"}), 500

    # Generate SQL
    sql = generate_sql(user_message)
    if not sql:
        return jsonify({"error": "Failed to generate SQL – possible prompt issue or LLM error. Try simpler query."}), 500

    # Validate SQL
    valid_sql, error = validate_sql(sql)
    if not valid_sql:
        suggestions = generate_smart_suggestions(user_message, [], [], error)
        return jsonify({"error": error, "suggestions": suggestions}), 400

    # Execute SQL (no LIMIT for aggregate – check if SELECT without FROM or simple SUM/COUNT)
    response = {"message_id": str(datetime.now().timestamp())}
    conn = None
    is_aggregate = 'SUM' in sql.upper() or 'COUNT' in sql.upper() or 'AVG' in sql.upper() or 'MIN' in sql.upper() or 'MAX' in sql.upper()
    exec_sql = valid_sql if is_aggregate or "LIMIT" in valid_sql.upper() else valid_sql + " LIMIT 100"
    try:
        cols, rows = execute_sql(exec_sql)
    except Exception as e:
        response["error"] = f"Query runtime error: {str(e)}"
        return jsonify(response), 500
    finally:
        if conn:
            pg_pool.putconn(conn)

    if not rows:
        response["type"] = "no_data"
        response["summary"] = "Query executed successfully but returned no rows. The table might be empty or no matching data."
        response["suggestions"] = ["Show table list", "Show columns of zuora_QA_invoiceitemadjustment", "Refresh schema"]
        return jsonify(response)

    # Summarize results via LLM
    sample_display = rows[:3]
    summary_prompt = f"Summarize the results (1-2 short sentences). Question: {user_message}\nColumns: {', '.join(cols[:8])}\nRows returned: {len(rows)}\nSample: {sample_display}"
    summary = ask_llm(summary_prompt, max_tokens=80)

    response["type"] = "data"
    response["summary"] = summary or f"Found {len(rows)} rows."
    response["data"] = {"columns": cols, "rows": rows, "total_rows": len(rows)}  # Send all rows for download, UI shows limited
    response["suggestions"] = generate_smart_suggestions(user_message, rows, cols, "")

    logger.info("Served SQL for user: %s (SQL: %s)", user_message[:120], exec_sql)
    return jsonify(response)

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json or {}
    message_id = data.get("message_id", "")
    rating = (data.get("rating") or "").strip()
    feedback_text = (data.get("feedback_text") or "").strip()
    user_question = data.get("user_question", "")
    sql_query = data.get("sql_query", "")
    session_id = data.get("session_id", "")
    if not message_id:
        return jsonify({"error": "Invalid message id"}), 400
    if not rating and not feedback_text:
        return jsonify({"error": "Empty feedback"}), 400

    # Store feedback (no schema tuning in strict mode)
    conn = None
    try:
        conn = pg_pool.getconn()
        with conn.cursor() as cur:
            # Create table if not exists (safe, schema-locked)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {DEFAULT_SCHEMA}.user_feedback (
                    id SERIAL PRIMARY KEY,
                    feedback_id VARCHAR(100) NOT NULL,
                    user_question TEXT,
                    sql_query TEXT,
                    rating VARCHAR(20),
                    feedback_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id VARCHAR(100),
                    message_id VARCHAR(100)
                );
            """)
            cur.execute(f"""
                INSERT INTO {DEFAULT_SCHEMA}.user_feedback
                (feedback_id, message_id, user_question, sql_query, rating, feedback_text, session_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (message_id, message_id, user_question, sql_query, rating, feedback_text, session_id))
            conn.commit()
    except Exception as e:
        logger.exception("Failed to store feedback: %s", e)
        return jsonify({"error": "Failed to store feedback"}), 500
    finally:
        if conn:
            pg_pool.putconn(conn)
    return jsonify({"message": "Thank you for your feedback!"})

@app.route("/download_csv", methods=["POST"])
def download_csv():
    data = request.json or {}
    cols = data.get("columns") or []
    rows = data.get("rows") or []
    if not cols or not rows:
        return jsonify({"error": "no data"}), 400
    csv_io = generate_csv(rows, cols)
    csv_io.seek(0)
    return send_file(io.BytesIO(csv_io.getvalue().encode()), mimetype="text/csv",
                     as_attachment=True,
                     download_name=f"solis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

@app.route("/visualize", methods=["POST"])
def visualize():
    data = request.json or {}
    chart_type = data.get("chart_type", "bar")
    rows = data.get("rows", [])
    cols = data.get("cols", [])
    if not rows or not cols:
        return jsonify({"error": "No data provided"}), 400
    try:
        numeric_indices = []
        text_indices = []
        for idx, col in enumerate(cols):
            sample_val = rows[0][idx] if rows else None
            if isinstance(sample_val, (int, float)):
                numeric_indices.append(idx)
            else:
                text_indices.append(idx)
        label_idx = text_indices[0] if text_indices else 0
        value_idx = numeric_indices[0] if numeric_indices else (1 if len(cols) > 1 else 0)
        labels = [str(row[label_idx]) for row in rows[:50]]
        values = []
        for row in rows[:50]:
            try:
                values.append(float(row[value_idx]) if row[value_idx] is not None else 0)
            except:
                values.append(0)
        chart_data = {"labels": labels, "datasets": [{"label": cols[value_idx], "data": values}], "chart_type": chart_type, "title": f"{cols[value_idx]} by {cols[label_idx]}"}
        return jsonify(chart_data)
    except Exception as e:
        logger.exception("Visualization failed: %s", e)
        return jsonify({"error": "Failed to generate visualization"}), 500

# ---------------------------
# Startup
# ---------------------------
if __name__ == "__main__":
    logger.info("Starting solis_ai STRICT MODE app with schema '%s'", SCHEMA_NAME)
    logger.info("Loaded %d tables into cache from DB for schema '%s'.", len(schema_table_list), SCHEMA_NAME)
    app.run(host="0.0.0.0", port=5000, debug=False)
