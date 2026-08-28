import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(
        dbname = os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )


def execute_query(query: str, parameters=None):
    connection = get_connection()
    connection.set_session(readonly=True)
    cursor = connection.cursor()
    
    try:
        query = query.strip()

        cursor.execute(query, parameters)

        columns = [description[0] for description in cursor.description]

        rows = cursor.fetchall()
        
        result = [
            dict(zip(columns, row))
            for row in rows
            
         ]
    
        return result
    
    finally:
        cursor.close()
        connection.close()

def execute_write(query: str, parameters=None):
    connection = get_connection()
    cursor = connection.cursor()

    try:
        query = query.strip()

        cursor.execute(query, parameters)

        affected_rows = cursor.rowcount

        connection.commit()

        return affected_rows

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def validate_write_query(query: str):
    normalized = query.strip().lower()

    forbidden_keywords = {
        "drop",
        "alter",
        "truncate",
        "create",
        "grant",
        "revoke"
    }

    if ";" in normalized.rstrip(";"):
        return "Multiple SQL statements are not allowed."

    if any(keyword in normalized.split() for keyword in forbidden_keywords):
        return "This SQL operation is not allowed."

    if normalized.startswith("insert"):
        return None

    if normalized.startswith("update"):
        if " where " not in normalized:
            return "UPDATE queries must include a WHERE clause."
        return None

    if normalized.startswith("delete"):
        if " where " not in normalized:
            return "DELETE queries must include a WHERE clause."
        return None

    return "Only INSERT, UPDATE, and DELETE write queries are allowed."


def count_matching_rows(table: str, where_clause: str):
    connection = get_connection()
    connection.set_session(readonly=True)
    cursor = connection.cursor()

    try:
        query = f"SELECT COUNT(*) FROM {table} WHERE {where_clause}"
        cursor.execute(query)

        return cursor.fetchone()[0]

    finally:
        cursor.close()
        connection.close()


def get_database_schema():
    rows = execute_query("""
    SELECT
        table_name,
        column_name,
        data_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
    ORDER BY table_name, ordinal_position;""")


    schema_text =  ""

    for row in rows:
        schema_text += f'{row["table_name"]}.{row["column_name"]} ({row["data_type"]})\n'

    return schema_text
    
