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

