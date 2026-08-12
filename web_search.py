import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("SERPER_API_KEY")


def search_web(query: str):
 headers = {
    "X-API-KEY": api_key,
    "Content-Type": "application/json"
}

 payload = {
    "q": query
}

 response = requests.post(
     "https://google.serper.dev/search",
     headers=headers,
     json=payload
)
 data = response.json()

 results = data["organic"]
 
 return results[:5]

