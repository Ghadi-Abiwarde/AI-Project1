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

    try:
        response = requests.post(
            "https://google.serper.dev/not-real",
            headers=headers,
            json=payload,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        results = data.get("organic", [])

        return results[:5]

    except requests.RequestException:
        return []
