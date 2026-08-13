import os
import requests
from dotenv import load_dotenv
from web_search import search_web
load_dotenv()
api_key = os.getenv("SERPER_API_KEY")


results = search_web(
    "latest developments in artificial intelligence"
)
print(results)
#for result in results:
#    print("TITLE:", result["title"])
#    print("LINK:", result["link"])
#    print("SNIPPET:", result["snippet"])
  #  print("-----")