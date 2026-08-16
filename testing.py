from llm import create_llm
from web_search import search_web


results = search_web(
    "What changed in the latest Python release?"
)

for result in results:
    #print("TITLE:", result.get("title"))
    #print("LINK:", result.get("link"))
    #print("SNIPPET:", result.get("snippet"))
    #print("-----")
    print(results[0])