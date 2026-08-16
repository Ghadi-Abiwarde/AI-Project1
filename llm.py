import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def create_llm(model_name="openai/gpt-oss-120b"):

    return ChatGroq(
        model=model_name,
        temperature=0
    )