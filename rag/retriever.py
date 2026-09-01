from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def retrieve_documents(query: str):
    vector_store = FAISS.load_local(
        "rag/vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )

    results = vector_store.similarity_search_with_score(
        query,
        k=3
    )

    print(f"\nQUERY: {query}")

    for document, score in results:
        print("SCORE:",score)
        print("CONTENT:", document.page_content)
        print("-" * 50)

    relevant_documents = [
        document
        for document, score in results
        if score < 1.1
    ]   
    
    return relevant_documents
