from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local(
    "rag/vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)


def retrieve_documents(query: str):
    documents = vector_store.similarity_search(
    query,
    k=3
)

    return documents

results = retrieve_documents(
    "How many annual leave days do employees receive?"
)

for document in results:
    print("SOURCE:", document.metadata)
    print(document.page_content)
    print("-----")