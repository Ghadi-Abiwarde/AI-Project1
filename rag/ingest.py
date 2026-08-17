from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

documents = []

file_paths = [
    "rag/documents/employee_handbook.txt",
    "rag/documents/remote_work_policy.txt"
]

for file_path in file_paths:
    loader = TextLoader(file_path)
    documents.extend(loader.load())

print(f"Loaded {len(documents)} documents")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = FAISS.from_documents(
    chunks,
    embeddings
)
vector_store_path = "rag/vector_store"
vector_store.save_local(vector_store_path)

print("Vector store saved successfully")

