from rag.retriever import retrieve_documents

documents = retrieve_documents(
    "What happens if an employee is repeatedly late?"
)

print("DOCUMENTS:", documents)
print("NUMBER:", len(documents))