import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Change this

def search_vectorstore(query: str, k: int = 3) -> str:
    """
    Search the ingested FAISS vectorstore for a query.
    Returns top-k relevant text as a string, or 'NO_MATCH'.
    """
    # ✅ Use the same embeddings that were used to create the index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ✅ load FAISS db
    if not os.path.exists("faiss_index"):
        return "NO_MATCH"

    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # ✅ search query
    results = vectorstore.similarity_search(query, k=k)

    if not results or len(results) == 0:
        return "NO_MATCH"

    # ✅ join page contents instead of accessing .data
    return "\n\n".join([doc.page_content for doc in results])