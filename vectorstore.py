# vectorstore.py
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Add multiple embedding models
EMBEDDING_MODELS = {
    "miniLM": "sentence-transformers/all-MiniLM-L6-v2",  # Original model
    #"mpnet": "sentence-transformers/all-mpnet-base-v2"   # Additional model
}

def search_vectorstore(query: str, k: int = 3) -> str:
    """
    Search the ingested FAISS vectorstore for a query.
    Returns top-k relevant text as a string, or 'NO_MATCH'.
    Uses the original miniLM model (same as before).
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

def search_vectorstore_multi(query: str, k: int = 3, model_type: str = "miniLM") -> str:
    """
    Search with multiple embedding model options.
    First tries the specified model, falls back to miniLM.
    """
    # Default to miniLM if model_type is invalid
    if model_type not in EMBEDDING_MODELS:
        model_type = "miniLM"
    
    # Use the specified embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODELS[model_type])

    # ✅ load FAISS db (same index name - compatible with original)
    if not os.path.exists("faiss_index"):
        return "NO_MATCH"

    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # ✅ search query (same logic)
    results = vectorstore.similarity_search(query, k=k)

    if not results or len(results) == 0:
        return "NO_MATCH"

    # ✅ join page contents (same logic)
    return "\n\n".join([doc.page_content for doc in results])