# tools.py
from vectorstore import search_vectorstore

def vector_search(query: str) -> str:
    """Retrieve answer strictly from the PDF vectorstore."""
    docs = search_vectorstore(query)

    if not docs:
        return "NO_MATCH"

    # Extract text from Document objects
    result_texts = [doc.page_content for doc in docs]
    return "\n\n".join(result_texts)