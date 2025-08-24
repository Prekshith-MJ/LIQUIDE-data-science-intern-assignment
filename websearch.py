# websearch.py
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

def web_search(query: str) -> str:
    """Fallback to web search when no PDF answer is found."""
    try:
        return search.run(query)
    except Exception as e:
        return f"Web search failed: {e}"