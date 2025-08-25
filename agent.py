# agent.py
import os
from vectorstore import search_vectorstore
from websearch import web_search
from openai import OpenAI

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def query_llm(context: str, question: str) -> str:
    """
    Use OpenRouter LLM to generate a response based on context and question.
    LLM will automatically use financial style for financial content, general style otherwise.
    """
    try:
        system_prompt = """You are a helpful research assistant. Provide answers in bullet point format.
        
        OUTPUT FORMAT:
        • [Term]: [Definition text]
        
    	If the context contains financial content, use precise financial terminology and style.
		If the context contains general content, use clear, concise definitions.
		Always use bullet points for answers.
		For multiple terms, provide the answer for each term in a separate point.
		Do not include introductory or filler sentences — only structured responses."""
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:8501"),
                "X-Title": os.getenv("SITE_NAME", "Research Agent App"),
            },
            model="anthropic/claude-3-haiku",
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Reference content:\n{context}\n\nProvide definition for: {question}"
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

def search_with_multiple_models(query: str) -> str:
    """
    Try multiple embedding models for better retrieval.
    First try mpnet (better), then fallback to miniLM (original).
    """
    # Try mpnet model first (better quality)
    try:
        from vectorstore import search_vectorstore_multi
        mpnet_result = search_vectorstore_multi(query, model_type="mpnet")
        if mpnet_result and mpnet_result != "NO_MATCH":
            return mpnet_result
    except:
        pass
    
    # Fallback to original miniLM model
    original_result = search_vectorstore(query)
    return original_result

def query_agent(question: str) -> str:
    """
    First try to answer from PDF vectorstore.
    If no relevant match, fall back to web search.
    """
    # Clean the question
    clean_question = question.replace("What is", "").replace("what is", "").strip().strip('?')
    
    # Search vectorstore for relevant context
    pdf_context = search_with_multiple_models(clean_question)
    
    # If we have good PDF context, use it (LLM auto-detects financial style)
    if pdf_context and pdf_context != "NO_MATCH":
        pdf_answer = query_llm(pdf_context, clean_question)
        if pdf_answer and len(pdf_answer.strip()) > 20:
            return pdf_answer
    
    # Fall back to web search (LLM will use general style)
    web_context = web_search(clean_question)
    if web_context and web_context != "NO_MATCH" and not web_context.startswith("Web search failed:"):
        web_answer = query_llm(web_context, clean_question)
        if web_answer and len(web_answer.strip()) > 20:
            return web_answer
    
    return "• I couldn't find any relevant information to answer your question."

# For testing
if __name__ == "__main__":
    test_question = "What is a dealer?"
    result = query_agent(test_question)
    print(f"Result: {result}")