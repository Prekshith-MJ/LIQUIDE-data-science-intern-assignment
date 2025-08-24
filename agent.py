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

def query_llm(context: str, question: str, is_financial: bool = True) -> str:
    """
    Use OpenRouter LLM to generate a response based on context and question.
    """
    try:
        if is_financial:
            system_prompt = """You replicate financial glossary entries exactly. Provide ONLY the definition text without any additional commentary.
            
            OUTPUT FORMAT:
            [Term]
            [Definition text exactly as it appears in financial documentation]
            
            EXAMPLES:
            Input: "Regulatory arbitrage"
            Output: "Regulatory arbitrage\nA financial contract or a series of transactions undertaken, entirely or in part, because the transaction(s) enable(s) one or more of the counterparties to accomplish a financial or operating objective which is unavailable to them directly because of regulatory obstacles."
            
            Input: "Rematerialisation"
            Output: "Rematerialisation\nThe process of converting electronic holdings into physical securities through a Depository Participant."
            
            Follow this exact pattern for all terms."""
        else:
            system_prompt = """You provide clear, concise definitions in glossary format. Provide ONLY the definition text without any additional commentary.
            
            OUTPUT FORMAT:
            [Term]
            [Definition text]
            
            EXAMPLES:
            Input: "Music"
            Output: "Music\nAn art form and cultural activity consisting of sound and silence expressed through time."
            
            Input: "Technology"
            Output: "Technology\nThe application of scientific knowledge for practical purposes, especially in industry."
            
            Provide direct definitions without mentioning finance or source."""
        
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

def query_agent(question: str) -> str:
    """
    First try to answer from PDF vectorstore.
    If no relevant match, fall back to web search.
    """
    # Clean the question
    clean_question = question.replace("What is", "").replace("what is", "").strip().strip('?')
    
    # Search vectorstore for relevant context
    pdf_context = search_vectorstore(clean_question, k=5)
    
    # If we have good PDF context, try to get answer from it (financial)
    if pdf_context and pdf_context != "NO_MATCH" and len(pdf_context.strip()) > 30:
        pdf_answer = query_llm(pdf_context, clean_question, is_financial=True)
        # Check if the answer looks like a proper definition
        if (len(pdf_answer.strip()) > 20 and 
            "\n" in pdf_answer and
            not pdf_answer.startswith("I couldn't") and
            not pdf_answer.startswith("Error")):
            return pdf_answer
    
    # Fall back to web search (general definition)
    web_context = web_search(clean_question)
    if web_context and web_context != "NO_MATCH" and not web_context.startswith("Web search failed:"):
        web_answer = query_llm(web_context, clean_question, is_financial=False)
        return web_answer
    
    return "I couldn't find any relevant information to answer your question."

# For testing
if __name__ == "__main__":
    test_questions = ["What is music?", "What is a dealer?", "What is regulatory arbitrage?", "What is technology?"]
    for question in test_questions:
        result = query_agent(question)
        print(f"\nQ: {question}")
        print(f"A: {result}")
        print("-" * 50)