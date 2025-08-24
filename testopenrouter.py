# debug_openrouter.py
import os
from openai import OpenAI

def test_openrouter():
    print("=== Testing OpenRouter Connection ===")
    
    # Check if API key is set
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable is not set")
        return False
    
    print("✅ OPENROUTER_API_KEY is set")
    
    # Initialize client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    # Test 1: Simple hello with gpt-4o
    print("\n=== Test 1: Calling openai/gpt-4o ===")
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Debug Test",
            },
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hello, what model are you?"}]
        )
        print("✅ Success with gpt-4o!")
        print(f"Response: {completion.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Error with gpt-4o: {e}")
    
    # Test 2: Try a different model
    print("\n=== Test 2: Calling anthropic/claude-3-haiku ===")
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Debug Test",
            },
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": "Hello, what model are you?"}]
        )
        print("✅ Success with claude-3-haiku!")
        print(f"Response: {completion.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Error with claude-3-haiku: {e}")
        return False

if __name__ == "__main__":
    test_openrouter()