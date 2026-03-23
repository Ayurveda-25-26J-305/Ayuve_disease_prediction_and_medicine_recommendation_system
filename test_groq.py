"""
Quick test: verifies Groq API key + model response.
Run before starting the full backend:
    python test_groq.py
"""

import sys
import os
import yaml

# Add scripts/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
from groq_llm import GroqLLM

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Groq API Connection Test")
    print("=" * 60)
    print(f"Model : {config.get('groq_model')}")
    print(f"Key   : {config.get('groq_api_key', '')[:12]}...")
    print()

    # Initialise GroqLLM
    llm = GroqLLM(config)

    # Test 1: simple generate()
    print("[Test 1]  generate() - plain prompt")
    prompt = (
        "You are an Ayurvedic expert. "
        "Answer in 2 bullet points.\n\n"
        "Question: What are the main benefits of turmeric in Ayurveda?"
    )
    answer = llm.generate(prompt, max_new_tokens=150)
    print("Answer:")
    print(answer)
    print()

    # Test 2: generate_from_messages()
    print("[Test 2]  generate_from_messages() - chat format")
    messages = [
        {
            "role": "system",
            "content": "You are an Ayurvedic knowledge assistant. Be concise."
        },
        {
            "role": "user",
            "content": "What dosha does ginger help to balance?"
        }
    ]
    answer2 = llm.generate_from_messages(messages, max_new_tokens=100)
    print("Answer:")
    print(answer2)
    print()

    print("=" * 60)
    print("All tests passed! Groq API is working correctly.")
    print("You can now start the backend:  python backend/app_enhanced.py")
    print("=" * 60)

if __name__ == '__main__':
    main()
