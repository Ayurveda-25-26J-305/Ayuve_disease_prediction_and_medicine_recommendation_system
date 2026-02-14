"""
Simple test to verify answer generation is working
"""

from vector_db_setup import FAISSVectorDB
from llm_architecture import LLMArchitecture
import yaml

def test_simple():
    print("\n" + "=" * 70)
    print("SIMPLE GENERATION TEST")
    print("=" * 70)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load vector DB
    print("\n📂 Loading vector database...")
    vector_db = FAISSVectorDB(
        embedding_model_name=config['embedding_model'],
        index_path=config['index_path']
    )
    vector_db.load_index()
    print(f"✓ Loaded {vector_db.index.ntotal} documents")
    
    # Initialize LLM
    print("\n🤖 Initialize LLM...")
    llm = LLMArchitecture(config)
    print("✓ LLM initialized")
    
    # Test question
    question = "What are the benefits of Triphala?"
    print(f"\n❓ Question: {question}")
    
    # Retrieve docs
    print("\n🔍 Retrieving relevant sources...")
    retrieved_docs = vector_db.search(question, top_k=3)
    
    # Build simple context
    context_parts = []
    for i, doc in enumerate(retrieved_docs[:3], 1):
        text = doc['text'][:200] + "..."
        context_parts.append(f"[Source {i}] {text}")
    
    context_text = "\n\n".join(context_parts)
    
    # Create prompt
    prompt = f"""Question: {question}

Context:
{context_text}

Write a comprehensive 2-3 paragraph answer synthesizing the information above:
"""
    
    print(f"\n📝 Prompt length: {len(prompt)} characters")
    
    # Generate answer
    print("\n💭 Generating answer...")
    answer = llm.generate(prompt, max_new_tokens=300)
    
    # Display result
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"\n{answer}\n")
    print("=" * 70)

if __name__ == "__main__":
    test_simple()
