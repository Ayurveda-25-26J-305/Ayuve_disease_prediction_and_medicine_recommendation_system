"""
Local Test - Full Quality Answers
Works within RTX 2050 4GB VRAM limits by running components sequentially
"""

from vector_db_setup import FAISSVectorDB
from llm_architecture import LLMArchitecture
import yaml


def test_full_quality_answer():
    """Generate full quality answer locally"""
    print("\n" + "=" * 70)
    print("LOCAL FULL QUALITY TEST")
    print("=" * 70)
    print("\nGenerating answer with your RTX 2050 GPU...")
    
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
    print("\n🤖 Initializing LLM...")
    llm = LLMArchitecture(config)
    print("✓ LLM ready")
    
    # Test questions
    questions = [
        "What are the benefits of Triphala?",
        "How to balance Pitta dosha?",
        "What is Ayurvedic morning routine?"
    ]
    
    for i, question in enumerate(questions, 1):
        print("\n" + "-" * 70)
        print(f"\n🔍 Question {i}: {question}")
        print("-" * 70)
        
        # Retrieve relevant documents
        print("📚 Retrieving relevant sources...")
        docs = vector_db.search(question, top_k=3)
        
        # Build context
        context_parts = []
        citations = []
        
        for idx, doc in enumerate(docs, 1):
            text = doc['text']
            source = doc.get('source', 'Unknown')
            meta = doc.get('metadata', {})
            chapter = meta.get('chapter', 'N/A')
            
            context_parts.append(f"[Source {idx}] {text}")
            citations.append(f"{idx}. {source}, Chapter {chapter}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Question: {question}

Context:
{context}

Write a comprehensive 2-3 paragraph answer synthesizing the information above:
"""
        
        # Generate answer with good token count
        print("💭 Generating answer...")
        answer = llm.generate(prompt, max_new_tokens=150)
        
        # Display results
        print("\n" + "=" * 70)
        print("📝 ANSWER:")
        print("=" * 70)
        print(f"\n{answer}\n")
        
        print("=" * 70)
        print("📖 SOURCES:")
        print("=" * 70)
        for citation in citations:
            print(f"   {citation}")
        
        print("\n" + "=" * 70)
        
        # Wait for user
        if i < len(questions):
            input("\nPress Enter for next question...")


if __name__ == "__main__":
    test_full_quality_answer()
