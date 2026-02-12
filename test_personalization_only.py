"""
Quick test for personalization feature only
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_rag_gpu import EnhancedAyurvedicRAG
from vector_db_setup import FAISSVectorDB
import yaml


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def test_personalization():
    """Test answer personalization"""
    print("\n" + "=" * 70)
    print("TEST: Answer Personalization")
    print("=" * 70)
    
    config = load_config()
    
    # Load vector DB
    print("\n📂 Loading vector database...")
    vector_db = FAISSVectorDB(
        embedding_model_name=config['embedding_model'],
        index_path=config['index_path']
    )
    vector_db.load_index()
    print(f"✓ Loaded {vector_db.index.ntotal} documents")
    
    # Initialize enhanced RAG (personalization only)
    print("\n🔧 Initializing Enhanced RAG...")
    rag = EnhancedAyurvedicRAG(
        llm_model_name=config['llm_model'],
        max_new_tokens=100,
        enable_validation=False,
        enable_personalization=True
    )
    print("✓ RAG initialized")
    
    # Create test user profile (Vata constitution)
    print("\n👤 Creating test user profile (Vata constitution)...")
    responses = {
        'q1': 'A', 'q2': 'A', 'q3': 'A', 'q4': 'A', 'q5': 'A',
        'q6': 'B', 'q7': 'A', 'q8': 'A', 'q9': 'A', 'q10': 'A'
    }
    
    profile = rag.create_user_profile(
        user_id='test_user_vata',
        prakriti_responses=responses
    )
    
    print(f"✓ Profile created:")
    print(f"   Dominant Dosha: {profile['dominant_dosha']}")
    print(f"   Season: {profile['current_season']}")
    
    # Test question
    question = "How to improve digestion?"
    print(f"\n❓ Question: {question}")
    
    # Get personalized answer
    print("\n🔍 Retrieving and generating personalized answer...")
    result = rag.answer_question(
        question=question,
        vector_db=vector_db,
        top_k=3,
        user_profile=profile
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    
    if 'base_answer' in result:
        print(f"\n📝 Base Answer:\n{result['base_answer']}\n")
    
    print(f"\n✨ Personalized Answer:\n{result['answer']}\n")
    
    if result.get('is_personalized'):
        print(f"✅ Answer was personalized!")
        print(f"👤 Personalized for: {result['user_info']['dominant_dosha']}")
        print(f"🌦️ Season: {result['user_info']['current_season']}")
    else:
        print(f"⚠️ Personalization was not applied")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETED")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    try:
        test_personalization()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
