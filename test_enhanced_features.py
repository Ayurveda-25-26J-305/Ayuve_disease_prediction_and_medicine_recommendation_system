import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# USE GPU VERSION (No quantization - works on Windows)
from enhanced_rag_gpu import EnhancedAyurvedicRAG
from vector_db_setup import FAISSVectorDB
import yaml


def load_config():
    """Load configuration"""
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_validation():
    """Test multi-source validation"""
    print("=" * 70)
    print("TEST 1: Multi-Source Validation")
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
    
    # Initialize enhanced RAG (validation only)
    print("\n🔧 Initializing Enhanced RAG...")
    rag = EnhancedAyurvedicRAG(
        llm_model_name=config['llm_model'],
        max_new_tokens=64,
        enable_validation=True,
        enable_personalization=False
    )
    print("✓ RAG initialized")
    
    # Test question
    question = "What are the benefits of Triphala?"
    print(f"\n❓ Question: {question}")
    
    # Get answer with validation
    result = rag.answer_question(
        question=question,
        vector_db=vector_db,
        top_k=3,
        validation_top_k=5
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    
    print(f"\n📝 Answer:\n{result['answer']}\n")
    
    validation = result['validation']
    print(f"✅ Confidence: {validation['confidence']}% ({validation['confidence_level']})")
    print(f"📚 Sources checked: {validation['sources_checked']}")
    print(f"✓ Sources agree: {validation['sources_agree']}/{validation['sources_checked']}")
    print(f"💡 Note: {validation['note']}")
    
    if validation['contradictions']:
        print(f"\n⚠️ Contradictions detected: {len(validation['contradictions'])}")
        for contra in validation['contradictions']:
            print(f"   - {contra['message']}")
    else:
        print(f"\n✓ No contradictions detected")
    
    print("\n📖 Source Agreement Details:")
    for agreement in validation['agreements']:
        emoji = '✓' if agreement['agrees'] else '◐'
        print(f"   {emoji} {agreement['source']}: {agreement['similarity']*100:.1f}% match ({agreement['agreement_level']})")
    
    return result


def test_personalization():
    """Test answer personalization"""
    print("\n\n" + "=" * 70)
    print("TEST 2: Answer Personalization")
    print("=" * 70)
    
    config = load_config()
    
    # Load vector DB
    print("\n📂 Loading vector database...")
    vector_db = FAISSVectorDB(
        embedding_model_name=config['embedding_model'],
        index_path=config['index_path']
    )
    vector_db.load_index()
    
    # Initialize enhanced RAG (personalization only for this test)
    print("\n🔧 Initializing Enhanced RAG...")
    rag = EnhancedAyurvedicRAG(
        llm_model_name=config['llm_model'],
        max_new_tokens=150,  # More tokens for personalized answer
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
    
    print(f"\n📝 Base Answer:\n{result['base_answer']}\n")
    print(f"\n✨ Personalized Answer:\n{result['answer']}\n")
    print(f"👤 Personalized for: {result['user_info']['dominant_dosha']}")
    print(f"🌦️ Season: {result['user_info']['current_season']}")
    
    return result


def test_combined():
    """Test both features together"""
    print("\n\n" + "=" * 70)
    print("TEST 3: Validation + Personalization Combined")
    print("=" * 70)
    
    config = load_config()
    
    # Load vector DB
    print("\n📂 Loading vector database...")
    vector_db = FAISSVectorDB(
        embedding_model_name=config['embedding_model'],
        index_path=config['index_path']
    )
    vector_db.load_index()
    
    # Initialize FULL enhanced RAG
    print("\n🔧 Initializing Enhanced RAG (BOTH features)...")
    rag = EnhancedAyurvedicRAG(
        llm_model_name=config['llm_model'],
        max_new_tokens=64,
        enable_validation=True,
        enable_personalization=True
    )
    print("✓ RAG initialized with Validation + Personalization")
    
    # Create test user profile (Pitta constitution)
    print("\n👤 Creating test user profile (Pitta constitution)...")
    responses = {
        'q1': 'B', 'q2': 'B', 'q3': 'B', 'q4': 'B', 'q5': 'B',
        'q6': 'B', 'q7': 'B', 'q8': 'B', 'q9': 'B', 'q10': 'A'
    }
    
    profile = rag.create_user_profile(
        user_id='test_user_pitta',
        prakriti_responses=responses
    )
    
    print(f"✓ Profile created: {profile['dominant_dosha']}")
    
    # Test question
    question = "What are the benefits of coconut water?"
    print(f"\n❓ Question: {question}")
    
    # Get FULLY enhanced answer
    result = rag.answer_question(
        question=question,
        vector_db=vector_db,
        top_k=3,
        user_profile=profile,
        validation_top_k=5
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("FINAL RESULTS (Complete Enhancement):")
    print("=" * 70)
    
    print(f"\n✨ Personalized & Validated Answer:\n")
    print(result['answer'])
    
    validation = result['validation']
    print(f"\n📊 Validation:")
    print(f"   Confidence: {validation['confidence']}% ({validation['confidence_level']})")
    print(f"   Sources agree: {validation['sources_agree']}/{validation['sources_checked']}")
    print(f"   Note: {validation['note']}")
    
    print(f"\n👤 Personalization:")
    print(f"   For: {result['user_info']['dominant_dosha']} constitution")
    print(f"   Season: {result['user_info']['current_season']}")
    
    print(f"\n📖 Citations: {len(result['citations'])} sources")
    
    return result


def main():
    """Run combined test only"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "ENHANCED FEATURE TESTING" + " " * 29 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        # Test 1: Validation (SKIPPED - run individually if needed)
        # test_validation()
        
        # Test 2: Personalization (SKIPPED - run individually if needed)
        # test_personalization()
        
        # Test 3: Combined (ONLY THIS RUNS)
        test_combined()
        
        print("\n\n" + "=" * 70)
        print("✅ COMBINED TEST COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n\n❌ ERROR during testing:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
