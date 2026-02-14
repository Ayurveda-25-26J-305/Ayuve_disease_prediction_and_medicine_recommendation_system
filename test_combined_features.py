"""
Combined Test: Validation + Personalization
Tests both enhanced features working together
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


def test_combined():
    """Test both validation and personalization together"""
    print("\n" + "=" * 70)
    print("COMBINED TEST: Validation + Personalization")
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
    
    # Initialize FULL enhanced RAG with BOTH features
    print("\n🔧 Initializing Enhanced RAG (Validation + Personalization)...")
    rag = EnhancedAyurvedicRAG(
        llm_model_name=config['llm_model'],
        max_new_tokens=64,  # Reduced for memory constraints
        enable_validation=True,
        enable_personalization=True
    )
    print("✓ RAG initialized with BOTH features enabled")
    
    # Create test user profile (Pitta constitution)
    print("\n👤 Creating test user profile (Pitta constitution)...")
    responses = {
        'q1': 'B', 'q2': 'B', 'q3': 'B', 'q4': 'B', 'q5': 'B',
        'q6': 'B', 'q7': 'B', 'q8': 'B', 'q9': 'B', 'q10': 'B'
    }
    
    profile = rag.create_user_profile(
        user_id='test_user_pitta',
        prakriti_responses=responses
    )
    
    print(f"✓ Profile created:")
    print(f"   Dominant Dosha: {profile['dominant_dosha']}")
    print(f"   Season: {profile['current_season']}")
    
    # Test question
    question = "What are the benefits of Triphala?"
    print(f"\n❓ Question: {question}")
    
    # Get validated AND personalized answer
    print("\n🔍 Processing with validation and personalization...")
    result = rag.answer_question(
        question=question,
        vector_db=vector_db,
        top_k=3,
        user_profile=profile,
        validation_top_k=5
    )
    
    # Display comprehensive results
    print("\n" + "=" * 70)
    print("COMBINED RESULTS")
    print("=" * 70)
    
    # Show base answer (before personalization) - CLEAN VERSION
    if 'base_answer' in result and result['base_answer']:
        print(f"\n📝 BASE ANSWER (Generated from Retrieved Sources):")
        print("-" * 70)
        print(result['base_answer'])
    
    # Show final answer (validated + personalized) - CLEAN VERSION  
    print(f"\n\n✨ FINAL ANSWER (Validated + Personalized):")
    print("-" * 70)
    print(result['answer'])
    
    # Validation metrics
    if 'validation' in result:
        validation = result['validation']
        print("\n" + "-" * 70)
        print("📊 VALIDATION METRICS:")
        print("-" * 70)
        print(f"✅ Confidence: {validation['confidence']}% ({validation['confidence_level']})")
        print(f"📚 Sources checked: {validation['sources_checked']}")
        print(f"✓ Sources agree: {validation['sources_agree']}/{validation['sources_checked']}")
        print(f"💡 {validation['note']}")
        
        if validation['contradictions']:
            print(f"\n⚠️ Contradictions detected: {len(validation['contradictions'])}")
            for contra in validation['contradictions']:
                print(f"   - {contra['message']}")
        
        print("\n📖 Source Agreement Details:")
        for agreement in validation['agreements'][:3]:  # Show top 3
            emoji = '✓' if agreement['agrees'] else '◐'
            print(f"   {emoji} {agreement['source']}: {agreement['similarity']*100:.1f}% ({agreement['agreement_level']})")
    
    # Personalization info
    if result.get('is_personalized'):
        print("\n" + "-" * 70)
        print("👤 PERSONALIZATION INFO:")
        print("-" * 70)
        print(f"✅ Answer was personalized for user")
        print(f"🧬 Dominant Dosha: {result['user_info']['dominant_dosha']}")
        print(f"🌦️ Current Season: {result['user_info']['current_season']}")
        print(f"📍 Location Context: Sri Lankan Ayurveda")
    
    # Citations
    if 'citations' in result and result['citations']:
        print("\n" + "-" * 70)
        print("📚 CITATIONS:")
        print("-" * 70)
        for i, citation in enumerate(result['citations'][:3], 1):
            print(f"{i}. {citation['source']}")
            if citation.get('chapter') != 'N/A':
                print(f"   {citation['chapter']}, Paragraph {citation['paragraph']}")
    
    print("\n" + "=" * 70)
    print("✅ COMBINED TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\n🎯 Summary:")
    print(f"   • Base answer generated from {len(result.get('citations', []))} sources")
    if 'validation' in result:
        print(f"   • Validated against {validation['sources_checked']} additional sources")
        print(f"   • Confidence level: {validation['confidence_level']}")
    if result.get('is_personalized'):
        print(f"   • Personalized for {result['user_info']['dominant_dosha']} constitution")
    print(f"   • Final answer quality: Enhanced with both features ✨")
    
    return result


if __name__ == "__main__":
    try:
        test_combined()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
