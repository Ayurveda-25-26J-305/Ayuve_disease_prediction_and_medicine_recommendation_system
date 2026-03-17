"""
Enhanced Flask Backend API for Ayurvedic QA System
Integrates:
- Multi-source validation (Option 1)
- Answer personalization (Option 2)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory and scripts/ to path to import our modules
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
sys.path.append(os.path.join(_root, 'scripts'))

from vector_db_setup import FAISSVectorDB
from enhanced_rag_gpu import EnhancedAyurvedicRAG  # Use GPU version with new features
from personalization_engine import PRAKRITI_QUESTIONS
import yaml

app = Flask(__name__)
# Enable CORS for frontend with credentials support
# Allow all origins for Localtunnel compatibility
CORS(app, resources={r"/*": {
    "origins": "*",  # Allow all origins (needed for Localtunnel)
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Bypass-Tunnel-Reminder", "ngrok-skip-browser-warning"],
    "supports_credentials": False  # Set to False when using "*" origin
}})

# Global variables for system components
vector_db = None
web_db = None        # NEW: Web knowledge FAISS index
rag_system = None
config = None


def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_system():
    """Initialize vector DB and Enhanced LLM on startup"""
    global vector_db, web_db, rag_system, config
    
    print("=" * 70)
    print("Initializing Enhanced Ayurvedic QA System Backend")
    print("=" * 70)
    
    # Load config
    config = load_config()
    print("\n✓ Config loaded")
    
    # Load BOOK vector database
    print("\n🔄 Loading book vector database...")
    
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    index_path = os.path.join(project_root, config['index_path'])
    
    print(f"   Index path: {index_path}")
    
    vector_db = FAISSVectorDB(
        embedding_model_name=config['embedding_model'],
        index_path=index_path
    )
    
    if vector_db.load_index():
        print(f"✓ Loaded book database with {vector_db.index.ntotal} documents")
    else:
        print("❌ Failed to load book vector database")
        return False
    
    # Load WEB vector database (optional — only if it exists)
    web_index_path = os.path.join(project_root, "faiss_index_web")
    if os.path.exists(os.path.join(web_index_path, "index.faiss")):
        print("\n🔄 Loading web knowledge database...")
        web_db = FAISSVectorDB(
            embedding_model_name=config['embedding_model'],
            index_path=web_index_path
        )
        if web_db.load_index():
            print(f"✓ Loaded web database with {web_db.index.ntotal} web documents")
        else:
            print("⚠️  Web index found but failed to load — proceeding books-only")
            web_db = None
    else:
        print("\nℹ️  No web knowledge index found (faiss_index_web/)")
        print("   Run 'python run_web_pipeline.py' to build it.")
        web_db = None
    
    # Initialize Enhanced LLM with validation and personalization
    print("\n🔄 Loading Enhanced LLM (this may take a minute)...")
    rag_system = EnhancedAyurvedicRAG(
        llm_model_name=config['llm_model'],
        max_new_tokens=config.get('max_new_tokens', 128),
        enable_validation=True,
        enable_personalization=True,
        enable_translation=True,
        embedding_model=config['embedding_model']
    )
    print("✓ Enhanced LLM initialized (Validation + Personalization enabled)")
    rag_system._answer_cache.clear()
    print("✓ Answer cache cleared")
    
    print("\n" + "=" * 70)
    print(f"✓ Backend ready  |  Books: {vector_db.index.ntotal} docs  |  Web: {web_db.index.ntotal if web_db else 0} docs")
    print("=" * 70)
    
    return True


def format_citation(source_doc, validation_info=None):
    """
    Format citation for frontend display with validation info.
    Handles book sources, web sources, and QA dataset entries.
    """
    metadata = source_doc.get("metadata", {})
    source_name = source_doc.get("source", "Unknown")
    doc_type = source_doc.get("type", "unknown")
    similarity_percentage = source_doc.get("similarity_percentage", 0.0)

    citation = {
        "source": source_name,
        "type": doc_type,
        "similarity_percentage": similarity_percentage
    }

    if doc_type == "book":
        chapter = metadata.get("chapter", "N/A")
        paragraph = metadata.get("paragraph", metadata.get("verse", "N/A"))
        citation["chapter"] = str(chapter)
        citation["paragraph"] = str(paragraph)
        citation["formatted"] = (
            f"{source_name} – Chapter {chapter} – "
            f"Verse/Paragraph {paragraph} ({similarity_percentage}% match)"
        )

    elif doc_type == "web":
        # Web / curated source — expose URL for clickable links
        url = (source_doc.get("url", "")
               or metadata.get("url", "")
               or metadata.get("source_url", ""))
        authority = source_doc.get("authority", metadata.get("authority", 0.80))
        citation["url"] = url
        citation["authority"] = round(authority, 2)
        citation["formatted"] = (
            f"{source_name} ({similarity_percentage}% match)"
        )
        if url:
            citation["formatted_with_link"] = (
                f'<a href="{url}" target="_blank" rel="noopener">'
                f'{source_name}</a> ({similarity_percentage}% match)'
            )

    else:
        # QA dataset entry
        qa_id = metadata.get("question_id", "N/A")
        related_question = metadata.get("question", "N/A")
        citation["qa_id"] = str(qa_id)
        citation["related_question"] = related_question
        citation["formatted"] = f"{source_name} – Q&A #{qa_id} ({similarity_percentage}% match)"

    # Add validation info if provided
    if validation_info:
        citation["agreement"] = validation_info.get("agreement_level", "unknown")
        citation["similarity"] = validation_info.get("similarity", 0.0)

    return citation


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "service": "Enhanced Ayurvedic QA System API",
        "version": "2.0",
        "features": ["validation", "personalization"],
        "documents": vector_db.index.ntotal if vector_db else 0
    })


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """
    Main endpoint for asking questions with optional personalization
    
    Request body:
    {
        "question": "What causes Vata imbalance?",
        "user_id": "user123"  // Optional - for personalization
    }
    
    Response:
    {
        "success": true,
        "answer": "...",
        "citations": [...],
        "validation": {
            "confidence": 85.5,
            "confidence_level": "high",
            ...
        },
        "personalized": true,
        "user_info": {...}
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                "success": False,
                "error": "No question provided"
            }), 400
        
        question = data['question'].strip()
        user_id = data.get('user_id', None)
        dominant_dosha = data.get('dominant_dosha', None)
        
        if not question:
            return jsonify({
                "success": False,
                "error": "Question cannot be empty"
            }), 400
        
        print(f"\n" + "=" * 70)
        print(f"🔍 Question: {question}")
        print(f"🔍 Retrieving relevant sources...")
        
        # Load user profile if user_id provided
        user_profile = None
        if user_id:
            user_profile = rag_system.load_user_profile(user_id)
            if user_profile:
                print(f"👤 User profile loaded: {user_profile['dominant_dosha']}")
        
        print(f"💭 Generating answer...")
        
        # Get enhanced answer with hybrid validation and personalization
        response = rag_system.answer_question(
            question=question,
            vector_db=vector_db,
            top_k=config.get('top_k', 3),
            user_profile=user_profile,
            validation_top_k=5,
            dominant_dosha=dominant_dosha,
            web_db=web_db  # Pass web knowledge index for hybrid retrieval
        )

        # Handle domain-blocked questions
        if response.get('blocked'):
            return jsonify({
                "success": False,
                "blocked": True,
                "answer": response['answer'],
                "block_reason": response.get('block_reason', ''),
                "citations": [],
                "validation": {}
            }), 200
        
        # Format citations with validation info
        citations = []
        for citation_dict in response.get('citations', []):
            formatted = format_citation(citation_dict)
            citations.append(formatted)
        
        print(f"✅ Generation complete!")
        print(f"   Answer length: {len(response['answer'])} chars")
        print(f"   Answer preview: {response['answer'][:200]}...")
        print(f"✓ Validating answer across sources...")
        print(f"✓ Answer generated")
        print(f"  Confidence: {response.get('validation', {}).get('confidence', 'N/A')}%")
        print(f"  Personalized: {response['personalized']}")
        print(f"  Citations: {len(citations)}")
        print(f"  Answer preview: {response['answer'][:100]}...")
        
        result = {
            "success": True,
            "answer": response['answer'],  # In user's language (English or Sinhala)
            "answer_english": response.get('answer_english', response['answer']),  # Always English
            "base_answer": response.get('base_answer', ''),  # For comparison
            "original_question": response.get('original_question', question),
            "translated_question": response.get('translated_question'),  # English translation if Sinhala
            "detected_language": response.get('detected_language', 'en'),
            "citations": citations,
            "num_sources": response['num_sources'],
            "validation": response.get('validation', {}),
            "personalized": response['personalized'],
            "detected_dosha": response.get('detected_dosha', 'General'),
            "personalized_tips": response.get('personalized_tips', ''),
            "user_info": response.get('user_info', {})
        }
        
        print(f"  Response keys: {list(result.keys())}")
        print(f"  Answer length: {len(result['answer'])}")
        print(f"  Citations count: {len(result['citations'])}")
        
        # Detailed answer display
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(result['answer'])
        print("\n" + "=" * 60)
        print("VALIDATION METRICS:")
        print("=" * 60)
        validation = result.get('validation', {})
        print(f"✅ Confidence: {validation.get('confidence', 0):.1f}%")
        print(f"📊 Source Agreement: {validation.get('sources_agree', 0)}/{validation.get('sources_checked', 0)} sources agree")
        print(f"📈 Confidence Level: {validation.get('confidence_level', 'N/A')}")
        print("=" * 60 + "\n")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/prakriti/questions', methods=['GET'])
def get_prakriti_questions():
    """
    Get Prakriti assessment questions
    
    Response:
    {
        "success": true,
        "questions": [...]
    }
    """
    try:
        return jsonify({
            "success": True,
            "questions": PRAKRITI_QUESTIONS
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/prakriti/assess', methods=['POST'])
def assess_prakriti():
    """
    Assess user's Prakriti and create profile
    
    Request body:
    {
        "user_id": "user123",
        "responses": {
            "q1": "A",
            "q2": "B",
            ...
        }
    }
    
    Response:
    {
        "success": true,
        "profile": {
            "dominant_dosha": "vata",
            "prakriti": {...},
            ...
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'user_id' not in data or 'responses' not in data:
            return jsonify({
                "success": False,
                "error": "Missing user_id or responses"
            }), 400
        
        user_id = data['user_id']
        responses = data['responses']
        
        print(f"\n👤 Creating profile for user: {user_id}")
        
        # Create user profile
        profile = rag_system.create_user_profile(
            user_id=user_id,
            prakriti_responses=responses
        )
        
        print(f"✓ Profile created: {profile['dominant_dosha']}")
        
        return jsonify({
            "success": True,
            "profile": profile
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/prakriti/profile/<user_id>', methods=['GET'])
def get_user_profile(user_id):
    """
    Get existing user profile
    
    Response:
    {
        "success": true,
        "profile": {...}
    }
    """
    try:
        profile = rag_system.load_user_profile(user_id)
        
        if not profile:
            return jsonify({
                "success": False,
                "error": "Profile not found"
            }), 404
        
        return jsonify({
            "success": True,
            "profile": profile
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        book_stats = vector_db.get_stats() if vector_db else {}
        web_stats  = web_db.get_stats()   if web_db   else {}

        return jsonify({
            "success": True,
            # Flat fields — used by Cell 9 and frontend
            "book_documents":  book_stats.get('total_documents', 0),
            "web_documents":   web_stats.get('total_documents', 0),
            "web_knowledge":   web_db is not None,
            "domain_filter":   True,
            # Detailed nested stats
            "stats": {
                "total_book_documents": book_stats.get('total_documents', 0),
                "total_web_documents":  web_stats.get('total_documents', 0),
                "embedding_dimension":  book_stats.get('embedding_dimension', 0),
                "document_types":       book_stats.get('document_types', {}),
                "model": config.get('llm_model', 'Unknown') if config else 'Unknown',
                "features": {
                    "validation":      True,
                    "personalization": True,
                    "web_knowledge":   web_db is not None,
                    "domain_filter":   True,
                    "translation":     True,
                }
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    # Initialize system before starting server
    if initialize_system():
        print("\n🚀 Starting Flask server on http://localhost:5000")
        print("Press Ctrl+C to stop\n")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("\n❌ Failed to initialize system. Exiting.")
        sys.exit(1)
