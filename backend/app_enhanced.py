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

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
rag_system = None
config = None


def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_system():
    """Initialize vector DB and Enhanced LLM on startup"""
    global vector_db, rag_system, config
    
    print("=" * 70)
    print("Initializing Enhanced Ayurvedic QA System Backend")
    print("=" * 70)
    
    # Load config
    config = load_config()
    print("\n✓ Config loaded")
    
    # Load vector database
    print("\n🔄 Loading vector database...")
    
    # Get the project root directory (parent of backend)
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    index_path = os.path.join(project_root, config['index_path'])
    
    print(f"   Index path: {index_path}")
    
    vector_db = FAISSVectorDB(
        embedding_model_name=config['embedding_model'],
        index_path=index_path
    )
    
    if vector_db.load_index():
        print(f"✓ Loaded vector database with {vector_db.index.ntotal} documents")
    else:
        print("❌ Failed to load vector database")
        return False
    
    # Initialize Enhanced LLM with validation and personalization
    print("\n🔄 Loading Enhanced LLM (this may take a minute)...")
    rag_system = EnhancedAyurvedicRAG(
        llm_model_name=config['llm_model'],
        max_new_tokens=config.get('max_new_tokens', 64),
        enable_validation=True,
        enable_personalization=True,
        enable_translation=True,  # Enable Sinhala ↔ English translation
        embedding_model=config['embedding_model']
    )
    print("✓ Enhanced LLM initialized (Validation + Personalization + Translation enabled)")
    
    print("\n" + "=" * 70)
    print("✓ Backend ready to serve requests")
    print("=" * 70)
    
    return True


def format_citation(source_doc, validation_info=None):
    """
    Format citation for frontend display with validation info
    Handles both book sources and QA dataset entries
    
    Args:
        source_doc: Source document dictionary
        validation_info: Optional validation info for this source
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
    
    # Add type-specific fields
    if doc_type == "book":
        chapter = metadata.get("chapter", "N/A")
        paragraph = metadata.get("paragraph", metadata.get("verse", "N/A"))
        citation["chapter"] = str(chapter)
        citation["paragraph"] = str(paragraph)
        citation["formatted"] = f"{source_name} – Chapter {chapter} – Verse/Paragraph {paragraph} ({similarity_percentage}% match)"
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
        
        if not question:
            return jsonify({
                "success": False,
                "error": "Question cannot be empty"
            }), 400
        
        print(f"\n🔍 Question: {question}")
        
        # Load user profile if user_id provided
        user_profile = None
        if user_id:
            user_profile = rag_system.load_user_profile(user_id)
            if user_profile:
                print(f"👤 User profile loaded: {user_profile['dominant_dosha']}")
        
        # Get enhanced answer with validation and personalization
        response = rag_system.answer_question(
            question=question,
            vector_db=vector_db,
            top_k=config.get('top_k', 3),
            user_profile=user_profile,
            validation_top_k=5
        )
        
        # Format citations with validation info
        citations = []
        for citation_dict in response.get('citations', []):
            formatted = format_citation(citation_dict)
            citations.append(formatted)
        
        print(f"✓ Answer generated")
        print(f"  Detected language: {response.get('detected_language', 'en').upper()}")
        print(f"  Confidence: {response['validation']['confidence']}%")
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
            "user_info": response.get('user_info', {})
        }
        
        print(f"  Response keys: {list(result.keys())}")
        print(f"  Answer length: {len(result['answer'])}")
        print(f"  Citations count: {len(result['citations'])}")
        
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
        stats = vector_db.get_stats() if vector_db else {}
        
        return jsonify({
            "success": True,
            "stats": {
                "total_documents": stats.get('total_documents', 0),
                "embedding_dimension": stats.get('embedding_dimension', 0),
                "document_types": stats.get('document_types', {}),
                "model": config.get('llm_model', 'Unknown') if config else 'Unknown',
                "features": {
                    "validation": True,
                    "personalization": True
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
