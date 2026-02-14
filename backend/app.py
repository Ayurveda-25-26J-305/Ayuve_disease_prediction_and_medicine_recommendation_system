"""
Flask Backend API for Ayurvedic QA System
Green Forest Theme Integration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_db_setup import FAISSVectorDB
from llm_architecture import AyurvedicRAG
import yaml

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

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
    """Initialize vector DB and LLM on startup"""
    global vector_db, rag_system, config
    
    print("=" * 70)
    print("Initializing Ayurvedic QA System Backend")
    print("=" * 70)
    
    # Load config
    config = load_config()
    print("\n Config loaded")
    
    # Load vector database
    print("\n Loading vector database...")
    
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
        print(f" Loaded vector database with {vector_db.index.ntotal} documents")
    else:
        print(" Failed to load vector database")
        return False
    
    # Initialize LLM
    print("\n Loading LLM (this may take a minute)...")
    rag_system = AyurvedicRAG(
        llm_model_name=config['llm_model'],
        max_new_tokens=config.get('max_new_tokens', 64)
    )
    print(" LLM initialized")
    
    print("\n" + "=" * 70)
    print(" Backend ready to serve requests")
    print("=" * 70)
    
    return True


def format_citation(source_doc):
    """Format citation for frontend display"""
    metadata = source_doc.get("metadata", {})
    book = source_doc.get("source", "Unknown")
    chapter = metadata.get("chapter", "N/A")
    paragraph = metadata.get("paragraph", metadata.get("verse", "N/A"))
    
    return {
        "book": book,
        "chapter": str(chapter),
        "paragraph": str(paragraph),
        "formatted": f"{book} – Chapter {chapter} – Paragraph {paragraph}"
    }


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "service": "Ayurvedic QA System API",
        "version": "1.0",
        "documents": vector_db.index.ntotal if vector_db else 0
    })


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """
    Main endpoint for asking questions
    
    Request body:
    {
        "question": "What causes Vata imbalance?"
    }
    
    Response:
    {
        "success": true,
        "answer": "...",
        "citations": [...],
        "num_sources": 3
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
        
        if not question:
            return jsonify({
                "success": False,
                "error": "Question cannot be empty"
            }), 400
        
        print(f"\n🔍 Question: {question}")
        
        # Get answer from RAG system
        response = rag_system.answer_question(
            question=question,
            vector_db=vector_db,
            top_k=config.get('top_k', 5)
        )
        
        # Filter to book sources only
        book_sources = [s for s in response['sources'] if s.get('type') == 'book']
        sources_to_cite = book_sources if book_sources else response['sources'][:3]
        
        # Format citations
        citations = [format_citation(source) for source in sources_to_cite]
        
        print(f" Answer generated with {len(citations)} citations")
        
        return jsonify({
            "success": True,
            "answer": response['answer'],
            "citations": citations,
            "num_sources": len(citations)
        })
        
    except Exception as e:
        print(f" Error: {str(e)}")
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
                "model": config.get('llm_model', 'Unknown') if config else 'Unknown'
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
        print("\n Starting Flask server on http://localhost:5000")
        print("Press Ctrl+C to stop\n")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("\n Failed to initialize system. Exiting.")
        sys.exit(1)
