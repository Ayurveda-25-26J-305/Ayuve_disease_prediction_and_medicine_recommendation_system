"""
Web Knowledge Indexer
Indexes scraped Ayurvedic web content into a separate FAISS index.
Run this ONCE after scraping to build the web knowledge base.
"""

import os
import sys
import json
import logging

# Add project root and scripts to path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
sys.path.append(os.path.join(_root, 'scripts'))

from vector_db_setup import FAISSVectorDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
WEB_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_knowledge.json")
WEB_INDEX_PATH = os.path.join(_root, "faiss_index_web")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def index_web_knowledge(
    json_path: str = WEB_JSON_PATH,
    index_path: str = WEB_INDEX_PATH,
    embedding_model: str = EMBEDDING_MODEL
):
    """
    Load scraped web knowledge from JSON and index into FAISS.
    """
    print("\n" + "=" * 60)
    print("📦 WEB KNOWLEDGE INDEXER")
    print("=" * 60)

    # Load scraped data
    if not os.path.exists(json_path):
        print(f"❌ Web knowledge file not found: {json_path}")
        print("   ➜ Run 'python web_scraper/ayurveda_scraper.py' first")
        return False

    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("❌ No chunks found in web_knowledge.json")
        return False

    print(f"✓ Loaded {len(chunks)} web chunks from JSON")

    # Initialize a fresh FAISS DB for web sources
    print(f"\n🔧 Initializing FAISS index at: {index_path}")
    web_db = FAISSVectorDB(
        embedding_model_name=embedding_model,
        index_path=index_path
    )

    # Prepare documents in the format FAISSVectorDB.add_documents() expects
    documents = []
    for chunk in chunks:
        doc = {
            "text": chunk["text"],
            "source": chunk.get("source", "Unknown Web Source"),
            "type": "web",
            "url": chunk.get("url", ""),
            "authority": chunk.get("authority", 0.80),
            "metadata": {
                "source_name": chunk.get("source", "Unknown"),
                "url": chunk.get("url", ""),
                "authority": chunk.get("authority", 0.80),
                "scraped_at": chunk.get("metadata", {}).get("scraped_at", "")
            }
        }
        documents.append(doc)

    # Add to FAISS
    print(f"\n📊 Indexing {len(documents)} web documents...")
    web_db.add_documents(documents)

    # Save
    web_db.save_index()

    # Stats
    stats = web_db.get_stats()
    print("\n✅ Web Knowledge Index Built Successfully!")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Index saved to:  {index_path}")
    return True


if __name__ == "__main__":
    success = index_web_knowledge()
    if success:
        print("\n🎉 Done! Web knowledge base is ready.")
        print("   The hybrid RAG system will now use both book and web sources.")
    else:
        print("\n❌ Indexing failed. Check errors above.")
