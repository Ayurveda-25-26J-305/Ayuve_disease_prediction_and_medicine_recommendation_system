"""
Run Web Scraping Pipeline - One-shot script
Step 1: Scrape all trusted Ayurvedic sources
Step 2: Index into FAISS web knowledge base

Run this ONCE to build the web knowledge base.
After this, the hybrid RAG will automatically use both books + web.
"""

import os
import sys

# Add paths
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
sys.path.append(os.path.join(_root, 'scripts'))

# Check required libraries
try:
    import requests
    import bs4
    print("✓ requests and beautifulsoup4 available")
except ImportError:
    print("❌ Missing libraries. Run:")
    print("   pip install requests beautifulsoup4")
    sys.exit(1)

from web_scraper.ayurveda_scraper import scrape_all_sources
from web_scraper.index_web_knowledge import index_web_knowledge

def run_pipeline():
    print("=" * 60)
    print("🚀 AYURVEDIC WEB KNOWLEDGE PIPELINE")
    print("=" * 60)
    print("This will:")
    print("  Step 1: Scrape trusted Ayurvedic websites")
    print("  Step 2: Index into FAISS web knowledge base")
    print("  Result: Hybrid RAG using books + web sources")
    print("=" * 60)

    # Step 1: Scrape
    print("\n📡 STEP 1: SCRAPING AYURVEDIC WEBSITES...")
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "web_scraper", "web_knowledge.json")
    chunks = scrape_all_sources(output_path=json_path)

    if not chunks:
        print("❌ No content scraped. Check internet connection.")
        return False

    print(f"\n✅ Step 1 Complete: {len(chunks)} web chunks collected")

    # Step 2: Index
    print("\n📦 STEP 2: INDEXING INTO FAISS...")
    success = index_web_knowledge(json_path=json_path)

    if success:
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 60)
        print("Your hybrid RAG is ready.")
        print("Both book knowledge and web knowledge will now be used.")
        print("\nWeb index saved to: faiss_index_web/")
        return True
    else:
        print("❌ Indexing failed.")
        return False


if __name__ == "__main__":
    run_pipeline()
