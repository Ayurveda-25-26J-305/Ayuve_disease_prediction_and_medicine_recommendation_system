#!/bin/bash
set -e

echo "============================================================"
echo "  Ayurvedic QA System - Railway Startup"
echo "============================================================"

# Mirrors the PYTHONPATH Colab sets so all modules resolve correctly
export PYTHONPATH="$(pwd):$(pwd)/scripts:${PYTHONPATH}"

# --- Book FAISS index (4839 docs from books + QA dataset) ---
if [ ! -f "faiss_index/index.faiss" ]; then
    echo ""
    echo ">>> Book FAISS index not found. Building (~2-3 min)..."
    python scripts/build_vector_db.py
    echo ">>> Book FAISS index built!"
else
    echo ">>> Book FAISS index found, skipping build."
fi

# --- Web knowledge index (159 curated docs) ---
if [ ! -f "faiss_index_web/index.faiss" ]; then
    echo ""
    echo ">>> Web knowledge index not found. Building (~1-2 min)..."
    python -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from web_scraper.generate_curated_knowledge import generate_curated_knowledge
from web_scraper.index_web_knowledge import index_web_knowledge
docs = generate_curated_knowledge('web_scraper/web_knowledge.json')
print(f'Generated {len(docs)} web documents')
index_web_knowledge(json_path='web_scraper/web_knowledge.json', index_path='faiss_index_web')
print('Web index built!')
" || echo ">>> Web index build failed — continuing with books only."
else
    echo ">>> Web knowledge index found, skipping build."
fi

echo ""
echo ">>> Starting Flask server on port ${PORT:-5000}..."
exec python backend/app_enhanced.py
