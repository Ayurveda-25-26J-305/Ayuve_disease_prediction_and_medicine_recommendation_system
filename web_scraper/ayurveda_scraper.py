"""
Ayurveda Web Scraper - Domain-Specific Knowledge Collection
Scrapes trusted Ayurvedic websites and prepares data for FAISS indexing.
Only collects from verified, authoritative Ayurvedic sources.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Trusted Ayurvedic Sources Configuration
# Authority weights:
#   1.0 = Classical text / official government
#   0.95 = National Ayurvedic institute
#   0.90 = Peer-reviewed Ayurvedic journal
#   0.80 = Reputable Ayurvedic education site
# ─────────────────────────────────────────────
TRUSTED_SOURCES = [
    {
        "name": "WisdomLib - Ayurveda",
        "base_url": "https://www.wisdomlib.org",
        "urls": [
            "https://www.wisdomlib.org/ayurveda",
            "https://www.wisdomlib.org/definition/vata",
            "https://www.wisdomlib.org/definition/pitta",
            "https://www.wisdomlib.org/definition/kapha",
            "https://www.wisdomlib.org/definition/tridosha",
            "https://www.wisdomlib.org/definition/triphala",
            "https://www.wisdomlib.org/definition/ashwagandha",
            "https://www.wisdomlib.org/definition/brahmi",
            "https://www.wisdomlib.org/definition/panchakarma",
            "https://www.wisdomlib.org/definition/prakriti",
            "https://www.wisdomlib.org/definition/agni",
            "https://www.wisdomlib.org/definition/rasayana",
        ],
        "authority": 0.85,
        "type": "web"
    },
    {
        "name": "Ayurveda Journal (AYU)",
        "base_url": "https://www.ayujournal.org",
        "urls": [
            "https://www.ayujournal.org/text.asp?2011/32/2/173/92554",  # Tridosha concept
        ],
        "authority": 0.90,
        "type": "web"
    },
    {
        "name": "National Ayurvedic Medical Association",
        "base_url": "https://www.ayurvedanama.org",
        "urls": [
            "https://www.ayurvedanama.org/what-is-ayurveda",
            "https://www.ayurvedanama.org/principles-of-ayurveda",
        ],
        "authority": 0.90,
        "type": "web"
    },
    {
        "name": "Chopra Center - Ayurveda",
        "base_url": "https://chopra.com",
        "urls": [
            "https://chopra.com/articles/what-is-ayurveda",
            "https://chopra.com/articles/vata-dosha",
            "https://chopra.com/articles/pitta-dosha",
            "https://chopra.com/articles/kapha-dosha",
            "https://chopra.com/articles/the-three-doshas",
            "https://chopra.com/articles/triphala-the-ayurvedic-wonder-herb",
            "https://chopra.com/articles/ashwagandha-herb-of-the-season",
        ],
        "authority": 0.80,
        "type": "web"
    },
    {
        "name": "Banyan Botanicals - Ayurveda",
        "base_url": "https://www.banyanbotanicals.com",
        "urls": [
            "https://www.banyanbotanicals.com/info/ayurvedic-living/learning-ayurveda/",
            "https://www.banyanbotanicals.com/info/ayurvedic-living/learning-ayurveda/vata-dosha/",
            "https://www.banyanbotanicals.com/info/ayurvedic-living/learning-ayurveda/pitta-dosha/",
            "https://www.banyanbotanicals.com/info/ayurvedic-living/learning-ayurveda/kapha-dosha/",
            "https://www.banyanbotanicals.com/info/ayurvedic-living/learning-ayurveda/the-three-doshas/",
            "https://www.banyanbotanicals.com/info/herbal-encyclopedia/",
        ],
        "authority": 0.80,
        "type": "web"
    },
]

# Minimum text length for a chunk to be considered valid
MIN_CHUNK_LENGTH = 80
MAX_CHUNK_LENGTH = 500

# Output file for scraped data
DEFAULT_OUTPUT_PATH = "web_scraper/web_knowledge.json"


def clean_text(text: str) -> str:
    """Clean scraped text — remove excess whitespace, special chars."""
    if not text:
        return ""
    # Remove multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\u0D80-\u0DFF]', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove HTML entities
    text = re.sub(r'&\w+;', ' ', text)
    return text.strip()


def is_ayurvedic_content(text: str) -> bool:
    """Check if scraped text is genuinely Ayurvedic."""
    ayurveda_terms = [
        'ayurveda', 'dosha', 'vata', 'pitta', 'kapha', 'herb',
        'prakriti', 'panchakarma', 'agni', 'ojas', 'dhatu',
        'rasayana', 'healing', 'medicine', 'triphala', 'ashwagandha'
    ]
    text_lower = text.lower()
    matches = sum(1 for term in ayurveda_terms if term in text_lower)
    return matches >= 1


def chunk_text(text: str, chunk_size: int = MAX_CHUNK_LENGTH, overlap: int = 50) -> List[str]:
    """Split long text into overlapping chunks."""
    words = text.split()
    if len(words) <= chunk_size // 5:  # Short enough — return as-is
        return [text] if len(text) >= MIN_CHUNK_LENGTH else []

    chunks = []
    chunk_words = chunk_size // 5  # Approx words per chunk
    step = chunk_words - (overlap // 5)

    for i in range(0, len(words), step):
        chunk = ' '.join(words[i:i + chunk_words])
        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)

    return chunks


def scrape_page(url: str, source_name: str, authority: float) -> List[Dict[str, Any]]:
    """
    Scrape a single Ayurvedic page and extract text chunks.

    Returns:
        List of document dicts ready for FAISS indexing
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        print(f"   Fetching: {url}")
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            print(f"   ❌ HTTP {response.status_code} for {url}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove nav, footer, scripts, ads
        for tag in soup.find_all(['nav', 'footer', 'script', 'style', 'header',
                                   'aside', 'form', 'button', 'iframe']):
            tag.decompose()

        # Extract paragraphs
        paragraphs = soup.find_all(['p', 'li', 'h2', 'h3'])
        raw_texts = [p.get_text(separator=' ') for p in paragraphs]

        chunks_out = []
        for raw in raw_texts:
            cleaned = clean_text(raw)

            if len(cleaned) < MIN_CHUNK_LENGTH:
                continue

            if not is_ayurvedic_content(cleaned):
                continue

            # Split into smaller chunks if too long
            sub_chunks = chunk_text(cleaned)
            for chunk in sub_chunks:
                chunks_out.append({
                    "text": chunk,
                    "source": source_name,
                    "url": url,
                    "authority": authority,
                    "type": "web",
                    "metadata": {
                        "source_name": source_name,
                        "url": url,
                        "authority": authority,
                        "scraped_at": datetime.now().isoformat()
                    }
                })

        print(f"   ✓ Extracted {len(chunks_out)} chunks from {source_name}")
        return chunks_out

    except requests.exceptions.Timeout:
        print(f"   ⚠️ Timeout for {url}")
        return []
    except Exception as e:
        print(f"   ❌ Error scraping {url}: {e}")
        return []


def scrape_all_sources(output_path: str = DEFAULT_OUTPUT_PATH) -> List[Dict[str, Any]]:
    """
    Scrape all trusted Ayurvedic sources and save to JSON.

    Returns:
        List of all document chunks
    """
    print("\n" + "=" * 60)
    print("🌐 AYURVEDIC WEB SCRAPER")
    print("=" * 60)
    print(f"Sources to scrape: {len(TRUSTED_SOURCES)}")
    print("=" * 60)

    all_chunks = []

    for source in TRUSTED_SOURCES:
        print(f"\n📚 Source: {source['name']} (authority: {source['authority']})")
        source_chunks = []

        for url in source["urls"]:
            chunks = scrape_page(url, source["name"], source["authority"])
            source_chunks.extend(chunks)
            time.sleep(1.5)  # Polite delay between requests

        print(f"   Total from {source['name']}: {len(source_chunks)} chunks")
        all_chunks.extend(source_chunks)

    # Remove near-duplicate chunks (same first 100 chars)
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        key = chunk["text"][:100].lower().strip()
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    print(f"\n✅ Total unique chunks: {len(unique_chunks)} (removed {len(all_chunks) - len(unique_chunks)} duplicates)")

    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique_chunks, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved to: {output_path}")
    return unique_chunks


if __name__ == "__main__":
    # Run the scraper
    chunks = scrape_all_sources()
    print(f"\n✅ Done! {len(chunks)} Ayurvedic web chunks collected.")
