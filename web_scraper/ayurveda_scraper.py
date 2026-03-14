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
        "name": "National Ayurvedic Medical Association",
        "urls": [
            "https://www.ayurvedanama.org/what-is-ayurveda",
            "https://www.ayurvedanama.org/about-ayurveda",
        ],
        "authority": 0.90,
        "type": "web"
    },
    {
        "name": "Banyan Botanicals - Ayurveda",
        "urls": [
            "https://www.banyanbotanicals.com/info/ayurvedic-living/learning-ayurveda/",
        ],
        "authority": 0.80,
        "type": "web"
    },
    {
        "name": "Kerala Ayurveda - Herbs",
        "urls": [
            "https://www.keralaayurveda.biz/blog/ashwagandha",
            "https://www.keralaayurveda.biz/blog/brahmi",
            "https://www.keralaayurveda.biz/blog/triphala",
            "https://www.keralaayurveda.biz/blog/neem",
            "https://www.keralaayurveda.biz/blog/turmeric",
            "https://www.keralaayurveda.biz/blog/ginger",
            "https://www.keralaayurveda.biz/blog/tulsi",
            "https://www.keralaayurveda.biz/blog/vata-dosha",
            "https://www.keralaayurveda.biz/blog/pitta-dosha",
            "https://www.keralaayurveda.biz/blog/kapha-dosha",
        ],
        "authority": 0.82,
        "type": "web"
    },
    {
        "name": "Ayur Times - Ayurvedic Herbs",
        "urls": [
            "https://www.ayurtimes.com/ashwagandha/",
            "https://www.ayurtimes.com/brahmi-bacopa-monnieri/",
            "https://www.ayurtimes.com/triphala/",
            "https://www.ayurtimes.com/turmeric-curcuma-longa/",
            "https://www.ayurtimes.com/neem-azadirachta-indica/",
            "https://www.ayurtimes.com/tulsi-ocimum-sanctum/",
            "https://www.ayurtimes.com/ginger-zingiber-officinale/",
            "https://www.ayurtimes.com/shatavari/",
            "https://www.ayurtimes.com/amla-indian-gooseberry/",
            "https://www.ayurtimes.com/vata-dosha/",
            "https://www.ayurtimes.com/pitta-dosha/",
            "https://www.ayurtimes.com/kapha-dosha/",
            "https://www.ayurtimes.com/panchakarma/",
            "https://www.ayurtimes.com/prakriti-body-type/",
        ],
        "authority": 0.80,
        "type": "web"
    },
    {
        "name": "Himalaya Wellness - Herbs",
        "urls": [
            "https://www.himalayawellness.com/ingredients/ashwagandha.htm",
            "https://www.himalayawellness.com/ingredients/brahmi.htm",
            "https://www.himalayawellness.com/ingredients/neem.htm",
            "https://www.himalayawellness.com/ingredients/turmeric.htm",
            "https://www.himalayawellness.com/ingredients/triphala.htm",
        ],
        "authority": 0.85,
        "type": "web"
    },
    {
        "name": "Healthline - Ayurvedic Herbs",
        "urls": [
            "https://www.healthline.com/nutrition/ashwagandha",
            "https://www.healthline.com/nutrition/triphala",
            "https://www.healthline.com/nutrition/turmeric-and-black-pepper",
            "https://www.healthline.com/nutrition/holy-basil",
            "https://www.healthline.com/health/ayurvedic-treatment-for-diabetes",
        ],
        "authority": 0.75,
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
