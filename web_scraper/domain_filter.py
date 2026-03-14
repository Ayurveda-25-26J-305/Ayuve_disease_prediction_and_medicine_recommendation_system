"""
Domain Filter - Ayurveda-Only Question Validator
Blocks non-Ayurvedic questions from entering the pipeline.
"""

import re
from typing import Tuple


# ─────────────────────────────────────────────
# Core Ayurvedic keyword list
# ─────────────────────────────────────────────
AYURVEDA_KEYWORDS = [
    # Doshas & constitution
    "vata", "pitta", "kapha", "dosha", "tridosha", "prakriti", "vikriti",
    # Classical texts & system
    "ayurveda", "ayurvedic", "charaka", "sushruta", "ashtanga", "hridaya",
    "samhita", "nidana", "rasayana", "panchakarma", "dinacharya", "ritucharya",
    # Body concepts
    "dhatu", "agni", "ama", "ojas", "prana", "srotas", "marma",
    # Common Ayurvedic herbs / ingredients
    "triphala", "ashwagandha", "brahmi", "neem", "turmeric", "haldi",
    "ginger", "tulsi", "amla", "amalaki", "shatavari", "guduchi",
    "giloy", "licorice", "yashtimadhu", "haritaki", "bibhitaki",
    "manjistha", "noni", "moringa", "ghee", "sesame", "mustard",
    "coriander", "fenugreek", "cumin", "cardamom", "cinnamon",
    "pepper", "trikatu", "chyawanprash", "hingvastak",
    # Treatments & therapies
    "abhyanga", "shirodhara", "basti", "nasya", "virechana", "vamana",
    "massage", "oil massage", "detox", "cleanse", "purgation",
    # Health concepts in Ayurveda
    "digestion", "metabolism", "immunity", "liver", "inflammation",
    "constipation", "flatulence", "bloating", "acidity", "indigestion",
    "skin", "hair", "joint", "arthritis", "diabetes", "fever",
    "cold", "cough", "respiratory", "sleep", "insomnia", "stress",
    "anxiety", "depression", "weight", "obesity", "blood sugar",
    # Sinhala Ayurvedic terms (romanized)
    "hela osu", "deshiya", "weda", "wedakama", "osu",
    "kurundu", "inguru", "gotu kola", "polpala", "ranawara",
    "iramusu", "beli", "goraka", "kohomba", "kothala himbutu",
]

# Non-Ayurvedic block patterns (explicit rejection)
NON_AYURVEDA_PATTERNS = [
    r'\b(python|java|javascript|coding|program|algorithm|code|software|database)\b',
    r'\b(cricket|football|sports|game|match|tournament|player)\b',
    r'\b(politics|election|president|government|parliament|minister)\b',
    r'\b(movie|film|actor|actress|music|song|album|artist)\b',
    r'\b(recipe|pizza|burger|pasta|noodles|sushi|fast food)\b',
    r'\b(stock|share|invest|finance|crypto|bitcoin|economy)\b',
    r'\b(history|geography|science|physics|chemistry|biology|math)\b',
    r'\b(weather|climate|temperature|forecast|rain|snow)\b',
]


def is_ayurvedic_question(question: str) -> Tuple[bool, str]:
    """
    Check if a question is Ayurveda-related.

    Returns:
        (is_ayurvedic: bool, reason: str)
    """
    if not question or not question.strip():
        return False, "Empty question"

    q_lower = question.lower().strip()

    # 1. Check explicit non-Ayurvedic patterns first
    for pattern in NON_AYURVEDA_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            return False, f"Question is not related to Ayurveda (detected non-Ayurvedic topic)"

    # 2. Check for Ayurvedic keywords
    for keyword in AYURVEDA_KEYWORDS:
        if keyword in q_lower:
            return True, f"Ayurvedic keyword detected: '{keyword}'"

    # 3. Check for general health questions that could be Ayurvedic
    general_health = [
        r'\b(herb|herbal|plant|natural remedy|natural medicine|traditional medicine)\b',
        r'\b(heal|cure|treat|remedy|medicine|medicinal)\b',
        r'\b(body|health|wellness|wellbeing|holistic)\b',
    ]
    for pattern in general_health:
        if re.search(pattern, q_lower, re.IGNORECASE):
            return True, "General health/herbal query — processed as Ayurvedic context"

    return False, "Question does not appear to be related to Ayurveda"


def get_rejection_message() -> str:
    """Return a user-friendly rejection message."""
    return (
        "🌿 This system is specialized for Ayurvedic health knowledge only. "
        "Please ask questions related to Ayurveda, such as herbs, doshas (Vata/Pitta/Kapha), "
        "Ayurvedic treatments, or traditional health remedies."
    )
