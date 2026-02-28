"""
Force GPU Execution - Enhanced RAG
This version forces the model to stay on GPU (may use more memory)
"""

import torch
import logging
from typing import Dict, Any, Optional

# Use the working LLM from main system (no cache issues)
from llm_architecture import LLMArchitecture

# Import the new engines
from validation_engine import ValidationEngine
from personalization_engine import PersonalizationEngine
from translation_service import TranslationService

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Query Expansion Map
# Maps everyday health words → Ayurvedic synonyms so FAISS finds the right
# passages even when the user's phrasing differs from the book vocabulary.
# ─────────────────────────────────────────────────────────────────────────────
QUERY_EXPANSION_MAP = {
    # Head / Neurological
    'headache':     'headache shiras shirashula head pain migraine treatment remedy',
    'head':         'head shiras skull brain headache treatment',
    'migraine':     'migraine headache shiras shirashula pain',
    'dizziness':    'dizziness vertigo bhrama head spinning treatment',
    'insomnia':     'insomnia sleep nidra sleeplessness treatment',
    'memory':       'memory medhya brain mental cognition treatment',
    # Pain / Musculoskeletal
    'pain':         'pain vedana ache treatment remedy relief vata',
    'shoulder':     'shoulder amsa skandha joint pain treatment oil massage',
    'joint':        'joint sandhi amavata arthritis pain treatment oil',
    'back':         'back pain kati kativata spine treatment',
    'knee':         'knee janu joint pain treatment',
    'arthritis':    'arthritis amavata joint pain sandhi treatment',
    'inflammation': 'inflammation shotha swelling pitta treatment remedy',
    'stiff':        'stiffness vata joint pain oil massage',
    # Digestive
    'digestion':    'digestion agni digestive fire jatharagni treatment',
    'constipation': 'constipation vibandha bowel vata treatment',
    'diarrhea':     'diarrhea atisara loose motion bowel treatment',
    'stomach':      'stomach udara abdomen agni digestion treatment',
    'gas':          'gas flatulence adhmana bloating vata treatment',
    'acidity':      'acidity amlapitta acid reflux pitta treatment',
    'nausea':       'nausea vomiting chardi chhardi treatment',
    # Respiratory
    'cough':        'cough kasa respiratory bronchial treatment remedy',
    'cold':         'cold pratishyaya rhinitis nasal congestion treatment',
    'fever':        'fever jwara temperature heat treatment remedy',
    'asthma':       'asthma shwasa tamaka breathing treatment',
    # Skin / Hair
    'skin':         'skin kushtha twak dermatitis treatment remedy',
    'rash':         'rash skin eruption kushtha pitta treatment',
    'hair':         'hair kesha loss scalp treatment oil',
    # Eyes
    'eye':          'eye netra akshi vision sight treatment',
    # General Wellness
    'fatigue':      'fatigue tiredness weakness kshaya ojas treatment',
    'stress':       'stress anxiety vata mental nervousness treatment',
    'anxiety':      'anxiety vata stress mental nervous treatment remedy',
    'weight':       'weight obesity kapha medovruddhi metabolism',
    'diabetes':     'diabetes prameha sugar blood glucose treatment',
    'immunity':     'immunity ojas bala strength resistance treatment',
    # Herbs commonly asked
    'cinnamon':     'cinnamon tvak cassia kurudu properties benefits uses treatment',
    'turmeric':     'turmeric haridra curcumin kaha properties benefits uses',
    'ginger':       'ginger sunthi ardrakam inguru properties benefits',
    'ashwagandha':  'ashwagandha withania adaptogen stress strength benefits',
    'neem':         'neem nimba antibacterial skin blood purifier benefits',
    'aloe':         'aloe vera kumari komarika skin digestive benefits',
    'pepper':       'pepper maricha gammiris piperine properties benefits',
    'cumin':        'cumin suduru jeeraka digestion properties benefits',
}


# Using LLMArchitecture from llm_architecture.py (imported above)
# This avoids the DynamicCache error


class EnhancedAyurvedicRAG:
    """Enhanced RAG - FORCE GPU"""
    
    def __init__(
        self, 
        llm_model_name: str,
        max_new_tokens: int = 64,
        enable_validation: bool = True,
        enable_personalization: bool = True,
        enable_translation: bool = True,
        embedding_model: str = "BAAI/bge-base-en-v1.5"
    ):
        # Initialize base LLM
        self.llm = LLMArchitecture({
            "llm_model": llm_model_name,
            "max_new_tokens": max_new_tokens,
        })
        
        # Initialize validation engine
        self.enable_validation = enable_validation
        if enable_validation:
            logger.info("Initializing ValidationEngine...")
            self.validator = ValidationEngine(embedding_model_name=embedding_model)
            logger.info("✓ ValidationEngine ready")
        else:
            self.validator = None
        
        # Initialize personalization engine
        self.enable_personalization = enable_personalization
        if enable_personalization:
            logger.info("Initializing PersonalizationEngine...")
            self.personalizer = PersonalizationEngine(llm_generator=self.llm)
            logger.info("✓ PersonalizationEngine ready")
        else:
            self.personalizer = None
        
        # Initialize translation service
        self.enable_translation = enable_translation
        if enable_translation:
            logger.info("Initializing TranslationService...")
            try:
                self.translator = TranslationService(device=self.llm.device)
                if self.translator.is_available():
                    logger.info("✓ TranslationService ready (English ↔ Sinhala)")
                else:
                    logger.warning("⚠️  Translation models not loaded, translation disabled")
                    self.translator = None
                    self.enable_translation = False
            except Exception as e:
                logger.warning(f"⚠️  Translation initialization failed: {e}")
                self.translator = None
                self.enable_translation = False
        else:
            self.translator = None
        
        logger.info("EnhancedAyurvedicRAG initialized (GPU FORCED)")
    
    # Phrases that indicate the model broke character and started talking about itself,
    # or is echoing source/author metadata instead of answering the question.
    _SELF_TALK_PATTERNS = [
        r"i apologize", r"apologies for", r"i'm sorry", r"sorry for",
        r"let me (re)?address", r"let'?s address", r"let me clarify",
        r"as an ai", r"as a language model", r"i need to clarify",
        r"i must clarify", r"i should note", r"please note that",
        r"i cannot", r"i can'?t provide", r"i will now",
        r"regarding your (request|question)",
        r"to address your", r"now regarding", r"let me now",
        r"for any confusion", r"any confusion earlier",
        # Author-echo patterns — model citing sources it shouldn't
        r"\w+'?s\s+(book|insights?|research|work|findings?|text|study|view)",
        r"according to \w+",
        r"as mentioned (in|by|across)",
        r"as stated (in|by)",
        r"as described (in|by)",
        # Knowledge-base disclaimer patterns
        r"note:.*assistant",
        r"outside.*knowledge base",
        r"beyond.*established facts",
        r"hypothetical scenarios",
        r"information.*april 20",
        r"knowledge cutoff",
        r"cannot address",
    ]

    def _clean_garbled_text(self, text: str) -> str:
        """
        Clean 4-bit quantized model output:
        1. Remove tokenization artifacts  (turmer03r, exac0stuate, btwn)
        2. Remove LLM self-talk lines     (I apologize, as an AI, let's address)
        3. Truncate a bullet at the point where self-talk begins mid-sentence
        4. Discard short/incomplete lines (<20 chars of real content)
        """
        import re as _re

        if not text:
            return text

        # Normalize common shorthand before any other processing
        text = text.replace("w/o ", "without ").replace("w/ ", "with ")
        text = _re.sub(r'\bw/', 'with', text)  # catch 'w/Pittahd' etc.
        # Remove possessive author-echo patterns like "Nithya Rajan's insights"
        text = _re.sub(
            r"\b[A-Z][a-z]+(\s+[A-Z][a-z]+)?'s\s+(book|insights?|research|work|findings?|text|study|view)s?",
            "", text, flags=_re.IGNORECASE
        )

        # Compile self-talk pattern once
        self_talk_re = _re.compile(
            '|'.join(self._SELF_TALK_PATTERNS), _re.IGNORECASE
        )

        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # --- SELF-TALK: skip the whole line if it IS self-talk ---
            content_lower = stripped.lower()
            if self_talk_re.search(content_lower) and not stripped.startswith('•'):
                continue

            # --- SELF-TALK: if a bullet contains self-talk mid-sentence, truncate ---
            if stripped.startswith('•'):
                m_st = self_talk_re.search(stripped)
                if m_st:
                    # Keep only the text before the self-talk kicks in
                    truncated = stripped[:m_st.start()].strip().rstrip(',:; ')
                    if len(truncated) < 22:  # too short after truncation — discard
                        continue
                    # Ensure it ends with a period
                    if truncated and truncated[-1] not in '.!?':
                        truncated += '.'
                    stripped = truncated
                    line = stripped

            # Extract bullet prefix if present
            prefix = ''
            content = stripped
            m = _re.match(r'^([•\-]\s*)', stripped)
            if m:
                prefix = m.group(1)
                content = stripped[m.end():]

            # --- GARBLED WORD REMOVAL ---
            words = content.split()
            good_words = []
            for word in words:
                pm = _re.search(r'[.,!?;:]+$', word)
                if pm:
                    core = word[:pm.start()]
                else:
                    core = word

                has_letter = bool(_re.search(r'[a-zA-Z]', core))
                has_digit  = bool(_re.search(r'\d', core))
                is_ordinal = bool(_re.match(r'^\d+(st|nd|rd|th)$', core, _re.I))
                is_number  = bool(_re.match(r'^\d+$', core))

                # Drop words with mixed letters+digits (e.g. turmer03r)
                if has_letter and has_digit and not is_ordinal and not is_number:
                    continue

                # Drop consonant-only clusters ≥3 chars (e.g. btwn, wth)
                if (len(core) >= 3 and has_letter and not has_digit
                        and not _re.search(r'[aeiouAEIOU]', core)
                        and core.lower() not in {'gym', 'dry', 'try', 'why',
                                                  'fly', 'sky', 'cry', 'fry',
                                                  'sly', 'spy', 'shy', 'thy',
                                                  'nth'}):
                    continue

                good_words.append(word)

            if not good_words:
                continue

            cleaned = prefix + ' '.join(good_words)
            cleaned = _re.sub(r'  +', ' ', cleaned).strip()

            # Discard very short lines (fragments)
            real_content = _re.sub(r'^[•\-]\s*', '', cleaned).strip()
            if len(real_content) < 20:
                continue

            # Ensure ends with punctuation
            if cleaned and cleaned[-1] not in '.!?':
                cleaned += '.'

            cleaned_lines.append(cleaned)

        return '\n'.join(cleaned_lines)

    def _build_context_with_citations(self, docs):
        """Build LLM context including citations.
        NOTE: Author/book names are intentionally excluded here to prevent the LLM
        from echoing them in the answer. Citation data is stored in doc metadata
        and added to the API response separately.
        """
        context_blocks = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            doc_type = doc.get("type", "unknown")
            
            if doc_type == "book":
                # 500 chars gives LLM richer content to synthesise informative bullets
                doc_text = doc.get("text", "")[:500]
                block = f"[Ayurvedic Knowledge {i}]\n{doc_text}"
            else:
                # QA dataset: include the related question as useful context
                qa_question = meta.get("question", "")
                doc_text = doc.get("text", "")[:500]
                if qa_question:
                    block = f"[Ayurvedic Q&A {i}]\nRelated: {qa_question}\n{doc_text}"
                else:
                    block = f"[Ayurvedic Q&A {i}]\n{doc_text}"
            
            context_blocks.append(block.strip())
        
        return "\n\n".join(context_blocks)
    
    def _format_answer(self, raw_answer: str) -> str:
        """
        Post-process generated answer to ensure quality formatting.
        Fixes run-on sentences, ensures bullet points, and improves readability.
        """
        if not raw_answer or not raw_answer.strip():
            return ""
        
        import re
        
        # Remove any leading/trailing whitespace
        answer = raw_answer.strip()
        
        # If answer doesn't have bullet points, try to add them
        if not any(answer.startswith(c) for c in ['-', '•', '*', '1.', '2.']):
            # Split by periods followed by capital letters (likely sentence boundaries)
            sentences = re.split(r'\.(?=[A-Z])', answer)
            if len(sentences) > 2:
                # Format as bullet points
                answer = '\n'.join([f"- {s.strip()}." for s in sentences if len(s.strip()) > 10])
            else:
                # Single sentence - just add bullet
                answer = f"- {answer}" if not answer.startswith('-') else answer
        
        # Fix excessive run-on sentences (sentences longer than 200 chars)
        lines = answer.split('\n')
        formatted_lines = []
        
        for line in lines:
            if len(line) > 200 and '.' not in line[-50:]:
                # This is a run-on sentence without proper ending
                # Split at conjunctions or commas
                parts = re.split(r'(?<=\w)(?:,\s+(?:and|while|also|by|through|including|thereby|hence|thus))', line, maxsplit=2)
                for part in parts[:3]:  # Limit to first 3 points
                    if len(part.strip()) > 15:
                        clean_part = part.strip()
                        if not clean_part.endswith('.'):
                            clean_part += '.'
                        if not clean_part.startswith(('-', '•', '*')):
                            clean_part = f"- {clean_part}"
                        formatted_lines.append(clean_part)
            else:
                formatted_lines.append(line)
        
        # Rejoin and clean up
        answer = '\n'.join(formatted_lines)
        
        # Ensure bullet points are consistent
        answer = re.sub(r'^\*\s+', '- ', answer, flags=re.MULTILINE)
        answer = re.sub(r'^•\s+', '- ', answer, flags=re.MULTILINE)
        
        # Remove excessive whitespace
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        
        # Capitalize first letter of each bullet point
        lines = answer.split('\n')
        formatted_lines = []
        for line in lines:
            if line.strip().startswith('-'):
                # Extract bullet and content
                match = re.match(r'^(-\s+)(.+)$', line)
                if match:
                    bullet, content = match.groups()
                    # Capitalize first letter
                    content = content[0].upper() + content[1:] if content else content
                    formatted_lines.append(f"{bullet}{content}")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        # === DEDUPLICATION: Remove similar/repetitive bullets ===
        unique_lines = []
        seen_keywords = set()
        
        for line in formatted_lines:
            if not line.strip() or not line.strip().startswith('-'):
                unique_lines.append(line)
                continue
            
            # Extract key terms (nouns, important words)
            content = re.sub(r'^-\s+', '', line.strip()).lower()
            # Extract main keywords (words longer than 4 chars, excluding common words)
            words = re.findall(r'\b\w{5,}\b', content)
            
            # Skip bullet if it's too similar to previous ones
            keywords = set(words)
            if keywords:
                # Check overlap with seen keywords
                overlap = len(keywords & seen_keywords) / len(keywords)
                if overlap < 0.6:  # Less than 60% overlap = unique enough
                    unique_lines.append(line)
                    seen_keywords.update(keywords)
            else:
                unique_lines.append(line)
        
        return '\n'.join(unique_lines).strip()
    
    # Sanskrit/Ayurvedic compound words that Google Translate garbles badly when
    # sent into Sinhala. Replace these with simpler English equivalents.
    _SANSKRIT_SIMPLIFY = [
        # Compound proper nouns from texts → readable description
        (r'\bPurisha Chaksma\b', 'colon cleansing'),
        (r'\bPratyaya\b', 'principle'),
        (r'\bKleshtana\b', 'toxin removal'),
        (r'\bLekhana\b', 'cleansing'),
        (r'\bAmrita Rasayana\b', 'Ayurvedic tonic'),
        (r'\bAgni\b', 'digestive fire'),
        (r'\bAma\b', 'toxins'),
        (r'\bOjas\b', 'vital energy'),
        (r'\bPrana\b', 'life force'),
        (r'\bDhatus?\b', 'body tissues'),
        (r'\bDosha\b', 'body constitution'),
        (r'\bVata\b', 'Vata'),
        (r'\bPitta\b', 'Pitta'),
        (r'\bKapha\b', 'Kapha'),
        # Remove overly long Sanskrit compound terms that don't translate
        (r'\b[A-Z][a-z]+(dh?atu|rasa|guna|karma|chikitsa|drav)?[A-Z][a-z]+\b', ''),
    ]

    def _simplify_for_translation(self, english_text: str) -> str:
        """
        Simplify English text before translation to improve Sinhala quality.
        - Replace Sanskrit compound terms with plain English equivalents
        - Break down complex sentences
        - Remove excessive technical jargon
        - Make sentences more translation-friendly
        """
        import re
        
        if not english_text or not english_text.strip():
            return ""

        # Replace known Sanskrit compounds with readable English
        for pattern, replacement in self._SANSKRIT_SIMPLIFY:
            english_text = re.sub(pattern, replacement, english_text, flags=re.IGNORECASE)

        # Remove leftover CamelCase compound Sanskrit terms that weren't caught
        # e.g. 'PachakapittamLekhana', 'KleshtnaKarma' — keep first capital word only
        english_text = re.sub(r'\b([A-Z][a-z]+)([A-Z][a-z]+){2,}\b', r'\1', english_text)

        # Clean up doubled spaces from removals
        english_text = re.sub(r'  +', ' ', english_text).strip()
        
        # Split into lines (bullets)
        lines = english_text.split('\n')
        simplified_lines = []
        
        for line in lines:
            if not line.strip():
                continue
            
            # Extract bullet marker and content
            if line.strip().startswith('•'):
                bullet = '• '
                content = re.sub(r'^•\s*', '', line.strip())
            elif line.strip().startswith('-'):
                bullet = '- '
                content = line.strip()[2:].strip()
            else:
                bullet = ''
                content = line.strip()
            
            # Simplify: Remove parenthetical phrases that don't translate well
            content = re.sub(r'\s*\([^)]+\)\s*', ' ', content)
            
            # Simplify: Break very long sentences (>150 chars) at conjunctions
            if len(content) > 150:
                # Try to keep first clause only (up to first major conjunction)
                parts = re.split(r',\s+(?:and|while|thereby|hence|thus|including)\s+', content, maxsplit=1)
                if parts and len(parts[0]) > 50:
                    content = parts[0]
                    if not content.endswith('.'):
                        content += '.'
            
            # Clean up extra spaces
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Ensure ends with period
            if content and not content.endswith(('.', '!', '?')):
                content += '.'
            
            if content and len(content) > 15:  # Skip very short fragments
                simplified_lines.append(f"{bullet}{content}")
        
        return '\n'.join(simplified_lines)
    
    def _cleanup_translated_answer(self, translated_text: str) -> str:
        """
        Post-process translated Sinhala text.
        Keeps any line that has meaningful Sinhala content.
        Previously over-strict vowel-ending checks were discarding valid translations.
        """
        import re

        if not translated_text or not translated_text.strip():
            return ""

        lines = translated_text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Count Sinhala Unicode characters in this line
            sinhala_chars = len(re.findall(r'[\u0D80-\u0DFF]', line))

            # Keep lines that either:
            # (a) have at least 4 Sinhala characters and ≥8 total chars, OR
            # (b) are section headings (contain ':'), OR
            # (c) are short but have some Sinhala content (≥2 chars) for numbered items
            is_heading = ':' in line and len(line) < 60
            has_number_prefix = bool(re.match(r'^\d+\.', line))
            has_bullet = line.startswith('•') or line.startswith('-')

            if is_heading and sinhala_chars >= 1:
                cleaned_lines.append(line)
            elif has_number_prefix and sinhala_chars >= 2:
                cleaned_lines.append(line)
            elif has_bullet and sinhala_chars >= 4:
                cleaned_lines.append(line)
            elif sinhala_chars >= 6 and len(line) >= 8:
                cleaned_lines.append(line)
            # Drop lines with almost no Sinhala (likely untranslated English fragments)

        if not cleaned_lines:
            # Nothing passed — return original so caller can decide
            return translated_text

        return '\n'.join(cleaned_lines)

    def _add_term_clarification(self, answer: str, original_question: str, is_romanized: bool) -> str:
        """
        Add term clarification to answer for romanized Singlish queries.
        Example: "Kurudu (known as cinnamon in English) has the following benefits:"
        
        Args:
            answer: The English answer
            original_question: The original romanized question
            is_romanized: Whether the question was romanized Singlish
            
        Returns:
            Answer with term clarification prepended if applicable
        """
        if not is_romanized or not self.translator:
            return answer
        
        # Import the dictionary
        from translation_service import SINHALA_TO_ENGLISH_DICT
        
        # Extract romanized keywords from the question
        import re
        words = re.findall(r'\b\w+\b', original_question.lower())
        
        # Find herb/ingredient keywords that were translated
        clarifications = []
        seen = set()
        
        for word in words:
            if word in SINHALA_TO_ENGLISH_DICT and word not in seen:
                english_word = SINHALA_TO_ENGLISH_DICT[word]
                
                # Only clarify nouns (herbs, ingredients, etc.) not question words
                question_words = {'monawada', 'mokadda', 'mokada', 'kohomada', 'kawuda', 
                                'kauda', 'kiyada', 'keyada', 'da', 'eka', 'wala', 'walata'}
                
                if word not in question_words and len(english_word) > 3:
                    # Capitalize for clarity
                    clarifications.append(f"{word.capitalize()} ({english_word})")
                    seen.add(word)
        
        # If we found terms to clarify, prepend to answer
        if clarifications:
            if len(clarifications) == 1:
                clarification_text = f"About {clarifications[0]}:\n\n"
            else:
                terms = ', '.join(clarifications[:-1]) + f" and {clarifications[-1]}"
                clarification_text = f"About {terms}:\n\n"
            
            return clarification_text + answer
        
        return answer

    def _detect_dosha_from_question(self, question: str, answer: str) -> str:
        """
        Detect the most relevant dosha from the question and answer text.
        Returns one of: 'Vata', 'Pitta', 'Kapha', or 'General'
        """
        text = (question + " " + answer).lower()

        vata_keywords = [
            'vata', 'wind', 'air', 'dry', 'anxious', 'anxiety', 'irregular',
            'constipation', 'insomnia', 'bloating', 'gas', 'nervous', 'thin',
            'cold', 'joint', 'pain', 'stiff', 'moving', 'mobile', 'variable'
        ]
        pitta_keywords = [
            'pitta', 'fire', 'heat', 'hot', 'inflammation', 'acid', 'acidity',
            'rash', 'skin', 'fever', 'burning', 'irritable', 'anger', 'liver',
            'digestion', 'metabolism', 'sharp', 'intense', 'focus', 'eye'
        ]
        kapha_keywords = [
            'kapha', 'water', 'earth', 'mucus', 'congestion', 'weight', 'heavy',
            'slow', 'lethargy', 'depression', 'cold', 'damp', 'lung', 'cough',
            'stable', 'steady', 'oily', 'smooth', 'sweet', 'excess'
        ]

        vata_score = sum(1 for w in vata_keywords if w in text)
        pitta_score = sum(1 for w in pitta_keywords if w in text)
        kapha_score = sum(1 for w in kapha_keywords if w in text)

        scores = {'Vata': vata_score, 'Pitta': pitta_score, 'Kapha': kapha_score}
        dominant = max(scores, key=scores.get)

        # Only return a dosha if there's meaningful signal
        if scores[dominant] == 0:
            return 'General'
        if scores[dominant] == max(vata_score, pitta_score, kapha_score):
            # Check for tie
            top_score = scores[dominant]
            tied = [d for d, s in scores.items() if s == top_score]
            if len(tied) > 1:
                return '-'.join(tied)

        return dominant

    def _generate_personalized_tips(self, question: str, answer: str, dosha: str) -> str:
        """
        Generate 2-3 short personalized tips relevant to the question/answer context,
        tailored to the detected dosha. Tips must be different from the answer.
        """
        print(f"💡 Generating personalized tips for {dosha} dosha...")
        try:
            import re as _re
            topic_line = question[:60].strip()
            # Only allow real Ayurvedic herbs — prevents hallucinated ingredients
            AYURVEDIC_HERBS = (
                "turmeric, ginger, ashwagandha, neem, tulsi, triphala, amla, brahmi, "
                "sesame oil, ghee, cardamom, cumin, coriander, fennel, guduchi, shatavari, "
                "licorice, pippali, haritaki, bibhitaki, sandalwood, coconut oil"
            )
            prompt = (
                f"You are an Ayurvedic doctor. Patient question: \"{question}\"\nDosha: {dosha}\n\n"
                f"Give 4 practical Ayurvedic lifestyle tips for a {dosha} type person.\n\n"
                f"STRICT RULES:\n"
                f"• Each tip: 1 complete sentence, 15–20 words, ending with a period.\n"
                f"• ONLY use herbs and practices from this list: {AYURVEDIC_HERBS}\n"
                f"• Do NOT invent fictional remedies or non-Ayurvedic ingredients.\n"
                f"• Each tip must be specific and actionable.\n"
                f"• Do NOT mention author names, book titles, or source labels.\n"
                f"• Do NOT use abbreviations like 'w/' — write full words only.\n"
                f"• Do NOT invent Sanskrit/Ayurvedic compound terms not in the herbs list.\n"
                f"• Start every tip with • symbol on its own line. NO introduction text.\n\n"
                f"Example (topic: digestion):\n"
                f"• Drink warm ginger tea before meals to stimulate Agni and ease digestion.\n"
                f"• Add a pinch of turmeric and cumin to cooked meals to balance Pitta dosha.\n"
                f"• Avoid cold drinks and raw food which aggravate Vata and slow digestion.\n"
                f"• Take one teaspoon of triphala powder with warm water before bed nightly.\n\n"
                f"Now give 4 tips for {dosha} dosha about: {topic_line}\n\n"
                f"•"
            )
            messages = [
                {"role": "user", "content": prompt}
            ]
            raw_tips = self.llm.generate_from_messages(
                messages, max_new_tokens=200, min_new_tokens=60, temperature=0.25
            )

            # Remove tokenization artifacts (e.g. 'turmer03r', 'btwn') from tips
            raw_tips = self._clean_garbled_text(raw_tips)

            # Strip any leading bullet/dash the model added (prompt ends with '•')
            # Prevents double-prefix like '• • text' or '• - text'
            raw_tips = _re.sub(r'^[\s•\-\*]+', '', raw_tips).strip()
            raw_tips = ("• " + raw_tips).strip()

            def _clip_bullet(b: str) -> str:
                b = b.strip()
                b = _re.sub(r'^(•\s*)[\-\*•]+\s*', r'\1', b)
                m = _re.search(r'(?<=[.!?])(?:\s|$)', b[40:])
                if m:
                    b = b[:40 + m.start() + 1].strip()
                if len(b) > 200:
                    b = b[:200].rsplit(' ', 1)[0].rstrip(',:;') + '.'
                return b

            if '•' in raw_tips:
                tips_raw = raw_tips
            else:
                converted = _re.sub(r'^\d+\.\s+', '• ', raw_tips, flags=_re.MULTILINE)
                tips_raw = converted if '•' in converted else '\n'.join(
                    f'• {s.strip()}' for s in _re.split(r'(?<=[.!?])\s+', raw_tips) if s.strip()
                )

            tip_lines = [l for l in tips_raw.splitlines() if l.strip().startswith('•')]
            tip_bullets = [_clip_bullet(t) for t in tip_lines[:4]]

            # Fallback: if model didn't produce 3+ bullet lines, split on sentence boundaries
            if len(tip_bullets) < 3:
                extra = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', raw_tips)
                         if len(s.strip()) > 20]
                used = set()
                for b in tip_bullets:
                    used.update(b.lower().split())
                for sent in extra:
                    if len(tip_bullets) >= 4:
                        break
                    clean = _re.sub(r'^[\s•\-\*\d\.]+', '', sent).strip()
                    if clean and len(clean) > 20 and len(set(clean.lower().split()) & used) <= 3:
                        tip_bullets.append(_clip_bullet(f'• {clean}'))
                        used.update(clean.lower().split())

            tips = '\n'.join(tip_bullets)

            print(f"✓ Tips generated: {tips[:100]}...")
            return tips
        except Exception as e:
            print(f"⚠️  Tip generation failed: {e}")
            return ""

    def _generate_structured_herb_answer(self, herb_name: str, question: str, context_summary: str) -> str:
        """
        Generate a structured multi-section herb answer using few-shot continuation.
        Phi-3-mini works reliably with completed examples — no placeholder syntax.
        """
        print(f"🌿 Generating structured answer for herb: '{herb_name}'")
        try:
            import re as _re_s
            h = herb_name  # short alias
            prompt = (
                f"You are an Ayurvedic doctor. Write a health guide about {h}.\n"
                f"Only write about {h}. Be specific and factual. No author names.\n\n"
                f"Ayurvedic context:\n{context_summary[:1000]}\n\n"
                f"--- EXAMPLE (topic: turmeric) ---\n"
                f"Benefits:\n"
                f"1. Reduces inflammation and eases joint pain and swelling effectively.\n"
                f"2. Improves digestion and stimulates bile production in the liver.\n"
                f"3. Boosts immunity and fights bacterial and viral infections naturally.\n"
                f"4. Balances blood sugar levels and supports healthy metabolism.\n\n"
                f"How to use:\n"
                f"- Add half teaspoon of turmeric powder to warm milk and drink at night.\n"
                f"- Mix turmeric with honey and take one teaspoon before meals.\n\n"
                f"Ayurvedic note:\n"
                f"- Balances Pitta and Kapha doshas and strengthens digestive fire (Agni).\n\n"
                f"Caution:\n"
                f"- Do not exceed one teaspoon per day. Avoid on empty stomach if you have acidity.\n"
                f"--- END EXAMPLE ---\n\n"
                f"Now write the same guide for {h}:\n\n"
                f"Benefits:\n"
            )
            messages = [{"role": "user", "content": prompt}]
            raw = self.llm.generate_from_messages(
                messages, max_new_tokens=380, min_new_tokens=150, temperature=0.25
            )
            raw = self._clean_garbled_text(raw)

            # Model may echo 'Benefits:' as first line — strip it
            raw_body = _re_s.sub(r'^\s*Benefits:\s*\n?', '', raw.lstrip(), count=1).lstrip()

            # Validate: structured answer must contain at least 2 numbered items
            if not _re_s.search(r'^\d+\.', raw_body, _re_s.MULTILINE):
                print("⚠️  Structured gen: no numbered items found, falling back")
                return ""

            result = (f"Benefits:\n" + raw_body).strip()
            print(f"   Structured answer length: {len(result)} chars")
            return result
        except Exception as e:
            print(f"⚠️  Structured herb generation failed: {e}")
            return ""

    def _translate_structured_to_sinhala(self, structured_text: str) -> str:
        """
        Translate a structured herb answer (with **Section:** headers and numbered/bullet items)
        to Sinhala by translating each unit (heading + individual items) separately.
        This produces much cleaner Sinhala than translating the whole block at once.
        """
        if not self.translator:
            return structured_text

        import re as _re_t

        # Map English section headings to Sinhala
        HEADING_MAP = {
            r'^Benefits:$':         'ප්‍රධාන ගුණ:',
            r'^Benefits of .+:$':   lambda m: m.group(0).replace('Benefits of ', '').replace(':', ' ගුණ:'),
            r'^How to use:$':       'භාවිතා කිරීම:',
            r'^Ayurvedic note:$':   'ආයුර්වේද සටහන:',
            r'^Caution:$':          'සැලකිල්ල:',
            r'^\*\*Benefits of (.+):\*\*$': lambda m: f'**{m.group(1)} ගුණ:**',
            r'^\*\*How to use:\*\*$':       '**භාවිතා කිරීම:**',
            r'^\*\*Ayurvedic note:\*\*$':    '**ආයුර්වේද සටහන:**',
            r'^\*\*Caution:\*\*$':           '**සැලකිල්ල:**',
        }

        lines = structured_text.split('\n')
        out = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                out.append('')
                continue

            # Handle section headings
            heading_replaced = False
            for pattern, replacement in HEADING_MAP.items():
                m = _re_t.match(pattern, stripped)
                if m:
                    if callable(replacement):
                        out.append(replacement(m))
                    else:
                        out.append(replacement)
                    heading_replaced = True
                    break
            if heading_replaced:
                continue

            # Handle numbered items: "1. Some benefit text"
            nm = _re_t.match(r'^(\d+)\.\s+(.+)$', stripped)
            if nm:
                num = nm.group(1)
                text = nm.group(2).strip()
                try:
                    text = self._simplify_for_translation(text)
                    text = _re_t.sub(r'^[•\-]?\s*', '', text).strip()
                    si = self.translator.translate_en_to_si(text)
                    si = self._cleanup_translated_answer(si).strip()
                    si = _re_t.sub(r'^[-•]\s*', '', si).strip()
                    out.append(f'{num}. {si}' if si else f'{num}. {text}')
                except Exception:
                    out.append(line)
                continue

            # Handle bullet items: "• Some text"
            bm = _re_t.match(r'^[•\-]\s*(.+)$', stripped)
            if bm:
                text = bm.group(1).strip()
                try:
                    text = self._simplify_for_translation(text)
                    text = _re_t.sub(r'^[•\-]?\s*', '', text).strip()
                    si = self.translator.translate_en_to_si(text)
                    si = self._cleanup_translated_answer(si).strip()
                    si = _re_t.sub(r'^[-•]\s*', '', si).strip()
                    out.append(f'• {si}' if si else f'• {text}')
                except Exception:
                    out.append(line)
                continue

            # Plain text line — translate directly
            try:
                txt = self._simplify_for_translation(stripped)
                si = self.translator.translate_en_to_si(txt)
                si = self._cleanup_translated_answer(si).strip()
                out.append(si if si else stripped)
            except Exception:
                out.append(line)

        return '\n'.join(out)

    def _expand_query_with_ayurvedic_terms(self, query: str) -> str:
        """
        Enrich the English search query with Ayurvedic synonyms so FAISS retrieves
        the most relevant passages even when the user used everyday language.
        E.g. 'my shoulder hurts' → '...shoulder amsa skandha joint pain oil massage'
        """
        query_lower = query.lower()
        expansions = []
        for term, expansion in QUERY_EXPANSION_MAP.items():
            if term in query_lower and expansion not in expansions:
                expansions.append(expansion)
        if expansions:
            expanded = query + ' ' + ' '.join(expansions)
            print(f"🔎 Query expanded with {len(expansions)} term group(s)")
            return expanded
        return query

    def answer_question(
        self,
        question: str,
        vector_db,
        top_k: int = 5,
        user_profile: Optional[Dict[str, Any]] = None,
        validation_top_k: int = 8
    ) -> Dict[str, Any]:
        """Answer question with validation, personalization, and translation"""
        logger.info(f"Processing question: {question[:50]}...")
        
        # === TRANSLATION: Detect language and translate question if needed ===
        original_question = question
        detected_language = 'en'  # Default to English
        is_romanized = False  # Track if romanized Singlish
        
        if self.enable_translation and self.translator:
            detected_language = self.translator.detect_language(question)
            print(f"🌐 Detected language: {detected_language.upper()}")
            
            if detected_language == 'si':
                # Check if romanized (no Sinhala Unicode chars) or Sinhala script
                is_romanized = not self.translator._has_sinhala_chars(question)
                
                if is_romanized:
                    print(f"📝 Romanized Singlish detected (e.g., 'kurudu wala guna')")
                
                # Translate Sinhala question to English for processing
                # If romanized, will be transliterated first inside translate_si_to_en
                question = self.translator.translate_si_to_en(question, is_romanized=is_romanized)
                print(f"🔄 Translated question: {question[:100]}...")
        
        # For Singlish queries, extract the primary herb/entity from the ORIGINAL romanized
        # words (before translation). This lets us pin the LLM prompt to the exact topic
        # even when FAISS retrieves documents about other herbs due to low-similarity guesses.
        primary_topic = None
        if is_romanized:
            import re as _re_pt
            from translation_service import SINHALA_TO_ENGLISH_DICT as _si_dict
            # Skip grammatical / question / adjective words — only look for nouns (herbs, body parts)
            _skip_words = {
                'monawada', 'mokadda', 'mokada', 'kohomada', 'kawuda', 'kauda',
                'kiyada', 'keyada', 'da', 'eka', 'wala', 'walata', 'ta', 'gen',
                'ekka', 'nisa', 'hinda', 'monada', 'guna', 'gunas', 'honda',
                'naraka', 'loku', 'podi', 'wadi', 'adu', 'me', 'meka', 'ara',
                'owa', 'mata', 'mama', 'api', 'oya', 'what', 'how', 'why',
                'karanna', 'ganna', 'bonawa', 'kannawa', 'thiyenawa',
            }
            for _w in original_question.lower().split():
                _cw = _re_pt.sub(r'[^\w]', '', _w)
                if _cw in _si_dict and _cw not in _skip_words:
                    _eng = _si_dict[_cw]
                    if len(_eng) > 3:  # skip single-char translations
                        primary_topic = _eng
                        break
            if primary_topic:
                print(f"🌿 Singlish primary topic pinned: '{primary_topic}'")

        # For Sinhala Unicode queries: extract primary_topic from the translated English
        # question so we can still pin the LLM and re-rank FAISS the same way.
        if not primary_topic and not is_romanized and detected_language == 'si':
            import re as _re_si
            _HERB_KEYWORDS = {
                'cinnamon', 'turmeric', 'ginger', 'neem', 'tulsi', 'ashwagandha',
                'triphala', 'amla', 'brahmi', 'ghee', 'sesame', 'cardamom',
                'cumin', 'coriander', 'fennel', 'licorice', 'pepper', 'garlic',
                'fenugreek', 'aloe', 'coconut', 'sandalwood', 'shatavari', 'guduchi',
            }
            for _tok in _re_si.findall(r'\b[a-z]+\b', question.lower()):
                if _tok in _HERB_KEYWORDS:
                    primary_topic = _tok
                    break
            if primary_topic:
                print(f"🌿 Sinhala Unicode primary topic pinned: '{primary_topic}'")

        # Expand query. If primary_topic was extracted, prepend it so FAISS finds
        # the most relevant passages even before the translated question words help.
        search_query = self._expand_query_with_ayurvedic_terms(
            (primary_topic + ' ' + question) if primary_topic else question
        )

        # Retrieve documents with similarity scores (prefer book sources)
        # Note: Always search in English since database is in English
        print("🔍 Retrieving relevant sources...")
        retrieved_docs = vector_db.search(search_query, top_k=validation_top_k, prefer_books=True)
        
        # Ensure we have book sources for better citations
        book_docs = [d for d in retrieved_docs if d.get("type") == "book"]
        qa_docs = [d for d in retrieved_docs if d.get("type") == "qa"]
        
        # Add similarity percentages
        for doc in retrieved_docs:
            if 'similarity' in doc:
                doc['similarity_percentage'] = round(doc['similarity'] * 100, 1)
        
        # Prefer book sources for context, but include high-similarity QA docs too.
        # For Singlish queries, boost documents that literally contain the primary herb/topic
        # so the LLM receives on-topic context instead of generic Ayurvedic passages.
        if primary_topic:
            _topic_lower = primary_topic.lower()
            def _score_with_topic_boost(d):
                bonus = 0.20 if _topic_lower in d.get('text', '').lower() else 0.0
                return d.get('similarity', 0) + bonus
            all_docs_sorted = sorted(retrieved_docs, key=_score_with_topic_boost, reverse=True)
            _topic_hits = sum(1 for d in all_docs_sorted[:5] if _topic_lower in d.get('text', '').lower())
            print(f"📌 Re-ranked for '{primary_topic}': {_topic_hits}/5 top docs contain it")
        else:
            all_docs_sorted = sorted(retrieved_docs, key=lambda d: d.get('similarity', 0), reverse=True)
        # Guarantee at least 2 book docs in context if available (for citation quality)
        top_book_docs = [d for d in all_docs_sorted if d.get('type') == 'book'][:2]
        other_top_docs = [d for d in all_docs_sorted if d not in top_book_docs][:max(0, top_k - len(top_book_docs))]
        top_context_docs = (top_book_docs + other_top_docs)[:top_k]
        print(f"📚 Context: {len([d for d in top_context_docs if d.get('type')=='book'])} book + "
              f"{len([d for d in top_context_docs if d.get('type')!='book'])} QA docs")
        
        # 320 tokens = comfortably fits 4 rich summary bullets of up to 25 words each
        dynamic_tokens = 320

        # Build context
        context_text = self._build_context_with_citations(top_context_docs)

        print(f"🔍 Context preview (first 300 chars): {context_text[:300]}...")

        # Rich summarization prompt — asks for 4 informative bullets covering
        # causes/benefits, treatments/herbs, lifestyle advice, and a key fact.
        context_summary = context_text[:2500]
        topic_hint = question[:80].strip()
        # Hard-pin constraint injected when we know the exact herb/topic from Singlish.
        # Without this, the LLM drifts to summarise whatever herbs appear in the FAISS context
        # rather than answering about the herb the user actually asked about.
        topic_pin = (
            f"⚠️  MANDATORY TOPIC: The patient specifically asked about '{primary_topic}'.\n"
            f"Every bullet point MUST be exclusively about {primary_topic}.\n"
            f"Do NOT mention Amalaki, Triphala, Shilajit, Ashwagandha, or any other herb "
            f"unless it is directly combined with or compared to {primary_topic}.\n\n"
        ) if primary_topic else ""
        messages = [
            {
                "role": "user",
                "content": (
                    f"You are a knowledgeable Ayurvedic doctor writing a clear summary for a patient.\n"
                    f"Patient question: \"{question}\"\n\n"
                    f"Ayurvedic Knowledge Sources:\n{context_summary}\n\n"
                    f"{topic_pin}"
                    f"Task: Write a SUMMARY with EXACTLY 4 bullet points that directly answer: {topic_hint}\n\n"
                    f"STRICT RULES:\n"
                    f"• Each bullet point = 1 complete informative sentence (15–25 words).\n"
                    f"• Each bullet MUST mention a specific herb, remedy, dosha, or Ayurvedic concept.\n"
                    f"• Cover different aspects: e.g. main benefit, remedy/herb, dosha effect, lifestyle tip.\n"
                    f"• Start EVERY bullet with the • symbol on its own line.\n"
                    f"• NO introduction sentence. NO conclusion. NO numbered lists. ONLY the 4 bullets.\n"
                    f"• Do NOT mention any author names, researcher names, book titles, or source labels.\n"
                    f"• Do NOT use abbreviations like 'w/' — write full words only.\n"
                    f"• Write as a knowledgeable Ayurvedic doctor, not as someone reading a book.\n\n"
                    f"Example (for a different topic):\n"
                    f"• Turmeric contains curcumin which reduces Pitta-driven inflammation in joints and tissues.\n"
                    f"• Daily intake of turmeric with warm milk improves digestion and detoxifies the liver.\n"
                    f"• Turmeric balances Kapha and Vata doshas and strengthens the immune system naturally.\n"
                    f"• Applying turmeric paste externally heals skin infections and reduces redness effectively.\n\n"
                    f"Now write 4 bullet point summary about: {topic_hint}\n\n"
                    f"•"
                )
            }
        ]
        # ── GENERATION STRATEGY ──────────────────────────────────────────────────
        # Singlish herb query (primary_topic known): use structured herb answer
        # directly — skip the flat-bullet generation entirely so we don't waste
        # an LLM call and don't produce off-topic flat bullets that leak through.
        # English/unknown queries: run existing flat-bullet pipeline.
        import re as _re

        is_structured = False
        base_answer = ""

        if primary_topic:
            # ── Structured path ──────────────────────────────────────────────────
            print(f"🌿 Structured generation for '{primary_topic}' (skipping flat pipeline)")
            structured = self._generate_structured_herb_answer(
                herb_name=primary_topic,
                question=question,
                context_summary=context_text[:1500]
            )
            if structured and len(structured.strip()) > 60:
                base_answer = structured
                is_structured = True
                print(f"✅ Structured herb answer ready ({len(base_answer)} chars)")
            else:
                print("⚠️  Structured gen returned empty — falling back to flat pipeline")

        if not is_structured:
            # ── Flat-bullet path (English / fallback) ────────────────────────────
            print("💭 Generating flat bullet answer...")
            raw_answer = self.llm.generate_from_messages(
                messages, max_new_tokens=dynamic_tokens, min_new_tokens=120, temperature=0.25
            )
            raw_answer = self._clean_garbled_text(raw_answer)

            raw_answer_clean = _re.sub(r'^[\s•\-\*]+', '', raw_answer).strip()
            raw_combined = ("• " + raw_answer_clean).strip()

            def _clip_bullet(b: str) -> str:
                b = b.strip()
                b = _re.sub(r'^(•\s*)[\-\*•]+\s*', r'\1', b)
                m = _re.search(r'(?<=[.!?])(?:\s|$)', b[40:])
                if m:
                    b = b[:40 + m.start() + 1].strip()
                if len(b) > 200:
                    b = b[:200].rsplit(' ', 1)[0].rstrip(',:;') + '.'
                return b

            if '•' in raw_combined:
                base_answer = raw_combined
            else:
                converted = _re.sub(r'^\d+\.\s+', '• ', raw_combined, flags=_re.MULTILINE)
                if '•' in converted:
                    base_answer = converted
                else:
                    sents = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', raw_combined) if s.strip()]
                    base_answer = '\n'.join(f'• {s}' for s in sents[:4])

            bullets = [line for line in base_answer.splitlines() if line.strip().startswith('•')]
            bullets = [_clip_bullet(b) for b in bullets[:4]]

            if len(bullets) < 3:
                extra_sents = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', raw_answer)
                               if len(s.strip()) > 25]
                used_words = set()
                for b in bullets:
                    used_words.update(b.lower().split())
                for sent in extra_sents:
                    if len(bullets) >= 4:
                        break
                    clean = _re.sub(r'^[\s•\ -\ *\ d\ .]+', '', sent).strip()
                    sent_words = set(clean.lower().split())
                    if len(sent_words & used_words) > 3:
                        continue
                    if clean and len(clean) > 25:
                        new_b = _clip_bullet(f'• {clean}')
                        bullets.append(new_b)
                        used_words.update(sent_words)

            if bullets:
                base_answer = '\n'.join(bullets)

        print(f"✅ Generation complete! ({'structured' if is_structured else 'flat bullets'})")
        print(f"   Answer length: {len(base_answer)} chars")
        print(f"   Answer preview: {base_answer[:200] if base_answer else '[EMPTY]'}...")
        
        if not base_answer or len(base_answer.strip()) < 10:
            print("⚠️  WARNING: Generated answer is empty or too short!")
            if not is_structured:
                _raw = locals().get('raw_answer', '')
                print(f"   Raw answer: {_raw[:300] if _raw else '[NONE]'}...")
        
        # Validate
        validation_result = None
        if self.enable_validation and self.validator:
            print("✓ Validating answer across sources...")
            validation_result = self.validator.validate_answer(
                primary_answer=base_answer,
                retrieved_docs=retrieved_docs,
                top_k=validation_top_k
            )
        
        # Personalize — skip for structured herb answers; the structured format is already
        # self-contained and personalizer produces garbled output on non-bullet text.
        final_answer = base_answer
        is_personalized = False

        if self.enable_personalization and self.personalizer and user_profile and not is_structured:
            print("✨ Personalizing answer...")
            final_answer = self.personalizer.personalize_answer(
                base_answer=base_answer,
                user_profile=user_profile,
                question=question
            )
            is_personalized = True

        # Detect dosha and generate personalized tips (always, regardless of user profile)
        detected_dosha = self._detect_dosha_from_question(question, base_answer)
        personalized_tips = self._generate_personalized_tips(question, base_answer, detected_dosha)
        print(f"🧬 Detected dosha: {detected_dosha}")
        
        # Format response with similarity percentages
        formatted_citations = []
        for doc in top_context_docs:
            meta = doc.get("metadata", {})
            doc_type = doc.get("type", "unknown")
            
            citation = {
                "source": doc.get("source", "Unknown"),
                "type": doc_type,
                "similarity_percentage": doc.get("similarity_percentage", 0.0),
                # Longer preview so users can locate the passage in the actual book
                "text_preview": doc.get("text", "")[:200].strip() + "..."
            }
            
            # Add type-specific fields
            if doc_type == "book":
                citation["chapter"] = meta.get("chapter", "N/A")
                # Fall back to 'verse' key — some books (e.g. Sanskrit texts) use 'verse' not 'paragraph'
                citation["paragraph"] = meta.get("paragraph", meta.get("verse", "N/A"))
            else:
                # QA entry
                citation["qa_id"] = meta.get("question_id", "N/A")
                citation["related_question"] = meta.get("question", "N/A")
            
            formatted_citations.append(citation)
        
        # === ADD TERM CLARIFICATION for romanized Singlish ===
        # Skip for structured answers — the herb name is already in the section heading.
        if detected_language == 'si' and is_romanized and not is_structured:
            print("📝 Adding term clarification for romanized Singlish...")
            final_answer = self._add_term_clarification(
                answer=final_answer,
                original_question=original_question,
                is_romanized=is_romanized
            )
        
        # === TRANSLATION: Translate answer back to user's language ===
        display_answer = final_answer  # Default: English version
        
        if self.enable_translation and self.translator and detected_language == 'si':
            # For ALL Sinhala inputs (romanized OR Unicode), translate answer to Sinhala
            if is_structured:
                # Structured herb answer: translate section-by-section
                print("🔄 Translating structured herb answer to Sinhala...")
                display_answer = self._translate_structured_to_sinhala(final_answer)
                if not display_answer or len(display_answer.strip()) < 20:
                    display_answer = final_answer
                print(f"✓ Structured translation complete: {display_answer[:100]}...")
            else:
                # Flat bullets: translate each bullet individually
                print("🔄 Translating answer to Sinhala (bullet-by-bullet)...")
                import re as _re2
                bullet_lines = [l for l in final_answer.splitlines() if l.strip().startswith('•')]
                if bullet_lines:
                    translated_bullets = []
                    for bl in bullet_lines:
                        content = _re2.sub(r'^•\s*', '', bl).strip()
                        # Simplify Sanskrit compound terms before sending to Google Translate
                        content = self._simplify_for_translation(content)
                        content = _re2.sub(r'^•\s*', '', content).strip()  # strip bullet added by simplify
                        # Ensure sentence ends with period for cleaner translation
                        if content and not content.endswith(('.', '!', '?')):
                            content += '.'
                        try:
                            si_content = self.translator.translate_en_to_si(content)
                            si_clean = self._cleanup_translated_answer(si_content).strip()
                            # Strip any bullet the cleanup may have added
                            si_clean = _re2.sub(r'^[-•]\s*', '', si_clean).strip()
                            if si_clean and len(si_clean) >= 5:
                                translated_bullets.append(f'• {si_clean}')
                        except Exception as _te:
                            print(f"⚠️  Bullet translation failed: {_te}")
                            translated_bullets.append(bl)  # keep English bullet as fallback
                    if translated_bullets:
                        display_answer = '\n'.join(translated_bullets)
                        print(f"✓ Translated {len(translated_bullets)} bullets to Sinhala")
                    else:
                        display_answer = final_answer
                else:
                    # No bullets — translate as single block
                    simplified_english = self._simplify_for_translation(final_answer)
                    raw_translation = self.translator.translate_en_to_si(simplified_english)
                    display_answer = self._cleanup_translated_answer(raw_translation)
                    if not display_answer or len(display_answer.strip()) < 20:
                        display_answer = final_answer
                print(f"✓ Translation complete: {display_answer[:100]}...")

        # Translate personalized tips to Sinhala if user asked in Singlish/Sinhala
        if self.enable_translation and self.translator and detected_language == 'si' and personalized_tips:
            print("🔄 Translating personalized tips to Sinhala (bullet-by-bullet)...")
            import re as _re3
            tip_lines = [l for l in personalized_tips.splitlines() if l.strip().startswith('•')]
            if tip_lines:
                translated_tips = []
                for tl in tip_lines:
                    content = _re3.sub(r'^•\s*', '', tl).strip()
                    # Simplify Sanskrit compounds before translation
                    content = self._simplify_for_translation(content)
                    content = _re3.sub(r'^•\s*', '', content).strip()
                    if content and not content.endswith(('.', '!', '?')):
                        content += '.'
                    try:
                        si_tip = self.translator.translate_en_to_si(content)
                        si_tip_clean = self._cleanup_translated_answer(si_tip).strip()
                        si_tip_clean = _re3.sub(r'^[-•]\s*', '', si_tip_clean).strip()
                        if si_tip_clean and len(si_tip_clean) >= 5:
                            translated_tips.append(f'• {si_tip_clean}')
                    except Exception:
                        translated_tips.append(tl)
                if translated_tips:
                    personalized_tips = '\n'.join(translated_tips)
                else:
                    print("⚠️  Tips translation failed, keeping English")

        response = {
            "answer": display_answer,  # Answer in Sinhala for all Sinhala/Singlish inputs
            "answer_english": final_answer,  # Always keep English version
            "base_answer": base_answer,
            "original_question": original_question,
            "translated_question": question if detected_language == 'si' else None,
            "detected_language": detected_language,
            "is_romanized": is_romanized if detected_language == 'si' else False,
            "citations": formatted_citations,
            "sources": top_context_docs,
            "all_sources": retrieved_docs,
            "num_sources": len(top_context_docs),
            "personalized": is_personalized,
            "detected_dosha": detected_dosha,
            "personalized_tips": personalized_tips,
        }
        
        if validation_result:
            response["validation"] = validation_result
        
        if is_personalized and user_profile:
            response["user_info"] = {
                "dominant_dosha": user_profile.get('dominant_dosha', 'N/A'),
                "current_season": user_profile.get('current_season', 'N/A')
            }
        
        return response
    
    def create_user_profile(self, user_id: str, prakriti_responses: Dict[str, str]) -> Dict[str, Any]:
        """Create user profile"""
        if not self.personalizer:
            raise RuntimeError("Personalization engine not enabled")
        return self.personalizer.create_user_profile(user_id, prakriti_responses)
    
    def load_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load user profile"""
        if not self.personalizer:
            return None
        return self.personalizer.load_user_profile(user_id)
