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
    
    def _build_context_with_citations(self, docs):
        """Build LLM context including citations"""
        context_blocks = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            book = doc.get("source", "Unknown Book")
            doc_type = doc.get("type", "unknown")
            
            # Handle book vs QA sources differently
            if doc_type == "book":
                chapter = meta.get("chapter", "N/A")
                paragraph = meta.get("paragraph", meta.get("verse", "N/A"))
                # Cap text to 200 chars to keep total prompt within token budget
                doc_text = doc.get("text", "")[:200]
                block = f"""[Source {i}]
Book: {book}
Chapter: {chapter}
Verse/Paragraph: {paragraph}

{doc_text}"""
            else:
                # QA dataset entry
                question = meta.get("question", "N/A")
                # Cap text to 200 chars to keep total prompt within token budget
                doc_text = doc.get("text", "")[:200]
                block = f"""[Source {i}]
Type: Ayurvedic Q&A Reference
Related Question: {question}

{doc_text}"""
            
            context_blocks.append(block.strip())
        
        return "\n\n".join(context_blocks)
    
    def _restructure_answer(self, raw: str, question: str = '') -> str:
        """
        Clean the raw LLM output into 2 concise, meaningful bullet points.
        Steps (all pure regex/string — no extra LLM call):
          1. Strip attribution preambles ("According to X", "Based on the sources")
          2. Strip inline source labels  ("(source one)", "Source 1 states")
          3. Strip quoted book titles    (\"Ayurvedhan\", 'Everyday Ayurveda')
          4. Split into sentences
          5. Reject filler / meta-commentary sentences
          6. Return first 2 clean sentences as bullet strings
        """
        import re

        if not raw or not raw.strip():
            return raw

        text = raw.strip()

        # 1. Strip leading attribution: "According to X, " / "Based on X, " / "X states that "
        text = re.sub(
            r'^(?:According\s+to|Based\s+on|As\s+per|Per|From)\s+[^,;.]{0,80}[,;]\s*',
            '', text, flags=re.IGNORECASE
        )
        # Also: "[Book name] (source one) states / mentions / says"
        text = re.sub(
            r'["\u2018\u201c][^"\u2019\u201d]{2,50}["\u2019\u201d]\s*(?:\([^)]{0,40}\))?\s*(?:states?|says?|mentions?|notes?|reports?|indicates?)\s+(?:that\s+)?',
            '', text, flags=re.IGNORECASE
        )

        # 2. Strip inline source labels everywhere
        text = re.sub(r'\([Ss]ource\s*(?:one|two|three|four|1|2|3|4)\)', '', text)
        text = re.sub(r'\b[Ss]ource\s*(?:one|two|three|1|2|3)\b', '', text)

        # 3. Strip quoted book titles left as orphan fragments
        text = re.sub(r'["\u2018\u201c][^"\u2019\u201d]{2,60}["\u2019\u201d]', '', text)

        # 4. Strip patterns like "(Curcuma longa Rhizome)" — raw taxonomy noise
        text = re.sub(r"\b[A-Z][a-z]+\s+[a-z]+\s+(?:[A-Z][a-z]+\s+)?[Rr]hiz\w*\b", '', text)

        # 5. Clean up punctuation artifacts left by stripping
        text = re.sub(r'\s*,\s*,', ',', text)        # double commas
        text = re.sub(r'^[,;:\s]+', '', text)         # leading punctuation
        text = re.sub(r'\s{2,}', ' ', text)           # multiple spaces
        text = text.strip()

        # 6. Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # 7. Also split on semicolons for run-on lists (take first segment only)
        expanded = []
        for s in sentences:
            # If a sentence has multiple semicolons it's a list — take first part
            parts = s.split(';')
            expanded.append(parts[0].strip())

        # 8. Filter out filler / meta-commentary / very short fragments
        FILLER_STARTS = [
            'all these', 'these factor', 'this information', 'the above', 'as mentioned',
            'in summary', 'in conclusion', 'therefore', 'thus,', 'hence,',
            'it is worth', 'it should be noted', 'please note', 'note that',
            'based on', 'according to', 'the source', 'outlined by',
            'outlined above', 'referring to', 'as per', 'as noted',
        ]
        clean = []
        for s in expanded:
            s = s.strip().rstrip('.,;- ')
            if not s:
                continue
            if len(s) < 15:
                continue
            low = s.lower()
            if any(low.startswith(f) for f in FILLER_STARTS):
                continue
            # Reject only sentences that are PRIMARILY about citing sources/references
            # (must contain source/citation/reference AND a verb like "states"/"mentions")
            if re.search(r'\b(source|citation|reference)\b', low) and \
               re.search(r'\b(states?|says?|mentions?|notes?|reports?|indicates?|describes?)\b', low):
                continue
            # Reject sentences that are pure bibliography/chapter/verse markers
            if re.search(r'\b(chapter|verse|bibliography|ibid)\b', low):
                continue
            # Ensure sentence ends with proper punctuation
            if not s[-1] in '.!?':
                s = s + '.'
            # Capitalize first letter
            s = s[0].upper() + s[1:]
            clean.append(s)

        if not clean:
            # Fallback: split raw into sentences and take first 3 as bullets
            fallback_sentences = re.split(r'(?<=[.!?])\s+', raw.strip())
            fallback_clean = []
            for s in fallback_sentences:
                s = s.strip()
                if len(s) > 15:
                    if s[-1] not in '.!?':
                        s = s + '.'
                    fallback_clean.append(s[0].upper() + s[1:])
            if fallback_clean:
                return '\n'.join(f'\u2022 {b}' for b in fallback_clean[:3])
            return raw.strip()

        # 9. Return up to 3 bullets
        bullets = clean[:3]
        return '\n'.join(f'\u2022 {b}' for b in bullets)

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
    
    def _simplify_for_translation(self, english_text: str) -> str:
        """
        Simplify English text before translation to improve Sinhala quality.
        - Break down complex sentences
        - Remove excessive technical jargon
        - Make sentences more translation-friendly
        """
        import re
        
        if not english_text or not english_text.strip():
            return ""
        
        # Split into lines (bullets)
        lines = english_text.split('\n')
        simplified_lines = []
        
        for line in lines:
            if not line.strip():
                continue
            
            # Extract bullet marker and content
            if line.strip().startswith('-'):
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
        Post-process translated Sinhala text to fix common issues:
        - Remove incomplete sentences
        - Fix formatting
        - Remove gibberish words
        """
        import re
        
        if not translated_text or not translated_text.strip():
            return ""
        
        # Split into lines
        lines = translated_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract bullet and content
            if line.startswith('-'):
                bullet = '- '
                content = line[2:].strip()
            else:
                bullet = ''
                content = line.strip()
            
            if not content:
                continue
            
            # Check for gibberish (words with unusual character patterns)
            # Look for words that don't follow Sinhala patterns (e.g., "නයිබර්")
            words = content.split()
            cleaned_words = []
            
            for word in words:
                # Check if word has Sinhala characters
                has_sinhala = bool(re.search(r'[\u0D80-\u0DFF]', word))
                
                if has_sinhala:
                    # Check for gibberish patterns (isolated vowel signs, excessive marks)
                    # Filter out words with unusual patterns like excessive ් (al-lakuna)
                    gibberish_pattern = r'([\u0DCA]{2,})|([ා-ෟ]{3,})'
                    if not re.search(gibberish_pattern, word):
                        cleaned_words.append(word)
                else:
                    # Keep non-Sinhala words (numbers, punctuation)
                    cleaned_words.append(word)
            
            if not cleaned_words:
                continue
            
            content = ' '.join(cleaned_words)
            
            # Check if sentence is complete
            # Sinhala sentences typically end with: ය, ා, ී, ෙ, ති, ද, ට, ම, න, ව, or punctuation
            sinhala_endings = r'[යාීෙතිදටමනව.!?]$'
            
            # Skip if too short and doesn't end properly
            if len(content) < 30:
                if not re.search(sinhala_endings, content):
                    continue
            
            # If reasonably long but doesn't end properly, skip it
            if len(content) >= 30 and len(content) < 100:
                if not re.search(sinhala_endings, content):
                    continue
            
            # For long text, check if it ends with Sinhala or punctuation
            if len(content) >= 100:
                if not re.search(r'[\u0D80-\u0DFF.!?]$', content):
                    # Incomplete, skip it
                    continue
            
            # Add cleaned line
            if content:
                cleaned_lines.append(f"{bullet}{content}")
        
        # Ensure we have at least some content
        if not cleaned_lines:
            return translated_text  # Return original if cleanup removed everything
        
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

    def _generate_personalized_tips(self, question: str, answer: str, dosha: str, original_question: str = '') -> str:
        """
        Generate 3 reliable dosha-specific Ayurvedic tips using templates.
        No LLM call — pure template system guarantees clean, relevant output.
        """
        import re as _re
        print(f"💡 Generating template tips for {dosha} dosha...")

        # --- Extract a clean topic keyword from the question ---
        # Use the original (pre-translation) question for topic extraction when available.
        # Translated questions can be garbled (e.g. "kurudu wala guna" → "Are They Useful")
        _source_q = question  # already translated to English at this point

        # Try to map romanized Sinhala herb names via the dictionary first
        try:
            from translation_service import SINHALA_TO_ENGLISH_DICT
            for sinhala_key, english_val in SINHALA_TO_ENGLISH_DICT.items():
                if sinhala_key.lower() in original_question.lower():
                    _source_q = f"what are the benefits of {english_val}"
                    break
        except Exception:
            pass

        topic_raw = _re.sub(
            r'^(what are the benefits of|what are the uses of|what is the use of|'
            r'what is|how does|benefits of|uses of|properties of|'
            r'tell me about|explain|describe)\s+',
            '', _source_q.lower(), flags=_re.IGNORECASE
        ).strip().strip('?').strip()

        # Remove trailing filler words
        topic_raw = _re.sub(r'\s+(in ayurveda|ayurvedic|for health|for body)$', '', topic_raw).strip()

        # Guard: if topic_raw is generic/bad (e.g. "are they useful", "it", "they"),
        # fall back to the original question directly
        _bad_topics = {'are', 'is', 'it', 'they', 'them', 'this', 'that', 'useful',
                       'benefits', 'guna', 'monawada', 'what', 'how', 'does', 'do'}
        topic_words = set(topic_raw.lower().split())
        if not topic_raw or len(topic_raw) < 3 or topic_words.issubset(_bad_topics):
            # Last resort: first noun-like word from original question
            _words = [w for w in _re.findall(r'\b[a-zA-Z]{4,}\b', original_question)
                      if w.lower() not in _bad_topics | {'wala', 'what', 'guna', 'monawada'}]
            topic_raw = _words[0] if _words else 'this herb'

        # Capitalize nicely (handles multi-word topics)
        topic = topic_raw.title() if topic_raw else question.strip('?').strip()
        topic_low = topic_raw if topic_raw else question.lower().strip('?').strip()

        # --- Dosha-specific tip templates ---
        # Pure Subject + Verb + Object sentences only.
        # No gerunds, no participials, no "to + infinitive" — avoids Google Translate Sinhala artefacts.
        templates = {
            'Vata': [
                f"Warm ghee improves the potency of {topic} and supports Vata balance every day.",
                f"{topic} stabilises Vata energy when consumed at the same time each morning.",
                f"Black pepper increases the warmth of {topic} and supports healthy Vata digestion.",
            ],
            'Pitta': [
                f"Coconut milk reduces the heating effect of {topic} and keeps Pitta dosha in balance.",
                f"{topic} works best in early morning, before Pitta energy rises at noon.",
                f"Fennel seeds cool the body when used together with {topic} for Pitta dosha.",
            ],
            'Kapha': [
                f"Black pepper and warm water activate the digestive benefits of {topic} for Kapha dosha.",
                f"{topic} reduces Kapha heaviness when consumed on an empty stomach each morning.",
                f"Dry ginger increases the warming energy of {topic} and reduces Kapha sluggishness.",
            ],
            'General': [
                f"{topic} supports Ayurvedic health and immunity when consumed daily with meals.",
                f"A warm cup of {topic} tea each morning increases its absorption in the body.",
                f"An Ayurvedic practitioner recommends the correct dose of {topic} for each body type.",
            ],
        }

        # Resolve compound doshas (e.g. "Vata-Pitta" → use Vata templates)
        key = 'General'
        for d in ['Vata', 'Pitta', 'Kapha']:
            if d in dosha:
                key = d
                break

        tips = templates[key]
        print(f"✓ Template tips generated for {key} dosha — topic: '{topic}'")
        return '\n'.join(f'{i+1}. {t}' for i, t in enumerate(tips))

    def answer_question(
        self, 
        question: str, 
        vector_db, 
        top_k: int = 5,
        user_profile: Optional[Dict[str, Any]] = None,
        validation_top_k: int = 5
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
            
            if detected_language in ('si', 'ta'):
                # Check if romanized (no Sinhala Unicode chars) or native script
                is_romanized = not self.translator._has_sinhala_chars(question)
                
                if is_romanized:
                    print(f"📝 Romanized input detected (e.g., 'kurudu wala guna')")
                
                # Translate question to English for processing
                question = self.translator.translate_si_to_en(question, is_romanized=is_romanized)
                print(f"🔄 Translated question: {question[:100]}...")
        
        # Retrieve documents with similarity scores (prefer book sources)
        # Note: Always search in English since database is in English
        print("🔍 Retrieving relevant sources...")
        # Build a targeted search query — if asking about benefits/uses, add context
        # so the vector search finds health-content chunks rather than book index pages
        search_query = question
        benefit_signals = [
            'benefit', 'use', 'good for', 'help', 'treat', 'property',
            'guna', 'effect', 'cure', 'purpose', 'health', 'medicinal'
        ]
        if any(w in question.lower() for w in benefit_signals):
            search_query = question + " health benefits medicinal properties"
            print(f"🎯 Enhanced search query: {search_query[:120]}...")
        retrieved_docs = vector_db.search(search_query, top_k=validation_top_k, prefer_books=True)
        
        # Ensure we have book sources for better citations
        book_docs = [d for d in retrieved_docs if d.get("type") == "book"]
        qa_docs = [d for d in retrieved_docs if d.get("type") == "qa"]
        
        # Add similarity percentages
        for doc in retrieved_docs:
            if 'similarity' in doc:
                doc['similarity_percentage'] = round(doc['similarity'] * 100, 1)
        
        # Prefer book sources for context, but include QA if relevant
        if book_docs:
            top_context_docs = book_docs[:top_k]
            print(f"📚 Using {len(top_context_docs)} book sources")
        else:
            top_context_docs = retrieved_docs[:top_k]
            print(f"⚠️  No book sources found, using QA entries")
        
        # Token budget — 150 tokens is enough for 2-3 clear sentences (Feb 23 approach)
        dynamic_tokens = 150
            
        # Build context
        context_text = self._build_context_with_citations(top_context_docs)
        
        print(f"🔍 Context preview (first 300 chars): {context_text[:300]}...")
        
        # Build messages — simple single-user-message prompt (Feb 23 proven approach).
        # Asking for 2-3 sentences produces clean prose; bullet-forcing caused
        # the model to emit meta-commentary and book-index garbage.
        context_summary = context_text[:600]  # ~150 tokens of context
        messages = [
            {
                "role": "user",
                "content": (
                    f"You are an Ayurvedic knowledge assistant. "
                    f"Using ONLY the sources below, write a clear answer in 2-3 sentences. "
                    f"State health facts directly. Do NOT say 'According to', do NOT name book titles or sources.\n\n"
                    f"Sources:\n{context_summary}\n\n"
                    f"Question: {question}"
                )
            }
        ]
        print(f"📏 Generating with max_new_tokens={dynamic_tokens}")

        # Generate answer
        print("💭 Generating answer...")
        raw_answer = self.llm.generate_from_messages(messages, max_new_tokens=dynamic_tokens)

        # Use raw answer directly — complex extractors were rejecting all valid
        # content and producing meta-commentary as output (Feb 23 proven approach)
        raw_answer = raw_answer.strip()

        # Restructure: strip attribution/source noise, extract 2 clean bullet facts
        base_answer = self._restructure_answer(raw_answer, question=question)

        print(f"✅ Generation complete!")
        print(f"   Answer length: {len(base_answer)} chars")
        print(f"   Answer preview: {base_answer[:200] if base_answer else '[EMPTY]'}...")
        
        if not base_answer or len(base_answer.strip()) < 10:
            print("⚠️  WARNING: Generated answer is empty or too short!")
            print(f"   Raw answer: {raw_answer[:300] if raw_answer else '[NONE]'}...")
            print(f"   Messages used: {str(messages)[:300]}...")
        
        # Validate
        validation_result = None
        if self.enable_validation and self.validator:
            print("✓ Validating answer across sources...")
            validation_result = self.validator.validate_answer(
                primary_answer=base_answer,
                retrieved_docs=retrieved_docs,
                top_k=validation_top_k
            )
        
        # Personalize
        final_answer = base_answer
        is_personalized = False
        
        if self.enable_personalization and self.personalizer and user_profile:
            print("✨ Personalizing answer...")
            final_answer = self.personalizer.personalize_answer(
                base_answer=base_answer,
                user_profile=user_profile,
                question=question
            )
            is_personalized = True

        # Detect dosha and generate personalized tips (always, regardless of user profile)
        detected_dosha = self._detect_dosha_from_question(question, base_answer)
        personalized_tips = self._generate_personalized_tips(question, base_answer, detected_dosha, original_question=original_question)
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
                "text_preview": doc.get("text", "")[:100] + "..."
            }
            
            # Add type-specific fields
            if doc_type == "book":
                citation["chapter"] = meta.get("chapter", "N/A")
                citation["paragraph"] = meta.get("paragraph", "N/A")
            else:
                # QA entry
                citation["qa_id"] = meta.get("question_id", "N/A")
                citation["related_question"] = meta.get("question", "N/A")
            
            formatted_citations.append(citation)
        
        # === ADD TERM CLARIFICATION for romanized Singlish ===
        # If user asked in romanized Singlish, explain what the terms mean
        if detected_language == 'si' and is_romanized:
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
            # Workflow: Singlish/Sinhala input → English processing → Sinhala output
            print("🔄 Translating answer to Sinhala...")
            
            # Step 1: Simplify English sentences for cleaner Google Translate output
            simplified_english = self._simplify_for_translation(final_answer)
            print(f"📝 Simplified for translation: {simplified_english[:100]}...")
            
            # Step 2: Translate
            raw_translation = self.translator.translate_en_to_si(simplified_english)
            
            # Step 3: Clean up translated Sinhala text
            cleaned_translation = self._cleanup_translated_answer(raw_translation)
            
            if cleaned_translation and len(cleaned_translation.strip()) >= 20:
                display_answer = cleaned_translation.strip()
                print(f"\u2713 Translation complete: {display_answer[:100]}...")
            elif raw_translation and raw_translation.strip():
                display_answer = raw_translation.strip()
                print(f"\u26a0\ufe0f  Cleanup over-filtered, using raw translation")
            else:
                print("⚠️  Translation returned empty, using English")
                display_answer = final_answer

        elif self.enable_translation and self.translator and detected_language == 'ta':
            # Tamil input → translate answer to Tamil
            print("🔄 Translating answer to Tamil...")
            simplified_english = self._simplify_for_translation(final_answer)
            raw_translation = self.translator.translate_en_to_ta(simplified_english)
            if raw_translation and raw_translation.strip():
                display_answer = raw_translation.strip()
                print(f"\u2713 Tamil translation complete: {display_answer[:100]}...")
            else:
                display_answer = final_answer

        # Translate tips line-by-line to the user's language (Sinhala or Tamil).
        # Per-line translation avoids the compound-sentence artefacts that appeared
        # when translating the whole block at once.
        import re as _re_tips
        if self.enable_translation and self.translator and detected_language in ('si', 'ta') and personalized_tips:
            print(f"🔄 Translating tips to {detected_language.upper()} (line by line)...")
            translated_lines = []
            for tip_line in personalized_tips.split('\n'):
                tip_line = tip_line.strip()
                if not tip_line:
                    continue
                # Strip number prefix, translate text, re-attach prefix
                m = _re_tips.match(r'^(\d+\.\s*)', tip_line)
                prefix = m.group(1) if m else ''
                text = tip_line[len(prefix):].strip() if m else tip_line
                simplified = self._simplify_for_translation(text)
                if detected_language == 'si':
                    translated = self.translator.translate_en_to_si(simplified)
                else:
                    translated = self.translator.translate_en_to_ta(simplified)
                if translated and len(translated.strip()) > 5:
                    translated_lines.append(f"{prefix}{translated.strip()}")
                else:
                    translated_lines.append(tip_line)  # keep English if fails
            if translated_lines:
                personalized_tips = '\n'.join(translated_lines)

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
