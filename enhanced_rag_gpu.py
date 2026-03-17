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

# Import domain filter for Ayurveda-only question validation
try:
    from web_scraper.domain_filter import is_ayurvedic_question, get_rejection_message
    _DOMAIN_FILTER_AVAILABLE = True
except ImportError:
    _DOMAIN_FILTER_AVAILABLE = False

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

        # In-memory cache: same question + same dosha → same answer every time
        self._answer_cache: Dict[str, Any] = {}
        
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

        # 0. If LLM returned bullet format (starts with '-' or '•'), parse directly
        import re as _re_bullet
        if _re_bullet.search(r'^\s*[-•]', text, _re_bullet.MULTILINE):
            bullet_lines = []
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Strip bullet marker
                line = _re_bullet.sub(r'^[-•\*]\s*', '', line).strip()
                # Strip [Source X] and metadata noise
                line = _re_bullet.sub(r'\[Source \d+\].*', '', line).strip()
                line = _re_bullet.sub(r'Book:.*|Chapter:.*|Verse.*', '', line).strip()
                if len(line) < 15:
                    continue
                low = line.lower()
                if any(low.startswith(f) for f in [
                    'according to', 'based on', 'as per', 'from source',
                    'source', 'note that', 'please', 'however',
                ]):
                    continue
                if not line[-1] in '.!?':
                    line = line + '.'
                line = line[0].upper() + line[1:]
                bullet_lines.append(line)
            if bullet_lines:
                return '\n'.join(f'\u2022 {b}' for b in bullet_lines[:3])

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
            # LLM hallucination / meta-commentary patterns
            'these details', 'this might not', 'this does not', 'there appears',
            'it does not', "it doesn't", 'not necessarily', 'if they don',
            'the information about', 'there is no', 'there are no',
            # More garbage patterns
            'all text', 'text extract', 'all the text', 'extract required',
            'personalized for', '**personalized', '*personalized',
            'in conclusion', 'to summarize', 'to sum up',
            # Informal / hallucination sentence starters
            'so there', 'there ya', 'but heed', 'heed my', 'those tiny',
            'mighty power', 'magic potion', 'just cuz', 'sorted down',
            'but remember', 'but note', 'folks don', 'as per source',
            'this way of', 'this herbal', 'maximally drink', 'follow this pattern',
            'secondary part', 'no longer', 'first thing in morning',
            # LLM apology / inability patterns
            "i'm sorry", "i am sorry", 'i apologize', 'unfortunately',
            'i cannot', 'i could not', 'i am unable', 'i\'m unable',
            'sorry, but', 'sorry but', 'regrettably', 'no information',
            'no specific', 'not able to', 'unable to find', 'not found',
            'cannot find', 'could not find', 'does not appear',
            'it appears no one', 'it appears that no',
            # Garbled LLM filler
            'however, based', 'however, as mentioned', 'however, the',
            'as mentioned, the', 'as mentioned above',
            # Meta-commentary about sources / text snippets
            'hence both', 'both text', 'text snippet', 'both snippet', 'both passage',
            'both source', 'both confirm', 'hence the text', 'hence the source',
            'both extracts', 'both of the', 'the two text', 'the two source',
            'snippets confirm', 'passages confirm', 'sources confirm',
            'both explicitly', 'both implicitly', 'both mention',
            'all told', 'all in all', 'taking all', 'putting it all',
            'finally these', 'finally, these', 'finally both',
        ]
        # Also reject sentences containing strongly informal/hallucinated markers mid-sentence
        INFORMAL_REJECT = re.compile(
            r'\b(ya\s+go|folks|heed\s+my|tiny\s+grain|mighty\s+power|magic\s+potion|'
            r'just\s+cuz|sorted\s+down|those\s+tiny|cow.s\s+udder|udder|'
            r'clockwork|there\s+ya|maximally\s+drink|no\s+longer\s+\.|'
            r'first\s+thing\s+in\s+morning)\b',
            re.IGNORECASE
        )
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
            # Reject sentences that are pure bibliography/chapter/verse/page markers
            if re.search(r'\b(chapter|verse|bibliography|ibid)\b', low):
                continue
            if re.search(r'\bpage\s*no\.?\b|\bline\s+number\b|\bpage\s+\d+\b', low):
                continue
            # Reject LLM meta-commentary: "extracted information", "pertaining", "constraints", etc.
            if re.search(r'\b(pertaining|verbatim|constraints?|extracted\s+information|pertainingsource|directly\s+mentioned)\b', low):
                continue
            # Reject sentences with informal/hallucinated language markers
            if INFORMAL_REJECT.search(s):
                continue
            # Reject run-on sentences with no benefit verb (>250 chars)
            if len(s) > 250 and not re.search(r'\b(helps?|reduces?|supports?|improves?|treats?|benefits?|boosts?|aids?|relieves?|contains?|promotes?)\b', low):
                continue
            # Ensure sentence ends with proper punctuation
            if not s[-1] in '.!?':
                s = s + '.'
            # Capitalize first letter
            s = s[0].upper() + s[1:]
            clean.append(s)

        if not clean:
            # Fallback: split raw into sentences — apply THE SAME filters as main path
            fallback_sentences = re.split(r'(?<=[.!?])\s+', raw.strip())
            # If one long paragraph with no sentence breaks, try comma+capital split
            if len(fallback_sentences) == 1 and len(fallback_sentences[0]) > 200:
                fallback_sentences = re.split(r',\s+(?=[A-Z])', raw.strip())
            fallback_clean = []
            for s in fallback_sentences:
                s = s.strip()
                if len(s) < 15:
                    continue
                low_s = s.lower()
                # Apply the same FILLER_STARTS check as the main path
                if any(low_s.startswith(f) for f in FILLER_STARTS):
                    continue
                # Apply the same INFORMAL_REJECT regex
                if INFORMAL_REJECT.search(s):
                    continue
                if re.search(r'\b(chapter|verse|bibliography|ibid)\b', low_s):
                    continue
                if re.search(r'\bpage\s*no\.?\b|\bline\s+number\b|\bpage\s+\d+\b', low_s):
                    continue
                if re.search(r'\b(source|citation|reference)\b', low_s) and \
                   re.search(r'\b(states?|says?|mentions?|notes?|reports?|indicates?|describes?)\b', low_s):
                    continue
                if len(s) > 250 and not re.search(r'\b(helps?|reduces?|supports?|improves?|treats?|benefits?|boosts?|aids?)\b', low_s):
                    continue
                if s[-1] not in '.!?':
                    s = s + '.'
                fallback_clean.append(s[0].upper() + s[1:])
            if fallback_clean:
                return '\n'.join(f'\u2022 {b}' for b in fallback_clean[:3])
            # Nothing survived — return a generic fallback rather than raw garbage
            return ''

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
                    # Reject English words mixed into Sinhala sentences
                    # e.g. 'curcumin(kur)', 'Guggilam', 'Ashwagandha' leaking in
                    # Allow: numbers, single letters, common punctuation, known loan words
                    if re.match(r'^[\d.,!?%;:()\-\/\\]+$', word):
                        cleaned_words.append(word)  # keep numbers/punctuation
                    elif len(word) <= 2:
                        cleaned_words.append(word)  # keep short tokens like "–"
                    # else: drop English words embedded in Sinhala lines
            
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

        result = '\n'.join(cleaned_lines)

        # Final guard: if more than 40% of characters are ASCII (English), the translation
        # failed badly — return empty so the caller falls back to English
        ascii_chars = sum(1 for c in result if ord(c) < 128 and c.isalpha())
        total_chars = sum(1 for c in result if c.isalpha())
        if total_chars > 0 and ascii_chars / total_chars > 0.4:
            return ''

        return result
    
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

    def _extract_followup_context(self, question: str) -> tuple[str, str]:
        """
        Split frontend follow-up marker:
        "What about side effects? (regarding: Turmeric benefits)"
        -> ("What about side effects?", "Turmeric benefits")
        """
        import re

        if not question:
            return "", ""

        match = re.search(r'\s*\(regarding:\s*(.*?)\)\s*$', question, flags=re.IGNORECASE)
        if not match:
            return question.strip(), ""

        main_q = re.sub(r'\s*\(regarding:.*?\)\s*$', '', question, flags=re.IGNORECASE).strip()
        regarding = (match.group(1) or '').strip()
        return main_q, regarding

    def _classify_question_intent(self, question: str) -> str:
        """Classify query intent so follow-ups get intent-specific answers."""
        import re

        q = (question or '').lower()
        patterns = [
            ('side_effects', r'\b(side\s*effects?|adverse|harm|danger|risk|unsafe|caution|warning|contraindication)\b'),
            ('dosage', r'\b(dosage|dose|how much|how many|how often|how long|amount|quantity|when to take|how to take)\b'),
            ('combination', r'\b(combine|combined|together|with|mix|can i take .* with|interaction)\b'),
            ('treatment', r'\b(treat|treatment|manage|relief|what to do|i have|suffering|symptom|fever|cold|cough|pain|infection|remedy)\b'),
            ('benefits', r'\b(benefit|benefits|uses|properties|guna|good for|helps|advantages)\b'),
            ('definition', r'\b(what is|what are|meaning|define|explain|describe)\b'),
        ]

        for intent, pattern in patterns:
            if re.search(pattern, q, flags=re.IGNORECASE):
                return intent
        return 'general'

    def _build_search_query(self, question: str, regarding_context: str, intent: str) -> str:
        """Build an intent-aware retrieval query from current + follow-up context."""
        base = question.strip()
        if regarding_context:
            base = f"{base} {regarding_context}".strip()

        intent_expansions = {
            'benefits': 'health benefits medicinal properties ayurveda evidence',
            'side_effects': 'side effects safety precautions contraindications toxicity interactions',
            'dosage': 'dosage amount frequency how to take with food timing ayurveda',
            'combination': 'combination with interactions synergy bioavailability compatibility',
            'treatment': 'symptoms causes ayurvedic treatment home remedies diet lifestyle',
            'definition': 'definition explanation ayurvedic concept basics',
            'general': 'ayurvedic guidance practical advice',
        }

        extra = intent_expansions.get(intent, intent_expansions['general'])
        return f"{base} {extra}".strip()

    def _generate_personalized_tips(self, question: str, answer: str, dosha: str, original_question: str = '') -> str:
        """
        Generate 3 dosha-specific Ayurvedic tips using question-type-aware templates.
        No LLM call — pure template system guarantees clean, relevant output.
        """
        import re as _re
        print(f"💡 Generating template tips for {dosha} dosha...")

        # --- Step 1: Extract a clean topic keyword ---
        _source_q = question

        # Map romanized Sinhala herb names via the dictionary first
        try:
            from translation_service import SINHALA_TO_ENGLISH_DICT
            for sinhala_key, english_val in SINHALA_TO_ENGLISH_DICT.items():
                if sinhala_key.lower() in original_question.lower():
                    _source_q = f"what are the benefits of {english_val}"
                    break
        except Exception:
            pass

        # Step 1: Extract topic — detect question type first (before stripping prefix)
        _q = _source_q.lower()

        # Remove the "(regarding: ...)" context suffix that buildQuestion appends
        _q = _re.sub(r'\s*\(regarding:.*?\)', '', _q, flags=_re.IGNORECASE).strip()

        # Detect question type BEFORE stripping topic (used for template selection)
        _is_dosage   = bool(_re.search(r'\b(how much|how many|how often|how long|dosage|dose|quantity|amount)\b', _q))
        _is_sideeff  = bool(_re.search(r'\b(side effect|harm|danger|safe|risk|caution|warning)\b', _q))
        _is_howto    = bool(_re.search(r'\b(how to use|how to take|how to consume|how to prepare)\b', _q))
        _is_treatment = bool(_re.search(r'\b(treat|treatment|manage|what to do|i have|suffering|symptom|fever|cold|cough|pain|infection|remedy)\b', _q))

        # Detect if question is about a dosha/concept rather than a specific herb
        _is_concept  = bool(_re.search(r'\b(what is|what are|explain|describe)\b', _q) and
                           _re.search(r'\b(vata|pitta|kapha|dosha|tridosha|ayurveda|prakriti|dhatu|agni)\b', _q))
        _is_which_herbs = bool(_re.search(r'\bwhich\s+herb|what\s+herbs?\b', _q))

        # Strip common question prefixes to isolate the herb/topic
        topic_raw = _re.sub(
            r'^(what are the (?:ayurvedic\s+)?(?:benefits|uses|properties|effects|qualities|guna)\s+of|'
            r'what are the benefits of|what are the uses of|what is the use of|'
            r'what are the|what are|what is the|what is|'
            r'how does|how is|how can|how to|how much|how many|how often|how long|'
            r'are there any|is there any|can i use|should i take|can i take|should i|'
            r'tell me about|tell me the|explain|describe|'
            r'benefits of|uses of|properties of|effects of|'
            r'what dosha does|which dosha|which herb|which herbs|'
            r'is there a|is it safe|is it good|is it ok|is it ok|'
            r'is\s+\w+\s+good|is\s+\w+\s+safe|is\s+\w+\s+ok|is\s+\w+\s+bad|'
            r'is\s+\w+\s+helpful|is\s+\w+\s+effective|is\s+\w+\s+beneficial|'
            r'are\s+\w+\s+good|are\s+\w+\s+safe|are\s+\w+\s+ok)\s+',
            '', _q, flags=_re.IGNORECASE
        ).strip().strip('?').strip()

        # If result has "of <herb>" inside, extract just the herb
        _of_match = _re.search(r'\bof\s+([a-zA-Z][a-zA-Z\s]{1,30}?)(?:\s+(?:daily|\?|\)|should|i|for|in|with|and|or|that|the).*)?$', topic_raw)
        if _of_match:
            topic_raw = _of_match.group(1).strip()

        # Strip trailing verb / conjunction phrases
        topic_raw = _re.sub(
            r'\s+(good|safe|helpful|beneficial|effective|useful|okay|ok|bad|harmful|'
            r'help\s*\w*|helps?|reduce\w*|balance\w*|support\w*|cure\w*|treat\w*|'
            r'do|does|is|are|used|with|for|should|take|daily|morning|each|every|and|or).*$',
            '', topic_raw, flags=_re.IGNORECASE
        ).strip()

        # Strip standalone pronoun/article left as leading artifact: "i ", "my ", "your ", "the "
        topic_raw = _re.sub(r'^(i|my|your|our|the|a|an)\s+', '', topic_raw, flags=_re.IGNORECASE).strip()

        # Remove trailing filler words and punctuation
        topic_raw = _re.sub(r'\s+(in ayurveda|ayurvedic|for health|for body|for|in|to|with|a|an|the)$', '', topic_raw, flags=_re.IGNORECASE).strip()
        topic_raw = topic_raw.strip('?)(.,')

        # Guard: if topic_raw is empty/generic/still a question fragment, use fallback
        _bad_topics = {
            'are', 'is', 'it', 'they', 'them', 'this', 'that', 'useful',
            'benefits', 'guna', 'monawada', 'what', 'how', 'does', 'do',
            'much', 'many', 'often', 'any', 'there', 'can', 'should',
            'side', 'effects', 'safe', 'dosage', 'dose', 'amount', 'long',
            # Dosha and concept words — not specific herbs
            'herbs', 'herb', 'vata', 'pitta', 'kapha', 'dosha', 'tridosha',
            'prakriti', 'dhatu', 'agni', 'ojas', 'prana', 'ayurveda', 'which',
            'food', 'foods', 'medicine', 'treatment', 'help', 'imbalance',
            # Action/question words that slip through prefix stripping
            'take', 'taking', 'taken', 'recommended', 'meaning', 'means',
            'combine', 'combined', 'combining', 'together', 'good', 'correct',
            'ratio', 'time', 'when', 'where', 'why', 'could', 'would', 'might',
            'mean', 'general', 'use', 'using', 'used', 'consume', 'consumption',
            'daily', 'morning', 'evening', 'right', 'proper', 'best',
            'health', 'body', 'blood', 'sugar', 'water', 'milk', 'intake',
            'have', 'having',
            # Multi-word question fragments that slip through
            'improve', 'digestion', 'digestions', 'diagestion', 'function',
            'sleep', 'stress', 'energy', 'weight', 'immunity', 'inflammation',
            'better', 'boost', 'increase', 'decrease', 'prevent', 'control',
            'manage', 'reduce', 'balance', 'detox', 'cleanse', 'heal',
        }
        topic_words = set(topic_raw.lower().split())
        _is_bad_topic = not topic_raw or len(topic_raw) < 3 or topic_words.issubset(_bad_topics)

        if _is_bad_topic:
            # Last resort: first known-herb word from original question
            _words = [w for w in _re.findall(r'\b[a-zA-Z]{4,}\b', original_question + ' ' + question)
                      if w.lower() not in _bad_topics | {'wala', 'what', 'guna', 'monawada', 'regarding',
                                                         'causes', 'cause', 'imbalance', 'question', 'answer'}]
            topic_raw = _words[0] if _words else ''
            _is_bad_topic = not topic_raw or len(topic_raw) < 3

        topic = topic_raw.title() if topic_raw else ''

        # --- Step 2: Concept tips (dosha/general questions with no specific herb) ---
        concept_tips = {
            'Vata': [
                "Keep regular daily routines — fixed meal times and sleep schedules are the best way to calm Vata.",
                "Eat warm, cooked, and slightly oily foods like rice with ghee, soups, and root vegetables.",
                "Gentle oil massage on the body before bathing helps reduce Vata dryness and anxiety.",
            ],
            'Pitta': [
                "Eat cooling foods like cucumber, coconut, and fresh greens to reduce Pitta heat in the body.",
                "Avoid spicy, fried, or very sour foods — these increase Pitta and cause irritation and acidity.",
                "Spending time in nature, cool showers, and short daily meditation help balance Pitta energy.",
            ],
            'Kapha': [
                "Eat light, warm, and dry foods — avoid heavy, oily, or sweet foods that slow down Kapha.",
                "Regular brisk exercise every morning is the best way to reduce Kapha heaviness and weight.",
                "Wake up before sunrise and avoid daytime naps to keep Kapha energy active and clear.",
            ],
            'General': [
                "Eat fresh, seasonal, and lightly cooked foods every day for good health in Ayurveda.",
                "Regular sleep, daily exercise, and consistent meal times form the foundation of Ayurvedic wellbeing.",
                "Consult an Ayurvedic practitioner to find the right diet and herbs for your body type.",
            ],
        }

        treatment_tips = {
            'Vata': [
                "For Vata-type symptoms, keep warm, hydrated, and rested; avoid cold food and irregular routines.",
                "Use light warm foods like rice gruel, thin mung soup, and ginger-tulsi tea during recovery.",
                "If fever or pain persists, consult a qualified clinician promptly for personalized care.",
            ],
            'Pitta': [
                "For Pitta-type symptoms, use cooling and light foods while avoiding spicy, fried, and sour meals.",
                "Hydrate well with warm water or room-temperature herbal fluids and prioritize rest.",
                "If symptoms are intense or persistent, seek professional medical care without delay.",
            ],
            'Kapha': [
                "For Kapha-type symptoms, prefer warm, light, and mildly spiced foods to reduce heaviness and congestion.",
                "Avoid cold dairy and heavy meals; use warm herbal drinks and gentle movement when able.",
                "Consult a healthcare professional if symptoms worsen or do not improve within a short time.",
            ],
            'General': [
                "During illness, prioritize hydration, rest, and easy-to-digest warm foods.",
                "Avoid heavy, oily, and very cold foods until digestion and energy recover.",
                "Seek timely medical care for high fever, breathing difficulty, dehydration, or persistent symptoms.",
            ],
        }

        # --- Step 2: Select templates based on dosha AND question type ---
        # Two flavour groups per benefit type — picked by topic's first letter so
        # the same herb always gets the same set, but different herbs get different tips.
        _grp = ord(topic[0].lower()) % 2 if topic else 0  # 0 or 1

        dosha_tips = {
            'Vata': {
                'benefit': [
                    # Flavour A — grounding / nourishment focus
                    [
                        f"{topic} reduces Vata dryness and supports joint and nerve health when taken with warm ghee.",
                        f"Mixing {topic} with warm sesame oil or milk helps the body absorb it and calms Vata anxiety.",
                        f"Take {topic} regularly in the morning with warm water to build a grounding Vata daily routine.",
                    ],
                    # Flavour B — warmth / circulation focus
                    [
                        f"{topic} warms and nourishes the Vata body by improving circulation and reducing cold sensitivity.",
                        f"For Vata types, {topic} works best as a warm drink with a pinch of cardamom before meals.",
                        f"Consistent daily use of {topic} for 4 weeks helps settle Vata restlessness and irregular digestion.",
                    ],
                ],
                'dosage': [
                    f"Take {topic} in small amounts — start with a pinch or quarter teaspoon and adjust slowly.",
                    f"Consume {topic} with warm water or ghee in the morning for the best absorption in Vata types.",
                    f"An Ayurvedic practitioner can set the exact daily dose of {topic} suited to your Vata constitution.",
                ],
                'sideeff': [
                    f"{topic} is generally well tolerated for Vata types when taken in small, consistent amounts.",
                    f"Avoid taking {topic} on an empty stomach if it causes gas or bloating, which Vata types can experience.",
                    f"Taking {topic} with warm ghee or milk reduces any drying or irritating effects on the Vata system.",
                ],
            },
            'Pitta': {
                'benefit': [
                    # Flavour A — cooling / anti-inflammation focus
                    [
                        f"{topic} helps reduce Pitta-related heat and inflammation — take it with cool water or coconut milk.",
                        f"For Pitta types, {topic} works best when taken in the morning before Pitta energy builds up.",
                        f"Pair {topic} with cooling foods like cucumber, mint, and fresh coconut to boost its Pitta-calming effect.",
                    ],
                    # Flavour B — digestion / steady use focus
                    [
                        f"{topic} supports healthy Pitta digestion and reduces acid build-up when taken after meals.",
                        f"For Pitta types, {topic} is most effective at room temperature — avoid hot water preparations.",
                        f"Use {topic} consistently for 2–4 weeks to see steady improvement in Pitta-related digestion and skin.",
                    ],
                ],
                'dosage': [
                    f"Take {topic} in moderate amounts — too high a dose can overheat the Pitta system.",
                    f"A cool or room-temperature preparation of {topic} works better than hot preparations for Pitta types.",
                    f"An Ayurvedic practitioner can recommend the right balanced dose of {topic} for your Pitta constitution.",
                ],
                'sideeff': [
                    f"{topic} is safe for Pitta when taken in moderate amounts and not during active inflammation flare-ups.",
                    f"If {topic} increases body heat or causes acidity, reduce the dose and take it with cooling fennel water.",
                    f"Avoid combining {topic} with spicy or very sour foods, as this can make Pitta side effects worse.",
                ],
            },
            'Kapha': {
                'benefit': [
                    # Flavour A — metabolism / digestion focus
                    [
                        f"{topic} reduces Kapha heaviness and stimulates metabolism and digestion when taken regularly.",
                        f"Taking {topic} with black pepper and warm water activates its digestive benefits for Kapha types.",
                        f"Use {topic} before breakfast on an empty stomach to clear Kapha sluggishness from the digestive tract.",
                    ],
                    # Flavour B — energy / weight focus
                    [
                        f"{topic} helps Kapha types by breaking down excess mucus and improving morning energy levels.",
                        f"Combine {topic} with a light diet and 20 minutes of brisk morning walking to add to its Kapha-clearing effect.",
                        f"For Kapha types, {topic} is most powerful with warm ginger water — avoid taking it with heavy meals.",
                    ],
                ],
                'dosage': [
                    f"Kapha types can take a slightly higher dose of {topic} as their metabolism handles it well.",
                    f"Take {topic} on an empty stomach each morning with warm water for the best Kapha-clearing effect.",
                    f"An Ayurvedic practitioner can confirm the right daily quantity of {topic} for your Kapha body type.",
                ],
                'sideeff': [
                    f"{topic} is well suited to Kapha types and side effects are rare when used in normal daily amounts.",
                    f"If {topic} causes heaviness or congestion, add dry ginger to your preparation to counteract Kapha buildup.",
                    f"Avoid taking {topic} with cold or sweet foods, as this increases Kapha and reduces its benefits.",
                ],
            },
            'General': {
                'benefit': [
                    # Flavour A
                    [
                        f"{topic} supports general Ayurvedic health and immunity when consumed daily with meals.",
                        f"A warm preparation of {topic} each morning increases its absorption and daily health benefits.",
                        f"An Ayurvedic practitioner can recommend the right form and dose of {topic} for your body type.",
                    ],
                    # Flavour B
                    [
                        f"{topic} is a well-known Ayurvedic herb used daily to support digestion, energy, and immunity.",
                        f"For best results, take {topic} consistently each day rather than on and off.",
                        f"Pair {topic} with plenty of warm water and a balanced diet to get its full health benefits.",
                    ],
                ],
                'dosage': [
                    f"Start with a small daily amount of {topic} and observe how your body responds before increasing.",
                    f"{topic} is best taken with warm water in the morning to get its full daily benefits.",
                    f"Consult an Ayurvedic practitioner for the correct dose of {topic} suited to your constitution.",
                ],
                'sideeff': [
                    f"{topic} is generally safe when used in traditional Ayurvedic amounts and standard preparations.",
                    f"If you feel any discomfort after taking {topic}, reduce the amount and always take it with food.",
                    f"Consult an Ayurvedic practitioner before long-term daily use of {topic} for your body type.",
                ],
            },
        }

        # Resolve compound doshas (e.g. "Vata-Pitta" → Vata)
        dosha_key = 'General'
        for d in ['Vata', 'Pitta', 'Kapha']:
            if d in dosha:
                dosha_key = d
                break

        # If no valid herb/topic was found (concept question like "what is pitta?"),
        # skip herb tips and use general dosha lifestyle advice
        if _is_treatment:
            tips = treatment_tips[dosha_key]
            print(f"✓ Treatment tips generated for {dosha_key}")
            return '\n'.join(f'{i+1}. {t}' for i, t in enumerate(tips))

        if _is_bad_topic or _is_concept or _is_which_herbs:
            tips = concept_tips[dosha_key]
            print(f"✓ Concept tips generated for {dosha_key} (no herb topic found)")
            return '\n'.join(f'{i+1}. {t}' for i, t in enumerate(tips))

        # Select template variant based on question type
        if _is_sideeff:
            q_type = 'sideeff'
        elif _is_dosage or _is_howto:
            q_type = 'dosage'
        else:
            q_type = 'benefit'

        tip_set = dosha_tips[dosha_key][q_type]
        # For benefit type, pick between flavour A (index 0) and B (index 1) by topic hash
        if q_type == 'benefit' and isinstance(tip_set[0], list):
            tips = tip_set[_grp]
        else:
            tips = tip_set
        print(f"✓ Tips generated for {dosha_key}/{q_type} (flavour {_grp}) — topic: '{topic}'")
        return '\n'.join(f'{i+1}. {t}' for i, t in enumerate(tips))

    def answer_question(
        self,
        question: str,
        vector_db,
        top_k: int = 5,
        user_profile: Optional[Dict[str, Any]] = None,
        validation_top_k: int = 5,
        dominant_dosha: Optional[str] = None,
        web_db=None  # Optional separate FAISS index for web knowledge
    ) -> Dict[str, Any]:
        """Answer question with hybrid retrieval (books + web), validation, personalization, and translation"""
        logger.info(f"Processing question: {question[:50]}...")

        # Extract optional frontend follow-up context marker.
        # Example: "What about side effects? (regarding: Turmeric benefits)"
        parsed_question, regarding_context = self._extract_followup_context(question)
        question_for_processing = parsed_question or question

        # === DOMAIN FILTER: Block non-Ayurvedic questions ===
        if _DOMAIN_FILTER_AVAILABLE:
            is_ayurvedic, reason = is_ayurvedic_question(question_for_processing)
            if not is_ayurvedic:
                print(f"🚫 Domain filter blocked: {question[:60]}... | Reason: {reason}")
                return {
                    "answer": get_rejection_message(),
                    "answer_english": get_rejection_message(),
                    "base_answer": "",
                    "original_question": question,
                    "translated_question": None,
                    "detected_language": "en",
                    "is_romanized": False,
                    "citations": [],
                    "sources": [],
                    "all_sources": [],
                    "num_sources": 0,
                    "personalized": False,
                    "detected_dosha": "General",
                    "personalized_tips": "",
                    "validation": {},
                    "user_info": {},
                    "blocked": True,
                    "block_reason": reason
                }

        # === ANSWER CACHE: same question + same dosha always returns same answer ===
        import copy as _copy
        _cache_key = f"{question.lower().strip()}::{(dominant_dosha or '').lower()}"
        if _cache_key in self._answer_cache:
            print(f"✅ Cache hit — returning cached answer for: {question[:60]}...")
            return _copy.deepcopy(self._answer_cache[_cache_key])
        
        # === TRANSLATION: Detect language and translate question if needed ===
        original_question = question
        detected_language = 'en'  # Default to English
        is_romanized = False  # Track if romanized Singlish
        question_for_processing_en = question_for_processing
        regarding_context_en = regarding_context
        
        if self.enable_translation and self.translator:
            detected_language = self.translator.detect_language(question_for_processing)
            print(f"🌐 Detected language: {detected_language.upper()}")
            
            if detected_language in ('si', 'ta'):
                # Check if romanized (no Sinhala Unicode chars) or native script
                is_romanized = not self.translator._has_sinhala_chars(question_for_processing)
                
                if is_romanized:
                    print(f"📝 Romanized input detected (e.g., 'kurudu wala guna')")
                
                # Translate question to English for processing
                question_for_processing_en = self.translator.translate_si_to_en(
                    question_for_processing,
                    is_romanized=is_romanized
                )
                if regarding_context:
                    regarding_context_en = self.translator.translate_si_to_en(
                        regarding_context,
                        is_romanized=is_romanized
                    )
                print(f"🔄 Translated question: {question_for_processing_en[:100]}...")

        # Classify question intent after translation for robust follow-up handling.
        combined_query_for_intent = f"{question_for_processing_en} {regarding_context_en}".strip()
        question_intent = self._classify_question_intent(combined_query_for_intent)
        print(f"🧭 Detected intent: {question_intent}")
        
        # === HYBRID RETRIEVAL: Books + Web ===
        print("🔍 Retrieving relevant sources (Books + Web)...")

        # Build intent-aware search query from current + follow-up context.
        search_query = self._build_search_query(
            question=question_for_processing_en,
            regarding_context=regarding_context_en,
            intent=question_intent,
        )
        print(f"🎯 Enhanced search query: {search_query[:120]}...")

        # Step 1: Retrieve from BOOK FAISS (existing)
        book_retrieved = vector_db.search(search_query, top_k=validation_top_k, prefer_books=True)
        print(f"   📚 Books: {len(book_retrieved)} docs retrieved")

        # Step 2: Retrieve from WEB FAISS (new — optional)
        web_retrieved = []
        if web_db is not None:
            try:
                web_retrieved = web_db.search(search_query, top_k=3, prefer_books=False)
                # Tag web docs clearly
                for doc in web_retrieved:
                    doc['type'] = 'web'
                print(f"   🌐 Web: {len(web_retrieved)} docs retrieved")
            except Exception as e:
                print(f"   ⚠️ Web retrieval failed: {e}")
                web_retrieved = []

        # Step 3: Merge — books first (higher authority), then web
        retrieved_docs = book_retrieved + web_retrieved

        # Separate by type for context building
        book_docs = [d for d in retrieved_docs if d.get("type") == "book"]
        web_docs  = [d for d in retrieved_docs if d.get("type") == "web"]
        qa_docs   = [d for d in retrieved_docs if d.get("type") == "qa"]

        # Add similarity percentages
        for doc in retrieved_docs:
            if 'similarity' in doc:
                doc['similarity_percentage'] = round(doc['similarity'] * 100, 1)

        # Build context: books take priority, append web sources as supplementary
        if book_docs:
            top_context_docs = book_docs[:top_k]
            # Append up to 2 web sources for richer context on complex questions
            top_context_docs = top_context_docs + web_docs[:2]
            print(f"📚 Context: {len(book_docs[:top_k])} book + {len(web_docs[:2])} web sources")
        else:
            top_context_docs = (web_docs + qa_docs)[:top_k]
            print(f"⚠️  No book sources — using {len(web_docs)} web + {len(qa_docs)} QA entries")
        
        # Token budget — 150 tokens is enough for 2-3 clear sentences (Feb 23 approach)
        dynamic_tokens = 150

        # ── CONCEPT KNOWLEDGE BASE ──────────────────────────────────────────────
        # For fundamental Ayurvedic concepts the vector DB may not have a direct
        # definition chunk, so we inject a curated paragraph as context.
        # The LLM uses ONLY this text as its source — no hallucination from irrelevant chunks.
        CONCEPT_KB = {
            'pitta': (
                "Pitta is one of the three doshas (body energies) in Ayurveda. "
                "It is made of fire and water elements. Pitta controls digestion, metabolism, "
                "body temperature, hunger, thirst, and intelligence. "
                "When Pitta is balanced, a person has good digestion, sharp thinking, and a warm personality. "
                "When Pitta is too high, it causes acidity, skin rashes, anger, inflammation, and burning sensations. "
                "Pitta is balanced by eating cooling foods like cucumber, coconut, and fresh greens, "
                "and by avoiding spicy, fried, and sour foods."
            ),
            'vata': (
                "Vata is one of the three doshas in Ayurveda. "
                "It is made of air and space elements. Vata controls all movement in the body — "
                "breathing, blood circulation, nerve signals, and digestion. "
                "When Vata is balanced, a person feels energetic, creative, and mentally clear. "
                "When Vata is too high, it causes dry skin, constipation, anxiety, joint pain, and irregular digestion. "
                "Vata is balanced by eating warm, cooked, oily foods like soups and rice with ghee, "
                "and by keeping a regular daily routine."
            ),
            'kapha': (
                "Kapha is one of the three doshas in Ayurveda. "
                "It is made of earth and water elements. Kapha gives the body structure, strength, and stability. "
                "It controls body weight, immune strength, joint lubrication, and calm emotions. "
                "When Kapha is balanced, a person is strong, calm, and steady. "
                "When Kapha is too high, it causes weight gain, laziness, congestion, slow digestion, and depression. "
                "Kapha is balanced by eating light, warm, spicy foods, exercising regularly, and waking up early."
            ),
            'dosha': (
                "In Ayurveda, a dosha is one of three fundamental body energies — Vata, Pitta, and Kapha. "
                "Every person has all three doshas, but one or two are usually dominant and form their Prakriti (body type). "
                "Vata (air+space) controls movement. Pitta (fire+water) controls digestion and metabolism. "
                "Kapha (earth+water) gives structure and strength. "
                "Good health in Ayurveda means keeping your doshas in their natural balance through diet, lifestyle, and herbs."
            ),
            'tridosha': (
                "Tridosha means the three doshas together — Vata, Pitta, and Kapha. "
                "These three energies control every function in the human body and mind. "
                "Vata controls movement and breathing. Pitta controls digestion and body heat. "
                "Kapha gives strength and stability. "
                "Ayurvedic treatment works by finding which dosha is out of balance and correcting it with the right food, "
                "herbs, and daily habits."
            ),
            'prakriti': (
                "Prakriti is your natural body type in Ayurveda, determined at birth by the balance of Vata, Pitta, and Kapha. "
                "There are seven Prakriti types: Vata, Pitta, Kapha, Vata-Pitta, Pitta-Kapha, Vata-Kapha, and Tridoshic. "
                "Knowing your Prakriti helps you choose the right diet, lifestyle, and herbs to stay healthy. "
                "Your Prakriti does not change throughout life, but the doshas can go out of balance due to diet, stress, or season."
            ),
            'agni': (
                "Agni means digestive fire in Ayurveda. "
                "It is the body's power to digest food, absorb nutrients, and remove waste. "
                "Strong Agni means good digestion, energy, and clear thinking. "
                "Weak Agni causes bloating, tiredness, and toxin buildup (Ama) in the body. "
                "Agni is kept strong by eating at regular times, avoiding cold or heavy foods, "
                "and using digestive herbs like ginger, cumin, and fennel."
            ),
            'ojas': (
                "Ojas is the vital life essence in Ayurveda. "
                "It is the purest result of good digestion and nourishment. "
                "Strong Ojas gives immunity, energy, a calm mind, and a glowing appearance. "
                "Ojas is reduced by stress, poor sleep, overwork, and unhealthy food. "
                "It is built up by eating fresh natural foods, getting enough sleep, practising meditation, "
                "and using herbs like Ashwagandha and Shatavari."
            ),
            'ama': (
                "Ama means undigested toxins in Ayurveda. "
                "It forms in the body when Agni (digestive fire) is weak and food is not fully processed. "
                "Ama looks like a white coating on the tongue and causes heaviness, tiredness, and blocked channels. "
                "Most diseases in Ayurveda start from Ama buildup. "
                "It is removed by improving digestion, eating light foods, fasting occasionally, "
                "and using detoxifying herbs like Triphala and dry ginger."
            ),
            'dhatu': (
                "Dhatu means body tissue in Ayurveda. "
                "There are seven Dhatus: Rasa (plasma), Rakta (blood), Mamsa (muscle), Meda (fat), "
                "Asthi (bone), Majja (marrow/nerve), and Shukra (reproductive tissue). "
                "Each Dhatu is nourished in order from food — first Rasa, then Rakta, and so on. "
                "Healthy Dhatus depend on strong Agni and a balanced diet. "
                "Weak Dhatus cause various diseases depending on which tissue is affected."
            ),
            'ayurveda': (
                "Ayurveda is a traditional system of medicine from India that is over 5000 years old. "
                "The word means 'knowledge of life' in Sanskrit. "
                "Ayurveda teaches that health comes from balance between the body, mind, and spirit. "
                "It uses diet, herbs, yoga, massage, and daily routines to prevent and treat disease. "
                "The three doshas — Vata, Pitta, and Kapha — are the foundation of Ayurvedic health theory. "
                "Ayurveda is still widely used in Sri Lanka, India, and around the world."
            ),
        }

        # Detect if question is about a known Ayurvedic concept
        import re as _re_kb
        _q_lower = question_for_processing_en.lower()
        _injected_context = None
        for _concept_key, _concept_text in CONCEPT_KB.items():
            if _re_kb.search(r'\b' + _concept_key + r'\b', _q_lower):
                _injected_context = _concept_text
                print(f"📖 Concept question detected: '{_concept_key}' — using curated KB context")
                break
        # ── END CONCEPT KNOWLEDGE BASE ───────────────────────────────────────────

        # ── CONDITION KNOWLEDGE BASE (general symptom questions) ────────────────
        CONDITION_KB = {
            'fever': {
                'treatment': (
                    "In Ayurveda, fever (Jvara) is managed with light, warm fluids and herbs that support digestion and immunity. "
                    "Useful home support includes warm Tulsi-ginger tea, adequate rest, and easily digestible foods like rice gruel or thin mung soup. "
                    "If fever is high, persistent, or associated with dehydration or breathing difficulty, seek medical care promptly in addition to Ayurvedic support."
                )
            },
            'cough': {
                'treatment': (
                    "Ayurvedic care for cough focuses on reducing mucus and soothing irritated airways with warm, spiced liquids. "
                    "Tulsi, dry ginger, black pepper, and honey are traditionally used in small amounts to support relief. "
                    "Avoid cold, oily, and heavy foods during cough episodes, and seek medical evaluation if cough persists or worsens."
                )
            },
            'cold': {
                'treatment': (
                    "For common cold, Ayurveda emphasizes warm hydration, steam inhalation, and light foods to protect Agni (digestive fire). "
                    "Tulsi-ginger-pepper decoction and warm soups are commonly used to reduce congestion and support recovery. "
                    "Rest well and avoid chilled foods; consult a clinician if symptoms are severe or prolonged."
                )
            }
        }
        if not _injected_context:
            for _cond, _data in CONDITION_KB.items():
                if _re_kb.search(r'\b' + _re_kb.escape(_cond) + r'\b', combined_query_for_intent.lower()):
                    _cond_key = 'treatment' if question_intent in ('treatment', 'general') else question_intent
                    _injected_context = _data.get(_cond_key) or _data.get('treatment')
                    if _injected_context:
                        print(f"🩺 Condition fast-path: '{_cond}'/{_cond_key}")
                    break
        # ── END CONDITION KNOWLEDGE BASE ───────────────────────────────────────

        # ── HERB KNOWLEDGE BASE ──────────────────────────────────────────────────
        # Curated 3-sentence facts for common Ayurvedic herbs — same fast-path as
        # CONCEPT_KB. Guarantees stable answers regardless of vector search quality.
        # Covers English names, misspellings, and romanized Sinhala aliases.
        HERB_KB = {
            'Turmeric': (
                "Turmeric is used in Ayurveda to reduce inflammation, support liver health, and purify the blood. "
                "Its active compound curcumin has powerful antioxidant and anti-inflammatory effects that help joint pain and digestion. "
                "Turmeric balances all three doshas and is best taken with warm milk, ghee, or black pepper to improve absorption."
            ),
            'Cinnamon': (
                "Cinnamon is a warming digestive herb in Ayurveda that stimulates Agni (digestive fire) and reduces gas and bloating. "
                "It helps control blood sugar levels, improves circulation in the body, and reduces Vata and Kapha imbalances. "
                "Cinnamon is taken with honey or warm water each morning in Ayurveda to improve metabolism and digestive health."
            ),
            'Ginger': (
                "Ginger is called the universal Ayurvedic medicine because it stimulates digestion, relieves nausea, and clears toxins. "
                "It reduces gas, bloating, and indigestion, and also helps with respiratory congestion and joint inflammation. "
                "Ginger balances Vata and Kapha doshas and is most effective when taken fresh with lemon juice and warm water."
            ),
            'Neem': (
                "Neem is a powerful Ayurvedic herb used for its antibacterial, antifungal, and blood-purifying properties. "
                "It helps treat skin conditions, reduces Pitta-related heat and rashes, and supports liver detoxification. "
                "Neem leaf juice, oil, or powder is used in Ayurveda to balance Pitta and Kapha doshas and boost immunity."
            ),
            'Ashwagandha': (
                "Ashwagandha is a powerful Ayurvedic adaptogen that reduces stress, anxiety, and chronic fatigue. "
                "It strengthens the immune system, supports muscle recovery, and improves sleep quality and mental focus. "
                "Ashwagandha is classified as a Rasayana herb in Ayurveda that builds long-term vitality and balances Vata dosha."
            ),
            'Triphala': (
                "Triphala is a classical Ayurvedic formula made of three fruits — Amalaki, Bibhitaki, and Haritaki. "
                "It supports gentle detoxification, relieves constipation, and improves digestion and nutrient absorption. "
                "Triphala balances all three doshas and is recommended as a daily tonic for digestive health and immunity."
            ),
            'Brahmi': (
                "Brahmi is an Ayurvedic herb known for its brain-nourishing and nervine tonic properties. "
                "It improves memory, concentration, and mental clarity, and significantly reduces anxiety and stress. "
                "Brahmi balances Vata and Pitta doshas and is applied as scalp oil or consumed as powder for brain health."
            ),
            'Amla': (
                "Amla is the richest natural source of Vitamin C and a key Ayurvedic Rasayana rejuvenating herb. "
                "It boosts immunity, improves digestion and liver function, and supports healthy skin and hair growth. "
                "Amla balances all three doshas and is a main ingredient in Triphala and Chyawanprash formulas."
            ),
            'Tulsi': (
                "Tulsi is a sacred Ayurvedic herb with strong antibacterial, antiviral, and immune-boosting properties. "
                "It relieves respiratory infections, reduces stress and anxiety, improves digestion, and purifies the blood. "
                "Tulsi balances Vata and Kapha doshas and is consumed daily as herbal tea or fresh juice for immune support."
            ),
            'Cardamom': (
                "Cardamom is a cooling digestive spice in Ayurveda that relieves gas, nausea, and acidity after meals. "
                "It freshens breath, supports respiratory health, and helps Pitta types manage excess heat and indigestion. "
                "Cardamom is added to herbal preparations and warm milk to improve digestion and enhance absorption of other herbs."
            ),
            'Pepper': (
                "Black pepper is called the king of spices in Ayurveda and is used to stimulate digestion and metabolism. "
                "It helps clear respiratory congestion, boosts nutrient absorption (especially curcumin from turmeric), and kills toxins. "
                "Black pepper balances Vata and Kapha doshas and is recommended in small daily amounts with food or herbal formulas."
            ),
            'Garlic': (
                "Garlic is a powerful Ayurvedic herb used to improve circulation, reduce cholesterol, and fight infections. "
                "It helps lower blood pressure, supports heart health, and has strong antibacterial and antifungal properties. "
                "Garlic balances Vata and Kapha doshas and is most potent when consumed raw or lightly cooked each morning."
            ),
        }
        # Aliases: English misspellings + romanized Sinhala → HERB_KB key
        _HERB_KB_ALIASES = {
            'turmeric': 'Turmeric', 'tumeric': 'Turmeric', 'kaha': 'Turmeric', 'haridra': 'Turmeric',
            'cinnamon': 'Cinnamon', 'kurudu': 'Cinnamon', 'dalchini': 'Cinnamon', 'twak': 'Cinnamon',
            'ginger': 'Ginger', 'inguru': 'Ginger', 'shunti': 'Ginger',
            'neem': 'Neem', 'kohomba': 'Neem', 'nimba': 'Neem',
            'ashwagandha': 'Ashwagandha', 'aswagandha': 'Ashwagandha',
            'triphala': 'Triphala',
            'brahmi': 'Brahmi', 'welpenela': 'Brahmi', 'gotukola': 'Brahmi',
            'amla': 'Amla', 'nelli': 'Amla', 'amalaki': 'Amla',
            'tulsi': 'Tulsi', 'basil': 'Tulsi',
            'cardamom': 'Cardamom', 'elachi': 'Cardamom',
            'pepper': 'Pepper',
            'garlic': 'Garlic',
        }
        # Check translated question first, then original (catches romanized Sinhala)
        import re as _re_herb_kb
        if not _injected_context:
            for _q_check in (question_for_processing_en.lower(), regarding_context_en.lower(), original_question.lower()):
                for _alias, _kb_key in _HERB_KB_ALIASES.items():
                    if _re_herb_kb.search(r'\b' + _re_herb_kb.escape(_alias) + r'\b', _q_check):
                        if question_intent == 'side_effects':
                            _injected_context = (
                                f"{_kb_key} is generally safe in traditional Ayurvedic amounts, but excess use can cause digestive discomfort in some people. "
                                f"People with sensitive stomach, gallbladder disease, bleeding disorders, or those taking blood thinners should use {_kb_key.lower()} cautiously and seek professional advice. "
                                f"To reduce side effects, start with a small dose, take it with food, and stop use if irritation or unusual symptoms appear."
                            )
                        elif question_intent == 'dosage':
                            _injected_context = (
                                f"Ayurveda usually recommends {_kb_key.lower()} in small, regular doses rather than large occasional doses. "
                                f"A practical starting approach is low-dose daily use with food or warm water, then adjusting based on digestion and tolerance. "
                                f"The exact dose depends on age, health conditions, and medicines, so personalized guidance from a qualified practitioner is best for long-term use."
                            )
                        elif question_intent == 'combination':
                            if _kb_key == 'Turmeric' and 'pepper' in combined_query_for_intent.lower():
                                _injected_context = (
                                    "Turmeric is commonly combined with black pepper in Ayurveda because piperine in pepper improves curcumin absorption. "
                                    "This combination is often used in warm milk, herbal decoctions, or food with a small amount of healthy fat like ghee. "
                                    "Use moderate amounts and avoid high doses if you have active gastritis, are on anticoagulants, or have gallbladder problems."
                                )
                            else:
                                _injected_context = (
                                    f"{_kb_key} can often be combined with compatible Ayurvedic spices to improve absorption and therapeutic effect. "
                                    f"Combining {_kb_key.lower()} with warm carriers such as ghee, milk, or ginger water is commonly practiced depending on constitution and condition. "
                                    f"For safety, check medicine interactions and use practitioner guidance when combining multiple herbs long-term."
                                )
                        else:
                            _injected_context = HERB_KB[_kb_key]
                        print(f"🌿 Herb KB fast-path: '{_kb_key}' with intent '{question_intent}'")
                        break
                if _injected_context:
                    break
        # ── END HERB KNOWLEDGE BASE ──────────────────────────────────────────────

        # Build context
        context_text = self._build_context_with_citations(top_context_docs)
        
        print(f"🔍 Context preview (first 300 chars): {context_text[:300]}...")
        
        # ── FAST PATH: concept questions bypass the LLM entirely ───────────────
        # The small LLM hallucinates even when given correct source text.
        # For Pitta/Vata/Kapha/etc. we just format the curated KB sentences as
        # bullet points directly — same pattern as personalised tips generation.
        import re as _re_raw
        if _injected_context:
            _kb_sentences = [s.strip() for s in _re_raw.split(r'(?<=[.!?])\s+', _injected_context.strip()) if len(s.strip()) > 20]
            # Take first 3 sentences max
            _kb_bullets = _kb_sentences[:3]
            base_answer = '\n'.join(f'\u2022 {s}' for s in _kb_bullets)
            print(f"⚡ KB fast-path answer (no LLM): {base_answer[:120]}...")
        else:
            # ── CONTEXT EXTRACTION PATH (no LLM) ─────────────────────────────────
            # Same deterministic approach as _generate_personalized_tips:
            # score every sentence from the retrieved context by word-overlap with
            # the question and return the top 2-3 as bullet points.
            # No LLM → no hallucination, no off-topic answers, always stable.
            import re as _re_ctx

            # 1. Detect canonical herb name (handles misspellings + romanized Sinhala)
            # These aliases map romanized Sinhala names and common misspellings to the
            # English canonical form that appears in the Ayurvedic book text.
            _HERB_ALIASES = {
                'tumeric':   'Turmeric',
                'kurudu':    'Cinnamon',
                'inguru':    'Ginger',
                'kaha':      'Turmeric',
                'kohomba':   'Neem',
                'welpenela': 'Brahmi',
                'gotukola':  'Brahmi',
                'nelli':     'Amla',
            }
            _HERB_PATTERN = (
                r'\b(turmeric|tumeric|cinnamon|ginger|neem|ashwagandha|triphala|tulsi|aloe vera|'
                r'brahmi|shatavari|cardamom|cumin|fenugreek|amla|nelli|kohomba|kurudu|'
                r'inguru|kaha|welpenela|gotukola|licorice|pepper|clove|nutmeg|garlic|'
                r'curry leaf|moringa|sesame|coconut|ghee)\b'
            )
            # Try translated question first; fall back to original (romanized) question
            _herb_match = _re_ctx.search(_HERB_PATTERN, question_for_processing_en.lower())
            if not _herb_match:
                _herb_match = _re_ctx.search(_HERB_PATTERN, original_question.lower())
            if _herb_match:
                _raw_herb = _herb_match.group(0)
                _herb_in_q = _HERB_ALIASES.get(_raw_herb.lower(), _raw_herb.title())
            else:
                _herb_in_q = ''

            # 2. Build scoring helpers
            _stopwords = {
                'what', 'that', 'this', 'with', 'from', 'have', 'been', 'will', 'which',
                'when', 'they', 'their', 'should', 'does', 'good', 'more', 'some', 'into',
                'than', 'about', 'help', 'cause', 'ayur', 'ayurvedic', 'herb', 'body',
                'health', 'used', 'also', 'both', 'well', 'very', 'such', 'each', 'than',
                'regarding', 'tell', 'please', 'need', 'want', 'know', 'question',
            }
            # Benefit-action verbs — sentences with these are almost always informative
            _BENEFIT_VERBS = _re_ctx.compile(
                r'\b(helps?|reduces?|supports?|improves?|treats?|benefits?|boosts?|aids?|'
                r'relieves?|contains?|promotes?|prevents?|cures?|heals?|stimulates?|'
                r'strengthens?|purifies?|balances?|calms?|increases?|decreases?|'
                r'useful|effective|beneficial|healing|medicinal|therapeutic|'
                r'anti.inflam|antiseptic|antioxidant|digestive|tonic|expectorant)\b',
                _re_ctx.IGNORECASE
            )
            _herb_lower = _herb_in_q.lower() if _herb_in_q else ''
            _query_terms = set(_re_ctx.findall(r'\b[a-z]{4,}\b', combined_query_for_intent.lower())) - _stopwords
            _intent_terms = {
                'side_effects': {'side', 'effect', 'risk', 'harm', 'safe', 'warning', 'caution'},
                'dosage': {'dose', 'dosage', 'amount', 'frequency', 'daily', 'take', 'timing'},
                'combination': {'combine', 'combination', 'with', 'interaction', 'mix', 'pepper'},
                'treatment': {'treat', 'treatment', 'symptom', 'manage', 'relief', 'fever', 'cough', 'cold'},
                'definition': {'what', 'definition', 'meaning', 'explain'},
                'benefits': {'benefit', 'use', 'helps', 'supports', 'improves'},
                'general': {'ayurveda', 'health'},
            }
            _intent_focus = _intent_terms.get(question_intent, set())

            # 3. Strip metadata header lines from context so we only score real text
            _clean_ctx = _re_ctx.sub(r'\[Source \d+\]', '', context_text)
            _clean_ctx = _re_ctx.sub(
                r'(?m)^(?:Book|Chapter|Verse|Verse/Paragraph|Type|Related Question):.*\n?', '', _clean_ctx
            )

            # 4. Score every sentence:
            #    +3  herb name appears in sentence  (handles "Turmeric reduces...")
            #    +2  benefit verb present           (handles "It reduces inflammation")
            #    +1  per extra question-keyword hit (handles general overlap)
            _ctx_sents = _re_ctx.split(r'(?<=[.!?])\s+', _clean_ctx.replace('\n', ' '))
            _scored_sents = []
            for _s in _ctx_sents:
                _s = _re_ctx.sub(r'\s+', ' ', _s).strip()
                if not (30 <= len(_s) <= 350):
                    continue
                _s_lower = _s.lower()
                _score = 0
                if _herb_lower and _herb_lower in _s_lower:
                    _score += 3
                if _BENEFIT_VERBS.search(_s):
                    _score += 2
                _s_words = set(_re_ctx.findall(r'\b[a-z]{4,}\b', _s_lower)) - _stopwords
                _score += len(_s_words & _query_terms) * 2
                _score += len(_s_words & _intent_focus)
                _score += len(_s_words & ({_herb_lower} if _herb_lower else set()))
                if _score > 0:
                    _scored_sents.append((_score, _s))

            _scored_sents.sort(key=lambda x: -x[0])

            # 5. Deduplicate: skip sentences sharing >55% keywords with already-chosen ones
            _deduped: list = []
            _seen_kws: set = set()
            for _sc, _s in _scored_sents:
                _kws = set(_re_ctx.findall(r'\b[a-z]{5,}\b', _s.lower())) - _stopwords
                if _kws:
                    if len(_kws & _seen_kws) / len(_kws) >= 0.55:
                        continue
                    _seen_kws.update(_kws)
                _deduped.append(_s)
                if len(_deduped) >= 3:
                    break

            # 6. Build bullet points
            if _deduped:
                _polished = []
                for _s in _deduped:
                    _s = _s[0].upper() + _s[1:]
                    if _s[-1] not in '.!?':
                        _s += '.'
                    _polished.append(_s)
                base_answer = '\n'.join(f'\u2022 {s}' for s in _polished)
                print(f"✅ Context extraction answer: {base_answer[:120]}...")
            else:
                base_answer = (
                    "• I could not find a strong direct match in the indexed Ayurvedic sources for this exact wording.\n"
                    "• Please rephrase with the main symptom, herb, or condition (for example: side effects, dosage, or treatment goal).\n"
                    "• If symptoms are severe or persistent, seek care from a qualified medical professional."
                )
                print("⚠️  No relevant sentences found in context.")
        
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
                question=question_for_processing_en
            )
            is_personalized = True

        # Strip "**Personalized for X Constitution:**" suffix that the personalizer LLM appends
        # — this is already shown separately via template tips; we don't want it in the answer bubble
        import re as _re_pers
        final_answer = _re_pers.sub(
            r'\n*\*{0,2}Personalized\s+for[^\n]*:?\*{0,2}[\s\S]*$',
            '', final_answer, flags=_re_pers.IGNORECASE
        ).strip()
        # Dosha priority: 1) passed in directly from frontend, 2) user profile, 3) detect from text
        if dominant_dosha and dominant_dosha.strip() not in ('', 'General', 'N/A', 'none'):
            detected_dosha = dominant_dosha.strip().capitalize()
            print(f"🧬 Using passed dominant_dosha: {detected_dosha}")
        elif user_profile and user_profile.get('dominant_dosha') and user_profile['dominant_dosha'] not in ('N/A', '', None):
            detected_dosha = user_profile['dominant_dosha'].capitalize()
            print(f"🧬 Using profile dosha: {detected_dosha}")
        else:
            detected_dosha = self._detect_dosha_from_question(question_for_processing_en, base_answer)
            print(f"🧬 Detected dosha from question: {detected_dosha}")
        personalized_tips = self._generate_personalized_tips(
            question_for_processing_en,
            base_answer,
            detected_dosha,
            original_question=original_question
        )
        
        # Format response with similarity percentages — include web source URLs
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

            if doc_type == "book":
                citation["chapter"] = meta.get("chapter", "N/A")
                citation["paragraph"] = meta.get("paragraph", "N/A")
            elif doc_type == "web":
                # Web source — expose URL and authority for frontend display
                citation["url"] = doc.get("url", meta.get("url", ""))
                citation["authority"] = doc.get("authority", meta.get("authority", 0.80))
                citation["formatted"] = (
                    f"{doc.get('source', 'Web Source')} "
                    f"({doc.get('similarity_percentage', 0.0)}% match)"
                )
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
            else:
                # cleaned_translation is empty — either cleanup over-filtered or ASCII guard fired.
                # Do NOT fall back to raw_translation (may still contain garbled English).
                print("⚠️  Translation filtered/empty, using English fallback")
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
            "translated_question": question_for_processing_en if detected_language == 'si' else None,
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

        # Always surface which dosha was used (so frontend can display it)
        response["user_info"] = response.get("user_info") or {}
        response["user_info"]["dominant_dosha"] = detected_dosha

        # Store in cache so the same question always returns the same answer
        self._answer_cache[_cache_key] = response

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
