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
                # 350 chars gives LLM enough content to synthesise a real answer (was 200)
                doc_text = doc.get("text", "")[:350]
                block = f"""[Source {i}]
Book: {book}
Chapter: {chapter}
Verse/Paragraph: {paragraph}

{doc_text}"""
            else:
                # QA dataset entry
                qa_question = meta.get("question", "N/A")
                # Use 350 chars per source (was 200)
                doc_text = doc.get("text", "")[:350]
                block = f"""[Source {i}]
Type: Ayurvedic Q&A Reference
Related Question: {qa_question}

{doc_text}"""
            
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

    def _generate_personalized_tips(self, question: str, answer: str, dosha: str) -> str:
        """
        Generate 2-3 short personalized tips relevant to the question/answer context,
        tailored to the detected dosha. Tips must be different from the answer.
        """
        print(f"💡 Generating personalized tips for {dosha} dosha...")
        try:
            import re as _re
            topic_line = question[:60].strip()
            prompt = (
                f"Patient question: \"{question}\"\nDosha: {dosha}\n\n"
                f"Give 3 Ayurvedic lifestyle tips for a {dosha} person about: {topic_line}\n"
                f"STRICT RULES:\n"
                f"• Each tip: 1 sentence, MAXIMUM 15 WORDS, end with a period.\n"
                f"• Start every tip with • symbol on its own line.\n"
                f"• Tips must be DIFFERENT from: {answer[:80]}\n"
                f"• NO introductions, NO paragraphs.\n"
                f"Example:\n"
                f"• Drink warm water with ginger each morning to aid digestion.\n"
                f"• Avoid spicy foods that aggravate {dosha} dosha imbalances.\n"
                f"• Practice daily oil massage with warm sesame oil for grounding.\n\n"
                f"Now give 3 tips for {dosha} about: {topic_line}\n\n"
                f"•"
            )
            messages = [
                {"role": "user", "content": prompt}
            ]
            raw_tips = self.llm.generate_from_messages(messages, max_new_tokens=160)

            # Strip any leading bullet/dash the model added (prompt ends with '•')
            # Prevents double-prefix like '• • text' or '• - text'
            raw_tips = _re.sub(r'^[\s•\-\*]+', '', raw_tips).strip()
            raw_tips = ("• " + raw_tips).strip()

            def _clip_bullet(b: str) -> str:
                b = b.strip()
                b = _re.sub(r'^(•\s*)[\-\*•]+\s*', r'\1', b)
                m = _re.search(r'(?<=[.!?])(?:\s|$)', b[30:])
                if m:
                    b = b[:30 + m.start() + 1].strip()
                if len(b) > 130:
                    b = b[:130].rsplit(' ', 1)[0].rstrip(',:;') + '.'
                return b

            if '•' in raw_tips:
                tips_raw = raw_tips
            else:
                converted = _re.sub(r'^\d+\.\s+', '• ', raw_tips, flags=_re.MULTILINE)
                tips_raw = converted if '•' in converted else '\n'.join(
                    f'• {s.strip()}' for s in _re.split(r'(?<=[.!?])\s+', raw_tips) if s.strip()
                )

            tip_lines = [l for l in tips_raw.splitlines() if l.strip().startswith('•')]
            tips = '\n'.join(_clip_bullet(t) for t in tip_lines[:3])

            print(f"✓ Tips generated: {tips[:100]}...")
            return tips
        except Exception as e:
            print(f"⚠️  Tip generation failed: {e}")
            return ""

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
        
        # Expand query with Ayurvedic synonyms before searching so FAISS finds
        # the right passages even when user phrasing differs from book vocabulary
        search_query = self._expand_query_with_ayurvedic_terms(question)

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
        # Sort ALL retrieved docs by similarity so the most relevant content wins
        # regardless of document type, while still keeping books at the front.
        all_docs_sorted = sorted(retrieved_docs, key=lambda d: d.get('similarity', 0), reverse=True)
        # Guarantee at least 2 book docs in context if available (for citation quality)
        top_book_docs = [d for d in all_docs_sorted if d.get('type') == 'book'][:2]
        other_top_docs = [d for d in all_docs_sorted if d not in top_book_docs][:max(0, top_k - len(top_book_docs))]
        top_context_docs = (top_book_docs + other_top_docs)[:top_k]
        print(f"📚 Context: {len([d for d in top_context_docs if d.get('type')=='book'])} book + "
              f"{len([d for d in top_context_docs if d.get('type')!='book'])} QA docs")
        
        # 220 tokens = comfortably fits 3 bullets of up to 20 words each
        dynamic_tokens = 220

        # Build context
        context_text = self._build_context_with_citations(top_context_docs)

        print(f"🔍 Context preview (first 300 chars): {context_text[:300]}...")

        # Strict short-bullet prompt. Prompt ends with '•' so model continues the list.
        context_summary = context_text[:1800]
        topic_hint = question[:70].strip()
        messages = [
            {
                "role": "user",
                "content": (
                    f"You are an Ayurvedic doctor. Patient question: \"{question}\"\n\n"
                    f"Knowledge:\n{context_summary}\n\n"
                    f"Write EXACTLY 3 bullet points answering: {topic_hint}\n"
                    f"STRICT RULES:\n"
                    f"• Each bullet: 1 sentence, MAXIMUM 15 WORDS, end with a period.\n"
                    f"• Start every bullet with • symbol on its own line.\n"
                    f"• Name a specific herb, oil or treatment.\n"
                    f"• NO paragraphs. NO introductions. NO extra text.\n"
                    f"Example format:\n"
                    f"• Turmeric reduces inflammation and purifies the blood.\n"
                    f"• Ginger tea aids digestion and relieves nausea.\n"
                    f"• Ashwagandha strengthens immunity and reduces stress.\n\n"
                    f"Now answer about: {topic_hint}\n\n"
                    f"•"
                )
            }
        ]
        print(f"📏 Generating with max_new_tokens={dynamic_tokens}")

        # Generate answer — min_new_tokens forces model to produce at least 3 bullets
        print("💭 Generating answer...")
        raw_answer = self.llm.generate_from_messages(messages, max_new_tokens=dynamic_tokens, min_new_tokens=80)

        # Strip any leading bullet/dash the model added (prompt already ends with '•')
        # This prevents double-prefix like '• - text' or '• • text'
        import re as _re
        raw_answer_clean = _re.sub(r'^[\s•\-\*]+', '', raw_answer).strip()
        raw_combined = ("• " + raw_answer_clean).strip()

        # Helper: hard-clip a bullet to its first sentence, min 30 chars before clipping
        import re as _re
        def _clip_bullet(b: str) -> str:
            b = b.strip()
            # Strip accidental double prefix like '• -' or '• •'
            b = _re.sub(r'^(•\s*)[\-\*•]+\s*', r'\1', b)
            # Find first sentence end after at least 30 chars
            m = _re.search(r'(?<=[.!?])(?:\s|$)', b[30:])
            if m:
                b = b[:30 + m.start() + 1].strip()
            # Absolute hard cap at 130 chars
            if len(b) > 130:
                b = b[:130].rsplit(' ', 1)[0].rstrip(',:;') + '.'
            return b

        # Guarantee • bullet format
        if '•' in raw_combined:
            base_answer = raw_combined
        else:
            converted = _re.sub(r'^\d+\.\s+', '• ', raw_combined, flags=_re.MULTILINE)
            if '•' in converted:
                base_answer = converted
            else:
                sents = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', raw_combined) if s.strip()]
                base_answer = '\n'.join(f'• {s}' for s in sents[:3])

        # Extract bullets, hard-truncate each one, keep max 3
        bullets = [line for line in base_answer.splitlines() if line.strip().startswith('•')]
        bullets = [_clip_bullet(b) for b in bullets[:3]]
        if bullets:
            base_answer = '\n'.join(bullets)
        
        print(f"✅ Generation complete!")
        print(f"   Answer length: {len(base_answer)} chars")
        print(f"   Answer preview: {base_answer[:200] if base_answer else '[EMPTY]'}...")
        
        if not base_answer or len(base_answer.strip()) < 10:
            print("⚠️  WARNING: Generated answer is empty or too short!")
            print(f"   Raw answer: {raw_answer[:300] if raw_answer else '[NONE]'}...")
            print(f"   Prompt used: {prompt[:500]}...")
        
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
            
            # Step 1: Simplify English for better translation
            simplified_english = self._simplify_for_translation(final_answer)
            print(f"📝 Simplified English: {simplified_english[:100]}...")
            
            # Step 2: Translate to Sinhala
            raw_translation = self.translator.translate_en_to_si(simplified_english)
            
            # Step 3: Clean up translated text (remove gibberish, incomplete sentences)
            display_answer = self._cleanup_translated_answer(raw_translation)
            print(f"✓ Translation complete: {display_answer[:100]}...")
            
            # Failsafe: If cleanup removed everything, use original English
            if not display_answer or len(display_answer.strip()) < 20:
                print("⚠️  Translation cleanup removed too much, using English")
                display_answer = final_answer

        # Translate personalized tips to Sinhala if user asked in Singlish/Sinhala
        if self.enable_translation and self.translator and detected_language == 'si' and personalized_tips:
            print("🔄 Translating personalized tips to Sinhala...")
            simplified_tips = self._simplify_for_translation(personalized_tips)
            raw_tips_si = self.translator.translate_en_to_si(simplified_tips)
            tips_si = self._cleanup_translated_answer(raw_tips_si)
            if tips_si and len(tips_si.strip()) >= 20:
                personalized_tips = tips_si
            else:
                print("⚠️  Tips translation cleanup removed too much, keeping English")

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
