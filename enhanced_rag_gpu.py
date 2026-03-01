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
            prompt = (
                f"You are an Ayurvedic health advisor. "
                f"A user asked: \"{question}\"\n"
                f"The answer was about: {answer[:300]}\n\n"
                f"The question relates to the {dosha} dosha.\n"
                f"Give exactly 2-3 short, practical personalized tips for someone with {dosha} "
                f"constitution that are RELATED to this topic but NOT already mentioned in the answer. "
                f"Format as a numbered list. Be concise (1-2 sentences each)."
            )
            messages = [
                {"role": "system", "content": "You are an Ayurvedic expert providing brief personalized health tips."},
                {"role": "user", "content": prompt}
            ]
            tips = self.llm.generate_from_messages(messages, max_new_tokens=180)
            tips = tips.strip()
            print(f"✓ Tips generated: {tips[:100]}...")
            return tips
        except Exception as e:
            print(f"⚠️  Tip generation failed: {e}")
            return ""

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
            
            if detected_language == 'si':
                # Check if romanized (no Sinhala Unicode chars) or Sinhala script
                is_romanized = not self.translator._has_sinhala_chars(question)
                
                if is_romanized:
                    print(f"📝 Romanized Singlish detected (e.g., 'kurudu wala guna')")
                
                # Translate Sinhala question to English for processing
                # If romanized, will be transliterated first inside translate_si_to_en
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
        
        # Token budget — 200 gives room for 3-4 clean bullet points
        dynamic_tokens = 200
            
        # Build context
        context_text = self._build_context_with_citations(top_context_docs)
        
        print(f"🔍 Context preview (first 300 chars): {context_text[:300]}...")
        
        # Build messages — use generate_from_messages() so special tokens are
        # encoded directly and never corrupted by a string round-trip.
        context_summary = context_text[:800]  # ~200 tokens of context
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an Ayurvedic health assistant. "
                    "Your job: read the sources and list specific health benefits, uses, or facts. "
                    "Format: exactly 3 bullet points using the • symbol. "
                    "Each bullet = one clear, specific benefit or use in one sentence. "
                    "IMPORTANT: Ignore any book titles, chapter names, verse numbers, "
                    "or table-of-contents text you see in the sources — only extract health facts. "
                    "Do NOT mention book names or source references in your answer. "
                    "Do NOT start with 'Ayurveda provides', 'Based on', or 'According to'. "
                    "Do NOT end with 'Follow these guidelines' or similar closings."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Sources:\n{context_summary}\n\n"
                    f"Question: {question}\n\n"
                    f"List 3 specific health benefits or uses that directly answer this question."
                )
            }
        ]
        print(f"📏 Generating with max_new_tokens={dynamic_tokens}")

        # Generate answer
        print("💭 Generating answer...")
        raw_answer = self.llm.generate_from_messages(messages, max_new_tokens=dynamic_tokens)

        # Use raw answer directly — _format_answer was converting paragraphs to broken bullets
        base_answer = raw_answer.strip()

        # Strip LLM-generated preamble and trailing boilerplate
        import re as _re
        _preamble_phrases = [
            'ayurveda provides', 'based on the sources', 'based on the context',
            'according to the sources', 'according to ayurveda', 'here are',
            'here is', 'health insights', 'following health', 'for your question',
            'follow these ayurvedic guidelines', 'these ayurvedic guidelines'
        ]
        _trailing_phrases = ['follow these', 'guidelines consistently', 'safe and effective', 'always use']

        lines = base_answer.split('\n')

        # Remove / trim leading preamble lines
        while lines:
            first = lines[0].lower().strip()
            if any(phrase in first for phrase in _preamble_phrases) or first.endswith(':'):
                # If there is real content after a colon on the same line, keep that part
                colon_idx = lines[0].find(':')
                after_colon = lines[0][colon_idx + 1:].strip() if colon_idx != -1 else ''
                # Strip leading bullet from rescued content
                after_colon = _re.sub(r'^[\u2022\-\*]\s*', '', after_colon).strip()
                lines.pop(0)
                if after_colon and len(after_colon) > 15:
                    lines.insert(0, after_colon)  # put rescued content back at front
                    break  # content found, stop stripping
            else:
                break

        # Remove trailing boilerplate lines
        while lines:
            last = lines[-1].lower().strip()
            if any(phrase in last for phrase in _trailing_phrases):
                lines.pop()
            else:
                break

        base_answer = '\n'.join(lines).strip()
        # Strip any stray leading bullet left over
        base_answer = _re.sub(r'^[\u2022\-\*]\s*', '', base_answer).strip()

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
            
            # Translate directly — no pre/post processing (cleanup was destroying valid Sinhala)
            raw_translation = self.translator.translate_en_to_si(final_answer)
            
            if raw_translation and raw_translation.strip():
                display_answer = raw_translation.strip()
                print(f"✓ Translation complete: {display_answer[:100]}...")
            else:
                print("⚠️  Translation returned empty, using English")
                display_answer = final_answer

        # Translate personalized tips to Sinhala if user asked in Singlish/Sinhala
        if self.enable_translation and self.translator and detected_language == 'si' and personalized_tips:
            print("🔄 Translating personalized tips to Sinhala...")
            raw_tips_si = self.translator.translate_en_to_si(personalized_tips)
            if raw_tips_si and raw_tips_si.strip():
                personalized_tips = raw_tips_si.strip()
            else:
                print("⚠️  Tips translation returned empty, keeping English")

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
