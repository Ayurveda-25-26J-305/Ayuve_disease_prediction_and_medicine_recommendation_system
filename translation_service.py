"""
Translation Service for Bilingual Ayurvedic Q&A System
Supports English ↔ Sinhala translation with automatic language detection

Uses Helsinki-NLP models from Hugging Face (FREE, runs on GPU):
- opus-mt-si-en: Sinhala → English
- opus-mt-en-si: English → Sinhala

Features:
- Automatic language detection (langdetect library)
- Romanized Singlish detection (e.g., "kurudu wala guna monawada?")
- Transliteration (romanized → Sinhala script)
- Bidirectional translation
- GPU acceleration
- Fallback to English if translation fails
"""

import logging
from typing import Dict, Tuple, Optional
import re

logger = logging.getLogger(__name__)


# Common Sinhala words in romanized form for Singlish detection
SINHALA_ROMANIZED_KEYWORDS = {
    # Question words
    'mokadda', 'monawada', 'monawa', 'mokada', 'kohomada', 'kohoma',
    'kiyada', 'keyada', 'kawuda', 'kauda', 'kewda', 'kaudda',
    'ehenam', 'mata', 'api', 'oya', 'oyala', 'meka', 'eka', 'me',
    'monada', 'mokakda', 'kawda',  # common shorthand variants

    # Possessive / pronouns
    'mage', 'ape', 'eya', 'eyage', 'mama',

    # Pain / symptom verbs (very common in health questions)
    'ridenawa', 'ridena', 'rida', 'arenawa', 'wedanawa',
    'thiyenawa', 'thiyenwa', 'thiyenne', 'thiynawa',
    'hithenawa', 'hithenne',

    # Common Ayurvedic/Health terms
    'leda', 'roga', 'behet', 'beheth', 'osuda', 'osuwa', 'wedakama',
    'aushadha', 'guna', 'wala', 'karanna', 'denne',
    'aragena', 'bonna', 'bonawa', 'gaththa', 'ganna', 'gannawa',

    # Body parts
    'oluwa', 'oluwata', 'oluwala',       # head
    'urahisa', 'urahis', 'urahisat',    # shoulder
    'pita', 'pitata',                   # back
    'hela', 'helata',                   # neck
    'kalawa', 'kalawata',               # stomach
    'wata', 'watata',                   # back/around
    'kapala', 'dugga', 'duggi',         # skull / forehead
    'kakula', 'kakule',                 # leg
    'atha', 'athe',                     # hand/arm
    'nethra', 'nethrata',               # eye
    'kana', 'kanata',                   # ear

    # Ability / modal verbs
    'puluwan', 'puluwanda', 'puluwanwada', 'bari',

    # Common herbs/foods
    'kurudu', 'kaha', 'inguru', 'suduru', 'karapincha', 'goraka',
    'thippili', 'welpenela', 'komarika', 'raththran', 'venivel',

    # Body/Health descriptors
    'riha', 'sathura', 'thalapola', 'gawwa', 'linda',
    'duka', 'wedana', 'hadada', 'hawa', 'una', 'seetha',

    # Common verbs/adjectives
    'honda', 'naraka', 'loku', 'podi', 'wadi', 'adu',
    'denne', 'bonawa', 'kannawa', 'yanawa',

    # Particles
    'da', 'ta', 'ekka', 'nisa', 'hinda', 'walata', 'wala', 'gen',
    'ekada', 'nemei', 'nehe', 'athi', 'nathi'
}

# Romanized to Sinhala character mapping (simplified)
ROMANIZED_TO_SINHALA = {
    'a': 'අ', 'aa': 'ආ', 'ae': 'ඇ', 'aae': 'ඈ',
    'i': 'ඉ', 'ii': 'ඊ', 'u': 'උ', 'uu': 'ඌ',
    'e': 'එ', 'ee': 'ඒ', 'ai': 'ඓ', 'o': 'ඔ', 'oo': 'ඕ', 'au': 'ඖ',
    
    'ka': 'ක', 'kha': 'ඛ', 'ga': 'ග', 'gha': 'ඝ', 'nga': 'ඞ',
    'cha': 'ච', 'chha': 'ඡ', 'ja': 'ජ', 'jha': 'ඣ', 'nya': 'ඤ',
    'ta': 'ට', 'tta': 'ට', 'tha': 'ථ', 'da': 'ද', 'dha': 'ධ', 'na': 'න',
    'pa': 'ප', 'pha': 'ෆ', 'ba': 'බ', 'bha': 'භ', 'ma': 'ම',
    'ya': 'ය', 'ra': 'ර', 'la': 'ල', 'wa': 'ව', 'va': 'ව',
    'sa': 'ස', 'sha': 'ශ', 'ha': 'හ', 'lla': 'ළ', 'fa': 'ෆ',
    
    # Common combinations
    'kurudu': 'කුරුඳු', 'kaha': 'කහ', 'inguru': 'ඉඟුරු',
    'behet': 'බෙහෙත්', 'osuda': 'ඖෂධ', 'guna': 'ගුණ',
    'wala': 'වල', 'monawada': 'මොනවද', 'mokadda': 'මොකද්ද',
    'kohomada': 'කොහොමද', 'mata': 'මට', 'me': 'මේ',
}

# Romanized Sinhala to English direct translation dictionary
SINHALA_TO_ENGLISH_DICT = {
    # Question words
    'monawada': 'what', 'mokadda': 'what', 'mokada': 'what',
    'monawa': 'what', 'kohomada': 'how', 'kohoma': 'how',
    'kiyada': 'how much', 'keyada': 'how much',
    'kawuda': 'who', 'kauda': 'who', 'kewda': 'who',
    'ehenam': 'if so', 'ehema': 'like that',
    
    # Pronouns
    'mata': 'me', 'mama': 'I', 'api': 'we',
    'oya': 'you', 'oyala': 'you all',
    'meka': 'this', 'eka': 'that', 'me': 'this',
    'ara': 'that', 'owa': 'that',
    
    # Health/Disease terms
    'leda': 'disease', 'roga': 'disease', 'behet': 'medicine', 'beheth': 'medicine',
    'osuda': 'medicine', 'osuwa': 'medicine', 'aushadha': 'medicine',
    'wedakama': 'treatment', 'thalapola': 'head',
    'riha': 'lungs', 'sathura': 'joints', 'gawwa': 'body',
    'hada': 'heart', 'linda': 'body',
    'duka': 'pain', 'wedana': 'pain',
    'ridenawa': 'is hurting', 'ridena': 'hurting', 'rida': 'pain',
    'arenawa': 'hurts', 'wedanawa': 'is painful',
    'una': 'fever', 'seetha': 'cold', 'hawa': 'cough',
    # Body parts
    'oluwa': 'head', 'oluwata': 'in the head',
    'urahisa': 'shoulder', 'urahis': 'shoulder',
    'hela': 'neck', 'kalawa': 'stomach',
    'kakula': 'leg', 'atha': 'arm', 'pita': 'back',
    'nethra': 'eye', 'kana': 'ear',
    # Pronouns / possession
    'mage': 'my', 'mama': 'I', 'ape': 'our',
    'monada': 'what', 'puluwan': 'can', 'bari': 'cannot',
    'thiyenawa': 'have', 'thiyenwa': 'have', 'thiyenne': 'has',
    
    # Common verbs
    'karanna': 'to do', 'karannada': 'to do',
    'ganna': 'to take', 'gannawa': 'take',
    'bonawa': 'to drink', 'bonna': 'to drink',
    'kannawa': 'to eat', 'kanna': 'to eat',
    'yanawa': 'to go', 'yanna': 'to go',
    'denna': 'to give', 'denne': 'give',
    'aragena': 'taking', 'gaththa': 'took',
    'thiyenawa': 'to have', 'thiyenne': 'have',
    
    # Properties/Adjectives
    'guna': 'benefits', 'gunas': 'properties',
    'honda': 'good', 'naraka': 'bad',
    'loku': 'big', 'podi': 'small',
    'wadi': 'more', 'adu': 'less',
    'sudu': 'white', 'kalu': 'black',
    'ratu': 'red', 'nil': 'blue',
    
    # Herbs and Foods
    'kurudu': 'cinnamon', 'kaha': 'turmeric',
    'inguru': 'ginger', 'suduru': 'cumin',
    'karapincha': 'curry leaves', 'goraka': 'garcinia',
    'thippili': 'long pepper', 'gammiris': 'black pepper',
    'welpenela': 'aloe vera', 'komarika': 'neem',
    'raththran': 'sandalwood', 'venivel': 'coscinium',
    'kottamalli': 'coriander', 'kothamalli': 'coriander',
    'nelli': 'gooseberry', 'nellie': 'gooseberry',
    'kohomba': 'neem', 'kohumba': 'neem',
    'karavila': 'bitter gourd', 'karawila': 'bitter gourd',
    'karela': 'bitter gourd',
    'nilkatarodumal': 'bitter gourd', 'nilkatarodumala': 'bitter gourd',
    'nil': 'blue',
    'mukunuwenna': 'sessile joyweed',
    'ranawara': 'senna',
    'beli': 'bael fruit',
    'thibbatu': 'turkey berry',
    'iramusu': 'false sarsaparilla',
    'polpala': 'aerva lanata',
    'hathawariya': 'asparagus',
    'thebu': 'costus',
    
    # Prepositions/Particles
    'wala': 'of', 'walata': 'for',
    'ta': 'to', 'gen': 'from',
    'ekka': 'with', 'nisa': 'because',
    'hinda': 'because of',
    'da': '?', 'nemei': 'is not', 'nehe': 'no',
    'athi': 'have', 'nathi': 'do not have',
    'one': 'need', 'ekada': 'is it',
    'ekata': 'for that',
}



class TranslationService:
    """
    Bilingual translation service for Ayurvedic Q&A
    
    Workflow:
    1. Detect input language (English vs Sinhala)
    2. Translate question to English if needed (for RAG processing)
    3. Process answer in English
    4. Translate answer back to original language
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize translation service
        
        Args:
            device: Device to run models on ("cuda" or "cpu")
        """
        logger.info("Initializing TranslationService...")
        self.device = device
        self.si_to_en_model = None
        self.en_to_si_model = None
        self.si_to_en_tokenizer = None
        self.en_to_si_tokenizer = None
        self.language_detector = None
        
        try:
            self._load_models()
            logger.info("✓ TranslationService initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️  Translation models not loaded: {e}")
            logger.warning("Translation will be disabled. Install: pip install transformers sentencepiece langdetect")
    
    def _load_models(self):
        """Load translation backend - uses Google Translate via deep_translator (no model download)"""
        try:
            from deep_translator import GoogleTranslator
            import langdetect

            # Verify connectivity with a quick test
            _ = GoogleTranslator(source='en', target='si').translate('test')

            self._google_translator = GoogleTranslator
            self.language_detector = langdetect

            # Set dummy model flags so is_available() returns True
            self.si_to_en_model = True
            self.en_to_si_model = True
            self.si_to_en_tokenizer = True
            self.en_to_si_tokenizer = True

            logger.info("✓ Translation ready via Google Translate (deep_translator)")

        except ImportError:
            logger.error("Missing deep_translator. Install: pip install deep-translator langdetect")
            raise
        except Exception as e:
            logger.error(f"Translation init failed: {e}")
            raise
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of input text (including romanized Singlish)
        
        Args:
            text: Input text
            
        Returns:
            Language code: 'si' (Sinhala/Singlish), 'en' (English), or 'unknown'
        """
        if not self.language_detector:
            return 'unknown'
        
        try:
            # Clean text for detection
            clean_text = re.sub(r'[^\w\s]', '', text).strip()
            
            if not clean_text:
                return 'unknown'
            
            # FIRST: Check for Sinhala Unicode characters (highest priority)
            if self._has_sinhala_chars(text):
                return 'si'
            
            # SECOND: Check for romanized Singlish (before langdetect)
            if self._is_romanized_singlish(text):
                logger.info("Detected romanized Singlish (e.g., 'kurudu wala guna monawada?')")
                return 'si'
            
            # THIRD: Use langdetect for standard detection
            detected = self.language_detector.detect(clean_text)
            
            # Map to supported languages
            if detected in ['si']:  # Sinhala
                return 'si'
            elif detected in ['ta']:  # Tamil
                return 'ta'
            elif detected == 'en':
                return 'en'
            else:
                return 'en'  # Default to English
                
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            # Fallback: check for Sinhala characters
            if self._has_sinhala_chars(text):
                return 'si'
            return 'en'
    
    def _has_sinhala_chars(self, text: str) -> bool:
        """
        Check if text contains Sinhala Unicode characters
        
        Args:
            text: Input text
            
        Returns:
            True if Sinhala characters detected
        """
        # Sinhala Unicode range: U+0D80 to U+0DFF
        sinhala_pattern = re.compile(r'[\u0D80-\u0DFF]')
        return bool(sinhala_pattern.search(text))
    
    def _is_romanized_singlish(self, text: str) -> bool:
        """
        Detect if text is romanized Singlish (Sinhala words written in English letters)
        
        Examples:
            "kurudu wala guna monawada?" → True (Singlish)
            "what are the benefits?" → False (English)
        
        Args:
            text: Input text
            
        Returns:
            True if romanized Singlish detected
        """
        # Convert to lowercase and split into words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        if not words:
            return False
        
        # Count how many words match Sinhala keywords
        sinhala_word_count = sum(1 for word in words if word in SINHALA_ROMANIZED_KEYWORDS)
        
        # If 30% or more words are Sinhala keywords, consider it Singlish
        # OR if text has 2+ Sinhala keywords (for short questions)
        threshold_ratio = 0.30
        threshold_count = 2
        
        is_singlish = (
            (sinhala_word_count / len(words) >= threshold_ratio) or
            (sinhala_word_count >= threshold_count)
        )
        
        if is_singlish:
            logger.info(f"Romanized Singlish detected: {sinhala_word_count}/{len(words)} Sinhala words")
        
        return is_singlish
    
    def _translate_romanized_keywords(self, text: str) -> str:
        """
        Translate romanized Singlish to English using word-by-word dictionary mapping
        
        This avoids the garbled output from trying to transliterate to Sinhala script.
        Uses direct keyword translation for better accuracy.
        
        Args:
            text: Romanized Singlish text (e.g., "kurudu wala guna monawada?")
            
        Returns:
            English translation (e.g., "what are the benefits of cinnamon?")
        """
        # Clean and split into words
        words = text.lower().split()
        translated_words = []
        
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Translate using dictionary, keep original if not found
            if clean_word in SINHALA_TO_ENGLISH_DICT:
                translated = SINHALA_TO_ENGLISH_DICT[clean_word]
                translated_words.append(translated)
            else:
                # Keep as-is (might be English or proper noun)
                translated_words.append(word)
        
        # Join and do basic grammar improvements
        result = ' '.join(translated_words)
        
        # Clean up common patterns
        result = result.replace('? ?', '?')  # Remove extra ?
        result = result.replace('  ', ' ')    # Remove double spaces
        result = result.strip()
        
        logger.info(f"Keyword-based translation: '{text}' → '{result}'")
        return result

    def _score_english_query_candidate(self, candidate: str) -> int:
        """
        Heuristic scoring for translated query quality.
        Higher score means more likely useful for retrieval and intent detection.
        """
        if not candidate:
            return -10

        c = candidate.strip()
        c_low = c.lower()
        score = 0

        if len(c) >= 8:
            score += 1
        if re.search(r'[a-z]{3,}', c_low):
            score += 1
        if not re.search(r'[\u0D80-\u0DFF]', c):
            score += 1

        intent_terms = [
            'what', 'how', 'benefit', 'use', 'property', 'treatment', 'treat',
            'diet', 'food', 'medicine', 'recommend', 'pain', 'fever', 'cough',
            'cold', 'dosage', 'side effect', 'symptom'
        ]
        if any(t in c_low for t in intent_terms):
            score += 2

        herb_terms = [
            'bitter gourd', 'turmeric', 'cinnamon', 'ginger', 'neem', 'ashwagandha',
            'triphala', 'tulsi', 'amla', 'brahmi', 'garlic'
        ]
        if any(h in c_low for h in herb_terms):
            score += 2

        if len(c) > 220:
            score -= 1

        return score
    
    def _transliterate_to_sinhala(self, text: str) -> str:
        """
        Transliterate romanized Singlish to Sinhala script
        
        Uses word-level mapping for common terms, falls back to character mapping.
        
        Args:
            text: Romanized text (e.g., "kurudu wala guna monawada?")
            
        Returns:
            Sinhala script text (e.g., "කුරුඳු වල ගුණ මොනවද?")
        """
        # Split text into words
        words = text.lower().split()
        transliterated_words = []
        
        for word in words:
            # Clean word (remove punctuation)
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Try direct mapping first (for common words)
            if clean_word in ROMANIZED_TO_SINHALA:
                transliterated_words.append(ROMANIZED_TO_SINHALA[clean_word])
            else:
                # Fallback: character-by-character transliteration (simplified)
                # This is not perfect but gives reasonable approximation
                transliterated = self._simple_transliterate(clean_word)
                transliterated_words.append(transliterated)
        
        result = ' '.join(transliterated_words)
        logger.info(f"Transliterated: '{text}' → '{result}'")
        return result
    
    def _simple_transliterate(self, word: str) -> str:
        """
        Simple character-based transliteration for words not in dictionary
        
        This is a simplified approach - not perfect but reasonable.
        For production, consider using dedicated libraries like 'sinling' or 'icu'.
        
        Args:
            word: Romanized word
            
        Returns:
            Approximated Sinhala text
        """
        # Very basic mapping - just enough for common patterns
        # For better quality, use a proper transliteration library
        result = []
        i = 0
        while i < len(word):
            # Try 3-char combinations first
            if i + 2 < len(word):
                three = word[i:i+3]
                if three in ROMANIZED_TO_SINHALA:
                    result.append(ROMANIZED_TO_SINHALA[three])
                    i += 3
                    continue
            
            # Try 2-char combinations
            if i + 1 < len(word):
                two = word[i:i+2]
                if two in ROMANIZED_TO_SINHALA:
                    result.append(ROMANIZED_TO_SINHALA[two])
                    i += 2
                    continue
            
            # Single character
            single = word[i]
            if single in ROMANIZED_TO_SINHALA:
                result.append(ROMANIZED_TO_SINHALA[single])
            else:
                result.append(single)  # Keep as-is if no mapping
            i += 1
        
        return ''.join(result)
    
    def translate_si_to_en(self, text: str, is_romanized: bool = False) -> str:
        """
        Translate Sinhala (or romanized Singlish) to English.
        - Romanized Singlish: keyword dictionary first, then Google Translate fallback
        - Sinhala Unicode: Google Translate directly
        """
        if not self.si_to_en_model:
            return text

        try:
            if is_romanized:
                # Romanized Singlish can be noisy. Build multiple candidates and
                # choose the most retrieval-friendly one.
                keyword_candidate = self._translate_romanized_keywords(text)
                candidates = [keyword_candidate]

                try:
                    gt = self._google_translator(source='auto', target='en')
                    direct_gt = gt.translate(text)
                    candidates.append(direct_gt)

                    # Translate keyword-expanded phrase as an additional option.
                    # This often preserves herb names better.
                    if keyword_candidate and keyword_candidate != text:
                        expanded_gt = gt.translate(keyword_candidate)
                        candidates.append(expanded_gt)
                except Exception as e:
                    logger.warning(f"Google Translate failed for Singlish, using keyword candidate: {e}")

                scored = sorted(
                    ((self._score_english_query_candidate(c), c) for c in candidates if c and c.strip()),
                    key=lambda x: x[0],
                    reverse=True
                )
                if scored:
                    best = scored[0][1].strip()
                    logger.info(f"Selected Singlish translation candidate: '{best}'")
                    return best

                return text
            else:
                # Sinhala Unicode → English via Google Translate
                gt = self._google_translator(source='si', target='en')
                result = gt.translate(text)
                logger.info(f"Google-translated si→en: '{text[:60]}' → '{result[:60]}'")
                return result
        except Exception as e:
            logger.warning(f"translate_si_to_en failed: {e}")
            return text

    def translate_en_to_si(self, text: str) -> str:
        """
        Translate English answer to Sinhala Unicode using Google Translate.
        """
        if not self.en_to_si_model:
            return text

        try:
            gt = self._google_translator(source='en', target='si')
            result = gt.translate(text)
            logger.info(f"Google-translated en→si: '{text[:60]}' → '{result[:60]}'")
            return result
        except Exception as e:
            logger.warning(f"translate_en_to_si failed: {e}")
            return text  # fallback to English

    def translate_en_to_ta(self, text: str) -> str:
        """
        Translate English text to Tamil using Google Translate.
        """
        try:
            from deep_translator import GoogleTranslator
            gt = GoogleTranslator(source='en', target='ta')
            result = gt.translate(text)
            logger.info(f"Google-translated en→ta: '{text[:60]}' → '{result[:60]}'")
            return result
        except Exception as e:
            logger.warning(f"translate_en_to_ta failed: {e}")
            return text  # fallback to English

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text between languages
        
        Args:
            text: Input text
            source_lang: Source language code ('en' or 'si')
            target_lang: Target language code ('en' or 'si')
            
        Returns:
            Translated text
        """
        # If same language, return original
        if source_lang == target_lang:
            return text
        
        # Translate based on direction
        if source_lang == 'si' and target_lang == 'en':
            return self.translate_si_to_en(text)
        elif source_lang == 'en' and target_lang == 'si':
            return self.translate_en_to_si(text)
        else:
            logger.warning(f"Unsupported translation: {source_lang} → {target_lang}")
            return text
    
    def process_question_and_answer(
        self, 
        question: str,
        answer: str
    ) -> Dict[str, str]:
        """
        Process question and answer with automatic language detection and translation
        
        Workflow:
        1. Detect question language
        2. Translate question to English if needed (for RAG processing)
        3. Answer is generated in English
        4. Translate answer back to original language
        
        Args:
            question: User's question (any language)
            answer: Generated answer (in English)
            
        Returns:
            Dict with:
                - original_question: Original question
                - translated_question: Question in English (for processing)
                - detected_language: Detected language code
                - answer: Answer in user's original language
        """
        # Detect question language
        detected_lang = self.detect_language(question)
        
        logger.info(f"🌐 Detected language: {detected_lang}")
        
        # Translate question to English if needed
        if detected_lang == 'si':
            translated_question = self.translate_si_to_en(question)
        else:
            translated_question = question  # Already English
        
        # Translate answer back to user's language if needed
        if detected_lang == 'si':
            translated_answer = self.translate_en_to_si(answer)
        else:
            translated_answer = answer  # Keep English
        
        return {
            'original_question': question,
            'translated_question': translated_question,
            'detected_language': detected_lang,
            'answer': translated_answer,
            'answer_english': answer  # Always keep English version for reference
        }
    
    def is_available(self) -> bool:
        """
        Check if translation service is available
        
        Returns:
            True if models are loaded and ready
        """
        return (
            self.si_to_en_model is not None and 
            self.en_to_si_model is not None and
            self.language_detector is not None
        )


# Convenience function for standalone use
def create_translation_service(device: str = "cuda") -> Optional[TranslationService]:
    """
    Create translation service instance
    
    Args:
        device: Device to run on ("cuda" or "cpu")
        
    Returns:
        TranslationService instance or None if initialization fails
    """
    try:
        service = TranslationService(device=device)
        if service.is_available():
            return service
        else:
            logger.warning("Translation service not available")
            return None
    except Exception as e:
        logger.error(f"Failed to create translation service: {e}")
        return None
