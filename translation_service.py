"""
Translation Service for Bilingual Ayurvedic Q&A System
Supports English ↔ Sinhala translation with automatic language detection

Uses Helsinki-NLP models from Hugging Face (FREE, runs on GPU):
- opus-mt-si-en: Sinhala → English
- opus-mt-en-si: English → Sinhala

Features:
- Automatic language detection (langdetect library)
- Bidirectional translation
- GPU acceleration
- Fallback to English if translation fails
"""

import logging
from typing import Dict, Tuple, Optional
import re

logger = logging.getLogger(__name__)


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
        """Load translation models and language detector"""
        try:
            from transformers import MarianMTModel, MarianTokenizer
            import langdetect
            
            logger.info("Loading Sinhala ↔ English translation models...")
            
            # Sinhala → English
            logger.info("  Loading opus-mt-si-en (Sinhala → English)...")
            self.si_to_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-si-en")
            self.si_to_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-si-en").to(self.device)
            
            # English → Sinhala  
            logger.info("  Loading opus-mt-en-si (English → Sinhala)...")
            self.en_to_si_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-si")
            self.en_to_si_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-si").to(self.device)
            
            # Language detector
            self.language_detector = langdetect
            
            logger.info("✓ All translation models loaded successfully")
            
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            logger.error("Install with: pip install transformers sentencepiece langdetect")
            raise
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of input text
        
        Args:
            text: Input text
            
        Returns:
            Language code: 'si' (Sinhala), 'en' (English), or 'unknown'
        """
        if not self.language_detector:
            return 'unknown'
        
        try:
            # Clean text for detection
            clean_text = re.sub(r'[^\w\s]', '', text).strip()
            
            if not clean_text:
                return 'unknown'
            
            # Detect language
            detected = self.language_detector.detect(clean_text)
            
            # Map to supported languages
            if detected in ['si', 'ta']:  # Sinhala or Tamil (treat as Sinhala)
                return 'si'
            elif detected == 'en':
                return 'en'
            else:
                # Check for Sinhala Unicode characters
                if self._has_sinhala_chars(text):
                    return 'si'
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
    
    def translate_si_to_en(self, text: str) -> str:
        """
        Translate Sinhala text to English
        
        Args:
            text: Sinhala text
            
        Returns:
            English translation
        """
        if not self.si_to_en_model:
            logger.warning("Translation model not loaded, returning original text")
            return text
        
        try:
            # Tokenize
            inputs = self.si_to_en_tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate translation
            outputs = self.si_to_en_model.generate(**inputs)
            
            # Decode
            translated = self.si_to_en_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Translated (si→en): {text[:50]}... → {translated[:50]}...")
            return translated.strip()
            
        except Exception as e:
            logger.error(f"Translation error (si→en): {e}")
            return text  # Fallback to original
    
    def translate_en_to_si(self, text: str) -> str:
        """
        Translate English text to Sinhala
        
        Args:
            text: English text
            
        Returns:
            Sinhala translation
        """
        if not self.en_to_si_model:
            logger.warning("Translation model not loaded, returning original text")
            return text
        
        try:
            # Tokenize
            inputs = self.en_to_si_tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate translation
            outputs = self.en_to_si_model.generate(**inputs)
            
            # Decode
            translated = self.en_to_si_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Translated (en→si): {text[:50]}... → {translated[:50]}...")
            return translated.strip()
            
        except Exception as e:
            logger.error(f"Translation error (en→si): {e}")
            return text  # Fallback to original
    
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
