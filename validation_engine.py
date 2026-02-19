"""
Validation Engine for Multi-Source Answer Validation
Implements Option 1: Multi-Source Validation & Confidence Scoring

Features:
- Multi-document retrieval from diverse sources
- Semantic similarity-based agreement scoring
- Source quality weighting (ancient texts weighted higher)
- Contradiction detection
- Confidence score calculation
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


# Source quality weights (ancient/classical texts weighted higher)
SOURCE_WEIGHTS = {
    'astanga_hridaya': 1.5,        # Classical Ayurvedic text
    'ayurveda_ancient_wisdom': 1.3, # Ancient wisdom
    'hela_osu_corpus': 1.2,         # Traditional Sri Lankan text
    'ayurveda_food_nutrition': 1.0, # Modern compilation
    'everyday_ayurveda': 0.9,       # Modern interpretation
    'qa_dataset': 0.7               # Secondary source
}


class ValidationEngine:
    """
    Multi-source validation engine for Ayurvedic Q&A answers
    """
    
    def __init__(self, embedding_model_name: str = "BAAI/bge-base-en-v1.5"):
        """
        Initialize validation engine
        
        Args:
            embedding_model_name: Name of the sentence transformer model to use
        """
        logger.info(f"Initializing ValidationEngine with {embedding_model_name} on CPU")
        # Force CPU to save GPU memory for the main LLM
        self.encoder = SentenceTransformer(embedding_model_name, device="cpu")
        logger.info("ValidationEngine initialized successfully (CPU)")
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Generate embeddings
        emb1 = self.encoder.encode(text1, normalize_embeddings=True)
        emb2 = self.encoder.encode(text2, normalize_embeddings=True)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2)
        
        return float(similarity)
    
    def get_source_weight(self, source_name: str) -> float:
        """
        Get quality weight for a source
        
        Args:
            source_name: Name of the source book
            
        Returns:
            Weight value (higher = more authoritative)
        """
        # Normalize source name for matching
        source_lower = source_name.lower().replace(' ', '_').replace('-', '_')
        
        # Find matching weight
        for key, weight in SOURCE_WEIGHTS.items():
            if key in source_lower or source_lower in key:
                return weight
        
        # Default weight for unknown sources
        return 1.0
    
    def calculate_agreement_scores(
        self, 
        primary_answer: str, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate how much each source agrees with the primary answer
        
        Args:
            primary_answer: The generated answer text
            retrieved_docs: List of retrieved document dictionaries
            
        Returns:
            List of agreement information for each source
        """
        agreements = []
        
        for doc in retrieved_docs:
            # Get source text
            source_text = doc.get('text', '')
            source_name = doc.get('source', 'Unknown')
            
            # Calculate semantic similarity
            similarity = self.calculate_semantic_similarity(
                primary_answer, 
                source_text
            )
            
            # Determine if source agrees (threshold: 0.55 for paraphrased/summarized answers)
            agrees = similarity > 0.55
            
            # Get source metadata
            metadata = doc.get('metadata', {})
            
            agreement_info = {
                'source': source_name,
                'chapter': metadata.get('chapter', 'N/A'),
                'paragraph': metadata.get('paragraph', 'N/A'),
                'similarity': round(similarity, 3),
                'agrees': agrees,
                'agreement_level': self._get_agreement_level(similarity),
                'source_weight': self.get_source_weight(source_name)
            }
            
            agreements.append(agreement_info)
        
        return agreements
    
    def _get_agreement_level(self, similarity: float) -> str:
        """
        Convert similarity score to agreement level
        
        Args:
            similarity: Similarity score (0-1)
            
        Returns:
            Agreement level string
        """
        if similarity >= 0.80:
            return 'strong'
        elif similarity >= 0.65:
            return 'high'
        elif similarity >= 0.55:
            return 'agrees'
        elif similarity >= 0.45:
            return 'partial'
        else:
            return 'weak'
    
    def detect_contradictions(
        self, 
        agreements: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Detect potential contradictions between sources
        
        Args:
            agreements: List of agreement information
            
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        # Count sources that don't agree
        non_agreeing = [a for a in agreements if not a['agrees']]
        
        if len(non_agreeing) >= 2:
            # Multiple sources disagree - potential contradiction
            contradiction = {
                'type': 'multiple_disagreement',
                'message': f"{len(non_agreeing)} sources show different perspectives",
                'sources': [a['source'] for a in non_agreeing]
            }
            contradictions.append(contradiction)
        
        # Check for very low similarity scores (< 0.4) which might indicate contradiction
        very_low = [a for a in agreements if a['similarity'] < 0.4]
        if very_low:
            contradiction = {
                'type': 'low_similarity',
                'message': "Some sources have significantly different information",
                'sources': [a['source'] for a in very_low]
            }
            contradictions.append(contradiction)
        
        return contradictions
    
    def calculate_confidence_score(
        self, 
        agreements: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall confidence score for the answer
        
        Formula (Research-Grade):
        confidence = (agreement_ratio × 0.35) + (weighted_quality × 0.50) + (consistency_bonus × 0.15)
        
        Optimized for summarized/paraphrased answers with threshold of 0.55 for agreement.
        Prioritizes semantic quality over strict agreement count for research deployment.
        
        Args:
            agreements: List of agreement information
            
        Returns:
            Confidence score (0-100)
        """
        if not agreements:
            return 0.0
        
        # Component 1: Agreement ratio (what % of sources agree?)
        agreeing_count = sum(1 for a in agreements if a['agrees'])
        agreement_ratio = agreeing_count / len(agreements)
        
        # Component 2: Weighted quality score (most important for research)
        weighted_similarities = []
        for agreement in agreements:
            weighted_sim = agreement['similarity'] * agreement['source_weight']
            weighted_similarities.append(weighted_sim)
        
        avg_weighted_quality = np.mean(weighted_similarities)
        
        # Normalize quality to 0-1 range (similarities are already 0-1)
        normalized_quality = avg_weighted_quality
        
        # Component 3: Consistency bonus (reward when all sources are similar quality)
        similarity_variance = np.var([a['similarity'] for a in agreements])
        # Lower variance = more consistent = higher bonus (max 1.0)
        consistency_bonus = max(0, 1.0 - (similarity_variance * 10))
        
        # Final confidence (35% agreement, 50% quality, 15% consistency)
        # This formula better reflects answer quality for paraphrased/summarized content
        confidence = (agreement_ratio * 0.35) + (normalized_quality * 0.50) + (consistency_bonus * 0.15)
        
        # Apply boost for high-quality sources (if avg similarity > 0.58, boost by up to 10%)
        if normalized_quality > 0.58:
            quality_boost = min(0.10, (normalized_quality - 0.58) * 0.5)
            confidence = min(1.0, confidence + quality_boost)
        
        # Convert to percentage
        confidence_percentage = confidence * 100
        
        return round(confidence_percentage, 1)
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        Convert confidence score to level label (research-grade thresholds)
        
        Args:
            confidence: Confidence score (0-100)
            
        Returns:
            Confidence level string
        """
        if confidence >= 80:
            return 'very_high'
        elif confidence >= 70:
            return 'high'
        elif confidence >= 55:
            return 'medium'
        elif confidence >= 40:
            return 'low'
        else:
            return 'very_low'
    
    def validate_answer(
        self, 
        primary_answer: str, 
        retrieved_docs: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Main validation function - validates answer against multiple sources
        
        Args:
            primary_answer: The generated answer text
            retrieved_docs: List of all retrieved documents
            top_k: Number of top sources to validate against
            
        Returns:
            Validation result dictionary with confidence, agreements, contradictions
        """
        logger.info(f"Validating answer against {len(retrieved_docs)} sources")
        
        # Use top_k sources for validation
        validation_docs = retrieved_docs[:top_k]
        
        # Calculate agreements
        agreements = self.calculate_agreement_scores(primary_answer, validation_docs)
        
        # Calculate confidence
        confidence = self.calculate_confidence_score(agreements)
        confidence_level = self.get_confidence_level(confidence)
        
        # Detect contradictions
        contradictions = self.detect_contradictions(agreements)
        
        # Count sources that agree
        sources_agree = sum(1 for a in agreements if a['agrees'])
        
        # Generate validation note
        note = self._generate_validation_note(confidence_level, agreements, contradictions)
        
        result = {
            'confidence': confidence,
            'confidence_level': confidence_level,
            'sources_checked': len(validation_docs),
            'sources_agree': sources_agree,
            'agreements': agreements,
            'contradictions': contradictions,
            'note': note
        }
        
        logger.info(f"Validation complete: {confidence:.1f}% confidence ({confidence_level})")
        
        return result
    
    def _generate_validation_note(
        self, 
        confidence_level: str, 
        agreements: List[Dict[str, Any]], 
        contradictions: List[Dict[str, str]]
    ) -> str:
        """
        Generate a human-readable validation note
        
        Args:
            confidence_level: Confidence level string
            agreements: List of agreement information
            contradictions: List of contradictions
            
        Returns:
            Validation note string
        """
        if confidence_level in ['very_high', 'high']:
            # Check for classical text support
            classical_sources = [
                a for a in agreements 
                if a['source_weight'] > 1.2 and a['agrees']
            ]
            
            if classical_sources:
                return "Strong consensus across classical and modern Ayurvedic texts"
            else:
                return "Good consensus across multiple Ayurvedic sources"
        
        elif confidence_level == 'medium':
            if contradictions:
                return "Moderate agreement with some variation across sources"
            else:
                return "Answer is supported by several sources with good agreement"
        
        else:  # low or very_low
            if contradictions:
                return "Limited consensus - different sources show varying perspectives"
            else:
                return "Limited source support - answer may need additional verification"


# ============================================================
# Usage Example
# ============================================================

def example_usage():
    """
    Example of how to use the ValidationEngine
    """
    # Initialize engine
    validator = ValidationEngine()
    
    # Example answer
    answer = "Triphala supports digestive health and detoxification."
    
    # Example retrieved documents
    docs = [
        {
            'text': 'Triphala is beneficial for digestion and cleansing the body.',
            'source': 'Astanga Hridaya',
            'metadata': {'chapter': 'Chapter 6', 'paragraph': 12}
        },
        {
            'text': 'The three fruits of Triphala aid in digestive function.',
            'source': 'Ayurveda Food and Nutrition',
            'metadata': {'chapter': 'Chapter 8', 'paragraph': 3}
        }
    ]
    
    # Validate
    result = validator.validate_answer(answer, docs, top_k=2)
    
    print(f"Confidence: {result['confidence']}%")
    print(f"Level: {result['confidence_level']}")
    print(f"Sources agree: {result['sources_agree']}/{result['sources_checked']}")
    print(f"Note: {result['note']}")


if __name__ == "__main__":
    example_usage()
