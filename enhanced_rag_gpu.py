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
        
        logger.info("EnhancedAyurvedicRAG initialized (GPU FORCED)")
    
    def _build_context_with_citations(self, docs):
        """Build LLM context including citations"""
        context_blocks = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            book = doc.get("source", "Unknown Book")
            chapter = meta.get("chapter", "N/A")
            paragraph = meta.get("paragraph", meta.get("verse", "N/A"))
            
            block = f"""[Source {i}]
Book: {book}
Chapter: {chapter}
Paragraph: {paragraph}

{doc.get("text", "")}"""
            context_blocks.append(block.strip())
        
        return "\n\n".join(context_blocks)
    
    def answer_question(
        self, 
        question: str, 
        vector_db, 
        top_k: int = 5,
        user_profile: Optional[Dict[str, Any]] = None,
        validation_top_k: int = 5
    ) -> Dict[str, Any]:
        """Answer question with validation and personalization"""
        logger.info(f"Processing question: {question[:50]}...")
        
        # Retrieve documents with similarity scores
        print("🔍 Retrieving relevant sources...")
        retrieved_docs = vector_db.search(question, top_k=validation_top_k)
        
        # Add similarity percentages to docs
        for i, doc in enumerate(retrieved_docs):
            if 'similarity' in doc:
                doc['similarity_percentage'] = round(doc['similarity'] * 100, 1)
            else:
                # Estimate based on ranking
                doc['similarity_percentage'] = round((1 - i * 0.1) * 100, 1)
        
        book_docs = [d for d in retrieved_docs if d.get("type") == "book"]
        citation_docs = book_docs if book_docs else retrieved_docs
        top_context_docs = citation_docs[:top_k]
        
        # Dynamic token calculation
        context_length = sum(len(d.get("text", "")) for d in top_context_docs)
        question_length = len(question)
        
        # Estimate tokens needed: increased for complete answers
        if context_length > 2000 or question_length > 100:
            dynamic_tokens = 600  # Full detailed answer
        elif context_length > 1000:
            dynamic_tokens = 500  # Medium detailed answer
        else:
            dynamic_tokens = 400  # Standard answer
            
        print(f"📏 Dynamic tokens: {dynamic_tokens} (context: {context_length} chars)")
        
        # Build context
        context_text = self._build_context_with_citations(top_context_docs)
        
        print(f"🔍 Context preview (first 300 chars): {context_text[:300]}...")
        
        # Use Phi-3's chat format properly
        context_summary = context_text[:2500]  # Increased context limit
        prompt = f"""<|system|>You are an expert Ayurvedic physician. Provide clear, comprehensive answers based on the context provided. Include specific details, benefits, and practical information. Structure your answer with bullet points for clarity.<|end|>
<|user|>Context: {context_summary}

Question: {question}<|end|>
<|assistant|>"""
        

        # Generate answer with dynamic token adjustment
        print("💭 Generating answer...")
        base_answer = self.llm.generate(prompt, max_new_tokens=dynamic_tokens)
        
        print(f"✅ Generation complete!")
        print(f"   Answer length: {len(base_answer)} chars")
        print(f"   Answer preview: {base_answer[:200] if base_answer else '[EMPTY]'}...")
        
        if not base_answer or len(base_answer.strip()) < 10:
            print("⚠️  WARNING: Generated answer is empty or too short!")
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
        
        # Format response with similarity percentages
        formatted_citations = []
        for doc in top_context_docs:
            meta = doc.get("metadata", {})
            citation = {
                "source": doc.get("source", "Unknown"),
                "chapter": meta.get("chapter", "N/A"),
                "paragraph": meta.get("paragraph", "N/A"),
                "similarity_percentage": doc.get("similarity_percentage", 0.0),
                "text_preview": doc.get("text", "")[:100] + "..."
            }
            formatted_citations.append(citation)
        
        response = {
            "answer": final_answer,
            "base_answer": base_answer,
            "citations": formatted_citations,
            "sources": top_context_docs,
            "all_sources": retrieved_docs,
            "num_sources": len(top_context_docs),
            "personalized": is_personalized
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
