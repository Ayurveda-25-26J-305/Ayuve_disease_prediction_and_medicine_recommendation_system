"""
Enhanced Ayurvedic RAG System
Integrates:
- Original RAG functionality
- Multi-source validation (Option 1)
- Answer personalization (Option 2)

This module extends the existing AyurvedicRAG class with the new features.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Dict, Any, Optional

# Import the new engines
from validation_engine import ValidationEngine
from personalization_engine import PersonalizationEngine

logger = logging.getLogger(__name__)


class LLMArchitecture:
    """LLM wrapper - same as before"""
    def __init__(self, config):
        self.config = config
        self.device = None  # Will be set after model loads
        self._initialize_model()

    def _initialize_model(self):
        model_name = self.config.get("llm_model")

        logger.info(f"Loading model: {model_name}")
        
        # Detect if CUDA is available
        has_cuda = torch.cuda.is_available()
        logger.info(f"CUDA available: {has_cuda}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True
        )

        # Load model - let it decide device placement
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if has_cuda else torch.float32,
            device_map="auto" if has_cuda else None,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            trust_remote_code=True
        )
        
        # Get actual device from model (might be CPU even if CUDA available due to size)
        if hasattr(self.model, 'device'):
            self.device = str(self.model.device)
        else:
            # Check where the first parameter is
            first_param = next(self.model.parameters())
            self.device = str(first_param.device)
        
        # Normalize device string
        if 'cuda' in self.device:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        # Ensure model is fully on the detected device
        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info(f"LLM initialized on device: {self.device}")

    def generate(self, prompt: str) -> str:
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Increased from 1024 to handle longer context
        ).to(self.device)

        print("🔄 Generating response...")

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 64),
                do_sample=False,
                use_cache=False,  # Disable cache to avoid compatibility issues
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode full output
        full_output = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        
        # Debug: Show what we got
        if len(full_output) < 50:
            print(f"⚠️ Warning: Output is very short ({len(full_output)} chars)")
            print(f"   First 200 chars of prompt: {prompt[:200]}...")
            print(f"   Output: '{full_output}'")
        
        # Extract only new generated text (remove prompt)
        # Try to find where the "Answer:" section starts
        answer_marker = "Answer:"
        if answer_marker in full_output:
            # Get everything after "Answer:"
            parts = full_output.split(answer_marker)
            if len(parts) > 1:
                generated_text = parts[-1].strip()
            else:
                generated_text = full_output.strip()
        elif full_output.startswith(prompt.strip()):
            # Prompt is at beginning, remove it
            generated_text = full_output[len(prompt.strip()):].strip()
        elif len(full_output) > len(prompt):
            # Fallback: assume prompt is at start based on length
            generated_text = full_output[len(prompt):].strip()
        else:
            # Last resort: use full output
            generated_text = full_output.strip()
        
        # If still empty or very short, return full output
        if not generated_text or len(generated_text) < 10:
            print("⚠️ Warning: Extracted answer is too short, using full output")
            generated_text = full_output.strip() if full_output.strip() else "Unable to generate answer. Please try with a simpler question."
        
        return generated_text


class EnhancedAyurvedicRAG:
    """
    Enhanced RAG system with validation and personalization
    """
    
    def __init__(
        self, 
        llm_model_name: str,
        max_new_tokens: int = 64,
        enable_validation: bool = True,
        enable_personalization: bool = True,
        embedding_model: str = "BAAI/bge-base-en-v1.5"
    ):
        """
        Initialize enhanced RAG system
        
        Args:
            llm_model_name: Name of LLM model to use
            max_new_tokens: Max tokens for generation
            enable_validation: Enable multi-source validation
            enable_personalization: Enable answer personalization
            embedding_model: Embedding model for validation
        """
        # Initialize base LLM
        self.llm = LLMArchitecture({
            "llm_model": llm_model_name,
            "max_new_tokens": max_new_tokens,
        })
        
        # Initialize validation engine (Option 1)
        self.enable_validation = enable_validation
        if enable_validation:
            logger.info("Initializing ValidationEngine...")
            self.validator = ValidationEngine(embedding_model_name=embedding_model)
            logger.info("✓ ValidationEngine ready")
        else:
            self.validator = None
        
        # Initialize personalization engine (Option 2)
        self.enable_personalization = enable_personalization
        if enable_personalization:
            logger.info("Initializing PersonalizationEngine...")
            self.personalizer = PersonalizationEngine(llm_generator=self.llm)
            logger.info("✓ PersonalizationEngine ready")
        else:
            self.personalizer = None
        
        logger.info("EnhancedAyurvedicRAG initialized successfully")
    
    def _build_context_with_citations(self, docs):
        """
        Build LLM context including book, chapter, and paragraph
        """
        context_blocks = []

        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})

            book = doc.get("source", "Unknown Book")
            chapter = meta.get("chapter", "N/A")
            paragraph = meta.get("paragraph", meta.get("verse", "N/A"))

            block = f"""
[Source {i}]
Book: {book}
Chapter: {chapter}
Paragraph: {paragraph}

{doc.get("text", "")}
"""
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
        """
        Answer question using RAG with validation and personalization
        
        Args:
            question: User's question
            vector_db: Vector database instance
            top_k: Number of documents for RAG context
            user_profile: Optional user profile for personalization
            validation_top_k: Number of sources for validation
            
        Returns:
            Enhanced answer dictionary with validation and personalization
        """
        logger.info(f"Processing question: {question[:50]}...")
        
        # STEP 1: Retrieve documents (original RAG)
        print("🔍 Retrieving relevant sources...")
        retrieved_docs = vector_db.search(question, top_k=validation_top_k)
        
        # Separate book and QA sources
        book_docs = [d for d in retrieved_docs if d.get("type") == "book"]
        qa_docs = [d for d in retrieved_docs if d.get("type") == "qa"]
        
        # Use book docs if available, otherwise fallback
        citation_docs = book_docs if book_docs else retrieved_docs
        
        # Limit context size for generation
        top_context_docs = citation_docs[:top_k]
        
        # Build context WITH citation metadata
        context_text = self._build_context_with_citations(top_context_docs)
        
        # Generate answer prompt
        prompt = f"""
You are an Ayurvedic knowledge assistant.
Answer the question strictly based on the context below.

Context:
{context_text}

Question:
{question}

Answer:
""".strip()
        
        # STEP 2: Generate base answer (original RAG)
        print("💭 Generating answer...")
        base_answer = self.llm.generate(prompt)
        
        # STEP 3: Validate answer (Option 1)
        validation_result = None
        if self.enable_validation and self.validator:
            print("✓ Validating answer across sources...")
            validation_result = self.validator.validate_answer(
                primary_answer=base_answer,
                retrieved_docs=retrieved_docs,
                top_k=validation_top_k
            )
            logger.info(f"Validation: {validation_result['confidence']}% confidence")
        
        # STEP 4: Personalize answer (Option 2)
        final_answer = base_answer
        is_personalized = False
        
        if self.enable_personalization and self.personalizer and user_profile:
            print("✨ Personalizing answer for your constitution...")
            final_answer = self.personalizer.personalize_answer(
                base_answer=base_answer,
                user_profile=user_profile,
                question=question
            )
            is_personalized = True
            logger.info(f"Answer personalized for {user_profile.get('dominant_dosha', 'N/A')}")
        
        # STEP 5: Format citations for frontend
        formatted_citations = []
        for doc in top_context_docs:
            meta = doc.get("metadata", {})
            citation = {
                "source": doc.get("source", "Unknown"),
                "chapter": meta.get("chapter", "N/A"),
                "paragraph": meta.get("paragraph", "N/A"),
                "text_preview": doc.get("text", "")[:100] + "..."
            }
            
            # Add validation info if available
            if validation_result:
                # Find matching agreement
                for agreement in validation_result.get('agreements', []):
                    if agreement['source'] == citation['source']:
                        citation['agreement'] = agreement['agreement_level']
                        citation['similarity'] = agreement['similarity']
                        break
            
            formatted_citations.append(citation)
        
        # STEP 6: Build final response
        response = {
            "answer": final_answer,
            "base_answer": base_answer,  # Include base for comparison
            "citations": formatted_citations,
            "sources": top_context_docs,
            "all_sources": retrieved_docs,
            "num_sources": len(top_context_docs),
            "personalized": is_personalized
        }
        
        # Add validation data if enabled
        if validation_result:
            response["validation"] = {
                "confidence": validation_result['confidence'],
                "confidence_level": validation_result['confidence_level'],
                "sources_checked": validation_result['sources_checked'],
                "sources_agree": validation_result['sources_agree'],
                "contradictions": validation_result['contradictions'],
                "note": validation_result['note'],
                "agreements": validation_result['agreements']  # Detailed breakdown
            }
        
        # Add user profile info if personalized
        if is_personalized and user_profile:
            response["user_info"] = {
                "dominant_dosha": user_profile.get('dominant_dosha', 'N/A'),
                "current_season": user_profile.get('current_season', 'N/A')
            }
        
        logger.info("✓ Enhanced answer generated successfully")
        
        return response
    
    def create_user_profile(
        self, 
        user_id: str, 
        prakriti_responses: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Create a user profile for personalization
        
        Args:
            user_id: Unique user identifier
            prakriti_responses: Dictionary of questionnaire responses
            
        Returns:
            User profile dictionary
        """
        if not self.personalizer:
            raise RuntimeError("Personalization engine not enabled")
        
        return self.personalizer.create_user_profile(
            user_id=user_id,
            prakriti_responses=prakriti_responses
        )
    
    def load_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Load existing user profile
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile or None
        """
        if not self.personalizer:
            return None
        
        return self.personalizer.load_user_profile(user_id)
    
    def get_prakriti_questions(self):
        """
        Get Prakriti assessment questions
        
        Returns:
            List of questions
        """
        if not self.personalizer:
            return []
        
        from personalization_engine import PRAKRITI_QUESTIONS
        return PRAKRITI_QUESTIONS


# ============================================================
# Example Usage
# ============================================================

def example_usage():
    """
    Example of how to use EnhancedAyurvedicRAG
    """
    # This is just for demonstration - you'd use your actual vector DB
    class DummyVectorDB:
        def search(self, query, top_k=5):
            return [
                {
                    'text': 'Triphala supports digestive health.',
                    'source': 'Astanga Hridaya',
                    'type': 'book',
                    'metadata': {'chapter': 'Chapter 6', 'paragraph': 12}
                },
                {
                    'text': 'The three fruits aid digestion.',
                    'source': 'Ayurveda Food and Nutrition',
                    'type': 'book',
                    'metadata': {'chapter': 'Chapter 8', 'paragraph': 3}
                }
            ]
    
    # Initialize enhanced RAG
    rag = EnhancedAyurvedicRAG(
        llm_model_name="microsoft/Phi-3-mini-4k-instruct",
        enable_validation=True,
        enable_personalization=True
    )
    
    # Create user profile
    profile = rag.create_user_profile(
        user_id='user123',
        prakriti_responses={
            'q1': 'A', 'q2': 'A', 'q3': 'A', 'q4': 'A', 'q5': 'A',
            'q6': 'B', 'q7': 'A', 'q8': 'A', 'q9': 'A', 'q10': 'A'
        }
    )
    
    # Answer question with validation and personalization
    result = rag.answer_question(
        question="What are the benefits of Triphala?",
        vector_db=DummyVectorDB(),
        user_profile=profile
    )
    
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['validation']['confidence']}%")
    print(f"Personalized: {result['personalized']}")


if __name__ == "__main__":
    example_usage()
