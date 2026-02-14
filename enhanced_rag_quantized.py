"""
4-bit Quantized Enhanced RAG - Runs fully on GPU
This version uses 4-bit quantization to fit Phi-3 on RTX 2050 (4GB)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
from typing import Dict, Any, Optional

from validation_engine import ValidationEngine
from personalization_engine import PersonalizationEngine

logger = logging.getLogger(__name__)


class LLMArchitecture:
    """4-bit Quantized LLM - Fits on 4GB GPU"""
    def __init__(self, config):
        self.config = config
        self.device = "cuda"
        self._initialize_model()

    def _initialize_model(self):
        model_name = self.config.get("llm_model")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for 4-bit quantization!")

        logger.info(f"Loading 4-bit quantized model: {model_name}")

        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True
        )

        # Load with 4-bit quantization
        # IMPORTANT: device_map="auto" is REQUIRED for quantized models
        # It prevents transformers from calling .to() internally
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )

        # Model is automatically on correct device from quantization_config
        self.device = "cuda"
        self.model.eval()
        
        logger.info(f"✓ 4-bit quantized model loaded on GPU")
        
        # Show memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def generate(self, prompt: str) -> str:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        print("🔄 Generating response (4-bit quantized)...")

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 64),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract answer
        answer_marker = "Answer:"
        if answer_marker in full_output:
            parts = full_output.split(answer_marker)
            generated_text = parts[-1].strip() if len(parts) > 1 else full_output.strip()
        elif full_output.startswith(prompt.strip()):
            generated_text = full_output[len(prompt.strip()):].strip()
        elif len(full_output) > len(prompt):
            generated_text = full_output[len(prompt):].strip()
        else:
            generated_text = full_output.strip()
        
        if not generated_text or len(generated_text) < 10:
            generated_text = full_output.strip() if full_output.strip() else "Unable to generate answer."
        
        return generated_text


class EnhancedAyurvedicRAG:
    """Enhanced RAG with 4-bit quantization"""
    
    def __init__(
        self, 
        llm_model_name: str,
        max_new_tokens: int = 64,
        enable_validation: bool = True,
        enable_personalization: bool = True,
        embedding_model: str = "BAAI/bge-base-en-v1.5"
    ):
        print("🚀 Initializing 4-bit Quantized Enhanced RAG...")
        
        self.llm = LLMArchitecture({
            "llm_model": llm_model_name,
            "max_new_tokens": max_new_tokens,
        })
        
        self.enable_validation = enable_validation
        if enable_validation:
            logger.info("Initializing ValidationEngine...")
            self.validator = ValidationEngine(embedding_model_name=embedding_model)
            logger.info("✓ ValidationEngine ready")
        else:
            self.validator = None
        
        self.enable_personalization = enable_personalization
        if enable_personalization:
            logger.info("Initializing PersonalizationEngine...")
            self.personalizer = PersonalizationEngine(llm_generator=None)  # Use templates only
            logger.info("✓ PersonalizationEngine ready")
        else:
            self.personalizer = None
        
        logger.info("✓ EnhancedAyurvedicRAG ready (4-bit quantized)")
    
    def _build_context_with_citations(self, docs):
        context_blocks = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            block = f"""[Source {i}]
Book: {doc.get("source", "Unknown")}
Chapter: {meta.get("chapter", "N/A")}
Paragraph: {meta.get("paragraph", "N/A")}

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
        logger.info(f"Processing: {question[:50]}...")
        
        print("🔍 Retrieving relevant sources...")
        retrieved_docs = vector_db.search(question, top_k=validation_top_k)
        
        book_docs = [d for d in retrieved_docs if d.get("type") == "book"]
        citation_docs = book_docs if book_docs else retrieved_docs
        top_context_docs = citation_docs[:top_k]
        
        context_text = self._build_context_with_citations(top_context_docs)
        
        prompt = f"""You are an Ayurvedic knowledge assistant.
Answer the question strictly based on the context below.

Context:
{context_text}

Question:
{question}

Answer:""".strip()
        
        print("💭 Generating answer...")
        base_answer = self.llm.generate(prompt)
        
        validation_result = None
        if self.enable_validation and self.validator:
            print("✓ Validating answer...")
            validation_result = self.validator.validate_answer(
                primary_answer=base_answer,
                retrieved_docs=retrieved_docs,
                top_k=validation_top_k
            )
        
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
        
        formatted_citations = []
        for doc in top_context_docs:
            meta = doc.get("metadata", {})
            formatted_citations.append({
                "source": doc.get("source", "Unknown"),
                "chapter": meta.get("chapter", "N/A"),
                "paragraph": meta.get("paragraph", "N/A"),
                "text_preview": doc.get("text", "")[:100] + "..."
            })
        
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
        if not self.personalizer:
            raise RuntimeError("Personalization not enabled")
        return self.personalizer.create_user_profile(user_id, prakriti_responses)
    
    def load_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not self.personalizer:
            return None
        return self.personalizer.load_user_profile(user_id)
