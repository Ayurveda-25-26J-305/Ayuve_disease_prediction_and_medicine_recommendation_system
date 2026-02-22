import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)


def _patch_dynamic_cache():
    """
    Phi-3's bundled modeling_phi3.py was written for transformers ~4.40.
    Newer transformers removed two things that Phi-3 still uses:
      1. DynamicCache.from_legacy_cache()  → raises AttributeError on generate()
      2. DynamicCache.seen_tokens property → raises AttributeError mid-forward-pass
    Without this patch we must use use_cache=False which destroys position
    encoding in 4-bit models, producing garbage output.
    This patch adds both back as no-ops / shims so use_cache=True works normally.
    """
    try:
        from transformers.cache_utils import DynamicCache

        # 1. Restore from_legacy_cache class method if missing
        if not hasattr(DynamicCache, "from_legacy_cache"):
            @classmethod
            def from_legacy_cache(cls, past_key_values=None):
                cache = cls()
                if past_key_values is not None:
                    for layer_idx, (k, v) in enumerate(past_key_values):
                        cache.update(k, v, layer_idx)
                return cache
            DynamicCache.from_legacy_cache = from_legacy_cache
            logger.info("DynamicCache.from_legacy_cache patched ✓")

        # 2. Restore seen_tokens property if missing
        if not isinstance(getattr(DynamicCache, "seen_tokens", None), property):
            DynamicCache.seen_tokens = property(
                lambda self: self.get_seq_length()
            )
            logger.info("DynamicCache.seen_tokens patched ✓")

    except Exception as e:
        logger.warning(f"DynamicCache patch failed (non-fatal): {e}")


_patch_dynamic_cache()




class LLMArchitecture:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_model()

    def _initialize_model(self):
        model_name = self.config.get("llm_model")

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True
        )

        # Load config and fix rope_scaling issue for Phi-3
        model_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Fix rope_scaling configuration if present - disable it if misconfigured
        if hasattr(model_config, 'rope_scaling') and model_config.rope_scaling is not None:
            if isinstance(model_config.rope_scaling, dict) and 'type' not in model_config.rope_scaling:
                # Disable rope_scaling if it doesn't have a valid type
                model_config.rope_scaling = None
                logger.info("Disabled misconfigured rope_scaling")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_config,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            ) if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map={"": 0} if self.device == "cuda" else None,  # explicit GPU 0 avoids dispatch_model calling .to() on quantized model
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            trust_remote_code=True
        )

        self.model.eval()
        logger.info("LLM initialized successfully")

    def generate_from_messages(self, messages: list, max_new_tokens: int = None) -> str:
        """
        Preferred method for chat models — tokenizes directly from messages using
        apply_chat_template(tokenize=True) so special tokens are NEVER re-tokenized
        from a string (which would corrupt them).
        """
        # Tokenize directly — this guarantees <|user|>/<|end|>/<|assistant|> are
        # encoded as their special token IDs, not as plain text bytes.
        # apply_chat_template returns a BatchEncoding in newer transformers, so
        # always access input_ids explicitly.
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        # Handle both plain tensor and BatchEncoding return types
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded.to(self.device)
        else:
            input_ids = encoded["input_ids"].to(self.device)

        input_length = input_ids.shape[1]
        # Explicit attention mask — required when pad_token_id == eos_token_id
        # Without this, generation is non-deterministic and produces garbage.
        attention_mask = torch.ones_like(input_ids)
        print(f" Generating response (input tokens: {input_length})...")

        # use_cache=True: DynamicCache is now patched at module load (see _patch_dynamic_cache)
        # to restore from_legacy_cache() and .seen_tokens that Phi-3's modeling_phi3.py requires.
        # This gives correct position encoding; use_cache=False caused garbage/repetition loops.
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens or self.config.get("max_new_tokens", 200),
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        # Decode only new tokens
        generated_ids = output_ids[0][input_length:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"✅ Generated {len(generated_ids)} new tokens")
        return answer if answer else "Unable to generate an answer. Please try again."

    def generate(self, prompt: str, max_new_tokens: int = None) -> str:
        """Legacy plain-text prompt path (used only by AyurvedicRAG)."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1500
        ).to(self.device)

        print(f" Generating response (input tokens: {inputs['input_ids'].shape[1]})...")

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.config.get("max_new_tokens", 64),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True  # DynamicCache patched at module load for Phi-3 compatibility
            )

        # Decode only the newly generated tokens (excluding the input prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_ids = output_ids[0][input_length:]
        
        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True  # Add this to fix spacing
        ).strip()
        
        # Fix common spacing issues
        import re
        # Add space after punctuation if missing
        answer = re.sub(r'([.!?,;:])([A-Z])', r'\1 \2', answer)
        # Fix compressed words (add space between lowercase and uppercase)
        answer = re.sub(r'([a-z])([A-Z])', r'\1 \2', answer)
        
        # Remove Phi-3 special tokens if present
        for token in ['<|system|>', '<|user|>', '<|assistant|>', '<|end|>']:
            answer = answer.replace(token, '')
        answer = answer.strip()
        
        # Clean up any remaining artifacts
        # Remove source citations if they appear
        if "[Source" in answer:
            answer = answer.split("[Source")[0].strip()
        
        # Remove any metadata lines
        lines = answer.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and metadata
            if line and not any(x in line for x in ["Chapter:", "Paragraph:", "Book:", "[Source"]):
                clean_lines.append(line)
        
        if clean_lines:
            answer = '\n'.join(clean_lines).strip()
        
        return answer if len(answer) > 10 else "Unable to generate a proper answer. Please try rephrasing your question."




class AyurvedicRAG:
    def __init__(self, llm_model_name, max_new_tokens=64):
        self.llm = LLMArchitecture({
            "llm_model": llm_model_name,
            "max_new_tokens": max_new_tokens,
        })

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

            # Truncate text to keep total prompt within 512 tokens
            text = doc.get("text", "")
            if len(text) > 300:
                text = text[:300] + "..."

            block = f"[Source {i}] {book}\n{text}"
            context_blocks.append(block.strip())

        return "\n\n".join(context_blocks)

    def answer_question(self, question: str, vector_db, top_k: int = 5):
        """
        Answer question using RAG with STRICT book-based citations
        """

        # Retrieve documents
        retrieved_docs = vector_db.search(question, top_k=top_k)

        # Separate book and QA sources
        book_docs = [d for d in retrieved_docs if d.get("type") == "book"]
        qa_docs = [d for d in retrieved_docs if d.get("type") == "qa"]

        # Use book docs if available, otherwise fallback
        citation_docs = book_docs if book_docs else retrieved_docs

        # Limit context size (memory safe)
        top_context_docs = citation_docs[:3]

        # Build context WITH citation metadata
        context_text = self._build_context_with_citations(top_context_docs)

        prompt = f"""
You are an Ayurvedic knowledge assistant.
Answer the question strictly based on the context below.

Context:
{context_text}

Question:
{question}

Answer:
""".strip()

        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": top_context_docs,     # ONLY book sources if available
            "all_sources": retrieved_docs,   # for debugging / evaluation
            "num_sources": len(top_context_docs)
        }
