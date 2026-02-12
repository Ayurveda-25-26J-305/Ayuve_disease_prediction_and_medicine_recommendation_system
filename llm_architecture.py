import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)




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

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            trust_remote_code=True
        )

        self.model.eval()
        logger.info("LLM initialized successfully")

    def generate(self, prompt: str, max_new_tokens: int = None) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        print(" Generating response...")

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.config.get("max_new_tokens", 64),
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the newly generated tokens (excluding the input prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_ids = output_ids[0][input_length:]
        
        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()
        
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

            block = f"""
[Source {i}]
Book: {book}
Chapter: {chapter}
Paragraph: {paragraph}

{doc.get("text", "")}
"""
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
