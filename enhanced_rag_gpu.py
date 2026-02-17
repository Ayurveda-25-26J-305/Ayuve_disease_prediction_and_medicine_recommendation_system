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
            doc_type = doc.get("type", "unknown")
            
            # Handle book vs QA sources differently
            if doc_type == "book":
                chapter = meta.get("chapter", "N/A")
                paragraph = meta.get("paragraph", meta.get("verse", "N/A"))
                block = f"""[Source {i}]
Book: {book}
Chapter: {chapter}
Verse/Paragraph: {paragraph}

{doc.get("text", "")}"""
            else:
                # QA dataset entry
                question = meta.get("question", "N/A")
                block = f"""[Source {i}]
Type: Ayurvedic Q&A Reference
Related Question: {question}

{doc.get("text", "")}"""
            
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
        
        return '\n'.join(formatted_lines).strip()
    
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
        
        # Retrieve documents with similarity scores (prefer book sources)
        print("🔍 Retrieving relevant sources...")
        retrieved_docs = vector_db.search(question, top_k=validation_top_k, prefer_books=True)
        
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
        
        # Dynamic token calculation - increased for complete answers
        context_length = sum(len(d.get("text", "")) for d in top_context_docs)
        question_length = len(question)
        
        # Production-grade token allocation for complete, coherent answers
        if context_length > 2000 or question_length > 100:
            dynamic_tokens = 450  # Detailed answers with proper structure
        elif context_length > 1000:
            dynamic_tokens = 400  # Moderate answers with complete points
        else:
            dynamic_tokens = 350  # Brief but complete answers
            
        print(f"📏 Dynamic tokens: {dynamic_tokens} (context: {context_length} chars)")
        
        # Build context
        context_text = self._build_context_with_citations(top_context_docs)
        
        print(f"🔍 Context preview (first 300 chars): {context_text[:300]}...")
        
        # Use Phi-3's chat format with strict formatting rules
        context_summary = context_text[:3000]  # More context for better understanding
        prompt = f"""<|system|>You are an Ayurvedic expert providing clear, professional answers based STRICTLY on the provided sources.

CONTENT RULES (MUST FOLLOW):
1. Use ONLY information from the Ayurvedic Knowledge sources provided below
2. Paraphrase the source content clearly but stay close to the original meaning
3. Do not add information not present in the sources
4. If sources mention specific terms, ingredients, or practices, include them in your answer

FORMATTING RULES (MUST FOLLOW):
1. Start each point with a bullet (•) or dash (-)
2. Write 3-5 separate points, each on a NEW LINE
3. Each point should be ONE complete sentence (15-30 words maximum)
4. Use simple, clear language - avoid technical jargon unless it appears in sources
5. DO NOT write run-on sentences or combine multiple ideas in one point
6. End each sentence with a period before starting the next point

EXAMPLE FORMAT:
- First benefit explained in one clear sentence using information from sources.
- Second benefit with specific details mentioned in the provided knowledge.
- Third benefit focusing on practical application as described in sources.<|end|>
<|user|>Ayurvedic Knowledge:
{context_summary}

Question: {question}

Based on the above sources, provide a well-structured answer with 3-5 bullet points:<|end|>
<|assistant|>
"""
        

        # Generate answer with dynamic token adjustment
        print("💭 Generating answer...")
        raw_answer = self.llm.generate(prompt, max_new_tokens=dynamic_tokens)
        
        # Post-process answer for quality and formatting
        base_answer = self._format_answer(raw_answer)
        
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
