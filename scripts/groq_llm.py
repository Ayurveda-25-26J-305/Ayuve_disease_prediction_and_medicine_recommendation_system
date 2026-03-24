"""
GroqLLM - Drop-in replacement for LLMArchitecture using the Groq cloud API.

Exposes the same interface as LLMArchitecture:
    .generate(prompt, max_new_tokens)       -> str
    .generate_from_messages(messages, ...) -> str
    .device                                -> "cpu"

No GPU required. Starts in < 2 seconds.
"""

import os
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Build Phi-3 special tokens at runtime to avoid encoding issues in this file
_SYS_TOK  = "<|" + "system|>"
_END_TOK  = "<|" + "end|>"
_USR_TOK  = "<|" + "user|>"
_ASST_TOK = "<|" + "assistant|>"
_PHI3_SPECIAL = [_SYS_TOK, _END_TOK, _USR_TOK, _ASST_TOK]


class GroqLLM:
    """
    Thin Groq SDK wrapper that acts as a drop-in for LLMArchitecture.
    EnhancedAyurvedicRAG, PersonalizationEngine, TranslationService all work
    without modification.
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): keys used:
                groq_api_key  - your Groq API key
                groq_model    - model id, e.g. "llama-3.3-70b-versatile"
                max_new_tokens - default generation budget
        """
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "groq package not installed. Run:  pip install groq"
            )

        config_api_key = str(config.get("groq_api_key", "")).strip()
        if config_api_key == "YOUR_API_KEY_HERE":
            config_api_key = ""
        self.api_key        = config_api_key or os.getenv("GROQ_API_KEY", "").strip()
        self.model          = config.get("groq_model", "llama-3.3-70b-versatile")
        self.max_new_tokens = int(config.get("max_new_tokens", 512))
        self.device         = "cpu"   # TranslationService reads this attribute

        if not self.api_key:
            raise ValueError(
                "groq_api_key is empty. Add it to config.yaml or set the "
                "GROQ_API_KEY environment variable."
            )

        self._client = Groq(api_key=self.api_key)
        logger.info(f"GroqLLM initialised  model={self.model}")
        print(f"[GroqLLM] Connected   model={self.model}  device=cpu")

    # ── private helpers ───────────────────────────────────────────────────────

    def _strip_phi3_tokens(self, text: str) -> str:
        """Remove Phi-3 chat-template tokens from a legacy prompt string."""
        for tok in _PHI3_SPECIAL:
            text = text.replace(tok, "")
        # Clean up excessive blank lines left behind
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _phi3_prompt_to_messages(self, prompt: str) -> List[Dict[str, str]]:
        """
        Parse a Phi-3 formatted prompt string into an OpenAI-style messages list.

        Phi-3 format (tokens shown symbolically):
            SYS_TOKEN ... system text ... END_TOKEN
            USR_TOKEN ... user text ... END_TOKEN
            ASST_TOKEN

        Returns list of {role, content} dicts understood by Groq API.
        """
        messages: List[Dict[str, str]] = []

        # Try structured parse first
        sys_match = re.search(
            re.escape(_SYS_TOK) + r"(.*?)" + re.escape(_END_TOK),
            prompt, re.DOTALL
        )
        usr_match = re.search(
            re.escape(_USR_TOK) + r"(.*?)" + re.escape(_END_TOK),
            prompt, re.DOTALL
        )

        if sys_match:
            sys_content = sys_match.group(1).strip()
            if sys_content:
                messages.append({"role": "system", "content": sys_content})

        if usr_match:
            usr_content = usr_match.group(1).strip()
            if usr_content:
                messages.append({"role": "user", "content": usr_content})

        # Fallback: treat the whole (stripped) prompt as a user message
        if not messages:
            clean = self._strip_phi3_tokens(prompt)
            messages.append({
                "role": "system",
                "content": (
                    "You are an expert Ayurvedic knowledge assistant. "
                    "Answer accurately and concisely using the provided context."
                )
            })
            messages.append({"role": "user", "content": clean})

        return messages

    def _call_groq(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float = 0.5,
        top_p: float = 0.9,
    ) -> str:
        """Make the actual Groq API call with retry on transient errors."""
        import time

        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False,
                )
                content = response.choices[0].message.content or ""
                return content.strip()
            except Exception as e:
                err = str(e)
                if "rate_limit" in err.lower() and attempt < 2:
                    wait = (attempt + 1) * 5
                    print(f"[GroqLLM] Rate limit hit — retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"Groq API error: {e}")
                    return ""
        return ""

    # ── public interface (mirrors LLMArchitecture) ────────────────────────────

    def generate(self, prompt: str, max_new_tokens: int = None) -> str:
        """
        Legacy plain-text prompt path.
        Called by _generate_dynamic_personalized_tips and other helpers.
        """
        max_tokens = max_new_tokens or self.max_new_tokens
        print(f"[GroqLLM] generate()  max_tokens={max_tokens}")

        # Convert Phi-3 style prompts to messages; otherwise wrap as user message
        if any(tok in prompt for tok in _PHI3_SPECIAL):
            messages = self._phi3_prompt_to_messages(prompt)
        else:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert Ayurvedic knowledge assistant. "
                        "Provide accurate, concise answers based on the context given."
                    )
                },
                {"role": "user", "content": prompt},
            ]

        answer = self._call_groq(messages, max_tokens)
        print(f"[GroqLLM] Generated {len(answer)} chars")
        return answer if answer else "Unable to generate an answer. Please try again."

    def generate_from_messages(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = None,
    ) -> str:
        """
        Preferred path for chat-style generation.
        Called by EnhancedAyurvedicRAG.answer_question() and
        PersonalizationEngine.
        """
        max_tokens = max_new_tokens or self.max_new_tokens
        print(f"[GroqLLM] generate_from_messages()  msgs={len(messages)}  max_tokens={max_tokens}")

        answer = self._call_groq(messages, max_tokens)
        print(f"[GroqLLM] Generated {len(answer)} chars")
        return answer if answer else "Unable to generate an answer. Please try again."
