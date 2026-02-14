"""
Quick Test Script - Test LLM Generation Only
This tests if the LLM can generate answers at all
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml


def test_basic_generation():
    """Test if the model can generate text"""
    print("=" * 70)
    print("BASIC LLM GENERATION TEST")
    print("=" * 70)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = config['llm_model']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n📦 Loading model: {model_name}")
    print(f"🖥️  Device: {device}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    print("✓ Model loaded")
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Simple prompt
    simple_prompt = """You are an Ayurvedic knowledge assistant.

Context:
Triphala is a traditional Ayurvedic formula consisting of three fruits. It supports digestive health and detoxification.

Question:
What is Triphala good for?

Answer:"""
    
    print(f"\n📝 Prompt ({len(simple_prompt)} chars):")
    print("-" * 70)
    print(simple_prompt)
    print("-" * 70)
    
    # Tokenize
    inputs = tokenizer(
        simple_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(device)
    
    print(f"\n🔢 Input tokens: {inputs['input_ids'].shape[1]}")
    
    # Generate
    print("\n🔄 Generating...")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,  # More tokens for testing
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("\n" + "=" * 70)
    print("FULL OUTPUT:")
    print("=" * 70)
    print(full_output)
    print("=" * 70)
    
    # Try to extract answer
    if "Answer:" in full_output:
        parts = full_output.split("Answer:")
        answer = parts[-1].strip()
    else:
        answer = full_output[len(simple_prompt):].strip()
    
    print("\n📤 EXTRACTED ANSWER:")
    print("=" * 70)
    print(answer)
    print("=" * 70)
    
    if answer and len(answer) > 10:
        print("\n✅ SUCCESS: Model is generating text correctly")
        return True
    else:
        print("\n❌ FAILED: Model is not generating proper answers")
        print(f"   Answer length: {len(answer)}")
        return False


if __name__ == "__main__":
    import sys
    success = test_basic_generation()
    sys.exit(0 if success else 1)
