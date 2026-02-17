"""
Translation Test Script
Run this in Colab to diagnose translation issues
"""

print("="*70)
print("TRANSLATION SERVICE DIAGNOSTIC TEST")
print("="*70)

# Test 1: Check dependencies
print("\n1. Checking dependencies...")
try:
    import transformers
    print("   ✓ transformers installed:", transformers.__version__)
except ImportError as e:
    print("   ✗ transformers NOT installed:", e)
    print("   → Run: !pip install transformers")

try:
    import sentencepiece
    print("   ✓ sentencepiece installed")
except ImportError as e:
    print("   ✗ sentencepiece NOT installed:", e)
    print("   → Run: !pip install sentencepiece")

try:
    import langdetect
    print("   ✓ langdetect installed")
except ImportError as e:
    print("   ✗ langdetect NOT installed:", e)
    print("   → Run: !pip install langdetect")

# Test 2: Import translation service
print("\n2. Importing translation_service...")
try:
    from translation_service import TranslationService
    print("   ✓ TranslationService imported successfully")
except Exception as e:
    print("   ✗ Import failed:", e)
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Initialize service
print("\n3. Initializing TranslationService...")
try:
    translator = TranslationService(device="cuda")
    if translator.is_available():
        print("   ✓ TranslationService initialized and available")
    else:
        print("   ✗ TranslationService initialized but NOT available")
        print("   → Models may not have loaded correctly")
except Exception as e:
    print("   ✗ Initialization failed:", e)
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Test romanized Singlish detection
print("\n4. Testing romanized Singlish detection...")
test_questions = [
    "kurudu wala guna monawada?",
    "කුරුඳු වල ගුණ මොනවද?",
    "what are the benefits of cinnamon?"
]

for question in test_questions:
    print(f"\n   Question: {question}")
    detected = translator.detect_language(question)
    print(f"   Detected language: {detected}")
    
    if detected == 'si':
        is_romanized = not translator._has_sinhala_chars(question)
        print(f"   Is romanized: {is_romanized}")
        
        if is_romanized:
            is_singlish = translator._is_romanized_singlish(question)
            print(f"   Is romanized Singlish: {is_singlish}")

# Test 5: Test keyword translation
print("\n5. Testing keyword-based translation...")
test_romanized = "kurudu wala guna monawada?"
print(f"   Input: {test_romanized}")

try:
    translated = translator._translate_romanized_keywords(test_romanized)
    print(f"   ✓ Keyword translation: {translated}")
except Exception as e:
    print(f"   ✗ Translation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test full translation pipeline
print("\n6. Testing full translation pipeline...")
print(f"   Input: {test_romanized}")

try:
    is_romanized = not translator._has_sinhala_chars(test_romanized)
    english_translation = translator.translate_si_to_en(test_romanized, is_romanized=is_romanized)
    print(f"   ✓ English translation: {english_translation}")
    
    # Try translating back to Sinhala
    sinhala_text = "Cinnamon helps regulate blood sugar levels and supports digestive health."
    sinhala_translation = translator.translate_en_to_si(sinhala_text)
    print(f"   ✓ Sinhala translation: {sinhala_translation[:100]}...")
    
    # Check if output is garbled
    if len(sinhala_translation) > 0:
        # Check for mixed scripts (sign of problem)
        has_sinhala = bool(__import__('re').search(r'[\u0D80-\u0DFF]', sinhala_translation))
        has_other = bool(__import__('re').search(r'[\u0B80-\u0BFF\u0C80-\u0CFF\u0D00-\u0D7F]', sinhala_translation))
        
        if has_sinhala and not has_other:
            print("   ✓ Sinhala translation looks clean (only Sinhala script)")
        elif has_other:
            print("   ✗ WARNING: Translation contains mixed scripts (garbled)")
        else:
            print("   ⚠️  Translation doesn't contain Sinhala (might be English)")
            
except Exception as e:
    print(f"   ✗ Pipeline test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
print("\nIf you see any ✗ errors above, fix them before running the backend.")
print("Common fixes:")
print("  - Install missing packages: !pip install transformers sentencepiece langdetect")
print("  - Restart runtime after installing")
print("  - Check GPU availability: import torch; print(torch.cuda.is_available())")
