# Troubleshooting Guide - Enhanced Features

## Error: 'DynamicCache' object has no attribute 'seen_tokens'

### Problem
This is a compatibility issue between transformers library versions and the Phi-3 model.

### ✅ Solution Applied

**Fixed in `enhanced_rag.py`:**
- Added `use_cache=False` to generation parameters
- Removed conflicting `temperature` and `top_p` parameters

### Quick Fix Commands

**Option 1: Try running again** (fix already applied):
```bash
py test_enhanced_features.py
```

**Option 2: Update transformers** (if still having issues):
```bash
pip install --upgrade transformers
```

Or specify exact version:
```bash
pip install "transformers>=4.40.0"
```

**Option 3: Use venv (if not activated)**:
```bash
venv\Scripts\activate   # On Windows
python test_enhanced_features.py
```

---

## Other Common Errors

### Error: Device mismatch (CUDA/CPU warning)
**Status:** ⚠️ Warning only, not critical
**Impact:** Slight performance decrease
**Solution:** Can be ignored for CPU-only systems

### Error: Generation takes too long
**Cause:** Running on CPU without quantization
**Solution:** 
- Expected time: 10-30 seconds per answer on CPU
- For faster results: Use GPU or quantize model (see `quantization_quickstart.md`)

### Error: Module not found
```bash
pip install -r requirements.txt
```

---

## Testing After Fix

Try running tests in order:

### 1. Basic LLM Test (fastest)
```bash
py test_basic_llm.py
```
Expected: ~1 minute, should show generated answer

### 2. Full Enhanced Features Test
```bash
py test_enhanced_features.py
```
Expected: ~5-7 minutes for all 3 tests

---

## Expected Output

### Test 1: Multi-Source Validation
```
✅ Confidence: 70-80% (medium to high)
📚 Sources checked: 5
✓ Sources agree: 4/5
```

### Test 2: Personalization
```
✨ Using template-based personalization (fast & reliable)
👤 Personalized for: vata constitution
```

### Test 3: Combined
```
Both features working together
Confidence: ~75%
Personalized: true
```

---

## If Still Failing

1. **Check transformers version:**
   ```bash
   pip show transformers
   ```
   Should be >= 4.40.0

2. **Reinstall if needed:**
   ```bash
   pip uninstall transformers
   pip install "transformers>=4.40.0"
   ```

3. **Try in clean venv:**
   ```bash
   python -m venv new_venv
   new_venv\Scripts\activate
   pip install -r requirements.txt
   py test_enhanced_features.py
   ```

---

## Performance Benchmarks

- **Embedding Generation:** ~1-2 seconds
- **LLM Generation (CPU):** ~10-30 seconds  
- **Validation:** ~2-3 seconds
- **Personalization:** <1 second (template-based)
- **Total per query:** ~15-40 seconds on CPU

GPU speeds everything up 5-10x.
