# GPU vs CPU Execution Guide

## Your Situation

You have an **RTX 2050 GPU** (4GB VRAM) and Phi-3-mini (3.8GB model).

The model is being **offloaded to CPU** because it's too large for your GPU memory.

---

## Options

### Option 1: Force GPU (May Crash)
**Risk:** OOM (Out of Memory) errors
**Speed:** Fastest if it works (~5-10x faster)

**Try this:**
```bash
# Use the GPU-forced version
# Edit test_enhanced_features.py, change line 16:
# from enhanced_rag import EnhancedAyurvedicRAG  # OLD
from enhanced_rag_gpu import EnhancedAyurvedicRAG  # NEW
```

Then run:
```bash
py test_enhanced_features.py
```

If you get OOM errors, go to Option 2.

---

### Option 2: Keep Auto Mode (Current - Safest)
**What happens:** Model uses GPU where possible, CPU when needed
**Speed:** Slower than pure GPU, but stable
**Status:** Already working

This is what's happening now. It's slower but won't crash.

---

### Option 3: Use 4-bit Quantization (BEST)
**Result:** Model shrinks to ~1GB, fits fully on GPU
**Speed:** Fastest + most memory efficient
**Quality:** Slightly lower but acceptable

**Steps:**
```bash
pip install bitsandbytes accelerate

# Edit enhanced_rag.py, add to model loading:
load_in_4bit=True,
bnb_4bit_compute_dtype=torch.float16,
```

See `quantization_quickstart.md` for full instructions.

---

## Recommendation

For your RTX 2050 (4GB):
1. **Best:** Use Option 3 (quantization) - fits on GPU, fast
2. **Current:** Keep auto mode - slower but stable
3. **Try if curious:** Option 1 (force GPU) - might OOM

Want me to implement Option 3 (quantization)?
