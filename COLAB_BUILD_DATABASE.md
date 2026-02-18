# Build Vector Database in Google Colab

Copy this cell into your Colab notebook **BEFORE** running the backend setup.

```python
"""
Build FAISS Vector Database in Colab
This builds the database from your Ayurvedic texts
"""

print("=" * 70)
print("📦 STEP 1: Upload Required Files")
print("=" * 70)

from google.colab import files
import os
import zipfile

# Create directories
os.makedirs('data/books', exist_ok=True)
os.makedirs('faiss_index', exist_ok=True)

print("\n📁 Please upload these files when prompted:")
print("  1. vector_db_setup.py")
print("  2. data_preprocessing.py")
print("  3. build_vector_db.py")
print("  4. All .txt files from data/books/ folder")
print("  5. Ayurvedic_QA_Dataset.csv")
print("\nYou can select multiple files at once!\n")

uploaded = files.upload()

# Organize files
for filename in uploaded.keys():
    if filename.endswith('.txt'):
        # Move book files to data/books/
        os.rename(filename, f'data/books/{filename}')
        print(f"✓ Moved {filename} to data/books/")
    elif filename == 'Ayurvedic_QA_Dataset.csv':
        # Move CSV to data/
        os.rename(filename, f'data/{filename}')
        print(f"✓ Moved {filename} to data/")
    else:
        # Python files stay in root
        print(f"✓ Uploaded {filename}")

print("\n" + "=" * 70)
print("🔧 STEP 2: Install Dependencies")
print("=" * 70)

!pip install -q sentence-transformers faiss-cpu pandas

print("✓ Dependencies installed")

print("\n" + "=" * 70)
print("🏗️ STEP 3: Build Vector Database")
print("=" * 70)
print("This will take 2-5 minutes...\n")

# Run the build script
!python build_vector_db.py

print("\n" + "=" * 70)
print("✅ VECTOR DATABASE IS READY!")
print("=" * 70)
print("\nYou can now run the backend setup cell.")
print("The RAG system will use this database to answer questions.")
print("=" * 70)
```

## Files You Need to Upload:

**From your project root:**

- `vector_db_setup.py`
- `data_preprocessing.py`
- `build_vector_db.py`

**From `data/books/` folder:**

- `astanga_hridaya.txt`
- `Ayurvedha_ancient_wisdom_for_modern_life.txt`
- `Ayurvedha_food_and_nutritions.txt`
- `everyday_Ayurvedha.txt`
- `Hela_osu_corpus.txt`

**From `data/` folder:**

- `Ayurvedic_QA_Dataset.csv`

**Total: 9 files** (select them all at once when Colab shows the upload dialog)

## What This Does:

1. ✅ Uploads all necessary files
2. ✅ Installs sentence-transformers and FAISS
3. ✅ Processes all 5 Ayurvedic books
4. ✅ Processes the QA dataset
5. ✅ Creates embeddings using BAAI/bge-base-en-v1.5
6. ✅ Builds FAISS index
7. ✅ Saves to `faiss_index/` folder
8. ✅ Tests the database with a sample query

## Expected Output:

```
Building Complete FAISS Vector Database
======================================================================

📚 Processing Books...
----------------------------------------------------------------------
Found 5 book files

  Processing: Astanga Hridaya
    ✓ Created 456 chunks
  Processing: Ayurvedha Ancient Wisdom For Modern Life
    ✓ Created 312 chunks
  ...

📄 Processing QA Dataset...
----------------------------------------------------------------------
  ✓ Loaded 500 QA pairs

💾 Building Vector Database...
----------------------------------------------------------------------
Total documents to add: 2921
Generating embeddings for 2921 texts...
✓ Added 2921 documents
✓ Saved FAISS index

📊 Database Statistics:
  Total Documents: 2921
  Embedding Dimension: 768
  Document Types:
    - book: 2421
    - qa: 500

✅ Vector Database Built Successfully!
======================================================================
```

The entire process takes about **3-5 minutes** in Colab.
