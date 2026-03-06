# Complete Colab Setup Guide

## CELL 1: Build Vector Database

**Copy and paste this into a new Colab cell:**

```python
from google.colab import files
import os

print("=" * 70)
print("📦 STEP 1: Upload Files")
print("=" * 70)

# Create directories
os.makedirs('data/books', exist_ok=True)
os.makedirs('backend', exist_ok=True)
os.makedirs('faiss_index', exist_ok=True)

print("\n📁 Upload these files (select all at once):")
print("  - vector_db_setup.py")
print("  - data_preprocessing.py")
print("  - build_vector_db.py")
print("  - config.yaml")
print("  - backend/app_enhanced.py")
print("  - enhanced_rag_gpu.py")
print("  - validation_engine.py")
print("  - personalization_engine.py")
print("  - 5 .txt book files")
print("  - Ayurvedic_QA_Dataset.csv\n")

uploaded = files.upload()

# Organize files
for filename in uploaded.keys():
    if filename.endswith('.txt'):
        os.rename(filename, f'data/books/{filename}')
        print(f"✓ {filename} → data/books/")
    elif filename == 'Ayurvedic_QA_Dataset.csv':
        os.rename(filename, f'data/{filename}')
        print(f"✓ {filename} → data/")
    elif filename == 'app_enhanced.py':
        os.rename(filename, f'backend/{filename}')
        print(f"✓ {filename} → backend/")
    else:
        print(f"✓ {filename}")

print("\n" + "=" * 70)
print("🔧 STEP 2: Install Dependencies")
print("=" * 70)
```

**Then run this in the same cell or next cell:**

```python
!pip install -q sentence-transformers faiss-cpu pandas transformers torch flask flask-cors pyyaml

print("✓ Dependencies installed")

print("\n" + "=" * 70)
print("🏗️ STEP 3: Build Vector Database")
print("=" * 70)
print("This takes 3-5 minutes...\n")

!python build_vector_db.py

print("\n" + "=" * 70)
print("✅ VECTOR DATABASE READY!")
print("=" * 70)
print("\nNow run CELL 2 to start the backend")
```

---

## CELL 2: Start Backend with Localtunnel

**Copy and paste this into a new Colab cell:**

```python
import subprocess
import time
import threading

print("=" * 70)
print("🔄 Starting Backend with Localtunnel")
print("=" * 70)

# Kill existing processes
print("\n1️⃣ Cleanup...")
subprocess.run(['pkill', '-f', 'lt'], stderr=subprocess.DEVNULL)
subprocess.run(['pkill', '-f', 'flask'], stderr=subprocess.DEVNULL)
time.sleep(2)
print("   ✅ Done")

# Install Localtunnel
print("\n2️⃣ Installing Localtunnel...")
subprocess.run(['apt-get', 'update', '-qq'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(['apt-get', 'install', '-y', '-qq', 'nodejs', 'npm'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(['npm', 'install', '-g', 'localtunnel'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("   ✅ Ready")

# Start Flask
print("\n3️⃣ Starting Flask...")

def run_flask():
    subprocess.run(['python', 'backend/app_enhanced.py'])

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
time.sleep(10)
print("   ✅ Flask running on localhost:5000")

# Start Localtunnel with FIXED subdomain
print("\n4️⃣ Starting Localtunnel...")
tunnel_process = subprocess.Popen(
    ['lt', '--port', '5000', '--subdomain', 'ayurvedic-qa'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True
)

time.sleep(5)

print("\n" + "=" * 70)
print("✅ BACKEND IS RUNNING!")
print("=" * 70)
print("\n   🌐 URL: https://ayurvedic-qa.loca.lt")
print("\n   Frontend is already configured!")
print("=" * 70)
print("\n🔄 Backend active... Keep this cell running!")
print("   Press ⏹️ to stop.\n")

try:
    tunnel_process.wait()
except KeyboardInterrupt:
    print("\n🛑 Stopping...")
    tunnel_process.kill()
    print("✅ Stopped")
```

---

## Files to Upload (14 total):

1. `vector_db_setup.py`
2. `data_preprocessing.py`
3. `build_vector_db.py`
4. `config.yaml`
5. `backend/app_enhanced.py`
6. `enhanced_rag_gpu.py`
7. `validation_engine.py`
8. `personalization_engine.py`
9. `astanga_hridaya.txt`
10. `Ayurvedha_ancient_wisdom_for_modern_life.txt`
11. `Ayurvedha_food_and_nutritions.txt`
12. `everyday_Ayurvedha.txt`
13. `Hela_osu_corpus.txt`
14. `Ayurvedic_QA_Dataset.csv`

---

## On Windows (After Colab is running):

```powershell
cd "c:\Users\Mahen Fernando\Desktop\Research - 4th year\Research- Vector_DB\frontend\ayurvedic-qa-app"
npm run dev
```

Then open browser at `http://localhost:3000` and test!

---

## URL is FIXED:

✅ Always: `https://ayurvedic-qa.loca.lt`  
✅ No need to update `.env.local` again  
✅ Works every time you restart Colab
