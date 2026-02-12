# Ayurvedic RAG System - Google Colab Deployment
# Easy to test and modify in real-time

## 📋 SETUP INSTRUCTIONS

### 1. Create New Colab Notebook
- Go to: https://colab.research.google.com
- File → New notebook
- Copy cells below into notebook

### 2. Upload Your Files
Upload these files when prompted:
- enhanced_rag_gpu.py
- llm_architecture.py
- validation_engine.py
- personalization_engine.py
- vector_db_setup.py
- config.yaml
- faiss_index/ folder (compressed as zip)
- data/ folder (compressed as zip)

---

## 🔧 COLAB NOTEBOOK CELLS

### Cell 1: Check GPU
```python
# Run this first to confirm you have GPU
!nvidia-smi
```

### Cell 2: Install Dependencies
```python
!pip install -q transformers==4.46.3
!pip install -q torch torchvision torchaudio
!pip install -q sentence-transformers
!pip install -q faiss-cpu
!pip install -q pyyaml
!pip install -q gradio

print("✅ Dependencies installed")
```

### Cell 3: Upload Project Files
```python
from google.colab import files
import zipfile
import os

# Upload your files
print("📁 Please upload your project files...")
uploaded = files.upload()

# Extract zips if any
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✅ Extracted {filename}")

print("✅ All files uploaded")
```

### Cell 4: Initialize System (EDITABLE - Test different configs here)
```python
import yaml
from vector_db_setup import FAISSVectorDB
from enhanced_rag_gpu import EnhancedAyurvedicRAG

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load vector DB
print("📂 Loading vector database...")
vector_db = FAISSVectorDB(
    embedding_model_name=config['embedding_model'],
    index_path=config['index_path']
)
vector_db.load_index()
print(f"✅ Loaded {vector_db.index.ntotal} documents")

# Initialize RAG
# 🔧 MODIFY THESE PARAMETERS TO TEST:
print("🤖 Initializing RAG...")
rag = EnhancedAyurvedicRAG(
    llm_model_name=config['llm_model'],
    max_new_tokens=150,  # ← CHANGE THIS to test different lengths
    enable_validation=True,  # ← TOGGLE to enable/disable
    enable_personalization=True  # ← TOGGLE to enable/disable
)
print("✅ System ready!")
```

### Cell 5: Create User Profile (EDITABLE - Test different Doshas)
```python
# 🔧 CHANGE RESPONSES TO TEST DIFFERENT DOSHAS:
# A = Vata, B = Pitta, C = Kapha

responses = {
    'q1': 'B', 'q2': 'B', 'q3': 'B', 'q4': 'B', 'q5': 'B',
    'q6': 'B', 'q7': 'B', 'q8': 'B', 'q9': 'B', 'q10': 'B'
}

profile = rag.create_user_profile(
    user_id='test_user',
    prakriti_responses=responses
)

print(f"✅ Profile: {profile['dominant_dosha']} constitution")
print(f"✅ Season: {profile['current_season']}")
```

### Cell 6: Test Single Question (EDITABLE - Test your questions)
```python
# 🔧 CHANGE QUESTION HERE:
question = "What are the benefits of Triphala?"

print(f"\n❓ Question: {question}")
print("⏳ Generating answer...\n")

result = rag.answer_question(
    question=question,
    vector_db=vector_db,
    top_k=3,  # ← CHANGE to retrieve more/fewer sources
    user_profile=profile,
    validation_top_k=5  # ← CHANGE validation sources
)

# Display results
print("=" * 70)
print("📝 ANSWER:")
print("=" * 70)
print(result['answer'])

print("\n" + "=" * 70)
print("📊 VALIDATION:")
print("=" * 70)
val = result['validation']
print(f"Confidence: {val['confidence']}% ({val['confidence_level']})")
print(f"Sources: {val['sources_agree']}/{val['sources_checked']}")

print("\n" + "=" * 70)
print("📚 SOURCES:")
print("=" * 70)
for i, cite in enumerate(result['citations'], 1):
    print(f"{i}. {cite['source']}, Chapter {cite['chapter']}")
```

### Cell 7: Interactive Gradio UI (OPTIONAL)
```python
import gradio as gr

def answer_question_ui(question, dosha_type):
    # Create profile based on dosha
    if dosha_type == "Vata":
        responses = {f'q{i}': 'A' for i in range(1, 11)}
    elif dosha_type == "Pitta":
        responses = {f'q{i}': 'B' for i in range(1, 11)}
    else:  # Kapha
        responses = {f'q{i}': 'C' for i in range(1, 11)}
    
    profile = rag.create_user_profile('ui_user', responses)
    
    # Get answer
    result = rag.answer_question(
        question=question,
        vector_db=vector_db,
        top_k=3,
        user_profile=profile,
        validation_top_k=5
    )
    
    # Format output
    output = f"""
📝 ANSWER:
{result['answer']}

📊 VALIDATION:
Confidence: {result['validation']['confidence']}% ({result['validation']['confidence_level']})
Sources checked: {result['validation']['sources_checked']}
Agreement: {result['validation']['sources_agree']}/{result['validation']['sources_checked']}

📚 SOURCES:
"""
    for i, cite in enumerate(result['citations'], 1):
        output += f"\n{i}. {cite['source']}, Chapter {cite['chapter']}"
    
    return output

# Create interface
demo = gr.Interface(
    fn=answer_question_ui,
    inputs=[
        gr.Textbox(label="Ask your Ayurvedic question", placeholder="What are the benefits of Triphala?"),
        gr.Radio(["Vata", "Pitta", "Kapha"], label="Your Constitution (Dosha)", value="Pitta")
    ],
    outputs=gr.Textbox(label="Answer", lines=20),
    title="🌿 Ayurvedic RAG System",
    description="Ask questions about Ayurveda with personalized answers validated across multiple sources"
)

# Launch with shareable link
demo.launch(share=True)  # ← Creates public link for 72 hours!
```

---

## 🔧 HOW TO ITERATE:

### **Testing Answer Quality:**
If answers are too short/long, edit Cell 4:
```python
# Change this line:
max_new_tokens=150,  # Try 100, 200, etc.
```
Then: Shift+Enter to re-run Cell 4, then re-run Cell 6

### **Testing Different Questions:**
Edit Cell 6:
```python
question = "How to balance Vata dosha?"  # Change question
```
Then: Shift+Enter to re-run only Cell 6

### **Testing Validation:**
Edit Cell 4:
```python
enable_validation=False,  # Disable to test without
```
Then: Re-run Cell 4 and Cell 6

### **Testing Different Files:**
If you modified a Python file locally:
1. Upload new version in Cell 3
2. Restart runtime (Runtime → Restart runtime)
3. Re-run all cells

---

## ✅ ADVANTAGES:

1. **Fast Iteration** - Edit and re-run in seconds
2. **No Re-deployment** - Changes are immediate
3. **Free GPU** - Test on T4 for free
4. **Shareable** - Share link with supervisors
5. **Version Control** - Save different notebook versions

---

## 🎯 WORKFLOW EXAMPLE:

```
1. Upload files → Run all cells → System works!
2. Notice answer too short → Edit max_new_tokens=200
3. Re-run Cell 4 & 6 → Better!
4. Want to test Vata → Edit Cell 5 responses to 'A'
5. Re-run Cell 5 & 6 → See Vata recommendations
6. Share public link with supervisor → Demo successful!
7. Supervisor suggests improvement → Edit code in Colab
8. Re-run cells → Show improved version immediately
```

---

## 💡 PRO TIPS:

**Save Your Work:**
- File → Save to Drive (automatic backups)
- Download → .ipynb (keep local copy)

**Faster Testing:**
- Don't restart runtime unless necessary
- Models stay loaded (saves 2-3 minutes)
- Just re-run changed cells

**Debugging:**
Add print statements anywhere:
```python
print(f"🔍 Generated {len(answer)} characters")
print(f"🔍 Confidence: {confidence}%")
```

---

Would you like me to create the complete `.ipynb` notebook file ready to upload to Colab? It will have all these cells pre-configured so you can start testing immediately!
