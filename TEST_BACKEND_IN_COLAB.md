# Test Backend Answer Generation in Colab

Paste this into a **new cell in Colab** (after the backend is running):

```python
"""
Test if backend is generating answers correctly
Run this in a separate cell while the backend is running
"""

import requests
import json

# Test question
test_question = "What are the benefits of turmeric?"

print("=" * 70)
print("🧪 TESTING BACKEND")
print("=" * 70)
print(f"\n❓ Question: {test_question}\n")

# Test the /api/ask endpoint
try:
    response = requests.post(
        'http://localhost:5000/api/ask',
        json={'question': test_question},
        headers={'Content-Type': 'application/json'},
        timeout=60
    )

    print(f"📡 Response Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()

        print("\n" + "=" * 70)
        print("✅ SUCCESS! BACKEND IS WORKING!")
        print("=" * 70)

        if data.get('success'):
            print(f"\n📝 ANSWER:")
            print("-" * 70)
            print(data.get('answer', 'No answer'))
            print("-" * 70)

            print(f"\n📊 VALIDATION:")
            if data.get('validation'):
                val = data['validation']
                print(f"  Confidence: {val.get('confidence', 'N/A')}%")
                print(f"  Level: {val.get('confidence_level', 'N/A')}")
                print(f"  Sources: {val.get('sources_agree', 0)}/{val.get('sources_checked', 0)}")

            print(f"\n📚 CITATIONS:")
            citations = data.get('citations', [])
            if citations:
                for i, cite in enumerate(citations[:3], 1):
                    print(f"  {i}. {cite.get('source', 'Unknown')}")
            else:
                print("  None")

            print("\n" + "=" * 70)
            print("✅ Backend is generating answers correctly!")
            print("=" * 70)
        else:
            print(f"\n❌ Backend returned error: {data.get('error', 'Unknown error')}")
    else:
        print(f"\n❌ HTTP Error: {response.status_code}")
        print(f"Response: {response.text}")

except requests.exceptions.ConnectionError:
    print("\n❌ CONNECTION ERROR")
    print("The Flask backend is not running on localhost:5000")
    print("Make sure Cell 2 (backend) is running!")

except requests.exceptions.Timeout:
    print("\n⏱️ TIMEOUT - Answer generation took too long")
    print("This might happen on first run while models load")

except Exception as e:
    print(f"\n❌ ERROR: {e}")

print("\n" + "=" * 70)
```

---

## What this does:

1. ✅ Sends a test question directly to Flask (bypassing tunnel)
2. ✅ Shows the full response including answer, validation, citations
3. ✅ Confirms if the backend is working correctly

## If it works in Colab but not in frontend:

That means the issue is with:

- The Localtunnel connection
- CORS headers
- Frontend fetch configuration

## If it doesn't work in Colab either:

That means the issue is with:

- Vector database not loaded
- Enhanced RAG system initialization
- Missing dependencies

Run this test and tell me what output you get! 🔍
