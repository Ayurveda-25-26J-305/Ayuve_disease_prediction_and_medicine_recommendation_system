"""
Custom RAG Accuracy Test - Uses Your Existing CSV Dataset
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import random

from vector_db_setup import FAISSVectorDB
from llm_architecture import AyurvedicRAG


def calculate_similarity(text1, text2, model):
    """Calculate semantic similarity"""
    emb1 = model.encode([text1])
    emb2 = model.encode([text2])
    return float(cosine_similarity(emb1, emb2)[0][0])


def test_with_csv_dataset(csv_path, num_samples=2, top_k=5):
    """
    Test accuracy using your existing CSV dataset
    
    Args:
        csv_path: Path to your Ayurvedic_QA_Dataset.csv
        num_samples: Number of random questions to test (default: 20)
        top_k: Number of documents to retrieve
    """
    
    print("\n" + "=" * 70)
    print(" RAG ACCURACY TEST - Using Your CSV Dataset")
    print("=" * 70)
    
    # Load your CSV dataset
    print(f"\n Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Found {len(df)} Q&A pairs")
    
    # Sample random questions
    if len(df) > num_samples:
        test_df = df.sample(n=num_samples, random_state=42)
        print(f" Testing with {num_samples} random samples")
    else:
        test_df = df
        print(f" Testing with all {len(df)} questions")
    
    # Load RAG system
    print("\n Loading vector database...")
    vector_db = FAISSVectorDB()
    if not vector_db.load_index():
        print(" No index found. Run main_pipeline.py first!")
        return
    
    print("\n Initializing RAG system...")
    rag_system = AyurvedicRAG(
        llm_model_name="microsoft/Phi-3-mini-4k-instruct",
        max_new_tokens=150
    )
    
    print("\n Loading evaluation model...")
    eval_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    
    # Test each question
    print("\n" + "=" * 70)
    print(" RUNNING TESTS")
    print("=" * 70)
    
    results = []
    
    for idx, row in test_df.iterrows():
        question = row['question']
        expected_answer = row['answer']
        
        print(f"\n[{len(results)+1}/{len(test_df)}] Testing: {question[:60]}...")
        
        try:
            # Get RAG answer
            response = rag_system.answer_question(
                question=question,
                vector_db=vector_db,
                top_k=top_k
            )
            
            generated_answer = response['answer']
            
            # Calculate semantic similarity
            similarity = calculate_similarity(expected_answer, generated_answer, eval_model)
            
            # Calculate word overlap (simple BLEU)
            expected_words = set(expected_answer.lower().split())
            generated_words = set(generated_answer.lower().split())
            
            if len(generated_words) > 0:
                bleu = len(expected_words & generated_words) / len(generated_words)
            else:
                bleu = 0.0
            
            # Calculate term coverage
            if len(expected_words) > 0:
                coverage = len(expected_words & generated_words) / len(expected_words)
            else:
                coverage = 0.0
            
            result = {
                'question': question,
                'expected_answer': expected_answer,
                'generated_answer': generated_answer,
                'semantic_similarity': similarity,
                'bleu_score': bleu,
                'term_coverage': coverage,
                'answer_length': len(generated_answer.split()),
                'retrieval_scores': [s['score'] for s in response['sources'][:3]]
            }
            
            results.append(result)
            
            print(f"   Semantic Similarity: {similarity:.3f}")
            print(f"   BLEU Score: {bleu:.3f}")
            print(f"   Term Coverage: {coverage:.3f}")
            
        except Exception as e:
            print(f"   Error: {str(e)}")
            continue
    
    # Calculate aggregate metrics
    print("\n" + "=" * 70)
    print(" FINAL RESULTS")
    print("=" * 70)
    
    avg_similarity = np.mean([r['semantic_similarity'] for r in results])
    avg_bleu = np.mean([r['bleu_score'] for r in results])
    avg_coverage = np.mean([r['term_coverage'] for r in results])
    avg_length = np.mean([r['answer_length'] for r in results])
    
    print(f"\n Answer Quality Metrics:")
    print(f"  • Semantic Similarity:  {avg_similarity:.3f}")
    print(f"  • BLEU Score:           {avg_bleu:.3f}")
    print(f"  • Term Coverage:        {avg_coverage:.3f}")
    print(f"  • Avg Answer Length:    {avg_length:.1f} words")
    
    # Interpretation
    print(f"\n Overall Performance:")
    if avg_similarity >= 0.80:
        grade = " EXCELLENT"
        status = "Production-ready! Your RAG system is highly accurate."
    elif avg_similarity >= 0.65:
        grade = " GOOD"
        status = "Solid performance. System works well."
    elif avg_similarity >= 0.50:
        grade = " FAIR"
        status = "Decent but could use optimization."
    else:
        grade = " POOR"
        status = "Needs significant improvement."
    
    print(f"  Grade: {grade}")
    print(f"  Status: {status}")
    
    # Score breakdown
    print(f"\n Score Distribution:")
    excellent = sum(1 for r in results if r['semantic_similarity'] >= 0.80)
    good = sum(1 for r in results if 0.65 <= r['semantic_similarity'] < 0.80)
    fair = sum(1 for r in results if 0.50 <= r['semantic_similarity'] < 0.65)
    poor = sum(1 for r in results if r['semantic_similarity'] < 0.50)
    
    print(f"   Excellent (≥0.80): {excellent}/{len(results)} ({excellent/len(results)*100:.1f}%)")
    print(f"   Good (0.65-0.79):  {good}/{len(results)} ({good/len(results)*100:.1f}%)")
    print(f"   Fair (0.50-0.64):  {fair}/{len(results)} ({fair/len(results)*100:.1f}%)")
    print(f"   Poor (<0.50):      {poor}/{len(results)} ({poor/len(results)*100:.1f}%)")
    
    # Save results
    output_file = 'my_accuracy_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'avg_semantic_similarity': avg_similarity,
                'avg_bleu_score': avg_bleu,
                'avg_term_coverage': avg_coverage,
                'avg_answer_length': avg_length,
                'num_tested': len(results),
                'grade': grade
            },
            'individual_results': results
        }, f, indent=2)
    
    print(f"\n Results saved to: {output_file}")
    
    # Show best and worst examples
    print(f"\n BEST Example (Highest Similarity):")
    best = max(results, key=lambda x: x['semantic_similarity'])
    print(f"  Q: {best['question'][:80]}...")
    print(f"  Similarity: {best['semantic_similarity']:.3f}")
    print(f"  Generated: {best['generated_answer'][:150]}...")
    
    print(f"\n WORST Example (Lowest Similarity):")
    worst = min(results, key=lambda x: x['semantic_similarity'])
    print(f"  Q: {worst['question'][:80]}...")
    print(f"  Similarity: {worst['semantic_similarity']:.3f}")
    print(f"  Expected: {worst['expected_answer'][:100]}...")
    print(f"  Generated: {worst['generated_answer'][:100]}...")
    
    print("\n" + "=" * 70)
    print(" Accuracy testing completed!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # Test with your CSV dataset
    # You can change num_samples to test more/fewer questions
    results = test_with_csv_dataset(
        csv_path='./data/Ayurvedic_QA_Dataset.csv',
        num_samples=2,  # Test 2 random questions
        top_k=5
    )