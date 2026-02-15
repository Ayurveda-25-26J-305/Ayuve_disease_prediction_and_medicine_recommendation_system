"""
Build Complete FAISS Vector Database
Loads all books and QA dataset
"""

import os
import pandas as pd
from vector_db_setup import FAISSVectorDB
from data_preprocessing import TextChunker, BookProcessor, QADatasetProcessor

def build_complete_database():
    """Build vector database from all books and QA dataset"""
    
    print("=" * 70)
    print("Building Complete FAISS Vector Database")
    print("=" * 70)
    
    # Initialize vector database
    vector_db = FAISSVectorDB(
        embedding_model_name="BAAI/bge-base-en-v1.5",
        index_path="faiss_index"
    )
    
    # Initialize processors
    book_processor = BookProcessor(chunk_size=300, overlap=50)
    qa_processor = QADatasetProcessor()
    
    all_documents = []
    
    # Process all books
    print("\n📚 Processing Books...")
    print("-" * 70)
    
    books_dir = "data/books"
    if os.path.exists(books_dir):
        book_files = [f for f in os.listdir(books_dir) if f.endswith('.txt')]
        print(f"Found {len(book_files)} book files")
        
        for book_file in book_files:
            book_path = os.path.join(books_dir, book_file)
            book_name = book_file.replace('.txt', '').replace('_', ' ').title()
            
            print(f"\n  Processing: {book_name}")
            try:
                chunks = book_processor.process_text_file(
                    file_path=book_path,
                    book_name=book_name
                )
                all_documents.extend(chunks)
                print(f"    ✓ Created {len(chunks)} chunks")
            except Exception as e:
                print(f"    ❌ Error: {e}")
    else:
        print(f"❌ Books directory not found: {books_dir}")
    
    # Process QA dataset
    print("\n\n📄 Processing QA Dataset...")
    print("-" * 70)
    
    qa_file = "data/Ayurvedic_QA_Dataset.csv"
    if os.path.exists(qa_file):
        try:
            qa_docs = qa_processor.process_csv(qa_file)
            all_documents.extend(qa_docs)
            print(f"  ✓ Loaded {len(qa_docs)} QA pairs")
        except Exception as e:
            print(f"  ❌ Error processing QA dataset: {e}")
    else:
        print(f"❌ QA dataset not found: {qa_file}")
    
    # Add all documents to vector database
    print("\n\n💾 Building Vector Database...")
    print("-" * 70)
    print(f"Total documents to add: {len(all_documents)}")
    
    if all_documents:
        vector_db.add_documents(all_documents)
        print(f"✓ Added {len(all_documents)} documents")
        
        # Save index
        vector_db.save_index()
        print("✓ Saved FAISS index")
        
        # Show statistics
        stats = vector_db.get_stats()
        print("\n📊 Database Statistics:")
        print(f"  Total Documents: {stats['total_documents']}")
        print(f"  Embedding Dimension: {stats['embedding_dimension']}")
        print(f"  Document Types:")
        for doc_type, count in stats['document_types'].items():
            print(f"    - {doc_type}: {count}")
        
        # Test search
        print("\n\n🔍 Testing Search...")
        print("-" * 70)
        test_query = "What causes Vata imbalance?"
        results = vector_db.search(test_query, top_k=3)
        
        print(f"Query: '{test_query}'")
        print("\nTop 3 Results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [Similarity: {result['score']:.4f}]")
            print(f"   Text: {result['text'][:150]}...")
            print(f"   Source: {result['source']}")
        
        print("\n" + "=" * 70)
        print("✅ Vector Database Built Successfully!")
        print("=" * 70)
    else:
        print("❌ No documents to process!")

if __name__ == "__main__":
    build_complete_database()
