"""
FAISS Vector Database Setup for Ayurvedic QA System
Author: Research Project - 4th Year
Date: December 30, 2025
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import json


class FAISSVectorDB:
    """
    FAISS Vector Database for storing and retrieving Ayurvedic text chunks
    """
    
    def __init__(self, embedding_model_name: str = "BAAI/bge-base-en-v1.5", 
                 index_path: str = "faiss_index"):
        """
        Initialize FAISS Vector Database
        
        Args:
            embedding_model_name: HuggingFace model name for embeddings
            index_path: Directory to save/load FAISS index
        """
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index_path = index_path
        
        # Initialize FAISS index (using Inner Product for cosine similarity)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Metadata storage
        self.metadata = []
        
        # Create index directory if it doesn't exist
        os.makedirs(index_path, exist_ok=True)
        
        print(f" FAISS Vector DB initialized (dimension: {self.embedding_dim})")
    
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            
        Returns:
            Normalized embeddings array
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True  # For cosine similarity
        )
        return embeddings
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector database
        
        Args:
            documents: List of documents with 'text', 'source', 'type', etc.
                      Example: {
                          "text": "Vata is aggravated by...",
                          "source": "Ashtanga Hridaya - Chapter 1",
                          "type": "book",
                          "metadata": {...}
                      }
        """
        if not documents:
            print(" No documents to add")
            return
        
        # Extract texts
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.create_embeddings(texts)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata.extend(documents)
        
        print(f" Added {len(documents)} documents to vector DB")
        print(f"  Total documents in DB: {self.index.ntotal}")
    
    def search(self, query: str, top_k: int = 5, prefer_books: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar documents with optional book preference
        
        Args:
            query: Query text
            top_k: Number of results to return
            prefer_books: If True, prioritize book sources over QA entries
            
        Returns:
            List of retrieved documents with scores and similarity
        """
        if self.index.ntotal == 0:
            print(" Vector DB is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.create_embeddings([query])
        
        # Search with more candidates if preferring books
        search_k = top_k * 3 if prefer_books else top_k
        scores, indices = self.index.search(query_embedding, min(search_k, self.index.ntotal))
        
        # Prepare results with similarity scores
        all_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(score)
                # Convert L2 distance to similarity (0-1 scale)
                # Lower distance = higher similarity
                result['similarity'] = 1.0 / (1.0 + float(score))
                all_results.append(result)
        
        # If prefer_books, rerank to boost book sources
        if prefer_books and all_results:
            book_results = [r for r in all_results if r.get('type') == 'book']
            qa_results = [r for r in all_results if r.get('type') == 'qa']
            
            # Take top books and fill remaining with QA if needed
            book_count = min(len(book_results), max(top_k - 1, int(top_k * 0.7)))
            qa_count = top_k - book_count
            
            results = book_results[:book_count] + qa_results[:qa_count]
            results = results[:top_k]
        else:
            results = all_results[:top_k]
        
        return results
    
    def save_index(self):
        """
        Save FAISS index and metadata to disk
        """
        # Save FAISS index
        index_file = os.path.join(self.index_path, "index.faiss")
        faiss.write_index(self.index, index_file)
        
        # Save metadata
        metadata_file = os.path.join(self.index_path, "metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f" Saved FAISS index to {self.index_path}")
        print(f"  Documents: {self.index.ntotal}")
    
    def load_index(self):
        """
        Load FAISS index and metadata from disk
        """
        index_file = os.path.join(self.index_path, "index.faiss")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")
        
        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            print(" No saved index found")
            return False
        
        # Load FAISS index
        self.index = faiss.read_index(index_file)
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f" Loaded FAISS index from {self.index_path}")
        print(f"  Documents: {self.index.ntotal}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database
        """
        stats = {
            "total_documents": self.index.ntotal,
            "embedding_dimension": self.embedding_dim,
            "document_types": {}
        }
        
        # Count document types
        for doc in self.metadata:
            doc_type = doc.get('type', 'unknown')
            stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1
        
        return stats


def main():
    """
    Example usage of FAISS Vector DB
    """
    # Initialize vector database
    vector_db = FAISSVectorDB(
        embedding_model_name="BAAI/bge-base-en-v1.5",
        index_path="faiss_index"
    )
    
    # Example documents (book chunks)
    example_docs = [
        {
            "text": "Vata is aggravated by dry, cold, light, and irregular habits. It governs all movement in the body.",
            "source": "Ashtanga Hridaya - Sutrasthana - Chapter 1",
            "type": "book",
            "metadata": {"chapter": 1, "verse": "1-5"}
        },
        {
            "text": "Pitta is increased by hot, sharp, oily, and acidic substances. It controls digestion and metabolism.",
            "source": "Charaka Samhita - Sutrasthana - Chapter 2",
            "type": "book",
            "metadata": {"chapter": 2, "verse": "10-15"}
        },
        {
            "text": "What causes Vata imbalance? Excessive travel, irregular eating, cold weather, and lack of sleep.",
            "source": "QA Dataset",
            "type": "qa",
            "metadata": {"question_id": 101}
        }
    ]
    
    # Add documents
    vector_db.add_documents(example_docs)
    
    # Search
    query = "What causes Vata problems?"
    print(f"\n Searching for: '{query}'")
    results = vector_db.search(query, top_k=3)
    
    print("\n Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [Score: {result['score']:.4f}]")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Source: {result['source']}")
        print(f"   Type: {result['type']}")
    
    # Save index
    vector_db.save_index()
    
    # Get stats
    stats = vector_db.get_stats()
    print(f"\n Database Statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
