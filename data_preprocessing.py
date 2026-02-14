"""
Data Preprocessing Utilities for Ayurvedic QA System
Handles text chunking, book processing, and dataset preparation
Author: Research Project - 4th Year
Date: December 30, 2025
"""

import re
import json
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path


class TextChunker:
    """
    Intelligent text chunking for Ayurvedic books
    """
    
    def __init__(self, 
                 chunk_size: int = 300,
                 overlap: int = 50,
                 separator: str = "\n"):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Target tokens per chunk (200-400 recommended)
            overlap: Number of overlapping tokens between chunks
            separator: Text separator for splitting
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation)
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4
    
    def chunk_text(self, 
                   text: str, 
                   source: str,
                   metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with automatic paragraph numbering
        
        Args:
            text: Input text to chunk
            source: Source identifier (e.g., book name, chapter)
            metadata: Additional metadata
            
        Returns:
            List of chunked documents
        """
        # Clean text
        text = text.strip()
        
        # Split by separator
        sentences = text.split(self.separator)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_number = 1
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_tokens = self.estimate_tokens(sentence)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                
                # Add chunk with paragraph number
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata['paragraph'] = chunk_number
                
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "type": "book",
                    "metadata": chunk_metadata
                })
                
                chunk_number += 1
                
                # Keep overlap
                overlap_tokens = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    overlap_tokens += self.estimate_tokens(sent)
                    overlap_sentences.insert(0, sent)
                    if overlap_tokens >= self.overlap:
                        break
                
                current_chunk = overlap_sentences
                current_tokens = overlap_tokens
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['paragraph'] = chunk_number
            
            chunks.append({
                "text": chunk_text,
                "source": source,
                "type": "book",
                "metadata": chunk_metadata
            })
        
        return chunks


class BookProcessor:
    """
    Process Ayurvedic books into structured chunks
    """
    
    def __init__(self, chunker: Optional[TextChunker] = None):
        """
        Initialize book processor
        
        Args:
            chunker: TextChunker instance
        """
        self.chunker = chunker or TextChunker()
    
    def process_text_file(self, 
                         file_path: str,
                         book_name: str) -> List[Dict[str, Any]]:
        """
        Process a text file into chunks with automatic chapter detection
        
        Args:
            file_path: Path to text file
            book_name: Name of the book
            
        Returns:
            List of document chunks
        """
        print(f"Processing: {book_name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Try to detect chapters automatically
        chapter_pattern = r'(?:Chapter|CHAPTER|Ch\.|Ch)\s+(\d+)'
        chapters = re.split(f'({chapter_pattern})', text, flags=re.IGNORECASE)
        
        all_chunks = []
        
        if len(chapters) > 1:
            # Book has chapter structure
            print(f"   Detected {(len(chapters)-1)//3} chapters")
            
            current_chapter = 0
            for i in range(0, len(chapters)):
                chunk_text = chapters[i].strip()
                
                # Check if this is a chapter marker
                chapter_match = re.match(chapter_pattern, chunk_text, re.IGNORECASE)
                if chapter_match:
                    current_chapter = int(chapter_match.group(1))
                    continue
                
                if chunk_text and current_chapter > 0:
                    # Process chapter content
                    chunks = self.chunker.chunk_text(
                        text=chunk_text,
                        source=book_name,
                        metadata={"chapter": current_chapter}
                    )
                    all_chunks.extend(chunks)
        else:
            # No chapter structure, just chunk the whole book
            all_chunks = self.chunker.chunk_text(
                text=text,
                source=book_name,
                metadata={}
            )
        
        print(f"  Created {len(all_chunks)} chunks")
        return all_chunks
    
    def process_structured_text(self,
                               file_path: str,
                               book_name: str) -> List[Dict[str, Any]]:
        """
        Process structured text with chapter/verse information
        
        Expected format:
        # Chapter 1
        ## Verse 1
        Text here...
        
        Args:
            file_path: Path to structured text file
            book_name: Name of the book
            
        Returns:
            List of document chunks
        """
        print(f"Processing structured: {book_name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse chapters and verses
        chapters = re.split(r'^# Chapter (\d+)', content, flags=re.MULTILINE)
        
        all_chunks = []
        
        for i in range(1, len(chapters), 2):
            chapter_num = chapters[i]
            chapter_content = chapters[i + 1] if i + 1 < len(chapters) else ""
            
            # Split by verses
            verses = re.split(r'^## Verse (\d+)', chapter_content, flags=re.MULTILINE)
            
            if len(verses) > 1:
                for j in range(1, len(verses), 2):
                    verse_num = verses[j]
                    verse_text = verses[j + 1].strip() if j + 1 < len(verses) else ""
                    
                    if verse_text:
                        all_chunks.append({
                            "text": verse_text,
                            "source": book_name,
                            "type": "book",
                            "metadata": {
                                "chapter": int(chapter_num),
                                "verse": int(verse_num)
                            }
                        })
            else:
                # No verse structure, chunk the chapter
                chunks = self.chunker.chunk_text(
                    text=chapter_content.strip(),
                    source=book_name,
                    metadata={"chapter": int(chapter_num)}
                )
                all_chunks.extend(chunks)
        
        print(f"   Created {len(all_chunks)} chunks")
        return all_chunks
    
    def process_directory(self, 
                         directory: str,
                         book_mapping: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Process all text files in a directory
        
        Args:
            directory: Path to directory containing book files
            book_mapping: Optional mapping of filename -> book name
            
        Returns:
            List of all document chunks
        """
        all_chunks = []
        path = Path(directory)
        
        for file_path in path.glob("*.txt"):
            book_name = book_mapping.get(file_path.name, file_path.stem) if book_mapping else file_path.stem
            chunks = self.process_text_file(str(file_path), book_name)
            all_chunks.extend(chunks)
        
        return all_chunks


class QADatasetProcessor:
    """
    Process Q&A datasets for the vector database
    """
    
    def __init__(self):
        """Initialize QA processor"""
        pass
    
    def process_json(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process JSON Q&A dataset
        
        Expected format:
        [
            {
                "question": "What causes Vata imbalance?",
                "answer": "Cold weather, irregular eating...",
                "id": 1
            },
            ...
        ]
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of Q&A documents
        """
        print(f"Processing Q&A dataset: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        for item in data:
            # Create combined Q&A text for better retrieval
            qa_text = f"Q: {item['question']}\nA: {item['answer']}"
            
            documents.append({
                "text": qa_text,
                "source": "QA Dataset",
                "type": "qa",
                "metadata": {
                    "question_id": item.get('id'),
                    "question": item['question'],
                    "answer": item['answer']
                }
            })
        
        print(f"   Processed {len(documents)} Q&A pairs")
        return documents
    
    def process_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process CSV Q&A dataset
        
        Expected columns: question, answer, (optional: id)
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of Q&A documents
        """
        print(f"Processing Q&A dataset: {file_path}")
        
        df = pd.read_csv(file_path)
        
        documents = []
        
        for idx, row in df.iterrows():
            qa_text = f"Q: {row['question']}\nA: {row['answer']}"
            
            documents.append({
                "text": qa_text,
                "source": "QA Dataset",
                "type": "qa",
                "metadata": {
                    "question_id": row.get('id', idx),
                    "question": row['question'],
                    "answer": row['answer']
                }
            })
        
        print(f"   Processed {len(documents)} Q&A pairs")
        return documents


def main():
    """
    Example usage of preprocessing utilities
    """
    print("=" * 60)
    print("Data Preprocessing - Example")
    print("=" * 60)
    
    # Example 1: Text chunking
    print("\n1. Text Chunking Example")
    print("-" * 60)
    
    chunker = TextChunker(chunk_size=200, overlap=50)
    
    sample_text = """
    Vata is the principle of movement and governs all biological activities.
    It is composed of air and ether elements.
    When balanced, Vata promotes creativity and flexibility.
    When imbalanced, it causes anxiety, dry skin, and digestive issues.
    Cold weather, irregular eating, and excessive travel aggravate Vata.
    """
    
    chunks = chunker.chunk_text(
        text=sample_text,
        source="Example Text",
        metadata={"chapter": 1}
    )
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} (Paragraph {chunk['metadata']['paragraph']}):")
        print(chunk['text'][:100] + "...")
    
    # Example 2: Book processing
    print("\n\n2. Book Processing Example")
    print("-" * 60)
    print("Expected usage:")
    print("""
    processor = BookProcessor()
    chunks = processor.process_text_file(
        file_path="path/to/ashtanga_hridaya.txt",
        book_name="Ashtanga Hridaya"
    )
    """)
    
    # Example 3: Q&A processing
    print("\n\n3. Q&A Dataset Processing Example")
    print("-" * 60)
    print("Expected usage:")
    print("""
    qa_processor = QADatasetProcessor()
    qa_docs = qa_processor.process_json("path/to/qa_dataset.json")
    # OR
    qa_docs = qa_processor.process_csv("path/to/qa_dataset.csv")
    """)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
