"""
Main Pipeline for Ayurvedic QA System
Complete workflow: Data Loading -> Vector DB Setup -> RAG System
Author: Research Project - 4th Year
Date: December 30, 2025
"""

import argparse
import json
import os
from typing import Optional

from vector_db_setup import FAISSVectorDB
from llm_architecture import AyurvedicRAG
from data_preprocessing import BookProcessor, QADatasetProcessor, TextChunker


# ============================================================
# Citation Utilities (BOOK-ONLY)
# ============================================================

def format_citation(source_doc: dict) -> str:
    """
    Format citation with book name, chapter and paragraph
    QA datasets are NEVER formatted as citations
    """
    metadata = source_doc.get("metadata", {})
    book = source_doc.get("source", "Unknown Book")

    chapter = metadata.get("chapter", "N/A")
    paragraph = metadata.get("paragraph", metadata.get("verse", "N/A"))

    return f"{book} – Chapter {chapter} – Paragraph {paragraph}"


def filter_book_sources(sources: list) -> list:
    """
    Return only book sources if available,
    otherwise fallback to original list
    """
    book_sources = [s for s in sources if s.get("type") == "book"]
    return book_sources if book_sources else sources


# ============================================================
# Pipeline Class
# ============================================================

class AyurvedicQAPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.vector_db = None
        self.rag_system = None

        print("=" * 70)
        print("Ayurvedic QA System Pipeline")
        print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: DATA PREPARATION
    # --------------------------------------------------------

    def step1_prepare_data(self):
        print("\n[STEP 1] Data Preparation")
        print("-" * 70)

        all_documents = []

        # Books
        if self.config.get("books_directory"):
            print("\n Processing Ayurvedic Books...")
            book_processor = BookProcessor(
                chunker=TextChunker(
                    chunk_size=self.config.get("chunk_size", 300),
                    overlap=self.config.get("overlap", 50)
                )
            )

            books_dir = self.config["books_directory"]
            if os.path.exists(books_dir):
                book_chunks = book_processor.process_directory(
                    directory=books_dir,
                    book_mapping=self.config.get("book_mapping")
                )
                all_documents.extend(book_chunks)
                print(f" Processed {len(book_chunks)} book chunks")
            else:
                print(f" Books directory not found: {books_dir}")

        # QA Dataset
        if self.config.get("qa_dataset_path"):
            print("\n Processing Q&A Dataset...")
            qa_processor = QADatasetProcessor()
            qa_path = self.config["qa_dataset_path"]

            if os.path.exists(qa_path):
                if qa_path.endswith(".json"):
                    qa_docs = qa_processor.process_json(qa_path)
                elif qa_path.endswith(".csv"):
                    qa_docs = qa_processor.process_csv(qa_path)
                else:
                    qa_docs = []

                all_documents.extend(qa_docs)
                print(f" Processed {len(qa_docs)} QA pairs")
            else:
                print(f" QA dataset not found: {qa_path}")

        print(f"\n Total documents prepared: {len(all_documents)}")
        return all_documents

    # --------------------------------------------------------
    # STEP 2: VECTOR DB
    # --------------------------------------------------------

    def step2_setup_vector_db(self, documents):
        print("\n[STEP 2] Vector Database Setup")
        print("-" * 70)

        self.vector_db = FAISSVectorDB(
            embedding_model_name=self.config.get("embedding_model"),
            index_path=self.config.get("index_path", "faiss_index")
        )

        if self.config.get("load_existing_index", False):
            print("\n Attempting to load existing index...")
            if self.vector_db.load_index():
                print(" Loaded existing index")
                return

        if documents:
            print(f"\n Adding {len(documents)} documents...")
            self.vector_db.add_documents(documents)

            print("\n Saving FAISS index...")
            self.vector_db.save_index()
        else:
            print(" No documents to add")

    # --------------------------------------------------------
    # STEP 3: LLM
    # --------------------------------------------------------

    def step3_initialize_llm(self):
        print("\n[STEP 3] LLM Initialization")
        print("-" * 70)

        self.rag_system = AyurvedicRAG(
            llm_model_name=self.config.get("llm_model"),
            max_new_tokens=self.config.get("max_new_tokens", 512),
           
        )

        print("✓ LLM initialized")

    # --------------------------------------------------------
    # STEP 4: TESTING
    # --------------------------------------------------------

    def step4_test_system(self):
        print("\n[STEP 4] System Testing")
        print("-" * 70)

        for question in self.config.get("test_questions", []):
            print("\n Question:", question)

            response = self.rag_system.answer_question(
                question=question,
                vector_db=self.vector_db,
                top_k=self.config.get("top_k", 5)
            )

            print("\n Answer:")
            print(response["answer"])

            sources = filter_book_sources(response["sources"])

            if sources:
                print("\n Citation Source:")
                print(" ", format_citation(sources[0]))
            else:
                print("\n No authoritative book source found")

    # --------------------------------------------------------
    # RUN PIPELINE
    # --------------------------------------------------------

    def run_pipeline(self):
        docs = self.step1_prepare_data()
        self.step2_setup_vector_db(docs)
        self.step3_initialize_llm()

        if self.config.get("run_tests", True):
            self.step4_test_system()

        print("\n Pipeline completed successfully")

    # --------------------------------------------------------
    # INTERACTIVE MODE
    # --------------------------------------------------------

    def interactive_mode(self):
        print("\n Interactive QA Mode (type 'quit' to exit)")
        print("=" * 70)

        while True:
            question = input("\n Your question: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                break

            response = self.rag_system.answer_question(
                question=question,
                vector_db=self.vector_db,
                top_k=self.config.get("top_k", 5)
            )

            print("\n Answer:")
            print(response["answer"])

            sources = filter_book_sources(response["sources"])

            if sources:
                print("\n Citation Source:")
                print(" ", format_citation(sources[0]))
            else:
                print("\n No authoritative book source found")


# ============================================================
# CONFIG LOADING + ENTRY
# ============================================================

def load_config(config_path: Optional[str] = None) -> dict:
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            if config_path.endswith(".json"):
                return json.load(f)
            else:
                import yaml
                return yaml.safe_load(f)

    return {}


def main():
    parser = argparse.ArgumentParser(description="Ayurvedic QA System")
    parser.add_argument("--config", type=str)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--load-index", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.load_index:
        config["load_existing_index"] = True

    pipeline = AyurvedicQAPipeline(config)
    pipeline.run_pipeline()

    if args.interactive:
        pipeline.interactive_mode()


if __name__ == "__main__":
    main()
