"""
Rebuild Vector Database with Improved Chapter Detection
Run this to rebuild the vector database with proper chapter/verse metadata
"""

import os
import shutil
from build_vector_db import main as build_db

def rebuild_database():
    """
    Rebuild the entire vector database
    """
    print("=" * 60)
    print("REBUILDING VECTOR DATABASE")
    print("=" * 60)
    
    # Backup old database if it exists
    if os.path.exists('vector_db'):
        backup_path = 'vector_db_backup'
        if os.path.exists(backup_path):
            print(f"Removing old backup: {backup_path}")
            shutil.rmtree(backup_path)
        
        print(f"Backing up current database to: {backup_path}")
        shutil.copytree('vector_db', backup_path)
        
        # Remove old database
        print("Removing old database...")
        shutil.rmtree('vector_db')
    
    # Build new database
    print("\nBuilding new vector database with improved chapter detection...")
    build_db()
    
    print("\n" + "=" * 60)
    print("✅ DATABASE REBUILD COMPLETE!")
    print("=" * 60)
    print("\nThe vector database has been rebuilt with improved:")
    print("- Chapter detection (supports multiple formats)")
    print("- Verse/section numbering")
    print("- Metadata extraction")
    print("\nRestart your Flask backend to use the new database.")

if __name__ == "__main__":
    rebuild_database()
