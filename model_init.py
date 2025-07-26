#!/usr/bin/env python3
"""
Model Initialization Script for PDF Analyzer
Downloads and caches required models for offline usage
"""

import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder

def download_models():
    """Download and cache all required models for the PDF analyzer"""
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("Starting model download and initialization...")
    print("This may take several minutes depending on your internet connection.")
    
    try:
        # Download BART model
        print("\n1. Downloading facebook/bart-base model...")
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/bart-base",
            cache_dir=str(models_dir)
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-base",
            cache_dir=str(models_dir),
            torch_dtype=torch.float32  # Ensure CPU compatibility
        )
        print("   BART model downloaded successfully")
        
        # Download sentence transformer model
        print("\n2. Downloading all-mpnet-base-v2 model...")
        sentence_models_dir = models_dir / "sentence_transformers"
        sentence_models_dir.mkdir(exist_ok=True)
        
        bi_encoder = SentenceTransformer(
            'all-mpnet-base-v2',
            cache_folder=str(sentence_models_dir),
            device='cpu'  # Ensure CPU usage
        )
        print("   Sentence transformer model downloaded successfully")
        
        # Download cross-encoder model (optional, skip if size constraints)
        print("\n3. Downloading cross-encoder model...")
        try:
            cross_encoder = CrossEncoder(
                'cross-encoder/ms-marco-MiniLM-L-6-v2',
                device='cpu'
            )
            print("   Cross-encoder model downloaded successfully")
        except Exception as e:
            print(f"   Warning: Cross-encoder download failed: {e}")
            print("   The system will work with bi-encoder only")
        
        print("\nModel initialization completed successfully!")
        print(f"Models cached in: {models_dir.absolute()}")
        
        # Verify model paths and check sizes
        print("\nVerifying model paths and sizes...")
        total_size = 0
        
        bart_snapshots = models_dir / "models--facebook--bart-base" / "snapshots"
        if bart_snapshots.exists():
            snapshots = list(bart_snapshots.iterdir())
            if snapshots:
                print(f"   BART model snapshot: {snapshots[0].name}")
                # Calculate size
                for file in snapshots[0].rglob('*'):
                    if file.is_file():
                        total_size += file.stat().st_size
        
        sentence_snapshots = sentence_models_dir / "models--sentence-transformers--all-mpnet-base-v2" / "snapshots"
        if sentence_snapshots.exists():
            snapshots = list(sentence_snapshots.iterdir())
            if snapshots:
                print(f"   Sentence transformer snapshot: {snapshots[0].name}")
                # Calculate size
                for file in snapshots[0].rglob('*'):
                    if file.is_file():
                        total_size += file.stat().st_size
        
        total_size_mb = total_size / (1024 * 1024)
        print(f"   Total model size: {total_size_mb:.1f} MB")
        
        if total_size_mb > 1000:  # 1GB constraint
            print("   Warning: Model size exceeds 1GB constraint")
        else:
            print("   Model size within 1GB constraint")
        
        print("\nModels are ready for offline usage!")
        
    except Exception as e:
        print(f"Error during model download: {e}")
        print("Please check your internet connection and try again.")
        sys.exit(1)

def check_models():
    """Check if models are already downloaded"""
    models_dir = Path("models")
    
    bart_path = models_dir / "models--facebook--bart-base"
    sentence_path = models_dir / "sentence_transformers" / "models--sentence-transformers--all-mpnet-base-v2"
    
    if bart_path.exists() and sentence_path.exists():
        print("Models already exist in the cache.")
        print("Use --force to re-download models.")
        return True
    return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize models for PDF Analyzer")
    parser.add_argument("--force", action="store_true", help="Force re-download even if models exist")
    parser.add_argument("--check", action="store_true", help="Check if models are already downloaded")
    args = parser.parse_args()
    
    if args.check:
        if check_models():
            print("All required models are available.")
        else:
            print("Models need to be downloaded. Run: python model_init.py")
        return
    
    if not args.force and check_models():
        return
    
    download_models()

if __name__ == "__main__":
    main()