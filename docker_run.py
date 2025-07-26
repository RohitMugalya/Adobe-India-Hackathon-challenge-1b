#!/usr/bin/env python3
"""
Docker execution script for Round 1B PDF Analysis
Processes all input collections and generates corresponding outputs
"""

import os
import sys
import json
import shutil
from pathlib import Path
from pdf_analyzer_transformers import PDFAnalyzer

def process_collections():
    """Process all collections from /app/input and generate outputs in /app/output"""
    
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print("Error: Input directory /app/input not found")
        sys.exit(1)
    
    # Find all collection directories in input
    collections = []
    for item in input_dir.iterdir():
        if item.is_dir():
            # Check if it's a valid collection (has challenge1b_input.json and PDFs directory)
            input_json = item / "challenge1b_input.json"
            pdfs_dir = item / "PDFs"
            if input_json.exists() and pdfs_dir.exists():
                collections.append(item)
    
    if not collections:
        print("Error: No valid collections found in /app/input")
        print("Each collection should have:")
        print("  - challenge1b_input.json")
        print("  - PDFs/ directory with PDF files")
        sys.exit(1)
    
    print(f"Found {len(collections)} collection(s) to process")
    
    # Initialize the PDF analyzer
    try:
        analyzer = PDFAnalyzer()
    except Exception as e:
        print(f"Error initializing PDF analyzer: {e}")
        sys.exit(1)
    
    # Process each collection
    for collection_path in collections:
        collection_name = collection_path.name
        print(f"\nProcessing collection: {collection_name}")
        
        try:
            # Create temporary working directory for this collection
            temp_collection = Path("/tmp") / collection_name
            if temp_collection.exists():
                shutil.rmtree(temp_collection)
            shutil.copytree(collection_path, temp_collection)
            
            # Process the collection
            analyzer.process_collection(str(temp_collection))
            
            # Copy the output file to the output directory
            output_file = temp_collection / "challenge1b_output_transformers.json"
            if output_file.exists():
                # Copy to output directory with collection name prefix
                final_output = output_dir / f"{collection_name}_output.json"
                shutil.copy2(output_file, final_output)
                print(f"Output saved to: {final_output}")
            else:
                print(f"Warning: No output generated for collection {collection_name}")
            
            # Clean up temporary directory
            shutil.rmtree(temp_collection)
            
        except Exception as e:
            print(f"Error processing collection {collection_name}: {e}")
            continue
    
    print(f"\nProcessing completed. Check /app/output for results.")

def main():
    """Main execution function"""
    print("Starting Round 1B PDF Analysis Docker Container")
    print("=" * 50)
    
    # Verify model files exist
    models_dir = Path("/app/models")
    required_models = [
        "models--facebook--bart-base",
        "sentence_transformers/models--sentence-transformers--all-mpnet-base-v2"
    ]
    
    for model_path in required_models:
        full_path = models_dir / model_path
        if not full_path.exists():
            print(f"Error: Required model not found: {full_path}")
            print("Models should have been downloaded during Docker build")
            sys.exit(1)
    
    print("All required models found")
    
    # Process collections
    process_collections()

if __name__ == "__main__":
    main()