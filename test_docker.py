#!/usr/bin/env python3
"""
Test script to verify Docker container functionality
"""

import json
import os
import tempfile
import shutil
from pathlib import Path

def create_test_input():
    """Create a test input structure for Docker testing"""
    
    # Create temporary input directory
    input_dir = Path("test_input")
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir()
    
    # Create test collection
    collection_dir = input_dir / "TestCollection"
    collection_dir.mkdir()
    
    # Create PDFs directory
    pdfs_dir = collection_dir / "PDFs"
    pdfs_dir.mkdir()
    
    # Create test input JSON
    test_input = {
        "challenge_info": {
            "challenge_id": "round_1b_test",
            "test_case_name": "docker_test"
        },
        "documents": [
            {"filename": "test_doc.pdf", "title": "Test Document"}
        ],
        "persona": {"role": "Test User"},
        "job_to_be_done": {"task": "Test the Docker container functionality"}
    }
    
    with open(collection_dir / "challenge1b_input.json", 'w') as f:
        json.dump(test_input, f, indent=2)
    
    print(f"Test input created at: {input_dir.absolute()}")
    print("Note: You need to add actual PDF files to test_input/TestCollection/PDFs/")
    
    return input_dir

def create_output_dir():
    """Create output directory for testing"""
    output_dir = Path("test_output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    
    print(f"Test output directory created at: {output_dir.absolute()}")
    return output_dir

def print_docker_commands():
    """Print the Docker commands for testing"""
    print("\nDocker Test Commands:")
    print("=" * 50)
    print("1. Build the image:")
    print("   docker build --platform linux/amd64 -t pdf-analyzer:test .")
    print()
    print("2. Run the container:")
    print("   docker run --rm \\")
    print("     -v $(pwd)/test_input:/app/input \\")
    print("     -v $(pwd)/test_output:/app/output \\")
    print("     --network none \\")
    print("     pdf-analyzer:test")
    print()
    print("3. Check results:")
    print("   ls -la test_output/")
    print("   cat test_output/TestCollection_output.json")

def main():
    """Main test setup function"""
    print("Setting up Docker test environment...")
    
    input_dir = create_test_input()
    output_dir = create_output_dir()
    
    print_docker_commands()
    
    print("\nTest setup complete!")
    print("Add your PDF files to test_input/TestCollection/PDFs/ and run the Docker commands above.")

if __name__ == "__main__":
    main()