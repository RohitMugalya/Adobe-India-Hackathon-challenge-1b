#!/usr/bin/env python3
"""
Runner script for PDF Analysis
Handles setup and execution of the PDF analysis tool
"""

import subprocess
import sys
import os
import time

def check_ollama():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Ollama is installed")
            return True
        else:
            print("✗ Ollama is not installed")
            return False
    except FileNotFoundError:
        print("✗ Ollama is not installed")
        print("Please install Ollama from: https://ollama.ai/download")
        return False

def ensure_model_available():
    """Ensure the gemma3:1b model is available"""
    print("Checking/installing gemma3:1b model...")
    try:
        # Check if model exists
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if 'gemma3:1b' in result.stdout:
            print("✓ gemma3:1b model is already available")
            return True
        
        # Pull the model if not available
        print("Downloading gemma3:1b model (this may take a few minutes)...")
        result = subprocess.run(['ollama', 'pull', 'gemma3:1b'], check=True)
        print("✓ Model installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install model: {e}")
        return False

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install requirements")
        return False

def ensure_ollama_running():
    """Ensure Ollama server is running"""
    try:
        # Try to connect using the ollama library
        import ollama
        client = ollama.Client()
        client.list()
        print("✓ Ollama server is running")
        return True
    except Exception:
        print("Starting Ollama server...")
        try:
            # Start ollama serve in background
            subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Wait for server to start
            time.sleep(3)
            
            # Test connection again
            client = ollama.Client()
            client.list()
            print("✓ Ollama server started successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to start/connect to Ollama server: {e}")
            return False

def main():
    print("PDF Analysis Tool Setup and Runner")
    print("=" * 40)
    
    # Check Ollama installation
    if not check_ollama():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Ensure model is available
    if not ensure_model_available():
        return
    
    # Ensure Ollama server is running
    if not ensure_ollama_running():
        return
    
    # Run the analysis
    print("\nStarting PDF analysis...")
    print("=" * 40)
    
    try:
        subprocess.run([sys.executable, 'pdf_analyzer.py'], check=True)
        print("\n✓ Analysis completed successfully!")
    except subprocess.CalledProcessError:
        print("\n✗ Analysis failed")

if __name__ == "__main__":
    main()