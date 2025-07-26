#!/bin/bash

# Build script for Round 1B PDF Analyzer Docker container
# Adobe India Hackathon 2025

set -e

echo "Building PDF Analyzer Docker Container for Round 1B"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Build the Docker image
echo "Building Docker image..."
docker build --platform linux/amd64 -t pdf-analyzer:round1b .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker image built successfully!"
    echo ""
    echo "Image details:"
    docker images pdf-analyzer:round1b
    echo ""
    echo "To run the container:"
    echo "docker run --rm \\"
    echo "  -v \$(pwd)/input:/app/input \\"
    echo "  -v \$(pwd)/output:/app/output \\"
    echo "  --network none \\"
    echo "  pdf-analyzer:round1b"
    echo ""
    echo "Make sure your input directory contains collection folders with:"
    echo "  - challenge1b_input.json"
    echo "  - PDFs/ directory with PDF files"
else
    echo "❌ Docker build failed!"
    exit 1
fi