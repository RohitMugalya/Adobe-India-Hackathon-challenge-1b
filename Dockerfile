# Use official Python runtime as base image with AMD64 architecture
FROM --platform=linux/amd64 python:3.11.4-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY pdf_analyzer_transformers.py .
COPY model_init.py .
COPY approach_explanation.md .

# Create necessary directories
RUN mkdir -p /app/input /app/output /app/models

# Download and cache models during build (offline operation)
RUN python model_init.py

# Copy the main execution script
COPY docker_run.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command to run the application
CMD ["python", "docker_run.py"]