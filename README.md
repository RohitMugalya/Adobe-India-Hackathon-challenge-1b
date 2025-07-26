# Challenge 1b: Multi-Collection PDF Analysis

## Overview
Advanced PDF analysis solution using transformer models and semantic similarity to extract relevant content from document collections based on specific personas and tasks. The system operates completely offline using locally cached models.

## Project Structure
```
Challenge_1b/
├── Collection 1/                    # Travel Planning
│   ├── PDFs/                       # South of France guides
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output_transformers.json  # Analysis results
├── Collection 2/                    # Adobe Acrobat Learning
│   ├── PDFs/                       # Acrobat tutorials
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output_transformers.json  # Analysis results
├── Collection 3/                    # Recipe Collection
│   ├── PDFs/                       # Cooking guides
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output_transformers.json  # Analysis results
├── models/                          # Cached transformer models
├── pdf_analyzer_transformers.py    # Main analysis tool
├── model_init.py                   # Model download script
├── approach_explanation.md         # Methodology documentation
├── requirements.txt                # Python dependencies
└── README.md
```

## Collections

### Collection 1: Travel Planning
- **Challenge ID**: round_1b_002
- **Persona**: Travel Planner
- **Task**: Plan a 4-day trip for 10 college friends to South of France
- **Documents**: 7 travel guides

### Collection 2: Adobe Acrobat Learning
- **Challenge ID**: round_1b_003
- **Persona**: HR Professional
- **Task**: Create and manage fillable forms for onboarding and compliance
- **Documents**: 15 Acrobat guides

### Collection 3: Recipe Collection
- **Challenge ID**: round_1b_001
- **Persona**: Food Contractor
- **Task**: Prepare vegetarian buffet-style dinner menu for corporate gathering
- **Documents**: 9 cooking guides

## Technical Approach

### Core Technologies
- **BART Model**: `facebook/bart-base` for text generation and content extraction
- **Sentence Transformers**: `all-mpnet-base-v2` for semantic similarity analysis
- **Cross-Encoder**: `ms-marco-MiniLM-L-6-v2` for document re-ranking (optional)
- **Cosine Similarity**: For document relevance detection and importance ranking

### Key Features
- **Semantic Document Filtering**: Uses cosine similarity to identify relevant documents
- **Hybrid Ranking System**: Combines bi-encoder and cross-encoder models for accurate importance ranking
- **CPU-Optimized**: Automatically uses all available CPU cores for parallel processing
- **Offline Operation**: All models cached locally, no internet required after setup
- **Generalized Approach**: Works with any persona/task combination without hardcoding

## Installation and Setup

### Prerequisites
- **Python 3.11.4** (recommended for optimal performance)
- **Docker** (for containerized deployment)
- **Internet connection** (for initial model download only)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Models (One-time setup)
```bash
# Download and cache all required models
python model_init.py

# Check if models are already downloaded
python model_init.py --check

# Force re-download models
python model_init.py --force
```

This will download and cache:
- `facebook/bart-base` (~500MB)
- `all-mpnet-base-v2` (~420MB)
- `ms-marco-MiniLM-L-6-v2` (~90MB, optional)

### Step 3: Run Analysis
```bash
# Process any collection (automatically uses all CPU cores)
python pdf_analyzer_transformers.py "Collection 1"
python pdf_analyzer_transformers.py "Collection 2"
python pdf_analyzer_transformers.py "Collection 3"
```

## Input/Output Format

### Input JSON Structure
```json
{
  "challenge_info": {
    "challenge_id": "round_1b_XXX",
    "test_case_name": "specific_test_case"
  },
  "documents": [{"filename": "doc.pdf", "title": "Title"}],
  "persona": {"role": "User Persona"},
  "job_to_be_done": {"task": "Use case description"}
}
```

### Output JSON Structure
```json
{
  "metadata": {
    "input_documents": ["list of all input documents"],
    "persona": "User Persona",
    "job_to_be_done": "Task description",
    "model_used": "facebook/bart-base",
    "similarity_model": "all-mpnet-base-v2 + cross-encoder",
    "device": "cpu",
    "parallel_workers": 8
  },
  "extracted_sections": [
    {
      "document": "relevant_doc.pdf",
      "section_title": "Most Important Section",
      "importance_rank": 1,
      "page_number": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "relevant_doc.pdf",
      "refined_text": "Detailed content analysis...",
      "page_number": 1
    }
  ]
}
```

## How It Works

### 1. Document Relevance Detection
- Converts document filenames to semantic embeddings
- Compares against task description using cosine similarity
- Filters out documents with similarity < 0.3 threshold

### 2. Importance Ranking
- Ranks relevant documents by semantic similarity to the task
- Applies cross-encoder re-ranking for improved accuracy
- Assigns unique ranks from 1 (most important) to N (least important)

### 3. Content Extraction
- Extracts section titles using regex patterns and BART generation
- Analyzes first 5 pages of each relevant document
- Generates coherent paragraph summaries for subsection analysis

### 4. Parallel Processing
- Automatically detects and uses all available CPU cores
- Thread-safe processing with concurrent document analysis
- Optimized for CPU-only environments

## Output Files

The tool generates `challenge1b_output_transformers.json` in each collection directory containing:
- **extracted_sections**: One entry per relevant document (no duplicates)
- **subsection_analysis**: Detailed content analysis for relevant documents only
- **metadata**: Processing information and model details

## Offline Operation

After initial setup, the system runs completely offline:
- All models cached in `models/` directory
- No internet connection required for analysis
- Portable across different machines

## Performance

- **CPU Optimized**: Uses all available cores automatically
- **Memory Efficient**: Loads models once, processes multiple documents
- **Fast Processing**: Parallel document analysis with semantic filtering
- **Scalable**: Handles collections of any size

## Troubleshooting

### Models Not Found
```bash
# Re-download models
python model_init.py --force
```

### Memory Issues
- Ensure sufficient RAM (minimum 4GB recommended)
- Close other applications during processing

### Slow Performance
- The system automatically uses all CPU cores
- Processing time depends on document count and CPU speed

## Docker Deployment (Adobe India Hackathon 2025 - Round 1B)

### Quick Start with Docker

#### 1. Build the Docker Image
```bash
# Build the Docker image for AMD64 architecture
docker build --platform linux/amd64 -t pdf-analyzer:round1b .
```

#### 2. Prepare Input Directory
Create an input directory with your collections:
```bash
mkdir -p input/Collection1/PDFs
mkdir -p input/Collection2/PDFs
mkdir -p output
```

#### 3. Run the Container
```bash
# Run the container with input/output volume mounts and no network access
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  pdf-analyzer:round1b
```

### Input Directory Structure
The input directory should contain collection folders, each with:
```
input/
├── Collection1/
│   ├── challenge1b_input.json      # Required: Contains persona and task
│   └── PDFs/                       # Required: Contains PDF documents
│       ├── document1.pdf
│       ├── document2.pdf
│       └── document3.pdf
├── Collection2/
│   ├── challenge1b_input.json
│   └── PDFs/
│       ├── guide1.pdf
│       └── guide2.pdf
└── Collection3/
    ├── challenge1b_input.json
    └── PDFs/
        └── manual.pdf
```

### Sample Input JSON Format
Each collection must have a `challenge1b_input.json` file:
```json
{
  "challenge_info": {
    "challenge_id": "round_1b_002",
    "test_case_name": "travel_planning"
  },
  "documents": [
    {"filename": "document1.pdf", "title": "Travel Guide"},
    {"filename": "document2.pdf", "title": "City Information"}
  ],
  "persona": {"role": "Travel Planner"},
  "job_to_be_done": {"task": "Plan a 4-day trip for 10 college friends"}
}
```

### Output Structure
The container generates output files in the output directory:
```
output/
├── Collection1_output.json         # Results for Collection1
├── Collection2_output.json         # Results for Collection2
└── Collection3_output.json         # Results for Collection3
```

### Sample Output JSON Format
```json
{
  "metadata": {
    "input_documents": ["document1.pdf", "document2.pdf"],
    "persona": "Travel Planner",
    "job_to_be_done": "Plan a 4-day trip for 10 college friends",
    "processing_timestamp": "2025-07-28T10:30:45.123456",
    "model_used": "facebook/bart-base",
    "similarity_model": "all-mpnet-base-v2 + cross-encoder"
  },
  "extracted_sections": [
    {
      "document": "document1.pdf",
      "section_title": "Travel Destinations",
      "importance_rank": 1,
      "page_number": 2
    }
  ],
  "subsection_analysis": [
    {
      "document": "document1.pdf",
      "refined_text": "Detailed analysis of travel destinations...",
      "page_number": 2
    }
  ]
}
```

### Hackathon Compliance (Round 1B Requirements)

#### ✅ Technical Constraints
- **Architecture**: AMD64 (x86_64) compatible
- **CPU Only**: No GPU dependencies, optimized for CPU processing
- **Model Size**: <1GB total (BART-base ~500MB + all-mpnet-base-v2 ~420MB)
- **Offline Operation**: No network access required after build
- **Processing Time**: <60 seconds per collection (3-5 documents)
- **Memory Usage**: Optimized for 16GB RAM systems
- **CPU Cores**: Automatically uses all 8 available CPU cores

#### ✅ Docker Requirements
- **Platform**: `--platform linux/amd64` compatible
- **Network**: Runs with `--network none` (offline)
- **Volumes**: Input/output directory mounting
- **Base Image**: python:3.11.4-slim for optimal performance

#### ✅ Execution Compliance
```bash
# Expected build command
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .

# Expected run command
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  mysolutionname:somerandomidentifier
```

### Container Specifications
- **Base Image**: python:3.11.4-slim (AMD64)
- **Python Version**: 3.11.4 (latest stable)
- **Models**: Pre-downloaded during build (offline ready)
- **Dependencies**: All Python packages included in container
- **Working Directory**: /app
- **Input Mount Point**: /app/input
- **Output Mount Point**: /app/output
- **Execution Script**: docker_run.py (automatic processing)

### Performance Optimization
- **Parallel Processing**: Utilizes all available CPU cores
- **Memory Efficient**: Loads models once, processes multiple collections
- **Fast Startup**: Models pre-cached during build
- **Optimized Libraries**: Uses CPU-optimized versions of PyTorch and Transformers

### Testing Your Setup
Use the provided test script to verify your Docker setup:
```bash
# Create test environment
python test_docker.py

# Build and test
./build_docker.sh

# Run test
docker run --rm \
  -v $(pwd)/test_input:/app/input \
  -v $(pwd)/test_output:/app/output \
  --network none \
  pdf-analyzer:round1b
```

### Troubleshooting

#### Build Issues
- Ensure Docker supports AMD64 platform
- Check internet connection for model downloads during build
- Verify sufficient disk space (>2GB for models and dependencies)

#### Runtime Issues
- Ensure input directory has correct structure
- Check that PDF files are readable and not corrupted
- Verify output directory has write permissions

#### Performance Issues
- Container automatically uses all available CPU cores
- Processing time scales with document count and complexity
- Memory usage peaks during model loading (~2-3GB)

---

**Note**: This solution uses state-of-the-art transformer models with semantic similarity analysis to provide accurate, persona-specific PDF analysis completely offline. Built for Adobe India Hackathon 2025 - Round 1B.

---

**Note**: This solution uses the official Ollama Python library with the gemma3:1b model to analyze PDFs based on specific personas and tasks, generating structured JSON outputs for each collection. 