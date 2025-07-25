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
- **Python 3.7+**
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

---

**Note**: This solution uses state-of-the-art transformer models with semantic similarity analysis to provide accurate, persona-specific PDF analysis completely offline.

---

**Note**: This solution uses the official Ollama Python library with the gemma3:1b model to analyze PDFs based on specific personas and tasks, generating structured JSON outputs for each collection. 