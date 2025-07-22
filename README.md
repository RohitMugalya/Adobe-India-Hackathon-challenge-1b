# Challenge 1b: Multi-Collection PDF Analysis

## Overview
Advanced PDF analysis solution that processes multiple document collections and extracts relevant content based on specific personas and use cases.

## Project Structure
```
Challenge_1b/
├── Collection 1/                    # Travel Planning
│   ├── PDFs/                       # South of France guides
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results
├── Collection 2/                    # Adobe Acrobat Learning
│   ├── PDFs/                       # Acrobat tutorials
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results
├── Collection 3/                    # Recipe Collection
│   ├── PDFs/                       # Cooking guides
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results
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
    "input_documents": ["list"],
    "persona": "User Persona",
    "job_to_be_done": "Task description"
  },
  "extracted_sections": [
    {
      "document": "source.pdf",
      "section_title": "Title",
      "importance_rank": 1,
      "page_number": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "source.pdf",
      "refined_text": "Content",
      "page_number": 1
    }
  ]
}
```

## Key Features
- **Persona-based content analysis** - Tailored analysis for specific roles and tasks
- **Parallel processing** - Fast document processing using multiple workers
- **Smart querying** - Individual value queries instead of JSON generation for reliability
- **Importance ranking** - AI-powered section ranking based on relevance
- **Command line interface** - Process specific collections with custom parameters
- **Structured JSON output** - Consistent output format with metadata

## Usage

### Prerequisites
1. **Ollama**: Install Ollama from https://ollama.ai/download
2. **Python 3.7+**: Ensure Python is installed on your system

### Quick Start
1. Run the setup script (optional):
   ```bash
   python run_analysis.py
   ```

2. Process a specific collection:
   ```bash
   python pdf_analyzer.py "Collection 1"
   ```

### Command Line Usage
```bash
# Process Collection 1 with default settings
python pdf_analyzer.py "Collection 1"

# Process Collection 2 with 8 parallel workers
python pdf_analyzer.py "Collection 2" --workers 8

# Process Collection 3 with a different model
python pdf_analyzer.py "Collection 3" --model gemma3:2b

# Get help
python pdf_analyzer.py --help
```

### Manual Setup
If you prefer manual setup:

1. Install Python requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Install and start Ollama:
   ```bash
   ollama pull gemma3:1b
   ollama serve
   ```

3. Run the analysis on a specific collection:
   ```bash
   python pdf_analyzer.py "Collection 1"
   ```

### Output
The tool generates `challenge1b_output_generated.json` files in each collection directory with:
- Extracted sections ranked by importance
- Refined text analysis based on the specific persona and task
- Metadata including processing timestamp

### Model Configuration
The tool uses the `gemma3:1b` model by default. You can modify the model in `pdf_analyzer.py` if needed.

### Dependencies
- **ollama**: Official Ollama Python library for seamless integration
- **PyPDF2**: PDF text extraction
- **Python 3.7+**: Core runtime

---

**Note**: This solution uses the official Ollama Python library with the gemma3:1b model to analyze PDFs based on specific personas and tasks, generating structured JSON outputs for each collection. 