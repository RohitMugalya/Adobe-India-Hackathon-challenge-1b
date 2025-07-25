# PDF Analysis Methodology

## Overview

Our PDF analysis system employs a hybrid approach combining semantic similarity analysis with transformer-based text generation to extract and rank relevant information from document collections based on specific personas and tasks.

## Core Methodology

### 1. Document Relevance Detection

The system uses **cosine similarity** with the `all-mpnet-base-v2` sentence transformer model to determine document relevance. Each document's filename is converted into a semantic representation and compared against the task description using the formula:

```
task_embedding = encode("A {persona} working on {task}")
doc_embedding = encode("A document about {document_type}")
similarity_score = cosine_similarity(task_embedding, doc_embedding)
```

Documents with similarity scores below a threshold (0.3) are filtered out as irrelevant.

### 2. Hybrid Document Ranking

We implement a two-stage ranking system:

**Stage 1: Bi-encoder Retrieval**
- Generate embeddings for all relevant documents
- Calculate cosine similarities with the task description
- Create initial ranking based on similarity scores

**Stage 2: Cross-encoder Re-ranking (Optional)**
- Apply cross-encoder model (`ms-marco-MiniLM-L-6-v2`) to top candidates
- Combine scores using weighted average (60% cross-encoder, 40% bi-encoder)
- Produce final importance rankings from 1 to N

### 3. Content Extraction and Analysis

**Section Title Identification:**
- Extract potential headings using regex patterns for capitalized text
- Filter out common non-section elements (page numbers, copyright notices)
- Generate fallback titles using BART model when extraction fails

**Subsection Analysis:**
- Process first 5 pages of each relevant document
- Apply keyword-based relevance filtering for efficiency
- Use BART model for content summarization and extraction
- Generate coherent paragraph summaries focused on persona-specific needs

### 4. Text Generation with BART

The system leverages `facebook/bart-base` for:
- Section title generation when PDF parsing fails
- Content relevance assessment through yes/no prompts
- Key information extraction and summarization
- Coherent paragraph generation for subsection analysis

**Generation Parameters:**
- Temperature: 0.7 for balanced creativity and consistency
- Top-p: 0.9 for nucleus sampling
- Repetition penalty: 1.1 to avoid redundant content
- Beam search with early stopping for quality

### 5. Parallel Processing Architecture

The system implements CPU-optimized parallel processing:
- Automatic detection of available CPU cores using `multiprocessing.cpu_count()`
- ThreadPoolExecutor for concurrent document processing
- Thread-safe logging and result aggregation
- Optimized for CPU-only environments as per problem constraints

### 6. Output Structure

The final output follows a strict JSON schema:
- **extracted_sections**: One section per relevant document with unique importance ranks
- **subsection_analysis**: Detailed content analysis for documents in extracted_sections
- **metadata**: Processing information including model details and execution metrics

## Technical Advantages

1. **Semantic Understanding**: Cosine similarity provides more accurate relevance detection than keyword matching
2. **Scalability**: Parallel processing adapts to available hardware resources
3. **Robustness**: Multiple fallback mechanisms ensure consistent output quality
4. **Offline Operation**: All models cached locally for deployment without internet dependency
5. **Generalization**: Works across different document types, personas, and tasks without hardcoded assumptions

This methodology ensures accurate, efficient, and scalable PDF analysis while maintaining high-quality output suitable for various professional applications.