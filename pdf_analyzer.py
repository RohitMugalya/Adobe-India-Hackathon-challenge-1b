#!/usr/bin/env python3
"""
PDF Analysis Tool using Ollama with gemma3:1b model
Processes multiple collections of PDFs based on specific personas and tasks
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import ollama
import PyPDF2
from typing import List, Dict, Any, Tuple

class PDFAnalyzer:
    def __init__(self, model_name="smollm:135m", max_workers=4):
        self.model_name = model_name
        self.client = ollama.Client()
        self.max_workers = max_workers
        self.print_lock = Lock()  # For thread-safe printing
        
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Extract text from PDF file, organized by pages"""
        text_by_page = {}
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        text_by_page[page_num] = text.strip()
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
        return text_by_page
    
    def call_ollama(self, prompt: str, system_prompt: str = "", max_tokens: int = None) -> str:
        """Make API call to Ollama using the official library with output limits"""
        try:
            options = {}
            if max_tokens:
                options["num_predict"] = max_tokens
            
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                system=system_prompt,
                stream=False,
                options=options
            )
            return response['response']
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return ""
    
    def create_system_prompt(self, persona: str, task: str) -> str:
        """Create system prompt based on persona and task"""
        return f"""You are a {persona}. Your specific task is: {task}

You are an expert at analyzing documents and extracting the most relevant information for your role and task. 

When analyzing documents:
1. Focus on content that directly relates to your task
2. Identify the most important sections that would help accomplish the task
3. Rank information by importance and relevance
4. Extract specific, actionable information
5. Consider practical implementation aspects

Respond in a structured, professional manner that demonstrates your expertise in your role."""
    
    def get_section_titles(self, pdf_text: Dict[str, str], filename: str, persona: str, task: str) -> List[str]:
        """Get section titles from the document by analyzing the full PDF content"""
        system_prompt = self.create_system_prompt(persona, task)
        full_text = "\n\n".join([f"Page {page}: {text}" for page, text in pdf_text.items()])
        
        prompt = f"""Read entire document: {filename}

{full_text[:6000]}

As {persona} for "{task}", find actual section headings/titles from this PDF content.

Look for real section titles, chapter names, or topic headings that exist in the document. Output only actual titles found, one per line:"""

        # Increased to 50 tokens for better section titles
        response = self.call_ollama(prompt, system_prompt, max_tokens=50)
        
        # Parse response into list of titles
        titles = [title.strip() for title in response.split('\n') if title.strip()]
        return titles[:3]  # Limit to top 3
    
    def get_section_page_number(self, pdf_text: Dict[str, str], section_title: str, filename: str) -> int:
        """Get the page number where a section appears"""
        for page_num, text in pdf_text.items():
            # Simple heuristic: if section title appears in page text
            if section_title.lower() in text.lower():
                return page_num
        return 1  # Default to page 1 if not found
    
    def analyze_document_sections(self, pdf_text: Dict[str, str], filename: str, persona: str, task: str) -> List[Dict]:
        """Analyze PDF and extract important sections by querying for specific values"""
        sections = []
        
        # Get section titles
        section_titles = self.get_section_titles(pdf_text, filename, persona, task)
        
        # Build sections with individual queries
        for rank, title in enumerate(section_titles, 1):
            page_num = self.get_section_page_number(pdf_text, title, filename)
            
            sections.append({
                "document": filename,
                "section_title": title,
                "importance_rank": rank,
                "page_number": page_num
            })
        
        return sections
    
    def get_relevant_content_from_page(self, text: str, filename: str, page_num: int, persona: str, task: str) -> str:
        """Extract relevant content from a specific page"""
        system_prompt = self.create_system_prompt(persona, task)
        
        prompt = f"""Page {page_num} from {filename}:

{text[:2000]}

Task: {persona} needs "{task}"

Extract key actionable information. Respond with only the direct facts. No explanations, no complete sentences, no introductory phrases. No "The important information is..." or "Key details include...". Just the core facts:"""

        # Limit to 80 tokens for refined content (increased to avoid truncation)
        response = self.call_ollama(prompt, system_prompt, max_tokens=80)
        return response.strip()
    
    def is_page_relevant(self, text: str, filename: str, page_num: int, persona: str, task: str) -> bool:
        """Check if a page contains relevant information for the task"""
        system_prompt = self.create_system_prompt(persona, task)
        
        prompt = f"""Page {page_num} from {filename}:

{text[:1000]}

Task: {persona} needs "{task}"

Is this page useful? Respond with only YES or NO:"""

        # Limit to 5 tokens for simple YES/NO response
        response = self.call_ollama(prompt, system_prompt, max_tokens=5)
        return "yes" in response.lower()
    
    def analyze_subsections(self, pdf_text: Dict[str, str], filename: str, persona: str, task: str) -> List[Dict]:
        """Analyze PDF and extract refined text for subsections by querying for specific values"""
        subsections = []
        
        # Analyze each page for relevant content
        for page_num, text in pdf_text.items():
            if len(text) < 100:  # Skip very short pages
                continue
            
            # First check if page is relevant
            if not self.is_page_relevant(text, filename, page_num, persona, task):
                continue
            
            # Get the relevant content from the page
            refined_content = self.get_relevant_content_from_page(text, filename, page_num, persona, task)
            
            if refined_content:
                subsections.append({
                    "document": filename,
                    "refined_text": refined_content,
                    "page_number": page_num
                })
        
        return subsections
    
    def safe_print(self, message: str):
        """Thread-safe printing"""
        with self.print_lock:
            print(message)
    
    def process_single_document_optimized(self, doc_info: Dict, pdf_dir: str, persona: str, task: str, doc_ranking: Dict[str, int]) -> Tuple[List[Dict], List[Dict]]:
        """Process a single PDF document with optimizations and proper ranking"""
        filename = doc_info["filename"]
        pdf_path = os.path.join(pdf_dir, filename)
        
        if not os.path.exists(pdf_path):
            self.safe_print(f"PDF not found: {pdf_path}")
            return [], []
        
        self.safe_print(f"Processing: {filename}")
        
        # Extract text from PDF
        pdf_text = self.extract_text_from_pdf(pdf_path)
        if not pdf_text:
            self.safe_print(f"No text extracted from {filename}")
            return [], []
        
        # Get document importance rank (should be 1-7 for Collection 1)
        doc_importance = doc_ranking.get(filename, len(doc_ranking) + 1)
        
        # SPEED OPTIMIZATION: Process only first few pages for less important docs
        if doc_importance > 3:  # For lower priority docs, process only first 3 pages
            pdf_text = {k: v for k, v in list(pdf_text.items())[:3]}
        
        # Get section titles with better analysis
        section_titles = self.get_section_titles(pdf_text, filename, persona, task)
        
        # Build sections with DOCUMENT-LEVEL importance ranking
        sections = []
        for local_rank, title in enumerate(section_titles, 1):
            page_num = self.get_section_page_number(pdf_text, title, filename)
            
            # FIXED: Use document importance rank directly (1-7 for Collection 1)
            # Each document gets its rank, sections within document don't affect overall ranking
            sections.append({
                "document": filename,
                "section_title": title,
                "importance_rank": doc_importance,  # Use document rank directly
                "page_number": page_num
            })
        
        # SPEED OPTIMIZATION: Process subsections only for top 3 documents
        subsections = []
        if doc_importance <= 3:
            subsections = self.analyze_subsections_optimized(pdf_text, filename, persona, task)
        
        self.safe_print(f"✓ Completed: {filename} (rank:{doc_importance}, {len(sections)} sections, {len(subsections)} subsections)")
        return sections, subsections
    
    def analyze_subsections_optimized(self, pdf_text: Dict[str, str], filename: str, persona: str, task: str) -> List[Dict]:
        """Optimized subsection analysis - process only relevant pages"""
        subsections = []
        
        # SPEED OPTIMIZATION: Batch relevance check for multiple pages
        relevant_pages = []
        page_texts = list(pdf_text.items())
        
        # Process pages in batches to reduce AI calls
        for page_num, text in page_texts[:5]:  # Limit to first 5 pages for speed
            if len(text) < 100:
                continue
            
            # Quick relevance check
            if self.is_page_relevant(text, filename, page_num, persona, task):
                relevant_pages.append((page_num, text))
        
        # Extract content from relevant pages only
        for page_num, text in relevant_pages[:3]:  # Limit to top 3 relevant pages
            refined_content = self.get_relevant_content_from_page(text, filename, page_num, persona, task)
            
            if refined_content:
                subsections.append({
                    "document": filename,
                    "refined_text": refined_content,
                    "page_number": page_num
                })
        
        return subsections
    
    def rank_documents_by_importance(self, documents: List[Dict], persona: str, task: str) -> Dict[str, int]:
        """Rank all documents in collection by importance for the task"""
        system_prompt = self.create_system_prompt(persona, task)
        
        # Create list of document names
        doc_names = [doc["filename"] for doc in documents]
        doc_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(doc_names)])
        
        prompt = f"""Documents in collection:
{doc_list}

Task: {persona} needs "{task}"

Rank these documents by importance for this task. Output only the document numbers in order of importance (most important first):

Example: 3 1 5 2 4"""

        # Limit tokens for ranking response
        response = self.call_ollama(prompt, system_prompt, max_tokens=30)
        
        # Parse ranking response and create proper ranking
        ranking = {}
        try:
            # Extract numbers from response
            numbers = [int(x.strip()) for x in response.split() if x.strip().isdigit()]
            
            # Assign ranks based on order in response
            for rank, doc_index in enumerate(numbers, 1):
                if 1 <= doc_index <= len(doc_names):
                    filename = doc_names[doc_index - 1]
                    ranking[filename] = rank
            
            # Fill in any missing documents with sequential ranks
            used_ranks = set(ranking.values())
            next_rank = max(used_ranks) + 1 if used_ranks else 1
            
            for doc in documents:
                if doc["filename"] not in ranking:
                    ranking[doc["filename"]] = next_rank
                    next_rank += 1
                    
        except Exception as e:
            print(f"Warning: Document ranking failed ({e}), using sequential fallback")
            # Fallback: assign sequential ranks
            for i, doc in enumerate(documents):
                ranking[doc["filename"]] = i + 1
        
        return ranking

    def process_collection(self, collection_path: str) -> None:
        """Process a single collection with optimized parallel processing and unique ranking"""
        print(f"\nProcessing collection: {collection_path}")
        
        # Read input configuration
        input_file = os.path.join(collection_path, "challenge1b_input.json")
        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
            return
            
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        persona = input_data["persona"]["role"]
        task = input_data["job_to_be_done"]["task"]
        documents = input_data["documents"]
        
        print(f"Persona: {persona}")
        print(f"Task: {task}")
        print(f"Documents to process: {len(documents)}")
        print(f"Using {self.max_workers} parallel workers")
        
        # Process PDFs
        pdf_dir = os.path.join(collection_path, "PDFs")
        if not os.path.exists(pdf_dir):
            print(f"PDF directory not found: {pdf_dir}")
            return
        
        # OPTIMIZATION 1: Get document ranking first (single AI call)
        print("🔄 Ranking documents by importance...")
        doc_ranking = self.rank_documents_by_importance(documents, persona, task)
        
        all_extracted_sections = []
        all_subsection_analysis = []
        
        # OPTIMIZATION 2: Process only most relevant pages per document
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all document processing tasks
            future_to_doc = {
                executor.submit(self.process_single_document_optimized, doc_info, pdf_dir, persona, task, doc_ranking): doc_info
                for doc_info in documents
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_doc):
                doc_info = future_to_doc[future]
                try:
                    sections, subsections = future.result()
                    all_extracted_sections.extend(sections)
                    all_subsection_analysis.extend(subsections)
                except Exception as e:
                    self.safe_print(f"Error processing {doc_info['filename']}: {e}")
        
        # Sort sections by unique importance rank
        all_extracted_sections.sort(key=lambda x: x.get("importance_rank", 999))
        
        # Create output JSON
        output_data = {
            "metadata": {
                "input_documents": [doc["filename"] for doc in documents],
                "persona": persona,
                "job_to_be_done": task,
                "processing_timestamp": datetime.now().isoformat(),
                "parallel_workers": self.max_workers
            },
            "extracted_sections": all_extracted_sections[:10],  # Limit to top 10
            "subsection_analysis": all_subsection_analysis[:20]  # Limit to top 20
        }
        
        # Save output
        output_file = os.path.join(collection_path, "challenge1b_output_generated.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"\n✓ Output saved to: {output_file}")
        print(f"✓ Extracted {len(all_extracted_sections)} sections and {len(all_subsection_analysis)} subsections")

def main():
    # START TIMING - Total execution time from command start
    total_start_time = datetime.now()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="PDF Analysis Tool using Ollama with smollm:135m model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_analyzer.py "Collection 1"
  python pdf_analyzer.py "Collection 2" --workers 8
  python pdf_analyzer.py "Collection 3" --model gemma3:2b
        """
    )
    
    parser.add_argument(
        "collection",
        help="Collection name to process (e.g., 'Collection 1', 'Collection 2', 'Collection 3')"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="smollm:135m",
        help="Ollama model to use (default: smollm:135m)"
    )
    
    args = parser.parse_args()
    
    # Validate collection exists
    if not os.path.exists(args.collection):
        print(f"✗ Collection '{args.collection}' not found")
        print("\nAvailable collections:")
        for collection in ["Collection 1", "Collection 2", "Collection 3"]:
            if os.path.exists(collection):
                print(f"  - {collection}")
        return
    
    print(f"🚀 Starting PDF analysis for {args.collection}...")
    
    # Initialize analyzer with specified parameters
    analyzer = PDFAnalyzer(model_name=args.model, max_workers=args.workers)
    
    # Check if Ollama is running and model is available
    print("🔍 Checking Ollama connection...")
    try:
        # Test connection and model availability
        analyzer.client.generate(
            model=analyzer.model_name,
            prompt="test",
            stream=False
        )
        print(f"✓ Ollama is running and {analyzer.model_name} is available")
    except ollama.ResponseError as e:
        if "model not found" in str(e).lower():
            print(f"✗ Model {analyzer.model_name} not found")
            print(f"Please install it with: ollama pull {analyzer.model_name}")
            return
        else:
            print(f"✗ Ollama error: {e}")
            return
    except Exception as e:
        print(f"✗ Ollama is not accessible: {e}")
        print("Please make sure Ollama is running with: ollama serve")
        return
    
    # Process the specified collection
    analyzer.process_collection(args.collection)
    
    # END TIMING - Calculate total execution time
    total_end_time = datetime.now()
    total_execution_time = (total_end_time - total_start_time).total_seconds()
    
    print(f"\n🎉 Analysis completed!")
    print(f"⏱️  Total execution time: {total_execution_time:.2f} seconds")
    print(f"📁 Collection: {args.collection}")
    print(f"🤖 Model: {args.model}")
    print(f"⚡ Workers: {args.workers}")

if __name__ == "__main__":
    main()