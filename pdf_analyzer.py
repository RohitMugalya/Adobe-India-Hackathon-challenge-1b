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
        """Get section titles from the document"""
        system_prompt = self.create_system_prompt(persona, task)
        full_text = "\n\n".join([f"Page {page}: {text}" for page, text in pdf_text.items()])
        
        prompt = f"""Document: {filename}

{full_text[:4000]}

Task: {persona} planning "{task}"

Find 3 important section titles. Respond with only the direct titles. No explanations, no complete sentences, no introductory phrases. No "Here is..." or "The titles are...". Just the section names, one per line:"""

        # Limit to 30 tokens for section titles (increased to avoid truncation)
        response = self.call_ollama(prompt, system_prompt, max_tokens=30)
        
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
    
    def process_single_document(self, doc_info: Dict, pdf_dir: str, persona: str, task: str) -> Tuple[List[Dict], List[Dict]]:
        """Process a single PDF document - designed for parallel execution"""
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
        
        # Analyze sections and subsections in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both analysis tasks
            sections_future = executor.submit(self.analyze_document_sections, pdf_text, filename, persona, task)
            subsections_future = executor.submit(self.analyze_subsections, pdf_text, filename, persona, task)
            
            # Get results
            sections = sections_future.result()
            subsections = subsections_future.result()
        
        self.safe_print(f"✓ Completed: {filename} ({len(sections)} sections, {len(subsections)} subsections)")
        return sections, subsections
    
    def process_collection(self, collection_path: str) -> None:
        """Process a single collection with parallel processing"""
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
        
        all_extracted_sections = []
        all_subsection_analysis = []
        
        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all document processing tasks
            future_to_doc = {
                executor.submit(self.process_single_document, doc_info, pdf_dir, persona, task): doc_info
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
        
        # Sort sections by importance rank
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="PDF Analysis Tool using Ollama with gemma3:1b model",
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
    
    # Initialize analyzer with specified parameters
    analyzer = PDFAnalyzer(model_name=args.model, max_workers=args.workers)
    
    # Check if Ollama is running and model is available
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
    start_time = datetime.now()
    analyzer.process_collection(args.collection)
    end_time = datetime.now()
    
    processing_time = (end_time - start_time).total_seconds()
    print(f"\n🚀 Total processing time: {processing_time:.2f} seconds")

if __name__ == "__main__":
    main()