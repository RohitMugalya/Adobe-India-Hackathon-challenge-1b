"""
Advanced PDF Analysis Tool with Robust Error Handling
"""

import json
import os
import sys
import argparse
import multiprocessing
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import PyPDF2
from typing import List, Dict, Any, Tuple, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

class PDFAnalyzer:
    def __init__(self, max_workers=None):
        self.model_name = "facebook/bart-base"
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.print_lock = Lock()
        
        self.bart_model_path = "models/models--facebook--bart-base/snapshots/aadd2ab0ae0c8268c7c9693540e9904811f36177"
        self.similarity_model_path = "models/sentence_transformers/models--sentence-transformers--all-mpnet-base-v2/snapshots/12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0"
        self.cross_encoder_path = "models/sentence_transformers/models--cross-encoder--ms-marco-MiniLM-L-6-v2"
        
        self.device = "cpu"
        print(f"Using device: {self.device}")
        self._load_models()

    def _load_models(self):
        """Load all required models from local paths"""
        print(f"Loading models from local paths...")
        try:
            if not os.path.exists(self.bart_model_path):
                print(f"BART model not found at: {self.bart_model_path}")
                sys.exit(1)
            if not os.path.exists(self.similarity_model_path):
                print(f"Similarity model not found at: {self.similarity_model_path}")
                sys.exit(1)
            
            print(f"Loading BART from: {self.bart_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.bart_model_path,
                local_files_only=True
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.bart_model_path,
                local_files_only=True,
                torch_dtype=torch.float32
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"Loading similarity model from: {self.similarity_model_path}")
            self.bi_encoder = SentenceTransformer(self.similarity_model_path)
            
            if os.path.exists(self.cross_encoder_path):
                print(f"Loading cross-encoder from: {self.cross_encoder_path}")
                self.cross_encoder = CrossEncoder(self.cross_encoder_path, device="cpu")
            else:
                print("Cross-encoder not found, using bi-encoder only")
                self.cross_encoder = None
            
            print(f"Models loaded successfully from local paths")
        except Exception as e:
            print(f"Error loading models: {e}")
            sys.exit(1)

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Robust PDF text extraction with error handling"""
        text_by_page = {}
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        text_by_page[page_num] = text.strip()
            return text_by_page
        except Exception as e:
            self.safe_print(f"Error reading PDF {pdf_path}: {e}")
            return {}

    def generate_response(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Improved text generation with better parameters"""
        try:
            set_seed(42)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True)
            

                
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    num_beams=2,
                    early_stopping=True,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3)
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._clean_response(response)
        except Exception as e:
            self.safe_print(f"Error generating response: {e}")
            return ""

    def _clean_response(self, text: str) -> str:
        """Enhanced response cleaning"""
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\ufffd', '').replace('\u200b', '')
        
        prefixes = ["Answer:", "Response:", "Summary:", "Information:", "Context:"]
        for prefix in prefixes:
            if prefix in text:
                text = text.split(prefix, 1)[-1].strip()
        
        return text.strip('"').strip("'").strip()

    def identify_section_titles(self, pdf_text: Dict[int, str]) -> List[str]:
        """Improved section title detection from PDF text"""
        potential_titles = set()
        heading_pattern = re.compile(r'^(?P<title>[A-Z][A-Za-z0-9 \-\–]+)(?::|\.\s|$)')
        
        for page_num, text in list(pdf_text.items())[:5]:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                match = heading_pattern.match(line)
                if match and 10 < len(line) < 100:
                    title = match.group('title').strip()
                    if not any(phrase in title.lower() for phrase in 
                             ["page", "copyright", "figure", "table"]):
                        potential_titles.add(title)
        
        return sorted(potential_titles, key=len, reverse=True)[:3] if potential_titles else []

    def rank_documents(self, documents: List[Dict], persona: str, task: str) -> Dict[str, int]:
        """Hybrid document ranking with bi-encoder + cross-encoder (if available)"""
        doc_embeddings = self._get_document_embeddings(documents)
        task_embedding = self.bi_encoder.encode(
            f"{persona} needs to {task}",
            convert_to_tensor=True)
        
        similarities = util.pytorch_cos_sim(task_embedding, doc_embeddings)[0]
        
        ranked_docs = sorted(
            zip(range(len(documents)), similarities),
            key=lambda x: x[1],
            reverse=True)
        
        if self.cross_encoder is not None:
            top_n = min(10, len(documents))
            pairs = [
                (f"{persona} needs to {task}", 
                 self._get_document_text(documents[idx]))
                for idx, _ in ranked_docs[:top_n]
            ]
            rerank_scores = self.cross_encoder.predict(pairs)
            
            final_scores = []
            for i in range(len(ranked_docs)):
                if i < top_n:
                    combined = 0.6 * rerank_scores[i] + 0.4 * ranked_docs[i][1]
                else:
                    combined = ranked_docs[i][1]
                final_scores.append((ranked_docs[i][0], combined))
            
            final_ranking = sorted(final_scores, key=lambda x: x[1], reverse=True)
        else:
            final_ranking = ranked_docs
        
        return {
            documents[idx]["filename"]: rank+1
            for rank, (idx, _) in enumerate(final_ranking)
        }

    def _get_document_embeddings(self, documents: List[Dict]) -> torch.Tensor:
        """Enhanced document representation for embedding"""
        texts = []
        for doc in documents:
            filename = doc["filename"]
            doc_type = self._get_document_type(filename)
            texts.append(
                f"Document titled '{filename}' about {doc_type} "
                f"with potential relevance to travel planning")
        return self.bi_encoder.encode(texts, convert_to_tensor=True)

    def _get_document_text(self, document: Dict) -> str:
        """Generate representative text for cross-encoder"""
        filename = document["filename"]
        doc_type = self._get_document_type(filename)
        return (
            f"Document titled '{filename}' about {doc_type}. "
            f"Contents may include travel destinations, "
            f"activities, or cultural information.")

    def _get_document_type(self, filename: str) -> str:
        """Extract clean document type from filename"""
        name = os.path.splitext(filename)[0]
        if " - " in name:
            return name.split(" - ")[-1].replace("_", " ")
        return name.replace("_", " ").title()

    def process_document(self, doc_info: Dict, pdf_dir: str, persona: str, task: str, doc_rank: int) -> Tuple[Dict, List[Dict]]:
        """Process a single document with improved relevance detection"""
        filename = doc_info["filename"]
        pdf_path = os.path.join(pdf_dir, filename)
        
        if not os.path.exists(pdf_path):
            self.safe_print(f"PDF not found: {pdf_path}")
            return None, []
            
        self.safe_print(f"Processing: {filename}")
        
        # Extract text from PDF
        pdf_text = self.extract_text_from_pdf(pdf_path)
        if not pdf_text:
            self.safe_print(f"No text extracted from {filename}")
            return None, []
        
        # Get the most relevant section
        section_titles = self.identify_section_titles(pdf_text)
        if not section_titles:
            section_titles = self._generate_section_titles(filename, persona, task)
        
        best_title = section_titles[0] if section_titles else "Main Content"
        best_page = self._find_section_page(pdf_text, best_title)
        
        section = {
            "document": filename,
            "section_title": best_title,
            "importance_rank": doc_rank,
            "page_number": best_page
        }
        
        # Find relevant content in first 5 pages
        subsections = []
        for page_num, text in list(pdf_text.items())[:5]:
            if self._is_content_relevant(text, filename, persona, task):
                content = self._extract_key_content(text, filename, persona, task)
                if content:
                    subsections.append({
                        "document": filename,
                        "refined_text": content,
                        "page_number": page_num
                    })
        
        self.safe_print(f"Completed: {filename} (rank:{doc_rank}, 1 section, {len(subsections)} subsections)")
        return section, subsections

    def _generate_section_titles(self, filename: str, persona: str, task: str) -> List[str]:
        """Generate relevant section titles when none can be identified"""
        doc_type = self._get_document_type(filename)
        prompt = f"""What are 3 most relevant section titles from a "{doc_type}" document that would help a {persona} accomplish: {task}?
Provide only the titles, one per line, without numbering."""
        
        response = self.generate_response(prompt)
        if not response:
            return [
                f"Key Information about {doc_type}",
                f"Important Details for {task}",
                f"Practical Guidance for {persona}s"
            ]
        
        titles = []
        for line in response.split('\n'):
            line = re.sub(r'^[\d\.\-\*]+', '', line).strip()
            if 5 < len(line) < 100:
                titles.append(line)
        
        return titles[:3] if titles else [
            f"Key Information about {doc_type}",
            f"Important Details for {task}",
            f"Practical Guidance for {persona}s"
        ]

    def _find_section_page(self, pdf_text: Dict[int, str], section_title: str) -> int:
        """Find page number for a section with improved matching"""
        for page_num, text in pdf_text.items():
            if section_title.lower() in text.lower():
                return page_num
        
        keywords = [word.lower() for word in section_title.split() if len(word) > 3]
        for page_num, text in pdf_text.items():
            text_lower = text.lower()
            if sum(keyword in text_lower for keyword in keywords) >= len(keywords)/2:
                return page_num
        
        return 1

    def _is_content_relevant(self, text: str, filename: str, persona: str, task: str) -> bool:
        """Improved relevance detection with hybrid approach"""
        if len(text) < 150:
            return False
            
        task_keywords = [
            'travel', 'trip', 'itinerary', 'plan', 'visit', 
            'destination', 'activity', 'guide', 'recommend'
        ]
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in task_keywords):
            return True
            
        prompt = f"""Is this content from "{filename}" relevant for a {persona} working on {task}?
Content: {text[:500]}
Answer only YES or NO."""
        
        response = self.generate_response(prompt, max_new_tokens=5).lower()
        return 'yes' in response

    def _extract_key_content(self, text: str, filename: str, persona: str, task: str) -> str:
        """Enhanced content extraction with fallbacks"""
        actionable = []
        for line in text.split('\n'):
            line = line.strip()
            if (any(word in line.lower() for word in ['how to', 'step', 'guide', 'itinerary', 'visit', 'recommend']) 
                and 20 < len(line) < 200):
                actionable.append(line)
        
        if actionable:
            return " ".join(actionable[:3])
        
        doc_type = self._get_document_type(filename)
        prompt = f"""Summarize the most relevant information from this "{doc_type}" document content for a {persona} working on {task}:
{text[:1000]}
Provide a concise 1-2 sentence summary focusing only on information that would help accomplish the task."""
        
        response = self.generate_response(prompt)
        return response if response else f"Relevant information about {doc_type} for {task}."

    def safe_print(self, message: str):
        """Thread-safe printing"""
        with self.print_lock:
            print(message)

    def process_collection(self, collection_path: str) -> None:
        """Process an entire collection with improved workflow"""
        print(f"\nProcessing collection: {collection_path}")
        
        # Load input data
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
        
        # Check PDF directory
        pdf_dir = os.path.join(collection_path, "PDFs")
        if not os.path.exists(pdf_dir):
            print(f"PDF directory not found: {pdf_dir}")
            return
            
        print("Determining document importance ranking...")
        doc_ranking = self.rank_documents(documents, persona, task)
        
        print(f"Using {self.max_workers} parallel workers")
        all_sections = []
        all_subsections = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.process_document,
                    doc,
                    pdf_dir,
                    persona,
                    task,
                    doc_ranking.get(doc["filename"], len(documents) + 1)
                ): doc for doc in documents
            }
            
            for future in as_completed(futures):
                doc = futures[future]
                try:
                    section, subsections = future.result()
                    if section:
                        all_sections.append(section)
                    all_subsections.extend(subsections)
                except Exception as e:
                    self.safe_print(f"Error processing {doc['filename']}: {e}")
        
        # Prepare output data
        output_data = {
            "metadata": {
                "input_documents": [doc["filename"] for doc in documents],
                "persona": persona,
                "job_to_be_done": task,
                "processing_timestamp": datetime.now().isoformat(),
                "parallel_workers": self.max_workers,
                "model_used": self.model_name,
                "device": self.device,
                "similarity_model": "all-mpnet-base-v2 + cross-encoder"
            },
            "extracted_sections": sorted(all_sections, key=lambda x: x["importance_rank"])[:5],
            "subsection_analysis": all_subsections[:5]
        }
        
        # Save output
        output_file = os.path.join(collection_path, "challenge1b_output_transformers.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"\nOutput saved to: {output_file}")
        print(f"Extracted {len(all_sections)} sections")
        print(f"Generated {len(all_subsections)} subsection analyses")

def main():
    total_start = datetime.now()
    parser = argparse.ArgumentParser(description="Advanced PDF Analysis Tool - Offline Version")
    parser.add_argument("collection", help="Path to collection directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.collection):
        print(f"Collection not found: {args.collection}")
        return
        
    print(f"Starting analysis for {args.collection}")
    print(f"Using hardcoded models: facebook/bart-base + all-mpnet-base-v2")
    analyzer = PDFAnalyzer()
    analyzer.process_collection(args.collection)
    
    total_time = (datetime.now() - total_start).total_seconds()
    print(f"\nAnalysis completed in {total_time:.2f} seconds")
    print(f"Model: facebook/bart-base")
    print(f"Workers: {analyzer.max_workers}")

if __name__ == "__main__":
    main()