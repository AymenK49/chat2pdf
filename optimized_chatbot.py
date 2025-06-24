#!/usr/bin/env python3
"""
Optimized AI-Powered PDF Chatbot
Enhanced version with better performance and memory management
"""

import os
import sys
import json
import logging
import time
import gc
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
import warnings
warnings.filterwarnings("ignore")

# Core imports
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import fitz  # PyMuPDF
from langdetect import detect
import re
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil

# Web interface
import gradio as gr

# Import our advanced models
from advanced_models import ModelManager, PerformanceOptimizer, BatchProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chatbot.log')
    ]
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Manage application configuration"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        default_config = {
            "models": {
                "embedding": "sentence-transformers/distiluse-base-multilingual-cased-v1",
                "qa": {
                    "en": "distilbert-base-cased-distilled-squad",
                    "fr": "etalab-ia/camembert-base-squadFR-fquad-piaf",
                    "ar": "aubmindlab/bert-base-arabertv02-squad2"
                },
                "generative": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            },
            "performance": {
                "max_memory_mb": 3000,
                "chunk_size": 500,
                "chunk_overlap": 50,
                "max_context_length": 400,
                "batch_size": 16,
                "max_threads": 4
            },
            "interface": {
                "default_port": 7860,
                "default_host": "0.0.0.0",
                "max_file_size_mb": 50
            },
            "cache": {
                "embeddings_cache": True,
                "model_cache_dir": "./models/cache"
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                self._deep_merge(default_config, user_config)
            except Exception as e:
                logger.warning(f"Error loading config: {e}, using defaults")
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def save_config(self):
        """Save current configuration"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

class OptimizedPDFProcessor:
    """Enhanced PDF processing with better performance"""
    
    def __init__(self, config: Dict):
        self.chunk_size = config['performance']['chunk_size']
        self.chunk_overlap = config['performance']['chunk_overlap']
        self.max_threads = config['performance']['max_threads']
        self.batch_processor = BatchProcessor(config['performance']['batch_size'])
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text with parallel processing"""
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            logger.info(f"Processing PDF with {total_pages} pages: {pdf_path}")
            
            # Process pages in parallel
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                futures = []
                for page_num in range(total_pages):
                    future = executor.submit(self._process_page, doc, page_num)
                    futures.append(future)
                
                all_chunks = []
                for future in futures:
                    page_chunks = future.result()
                    all_chunks.extend(page_chunks)
            
            doc.close()
            
            # Sort chunks by page and position
            all_chunks.sort(key=lambda x: (x['page'], x['chunk_id']))
            
            logger.info(f"Extracted {len(all_chunks)} chunks from {pdf_path}")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return []
    
    def _process_page(self, doc, page_num: int) -> List[Dict]:
        """Process a single page"""
        try:
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Enhanced text cleaning
            text = self._clean_text(text)
            
            if len(text.strip()) < 50:  # Skip pages with little content
                return []
            
            # Create chunks with metadata
            chunks = self._create_smart_chunks(text, page_num + 1)
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Page \d+.*$', '', text, flags=re.MULTILINE)
        
        # Keep multilingual characters
        text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF.,!?;:\-()"\']', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        return text.strip()
    
    def _create_smart_chunks(self, text: str, page_num: int) -> List[Dict]:
        """Create chunks with smart sentence boundaries"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        current_word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_words = len(sentence.split())
            
            # Check if adding this sentence would exceed chunk size
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'page': page_num,
                    'chunk_id': len(chunks),
                    'word_count': current_word_count
                })
                
                # Start new chunk with overlap
                overlap_words = ' '.join(current_chunk.split()[-self.chunk_overlap:])
                current_chunk = overlap_words + " " + sentence
                current_word_count = len(current_chunk.split())
            else:
                current_chunk += " " + sentence
                current_word_count += sentence_words
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'page': page_num,
                'chunk_id': len(chunks),
                'word_count': current_word_count
            })
        
        return chunks

class OptimizedEmbeddingManager:
    """Enhanced embedding management with caching"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config['models']['embedding']
        self.model = None
        self.index = None
        self.chunks = []
        self.embeddings = None
        self.cache_enabled = config['cache']['embeddings_cache']
        self.cache_dir = Path(config['cache']['model_cache_dir']) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self):
        """Load embedding model with optimization"""
        if self.model is not None:
            return
        
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Set cache directory
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = self.config['cache']['model_cache_dir']
            
            self.model = SentenceTransformer(self.model_name)
            
            # Optimize for CPU
            PerformanceOptimizer.optimize_torch_settings()
            
            logger.info("Embedding model loade