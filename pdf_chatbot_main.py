#!/usr/bin/env python3
"""
AI-Powered PDF Chatbot for CPU-only servers
Supports English, French, and Arabic with lightweight models
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
import warnings
warnings.filterwarnings("ignore")

# Core imports
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import fitz  # PyMuPDF
from langdetect import detect
import re

# Web interface
import gradio as gr
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handle PDF parsing and text extraction"""
    
    def __init__(self):
        self.chunk_size = 500
        self.chunk_overlap = 50
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF with metadata"""
        try:
            doc = fitz.open(pdf_path)
            chunks = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Clean and chunk the text
                text = self._clean_text(text)
                page_chunks = self._chunk_text(text, page_num + 1)
                chunks.extend(page_chunks)
            
            doc.close()
            logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF.,!?;:\-()"]', '', text)
        return text.strip()
    
    def _chunk_text(self, text: str, page_num: int) -> List[Dict]:
        """Split text into chunks with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text.strip()) > 50:  # Skip very short chunks
                chunks.append({
                    'text': chunk_text,
                    'page': page_num,
                    'chunk_id': len(chunks)
                })
        
        return chunks

class EmbeddingManager:
    """Handle text embeddings and vector search"""
    
    def __init__(self, model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v1"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        self.embeddings = None
        
    def load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Create embeddings for text chunks"""
        if not self.model:
            self.load_model()
        
        texts = [chunk['text'] for chunk in chunks]
        logger.info(f"Creating embeddings for {len(texts)} chunks...")
        
        # Process in batches to manage memory
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        self.embeddings = np.vstack(embeddings)
        self.chunks = chunks
        
        # Create FAISS index
        self._create_faiss_index()
        
        logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
        return self.embeddings
    
    def _create_faiss_index(self):
        """Create FAISS index for fast similarity search"""
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        logger.info(f"Created FAISS index with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks"""
        if not self.index or not self.model:
            return []
        
        # Create query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result['score'] = float(score)
                results.append(result)
        
        return results

class LanguageDetector:
    """Detect query language and manage language-specific processing"""
    
    def __init__(self):
        self.supported_languages = {'en': 'English', 'fr': 'French', 'ar': 'Arabic'}
    
    def detect_language(self, text: str) -> str:
        """Detect the language of input text"""
        try:
            lang = detect(text)
            if lang in self.supported_languages:
                return lang
            return 'en'  # Default to English
        except:
            return 'en'
    
    def get_language_name(self, lang_code: str) -> str:
        """Get full language name from code"""
        return self.supported_languages.get(lang_code, 'English')

class QAModel:
    """Handle question answering with multiple model options"""
    
    def __init__(self):
        self.extractive_models = {
            'en': 'distilbert-base-cased-distilled-squad',
            'fr': 'camembert-base-squadFR-fquad-piaf',
            'ar': 'CAMeL-Lab/bert-base-arabic-camelbert-msa-qadi'
        }
        self.loaded_models = {}
        self.generative_model = None
    
    def load_extractive_model(self, language: str):
        """Load extractive QA model for specific language"""
        if language in self.loaded_models:
            return self.loaded_models[language]
        
        model_name = self.extractive_models.get(language, self.extractive_models['en'])
        
        try:
            logger.info(f"Loading extractive QA model for {language}: {model_name}")
            qa_pipeline = pipeline(
                "question-answering",
                model=model_name,
                tokenizer=model_name,
                return_all_scores=False,
                device=-1  # CPU only
            )
            self.loaded_models[language] = qa_pipeline
            logger.info(f"Loaded extractive QA model for {language}")
            return qa_pipeline
            
        except Exception as e:
            logger.error(f"Error loading extractive model for {language}: {e}")
            # Fallback to English model
            if language != 'en':
                return self.load_extractive_model('en')
            return None
    
    def extractive_answer(self, question: str, context: str, language: str) -> Dict:
        """Get extractive answer from context"""
        qa_model = self.load_extractive_model(language)
        if not qa_model:
            return {"answer": "Model not available", "confidence": 0.0}
        
        try:
            # Truncate context if too long
            max_length = 512
            if len(context) > max_length:
                context = context[:max_length]
            
            result = qa_model(question=question, context=context)
            return {
                "answer": result['answer'],
                "confidence": result['score'],
                "start": result.get('start', 0),
                "end": result.get('end', 0)
            }
        except Exception as e:
            logger.error(f"Error in extractive QA: {e}")
            return {"answer": "Error processing question", "confidence": 0.0}

class PDFChatbot:
    """Main chatbot orchestrator"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.embedding_manager = EmbeddingManager()
        self.language_detector = LanguageDetector()
        self.qa_model = QAModel()
        self.pdf_chunks = []
        self.current_pdf = None
    
    def load_pdf(self, pdf_path: str) -> bool:
        """Load and process a PDF file"""
        try:
            logger.info(f"Loading PDF: {pdf_path}")
            
            # Extract text chunks
            chunks = self.pdf_processor.extract_text_from_pdf(pdf_path)
            if not chunks:
                return False
            
            # Create embeddings
            self.embedding_manager.create_embeddings(chunks)
            self.pdf_chunks = chunks
            self.current_pdf = pdf_path
            
            logger.info(f"Successfully loaded PDF with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            return False
    
    def answer_question(self, question: str, answer_type: str = "extractive") -> Dict:
        """Answer a question about the loaded PDF"""
        if not self.pdf_chunks:
            return {
                "answer": "No PDF loaded. Please load a PDF first.",
                "confidence": 0.0,
                "sources": [],
                "language": "en"
            }
        
        # Detect language
        detected_lang = self.language_detector.detect_language(question)
        
        # Search for relevant chunks
        relevant_chunks = self.embedding_manager.search(question, top_k=3)
        
        if not relevant_chunks:
            return {
                "answer": "No relevant information found in the PDF.",
                "confidence": 0.0,
                "sources": [],
                "language": detected_lang
            }
        
        # Combine contexts
        combined_context = " ".join([chunk['text'] for chunk in relevant_chunks])
        
        # Get answer based on type
        if answer_type == "extractive":
            result = self.qa_model.extractive_answer(question, combined_context, detected_lang)
            
            return {
                "answer": result['answer'],
                "confidence": result['confidence'],
                "sources": relevant_chunks,
                "language": detected_lang,
                "type": "extractive"
            }
        else:
            # For generative answers, we'll use a simple template for now
            # In a real implementation, you'd integrate TinyLlama here
            return {
                "answer": f"Based on the PDF content: {combined_context[:200]}...",
                "confidence": 0.8,
                "sources": relevant_chunks,
                "language": detected_lang,
                "type": "generative"
            }

def create_gradio_interface():
    """Create Gradio web interface"""
    chatbot = PDFChatbot()
    
    def load_pdf_interface(pdf_file):
        if pdf_file is None:
            return "Please select a PDF file", ""
        
        success = chatbot.load_pdf(pdf_file.name)
        if success:
            return f"Successfully loaded: {os.path.basename(pdf_file.name)}", f"PDF loaded with {len(chatbot.pdf_chunks)} text chunks"
        else:
            return "Error loading PDF", "Failed to process the PDF file"
    
    def answer_question_interface(question, answer_type):
        if not question.strip():
            return "Please enter a question"
        
        result = chatbot.answer_question(question, answer_type)
        
        # Format response
        response = f"**Answer ({result['language'].upper()}):** {result['answer']}\n\n"
        response += f"**Confidence:** {result['confidence']:.2f}\n\n"
        
        if result['sources']:
            response += "**Sources:**\n"
            for i, source in enumerate(result['sources'][:2], 1):
                response += f"{i}. Page {source['page']} (Score: {source['score']:.3f})\n"
                response += f"   {source['text'][:100]}...\n\n"
        
        return response
    
    # Create interface
    with gr.Blocks(title="PDF Chatbot", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🤖 AI-Powered PDF Chatbot")
        gr.Markdown("Upload a PDF and ask questions in English, French, or Arabic!")
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                load_btn = gr.Button("Load PDF", variant="primary")
                
                load_status = gr.Textbox(label="Status", interactive=False)
                load_info = gr.Textbox(label="PDF Info", interactive=False)
        
        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(
                    label="Ask a question",
                    placeholder="Enter your question in English, French, or Arabic...",
                    lines=2
                )
                answer_type = gr.Radio(
                    choices=["extractive", "generative"],
                    value="extractive",
                    label="Answer Type"
                )
                ask_btn = gr.Button("Ask Question", variant="primary")
        
        answer_output = gr.Markdown(label="Answer")
        
        # Event handlers
        load_btn.click(
            load_pdf_interface,
            inputs=[pdf_input],
            outputs=[load_status, load_info]
        )
        
        ask_btn.click(
            answer_question_interface,
            inputs=[question_input, answer_type],
            outputs=[answer_output]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["What is the main topic of this document?", "extractive"],
                ["Quel est le sujet principal de ce document?", "extractive"],
                ["ما هو الموضوع الرئيسي لهذه الوثيقة؟", "extractive"]
            ],
            inputs=[question_input, answer_type]
        )
    
    return app

def create_cli_interface():
    """Create command-line interface"""
    parser = argparse.ArgumentParser(description="PDF Chatbot CLI")
    parser.add_argument("--pdf", type=str, help="Path to PDF file")
    parser.add_argument("--question", type=str, help="Question to ask")
    parser.add_argument("--type", choices=["extractive", "generative"], 
                       default="extractive", help="Answer type")
    
    args = parser.parse_args()
    
    chatbot = PDFChatbot()
    
    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"Error: PDF file not found: {args.pdf}")
            return
        
        print(f"Loading PDF: {args.pdf}")
        success = chatbot.load_pdf(args.pdf)
        
        if not success:
            print("Error: Failed to load PDF")
            return
        
        print(f"Successfully loaded PDF with {len(chatbot.pdf_chunks)} chunks")
    
    # Interactive mode
    print("\n🤖 PDF Chatbot CLI")
    print("Commands: 'load <pdf_path>', 'ask <question>', 'quit'")
    print("Languages supported: English, French, Arabic\n")
    
    while True:
        try:
            user_input = input("chatbot> ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if user_input.startswith('load '):
                pdf_path = user_input[5:].strip()
                if os.path.exists(pdf_path):
                    success = chatbot.load_pdf(pdf_path)
                    if success:
                        print(f"✅ Loaded PDF with {len(chatbot.pdf_chunks)} chunks")
                    else:
                        print("❌ Failed to load PDF")
                else:
                    print("❌ PDF file not found")
            
            elif user_input.startswith('ask '):
                question = user_input[4:].strip()
                if question:
                    result = chatbot.answer_question(question, args.type)
                    
                    print(f"\n📝 Answer ({result['language'].upper()}):")
                    print(result['answer'])
                    print(f"\n🎯 Confidence: {result['confidence']:.2f}")
                    
                    if result['sources']:
                        print(f"\n📚 Sources:")
                        for i, source in enumerate(result['sources'][:2], 1):
                            print(f"{i}. Page {source['page']} (Score: {source['score']:.3f})")
                else:
                    print("❌ Please provide a question")
            
            else:
                print("❌ Unknown command. Use 'load <pdf>', 'ask <question>', or 'quit'")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI-Powered PDF Chatbot")
    parser.add_argument("--interface", choices=["cli", "gradio"], 
                       default="gradio", help="Interface type")
    parser.add_argument("--port", type=int, default=7860, help="Port for web interface")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for web interface")
    
    args = parser.parse_args()
    
    if args.interface == "cli":
        create_cli_interface()
    else:
        app = create_gradio_interface()
        app.launch(server_name=args.host, server_port=args.port, share=False)

if __name__ == "__main__":
    main()