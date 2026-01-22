import re
import PyPDF2
from typing import List, Dict
from src.config import Config

class PDFProcessor:
    def __init__(self):
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
    
    def extract_pages_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract text from all pages of a PDF"""
        pages = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    pages.append(page_text if page_text else "")
        except Exception as e:
            print(f"Error reading {pdf_path}: {str(e)}")
            raise
        return pages
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Keep only common punctuation and alphanumeric
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']', ' ', text)
        # Remove multiple spaces again
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to end at a sentence boundary
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_pdf(self, pdf_path: str, book_title: str) -> List[Dict]:
        """Process a PDF and return chunks with metadata"""
        pages = self.extract_pages_from_pdf(pdf_path)
        all_chunks = []
        
        for page_num, page in enumerate(pages, 1):
            page_text = self.clean_text(page)
            chunks = self.chunk_text(page_text)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append({
                    'book': book_title,
                    'page': page_num,
                    'text': chunk,
                    'chunk_index': chunk_idx
                })
        
        return all_chunks