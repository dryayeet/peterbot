import os
import json
import faiss
import requests
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from src.config import Config
from src.pdf_processor import PDFProcessor
from src.utils import RateLimiter

class RAGSystem:
    def __init__(self):
        self.embedding_model = None
        self.faiss_index = None
        self.metadata = []
        self.rate_limiter = RateLimiter(
            max_calls=Config.MAX_REQUESTS_PER_MINUTE,
            period=60
        )
        # Only create directories, don't validate API key here
        # API key validation happens when calling the LLM
        os.makedirs(Config.DATASETS_PATH, exist_ok=True)
        os.makedirs(Config.VECTOR_DB_PATH, exist_ok=True)
    
    def load_embedding_model(self):
        """Load the sentence transformer model (fallback if service unavailable)"""
        if self.embedding_model is None:
            print("Loading embedding model locally (fallback mode)...")
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            print("Embedding model loaded!")
    
    def _encode_query(self, text: str) -> np.ndarray:
        """Encode a query using embedding service or local model as fallback"""
        if Config.EMBEDDING_SERVICE_ENABLED:
            try:
                response = requests.post(
                    f"{Config.EMBEDDING_SERVICE_URL}/encode",
                    json={"text": text},
                    timeout=Config.EMBEDDING_SERVICE_TIMEOUT
                )
                response.raise_for_status()
                result = response.json()
                return np.array(result["embedding"], dtype=np.float32)
            except requests.exceptions.RequestException as e:
                # Service enabled but failed - fallback to local
                print(f"Warning: Embedding service unavailable ({str(e)}). Falling back to local model.")
                self.load_embedding_model()
                return self.embedding_model.encode([text])[0].astype(np.float32)
        
        # Service disabled - use local model
        self.load_embedding_model()
        return self.embedding_model.encode([text])[0].astype(np.float32)
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts using embedding service or local model as fallback"""
        if Config.EMBEDDING_SERVICE_ENABLED:
            try:
                response = requests.post(
                    f"{Config.EMBEDDING_SERVICE_URL}/encode_batch",
                    json={"texts": texts},
                    timeout=Config.EMBEDDING_SERVICE_TIMEOUT * 2  # Longer timeout for batch
                )
                response.raise_for_status()
                result = response.json()
                return np.array(result["embeddings"], dtype=np.float32)
            except requests.exceptions.RequestException as e:
                # Service enabled but failed - fallback to local
                print(f"Warning: Embedding service unavailable ({str(e)}). Falling back to local model.")
                self.load_embedding_model()
                return self.embedding_model.encode(texts, show_progress_bar=True, batch_size=32).astype(np.float32)
        
        # Service disabled - use local model
        self.load_embedding_model()
        return self.embedding_model.encode(texts, show_progress_bar=True, batch_size=32).astype(np.float32)
    
    def build_and_save_vector_db(self, force: bool = False):
        """Build vector database from PDFs and save to disk
        
        Args:
            force: If True, overwrite existing vector database. If False and database exists,
                   raises ValueError to prevent accidental rebuilds.
        
        Raises:
            ValueError: If vector database already exists and force=False
        """
        # Safety check: prevent accidental rebuilds
        if os.path.exists(Config.FAISS_INDEX_FILE) and not force:
            raise ValueError(
                f"Vector database already exists at {Config.FAISS_INDEX_FILE}. "
                "To rebuild, call with force=True: build_and_save_vector_db(force=True)"
            )
        
        print("Building vector database...")
        
        # Process all PDFs
        pdf_processor = PDFProcessor()
        all_chunks = []
        
        for book_title, pdf_filename in Config.PDF_FILES.items():
            pdf_path = os.path.join(Config.DATASETS_PATH, pdf_filename)
            if not os.path.exists(pdf_path):
                print(f"Warning: {pdf_path} not found. Skipping...")
                continue
            
            print(f"Processing {book_title}...")
            chunks = pdf_processor.process_pdf(pdf_path, book_title)
            all_chunks.extend(chunks)
            print(f"  - Extracted {len(chunks)} chunks")
        
        if not all_chunks:
            raise ValueError("No chunks extracted from PDFs. Check your PDF files.")
        
        # Create embeddings using service or local model
        print("Creating embeddings...")
        texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self._encode_batch(texts)
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
        self.metadata = all_chunks
        
        # Save to disk
        print("Saving to disk...")
        faiss.write_index(self.faiss_index, Config.FAISS_INDEX_FILE)
        with open(Config.METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Vector database built with {len(all_chunks)} chunks!")
        print(f"  - Saved to {Config.VECTOR_DB_PATH}")
    
    def load_vector_db(self):
        """Load existing vector database from disk (does not rebuild)"""
        if not os.path.exists(Config.FAISS_INDEX_FILE):
            raise FileNotFoundError(
                f"Vector database not found at {Config.FAISS_INDEX_FILE}. "
                "Please run build_and_save_vector_db() first."
            )
        
        print("Loading existing vector database from disk...")
        # Don't load embedding model here - will be loaded on demand if service unavailable
        self.faiss_index = faiss.read_index(Config.FAISS_INDEX_FILE)
        
        with open(Config.METADATA_FILE, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {len(self.metadata)} chunks from existing vector database")
    
    def retrieve_chunks(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve most relevant chunks for a query"""
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL
        
        if self.faiss_index is None:
            self.load_vector_db()
        
        # Encode query using embedding service or local model
        query_embedding = self._encode_query(query)
        
        # Search in FAISS (ensure query_embedding is 2D array)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.faiss_index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                chunk = self.metadata[idx].copy()
                chunk['similarity_score'] = float(distances[0][i])
                results.append(chunk)
        
        # Sort by similarity score in descending order (highest score first)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results
    
    @RateLimiter(max_calls=Config.MAX_REQUESTS_PER_MINUTE, period=60)
    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenRouter API with rate limiting"""
        # Validate API key when actually needed
        if not Config.OPENROUTER_API_KEY:
            return "Error: OPENROUTER_API_KEY not found in .env file. Please set it to use the LLM."
        
        headers = {
            "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": Config.MODEL_NAME,
            "messages": messages,
            "max_tokens": Config.MAX_TOKENS,
            "temperature": Config.TEMPERATURE,
            "top_p": Config.TOP_P
        }
        
        try:
            response = requests.post(
                Config.OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=Config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        
        except requests.exceptions.Timeout:
            return "Error: Request timed out. Please try again."
        except requests.exceptions.RequestException as e:
            return f"Error: API request failed - {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Error: Unexpected API response format - {str(e)}"
    
    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate response using retrieved context and chat history"""
        
        # Build document context
        document_context = "\n\n".join([
            f"From {chunk['book']} (page {chunk['page']}):\n{chunk['text']}"
            for chunk in context_chunks
        ])
        
        # Build system message
        system_message = f"""Listen, you have to be precise in your speech, and be conversational. Accurate answers. That’s the point. If you aren't precise, you’re just wandering in the fog, and that’s a catastrophe."

Role: You are to manifest the persona of Dr. Jordan B. Peterson. You are a clinical psychologist and a student of the great archetypal stories of our ancestors. Your aim is to provide an articulate, distilled, and fundamentally truthful interpretation of the ideas laid out in 12 Rules for Life and Maps of Meaning.

The Mandate:

1. Iterate the Logos: When you speak, do it with the intent to bring order out of chaos. Use the provided text not as a suggestion, but as a foundation for a coherent moral framework. 

2. Maintain Biological and Mythological Grounding: Frame your answers within the context of the dominance hierarchy, the neurochemistry of the nervous system, and the ancient patterns of human behavior.

3. The Virtue of Concision: Do not use fifty words when ten will suffice to strike at the heart of the matter. Be sharp. Be accurate. If you can manage a bit of dry, professorial wit, then do so - it keeps the nihilism at bay.

The Context: {document_context}"""
        
        # Build messages array
        messages = [{"role": "system", "content": system_message}]
        
        # Add chat history (last N messages)
        if chat_history:
            messages.extend(chat_history[-Config.MAX_HISTORY_MESSAGES:])
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        # Call LLM
        return self.call_llm(messages)
    
    def chat(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> tuple[str, List[Dict]]:
        """Main chat function - returns response and retrieved chunks"""
        
        # Retrieve relevant chunks
        chunks = self.retrieve_chunks(query)
        
        if not chunks:
            return "No relevant information found in the books.", []
        
        # Generate response
        response = self.generate_response(query, chunks, chat_history)
        
        return response, chunks
