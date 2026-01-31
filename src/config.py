import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL_NAME = "mistralai/mistral-large-2512"
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE = 20
    REQUEST_TIMEOUT = 30  # seconds
    
    # RAG Configuration
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    TOP_K_RETRIEVAL = 4
    
    # LLM Generation Parameters
    MAX_TOKENS = 200
    TEMPERATURE = 1.0
    TOP_P = 0.9
    
    # Paths
    DATASETS_PATH = "datasets"
    VECTOR_DB_PATH = "vector_db"
    FAISS_INDEX_FILE = os.path.join(VECTOR_DB_PATH, "index.faiss")
    METADATA_FILE = os.path.join(VECTOR_DB_PATH, "metadata.json")
    
    # PDF Configuration
    PDF_FILES = {
        "12 Rules for Life": "12-Rules-for-Life.pdf",
        "Maps of Meaning": "Maps-of-Meaning.pdf"
    }
    
    # Chat Configuration
    MAX_HISTORY_MESSAGES = 8  # Last 5 messages for context
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")
        
        # Create directories if they don't exist
        os.makedirs(cls.DATASETS_PATH, exist_ok=True)
        os.makedirs(cls.VECTOR_DB_PATH, exist_ok=True)
