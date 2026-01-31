"""
Embedding Service - FastAPI server for text embeddings

This service loads the embedding model once on startup and provides
HTTP endpoints for encoding text. It runs independently from the
Streamlit app, allowing the app to start instantly without loading models.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from sentence_transformers import SentenceTransformer
from src.config import Config

# Initialize FastAPI app
app = FastAPI(
    title="Embedding Service",
    description="Text embedding service for RAG system",
    version="1.0.0"
)

# Global model variable - loaded once on startup
embedding_model = None


class EncodeRequest(BaseModel):
    text: str


class EncodeBatchRequest(BaseModel):
    texts: List[str]


class EncodeResponse(BaseModel):
    embedding: List[float]


class EncodeBatchResponse(BaseModel):
    embeddings: List[List[float]]


@app.on_event("startup")
async def load_model():
    """Load embedding model on startup"""
    global embedding_model
    print(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    print("Embedding model loaded successfully!")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": embedding_model is not None,
        "model_name": Config.EMBEDDING_MODEL
    }


@app.post("/encode", response_model=EncodeResponse)
async def encode_text(request: EncodeRequest):
    """Encode a single text string into an embedding vector"""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    try:
        embedding = embedding_model.encode([request.text])[0]
        return EncodeResponse(embedding=embedding.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")


@app.post("/encode_batch", response_model=EncodeBatchResponse)
async def encode_batch(request: EncodeBatchRequest):
    """Encode multiple text strings into embedding vectors"""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="Empty texts list")
    
    try:
        embeddings = embedding_model.encode(
            request.texts,
            show_progress_bar=False,
            batch_size=32
        )
        return EncodeBatchResponse(embeddings=[emb.tolist() for emb in embeddings])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch encoding failed: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run embedding service")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    args = parser.parse_args()
    
    print(f"Starting embedding service on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

