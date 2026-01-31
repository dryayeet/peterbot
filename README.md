# Peterson Chatbot - RAG Systemmmm

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on Jordan Peterson's books "12 Rules for Life" and "Maps of Meaningggg".

## Features
- Semantic search using FAISS vector database
- Conversational AI with context memory (last 5 messages)
- Multi-book knowledge base
- Clean Streamlit interface
- Persistent vector database (no rebuild needed)
- Rate limiting for API calls
- **Separate embedding service** - Fast startup, model stays loaded

## Setup

### 1. Clone or Download the Project
```bash
cd peterson-chatbot
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Your PDFs
Place your PDF files in the `datasets/` folder:
- `12-Rules-for-Life.pdf`
- `Maps-of-Meaning.pdf`

### 5. Configure API Key and Embedding Service
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Embedding service configuration
EMBEDDING_SERVICE_URL=http://localhost:8000
EMBEDDING_SERVICE_ENABLED=true
```

Get your API key from: https://openrouter.ai/keys

**Note:** If `EMBEDDING_SERVICE_ENABLED=false`, the app will load the model locally (slower startup but no separate service needed).

### 6. Build Vector Database (First Time Only)
```bash
python -c "from src.rag_system import RAGSystem; rag = RAGSystem(); rag.build_and_save_vector_db()"
```

This will process the PDFs and save the vector database to `vector_db/`.

### 7. Start Embedding Service (Recommended)

The embedding service runs separately and keeps the model loaded in memory, allowing the Streamlit app to start instantly.

**Option A: Start services separately (recommended for development)**

Terminal 1 - Start embedding service:
```bash
python embedding_service.py
# Or use the helper script:
python start_embedding_service.py
```

Terminal 2 - Start Streamlit app:
```bash
streamlit run app.py
```

**Option B: Start both services together**

Windows:
```bash
start_all.bat
```

Linux/Mac:
```bash
chmod +x start_all.sh
./start_all.sh
```

The embedding service runs on `http://localhost:8000` by default.  
The Streamlit app opens at `http://localhost:8501`

**Note:** If the embedding service is not running, the app will automatically fall back to loading the model locally (slower startup).

## Usage

1. **Ask Questions**: Type your question in the chat input
2. **View Context**: Expand "View Retrieved Context" to see which book passages were used
3. **Clear Chat**: Click "Clear Chat History" to start fresh
4. **Chat History**: Last 5 messages are maintained for context

## Architecture

The system uses a **microservices architecture** with separate embedding service:

```
┌─────────────────────┐
│ Embedding Service   │  ← FastAPI server (port 8000)
│ (Long-running)      │  ← Model loaded once, stays in memory
│ - /encode           │
│ - /encode_batch     │
└──────────┬──────────┘
           │ HTTP API
           │
┌──────────▼──────────┐
│ Streamlit App       │  ← Streamlit (port 8501)
│ (Can restart freely) │  ← Starts instantly, no model loading
│ - RAGSystem         │
│ - Chat interface    │
└─────────────────────┘
```

**Benefits:**
- Streamlit app starts instantly (no model loading delay)
- Embedding service runs independently (survives app restarts)
- Model loaded once, reused for all requests
- Production-ready architecture

## Project Structure

- `embedding_service.py` - FastAPI embedding service (runs separately)
- `src/pdf_processor.py` - PDF text extraction and chunking
- `src/rag_system.py` - Vector database and LLM integration
- `src/config.py` - Configuration settings
- `src/utils.py` - Helper functions
- `app.py` - Streamlit web interface
- `start_embedding_service.py` - Helper script to start embedding service
- `start_all.bat` / `start_all.sh` - Scripts to start both services
- `datasets/` - PDF storage
- `vector_db/` - FAISS index and metadata

## Technical Details

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: Mistral Large 2 (via OpenRouter API)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Chunk Size**: 1000 characters with 200 character overlap
- **Retrieval**: Top 3 most relevant chunks per query

## Troubleshooting

### "Vector database not found"
Run the build command from step 6 above.

### "API key not found"
Check that your `.env` file exists and contains `OPENROUTER_API_KEY=...`

### Slow performance
- **With embedding service**: App starts instantly, model loads once in service
- **Without embedding service**: First run downloads the embedding model (~80MB)
- API calls may take 2-5 seconds depending on response length

### "Embedding service unavailable"
- Make sure the embedding service is running on port 8000
- Check: `curl http://localhost:8000/health`
- The app will automatically fall back to local model if service is unavailable
- To disable service and use local model: Set `EMBEDDING_SERVICE_ENABLED=false` in `.env`

## Deployment

### Local Deployment
The application runs on `http://localhost:8501` by default when using Streamlit.

### Production Deployment
For production deployment, consider:

1. **Streamlit Cloud**: Deploy directly from GitHub
   - Connect your repository to Streamlit Cloud
   - Set environment variables (OPENROUTER_API_KEY) in the dashboard
   - Ensure `requirements.txt` is up to date

2. **Docker Deployment**:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

3. **Environment Variables**:
   - Set `OPENROUTER_API_KEY` in your deployment environment
   - Set `EMBEDDING_SERVICE_URL` if embedding service is on different host/port
   - Ensure vector database files are accessible or rebuild on deployment

4. **Embedding Service Deployment**:
   - Deploy embedding service separately (e.g., as a Docker container)
   - Update `EMBEDDING_SERVICE_URL` in Streamlit app to point to deployed service
   - Service can be scaled independently based on load

### Security Considerations
- Never commit `.env` files or API keys to version control
- Use environment variables for sensitive configuration
- Consider implementing authentication for production use
- Monitor API usage and rate limits

## License
For educational purposes only. Respects the copyright of Jordan Peterson's works.
