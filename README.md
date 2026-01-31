# Peterson Chatbot - RAG Systemmmm

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on Jordan Peterson's books "12 Rules for Life" and "Maps of Meaningggg".

## Features
- Semantic search using FAISS vector database
- Conversational AI with context memory (last 5 messages)
- Multi-book knowledge base
- Clean Streamlit interface
- Persistent vector database (no rebuild needed)
- Rate limiting for API calls

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

### 5. Configure API Key
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Get your API key from: https://openrouter.ai/keys

### 6. Build Vector Database (First Time Only)
```bash
python -c "from src.rag_system import RAGSystem; rag = RAGSystem(); rag.build_and_save_vector_db()"
```

This will process the PDFs and save the vector database to `vector_db/`.

### 7. Run the App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Ask Questions**: Type your question in the chat input
2. **View Context**: Expand "View Retrieved Context" to see which book passages were used
3. **Clear Chat**: Click "Clear Chat History" to start fresh
4. **Chat History**: Last 5 messages are maintained for context

## Project Structure

- `src/pdf_processor.py` - PDF text extraction and chunking
- `src/rag_system.py` - Vector database and LLM integration
- `src/config.py` - Configuration settings
- `src/utils.py` - Helper functions
- `app.py` - Streamlit web interface
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
- First run downloads the embedding model (~80MB)
- Subsequent runs use cached model
- API calls may take 2-5 seconds depending on response length

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
   - Ensure vector database files are accessible or rebuild on deployment

### Security Considerations
- Never commit `.env` files or API keys to version control
- Use environment variables for sensitive configuration
- Consider implementing authentication for production use
- Monitor API usage and rate limits

## License
For educational purposes only. Respects the copyright of Jordan Peterson's works.
