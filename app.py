import streamlit as st
from src.rag_system import RAGSystem
from src.utils import format_chunk_display
from src.config import Config

# Page configuration
st.set_page_config(
    page_title="Peterson Chatbot",
    page_icon="📚",
    layout="centered"
)

# Custom CSS
st.markdown("""

    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .context-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }

""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.title('📚 Peterson Chatbot')
st.markdown('Ask questions about "12 Rules for Life" and "Maps of Meaning"')

# Initialize RAG system
@st.cache_resource
def load_rag_system():
    """Load RAG system (cached to avoid reloading)"""
    try:
        rag = RAGSystem()
        rag.load_vector_db()
        return rag
    except FileNotFoundError as e:
        st.error(f"Error: {str(e)}")
        st.info("Please run: `python -c \"from src.rag_system import RAGSystem; rag = RAGSystem(); rag.build_and_save_vector_db()\"`")
        st.stop()
    except Exception as e:
        st.error(f"Error loading system: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

if st.session_state.rag_system is None:
    with st.spinner("Loading chatbot..."):
        st.session_state.rag_system = load_rag_system()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("""
    This chatbot uses RAG (Retrieval-Augmented Generation) to answer questions based on:
    - 12 Rules for Life
    - Maps of Meaning
    
    The bot maintains context from your last 5 messages.
    """)
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"**Model:** {Config.MODEL_NAME}")
    st.markdown(f"**Top-K Retrieval:** {Config.TOP_K_RETRIEVAL}")
    st.markdown(f"**Context Messages:** {Config.MAX_HISTORY_MESSAGES}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show retrieved context if available
        if message["role"] == "assistant" and "chunks" in message:
            with st.expander("📖 View Retrieved Context"):
                for i, chunk in enumerate(message["chunks"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.markdown(format_chunk_display(chunk))
                    if i < len(message["chunks"]):
                        st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask a question about Peterson's work..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, chunks = st.session_state.rag_system.chat(
                prompt,
                st.session_state.chat_history
            )
        
        st.markdown(response)
        
        # Show retrieved context
        with st.expander("📖 View Retrieved Context"):
            for i, chunk in enumerate(chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.markdown(format_chunk_display(chunk))
                if i < len(chunks):
                    st.markdown("---")
    
    # Update chat history for LLM context
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Add assistant message to display
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "chunks": chunks
    })

# Footer
st.markdown("---")
st.markdown(
    ""
    "Built with Streamlit • Powered by Mistral Large & FAISS"
    "",
    unsafe_allow_html=True
)