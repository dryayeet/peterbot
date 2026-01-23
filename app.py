import streamlit as st
from src.rag_system import RAGSystem
from src.utils import format_chunk_display
from src.config import Config

# Page configuration
st.set_page_config(
    page_title="Peterson Chatbot",
    page_icon=":book:",
    layout="centered"
)

# Custom CSS - injected via style tag to prevent display
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Improve chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Context box styling */
    .context-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #1f77b4;
    }
    
    /* Improve sidebar styling */
    .css-1d391kg {
        padding-top: 3rem;
    }
    
    /* Better spacing for main content */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.title('Peterson Chatbot')
st.markdown('<p style="text-align: center; color: #666; font-size: 1.1em; margin-bottom: 2rem;">Ask questions about "12 Rules for Life" and "Maps of Meaning"</p>', unsafe_allow_html=True)

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
    st.header("Settings")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This chatbot uses RAG (Retrieval-Augmented Generation) to answer questions based on:
    
    • 12 Rules for Life  
    • Maps of Meaning
    
    The system maintains context from your last 5 messages for more coherent conversations.
    """)
    
    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Configuration")
    st.markdown(f"**Model:** `{Config.MODEL_NAME}`")
    st.markdown(f"**Top-K Retrieval:** {Config.TOP_K_RETRIEVAL}")
    st.markdown(f"**Context Messages:** {Config.MAX_HISTORY_MESSAGES}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show retrieved context if available
        if message["role"] == "assistant" and "chunks" in message:
            with st.expander("View Retrieved Context", expanded=False):
                for i, chunk in enumerate(message["chunks"], 1):
                    st.markdown(f"**Source {i}:**")
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
        with st.expander("View Retrieved Context", expanded=False):
            for i, chunk in enumerate(chunks, 1):
                st.markdown(f"**Source {i}:**")
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
    "<div style='text-align: center; color: #888; font-size: 0.85em; padding: 1rem 0;'>Built with Streamlit | Powered by Mistral Large & FAISS</div>",
    unsafe_allow_html=True
)