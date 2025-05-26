import streamlit as st
import os
import sys
from dotenv import load_dotenv
from typing import Tuple

# Add backend to path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(current_dir, '..', 'backend')
sys.path.insert(0, backend_path)

try:
    from rag_engine import ContractRAGEngine
except ImportError:
    st.error("❌ Cannot import RAG engine. Please check file structure.")
    st.stop()

# Load environment variables
load_dotenv('.env', override=True)

# Page configuration
st.set_page_config(
    page_title="Contract Analyzer Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}
    if "temp_openai_key" not in st.session_state:
        st.session_state.temp_openai_key = ""
    if "use_personal_api" not in st.session_state:
        st.session_state.use_personal_api = True  # Default to Personal mode
    if "api_status" not in st.session_state:
        st.session_state.api_status = {"valid": False, "message": ""}

def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """Validate API key format - SIMPLIFIED"""
    if not api_key or not api_key.strip():
        return False, "API key is empty"
    
    key = api_key.strip()
    
    # Check for the 2 actual common placeholders
    common_placeholders = [
        "your-openai-api-key-here",
        "sk-your-openai-api-key-here"
    ]
    
    if key.lower() in [p.lower() for p in common_placeholders]:
        return False, "API key is a placeholder - please replace with your real key"
    
    # Basic format validation
    if not key.startswith("sk-"):
        return False, "OpenAI API key should start with 'sk-'"
    
    if len(key) < 40:  # Real keys are 50+ chars
        return False, "API key appears to be too short"
    
    return True, "API key format is valid"

def create_rag_engine(use_azure: bool = False):
    """Create RAG engine with completely independent modes"""
    try:
        if use_azure:
            # Azure mode - only check Azure credentials
            azure_key = os.getenv("AZURE_OPENAI_KEY", "")
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
            
            # Simple Azure validation
            if not azure_key or not azure_endpoint:
                st.session_state.api_status = {"valid": False, "message": "❌ Azure credentials missing from environment"}
                return None
            
            # Check for Azure placeholders - BOTH key and endpoint
            if ("your-azure" in azure_key.lower() or 
                "your-resource" in azure_endpoint.lower()):
                st.session_state.api_status = {"valid": False, "message": "❌ Azure credentials are placeholders"}
                return None
            
            # Create Azure engine
            engine = ContractRAGEngine(use_azure=True)
            success, message = engine.test_api_connection()
            
            if success:
                st.session_state.api_status = {"valid": True, "message": "✅ Azure OpenAI connected"}
                return engine
            else:
                st.session_state.api_status = {"valid": False, "message": f"❌ Azure connection failed: {message}"}
                return None
                
        else:
            # Personal mode - only check Personal API key
            api_key = st.session_state.get("temp_openai_key", "")
            
            # If no key in session, try environment (but ignore placeholders)
            if not api_key:
                env_key = os.getenv("OPENAI_API_KEY", "")
                if env_key and not any(placeholder in env_key.lower() for placeholder in [
                    "your-openai-api-key-here", "sk-your-openai-api-key-here"
                ]):
                    api_key = env_key
            
            if not api_key:
                st.session_state.api_status = {"valid": False, "message": "❌ No Personal OpenAI API key provided"}
                return None
            
            # Validate Personal API key format
            is_valid, validation_msg = validate_api_key(api_key)
            if not is_valid:
                st.session_state.api_status = {"valid": False, "message": f"❌ {validation_msg}"}
                return None
            
            # Create Personal engine - but don't auto-test connection
            engine = ContractRAGEngine(api_key=api_key, use_azure=False)
            st.session_state.api_status = {"valid": True, "message": "✅ API key configured"}
            return engine
                
    except Exception as e:
        st.session_state.api_status = {"valid": False, "message": f"❌ Setup error: {str(e)}"}
        return None

def sidebar_ui():
    """Render sidebar UI with dropdown selection"""
    st.sidebar.markdown('<div class="sidebar-header">⚙️ API Configuration</div>', unsafe_allow_html=True)
    
    # API mode selection - DROPDOWN instead of toggle
    api_mode = st.sidebar.selectbox(
        "Select API Mode",
        options=["Personal OpenAI", "Azure OpenAI"],
        index=0 if st.session_state.use_personal_api else 1,
        help="Choose between Personal OpenAI API or Corporate Azure OpenAI"
    )
    
    # Convert dropdown selection to boolean
    use_personal = (api_mode == "Personal OpenAI")
    
    # Update session state if changed
    if use_personal != st.session_state.use_personal_api:
        st.session_state.use_personal_api = use_personal
        st.session_state.rag_engine = None  # Force recreation
        st.session_state.api_status = {"valid": False, "message": ""}
    
    config_valid = False
    
    if use_personal:
        # PERSONAL MODE ONLY
        with st.sidebar.expander("🔐 API Key Configuration", expanded=True):
            new_api_key = st.text_input(
                "Enter OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Your personal OpenAI API key",
                value=st.session_state.temp_openai_key
            )
            
            # Update key if changed
            if new_api_key != st.session_state.temp_openai_key:
                st.session_state.temp_openai_key = new_api_key
                st.session_state.rag_engine = None  # Force recreation
                st.session_state.api_status = {"valid": False, "message": ""}
            
            # Optional test connection button
            if st.button("🔄 Test Connection", help="Test your API key"):
                if st.session_state.rag_engine:
                    success, message = st.session_state.rag_engine.test_api_connection()
                    if success:
                        st.success("✅ Connection successful!")
                    else:
                        st.error(f"❌ {message}")
        
        # Try to create Personal engine
        if st.session_state.rag_engine is None:
            st.session_state.rag_engine = create_rag_engine(use_azure=False)
            
    else:
        # AZURE MODE ONLY
        st.sidebar.info("🔷 **Azure OpenAI Mode**")
        st.sidebar.info("Contact IT admin for setup")
        
        # Only try to create Azure engine if user explicitly requests it
        if st.sidebar.button("🔄 Test Azure Connection"):
            st.session_state.rag_engine = create_rag_engine(use_azure=True)
    
    # Show status for current mode only
    if st.session_state.api_status["message"]:
        if st.session_state.api_status["valid"]:
            config_valid = True
        else:
            st.sidebar.error(st.session_state.api_status["message"])
    
    st.sidebar.markdown("---")
    
    # Document upload section - works for both modes when valid
    if config_valid and st.session_state.rag_engine:
        st.sidebar.markdown('<div class="sidebar-header">📄 Document Upload</div>', unsafe_allow_html=True)
        
        uploaded_files = st.sidebar.file_uploader(
            "Choose contract files",
            type=['pdf', 'docx', 'doc'],
            accept_multiple_files=True,
            help="Upload contract documents (PDF, DOC, DOCX) - max 10 files"
        )
        
        if uploaded_files:
            if len(uploaded_files) > 10:
                st.sidebar.error("❌ Maximum 10 files allowed")
            else:
                process_uploaded_files(uploaded_files)
        
        # Clear data button
        if st.sidebar.button("🗑️ Clear All Data", help="Remove all files and embeddings"):
            clear_all_data()
    else:
        st.sidebar.markdown("📄 **Document Upload**")
        current_mode = "Personal API key" if use_personal else "Azure OpenAI connection"
        st.sidebar.info(f"👆 Complete {current_mode} setup first")
    
    return config_valid

def process_uploaded_files(uploaded_files):
    """Process uploaded files with simplified feedback"""
    for uploaded_file in uploaded_files:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # Check if already processed successfully - but don't show message
        if st.session_state.processed_files.get(file_key) == "success":
            continue
        
        # Show processing status
        with st.sidebar.container():
            status_placeholder = st.empty()
            status_placeholder.info(f"🔄 Processing {uploaded_file.name}...")
            
            try:
                # Save file (with replacement option)
                success, message = st.session_state.rag_engine.save_uploaded_file(
                    uploaded_file, 
                    replace_existing=True
                )
                
                if success:
                    # Process and store in vector database
                    file_path = os.path.join(st.session_state.rag_engine.data_dir, uploaded_file.name)
                    process_success, process_message = st.session_state.rag_engine.process_and_store_document(file_path)
                    
                    if process_success:
                        st.session_state.processed_files[file_key] = "success"
                        status_placeholder.success(f"✅ {uploaded_file.name}")
                    else:
                        st.session_state.processed_files[file_key] = "failed"
                        status_placeholder.error(f"❌ {uploaded_file.name} - Failed")
                else:
                    st.session_state.processed_files[file_key] = "failed"
                    status_placeholder.error(f"❌ {uploaded_file.name} - {message}")
                    
            except Exception as e:
                st.session_state.processed_files[file_key] = "failed"
                status_placeholder.error(f"❌ {uploaded_file.name} - Error")

def clear_all_data():
    """Clear all data with confirmation"""
    if st.session_state.rag_engine:
        if st.session_state.rag_engine.clear_all_data():
            st.session_state.processed_files.clear()
            st.session_state.chat_history.clear()
            st.sidebar.success("✅ All data cleared successfully")
            st.rerun()
        else:
            st.sidebar.error("❌ Failed to clear data")

def main_chat_ui(config_ready):
    """Main chat interface"""
    st.markdown('<div class="main-header">📄 Contract Analyzer Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered contract analysis and procurement insights</div>', unsafe_allow_html=True)
    
    if not config_ready:
        # Show configuration help
        st.info("👈 **Please complete the API configuration in the sidebar to get started**")
        
        # Show API status if there's an error
        if st.session_state.api_status["message"] and not st.session_state.api_status["valid"]:
            st.error("**Configuration Issue:**")
            st.error(st.session_state.api_status["message"])
            
            # Provide specific help based on the error
            if "placeholder" in st.session_state.api_status["message"].lower():
                if "azure" in st.session_state.api_status["message"].lower():
                    st.info("**How to fix:** Contact your IT admin for Azure OpenAI setup, or switch to Personal OpenAI mode.")
                else:
                    st.info("**How to fix:** Replace the placeholder in your .env file with a real API key, or enter your key in the sidebar.")
            elif "api key" in st.session_state.api_status["message"].lower():
                st.info("**How to fix:** Get your API key from https://platform.openai.com/account/api-keys and enter it in the sidebar.")
        
        # Show about section
        with st.expander("💡 **About Contract Analyzer Agent**", expanded=True):
            st.markdown("""
            **What it does:**
            - Analyzes IT contracts and SOWs (Service Level Agreements)
            - Answers procurement questions about vendors, costs, and technologies  
            - Provides source citations for all responses
            
            **Supported formats:** PDF, DOC, DOCX (max 10MB each)
            
            **Example questions:**
            - "How many Cognizant contracts are there?"
            - "What's the total value of all contracts?"
            - "Which SOWs address Gen AI projects?"
            - "What contracts are expiring soon?"
            """)
        return
    
    if not st.session_state.rag_engine:
        st.warning("⚠️ RAG engine not properly initialized. Please check the sidebar configuration.")
        return
    
    # Query Input Section
    st.markdown("### 💬 Ask Your Question")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Enter your contract question:",
            placeholder="e.g., How many contracts are expiring this year?",
            help="Ask about contract values, vendors, technologies, dates, etc.",
            label_visibility="collapsed"
        )
    
    with col2:
        analyze_button = st.button("🔍 **Analyze**", type="primary", use_container_width=True)
    
    # Process query on button click only
    if analyze_button and query.strip():
        process_query(query)
    elif analyze_button:
        st.warning("Please enter a question first.")
    
    # Chat History with pagination - show last 3 conversations
    # Always show this section (even if empty) so answers appear immediately
    with st.expander("📜 **Recent Conversations** (Last 3 conversations displayed)", expanded=True):
        if st.session_state.chat_history:
            # Show only last 3 conversations for better performance
            recent_chats = st.session_state.chat_history[-3:]
            for i, (query_hist, response, sources) in enumerate(recent_chats):
                # Simple numbering starting from the actual position
                question_num = len(st.session_state.chat_history) - len(recent_chats) + i + 1
                st.markdown(f"**🙋 Question {question_num}:** {query_hist}")
                st.markdown(f"**🤖 Answer:** {response}")
                if i < len(recent_chats) - 1:
                    st.markdown("---")
        else:
            st.info("💬 Your conversation history will appear here after you ask your first question.")

def process_query(query: str):
    """Process user query and display response"""
    if not st.session_state.rag_engine:
        st.error("❌ RAG engine not initialized")
        return
    
    # Check if documents are uploaded
    stats = st.session_state.rag_engine.get_collection_stats()
    if stats["total_files"] == 0:
        st.warning("📄 **Please upload some contract documents first using the sidebar.**")
        return
    
    if stats["total_chunks"] == 0:
        st.warning("📄 **No document content available. Please check if documents were processed successfully.**")
        return
    
    # Process the query
    with st.spinner("🔍 Analyzing contracts..."):
        try:
            response, sources = st.session_state.rag_engine.query_contracts(query)
            
            # Add to chat history
            st.session_state.chat_history.append((query, response, sources))
            
            # Force a rerun to immediately show the new conversation
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.error("Please check your API configuration and try again.")

def main():
    """Main application function"""
    initialize_session_state()
    
    # Sidebar configuration
    config_ready = sidebar_ui()
    
    # Main chat interface
    main_chat_ui(config_ready)

if __name__ == "__main__":
    main()