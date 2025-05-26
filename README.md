# 📄 Contract Advisor Assistant

> **Conversational AI for contract analysis and vendor insights**

A prototype RAG (Retrieval-Augmented Generation) application that helps procurement teams analyze IT contracts and SOWs (Service Level Agreements) through natural language queries.

## 🚀 Quick Start

### **Prerequisites**
- Python 3.12.6
- Virtual environment (recommended)
- OpenAI API key OR Azure OpenAI access

### **Local Setup (5 minutes)**
```bash
# 1. Clone/navigate to project
cd contract_rag

# 2. Create virtual environment
python -m venv ragenv
source ragenv/bin/activate  # On Windows: ragenv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your Azure OpenAI credentials (or leave for UI input)

# 5. Run the application
streamlit run frontend/app.py

# 6. Open browser
# Navigate to: http://localhost:8501
```

### **Test Setup**
```bash
# Verify everything is working
python test_setup.py
```

## 🏗️ Architecture

### **Technology Stack**
- **Frontend**: Streamlit (Python web framework)
- **Backend**: LangChain + OpenAI/Azure OpenAI
- **Vector Database**: ChromaDB (for document embeddings)
- **Document Processing**: PyPDF2, python-docx
- **Deployment**: Docker + Kubernetes

## 🔐 Security & API Management

### **Two API Modes**

#### **Default: Azure OpenAI (Recommended for Enterprise)**
- Uses corporate Azure OpenAI credentials
- Managed by IT department
- Configured via environment variables
- Enterprise-grade security and compliance

#### **Personal: OpenAI API (For Development)**
- Enter personal OpenAI API key via secure UI
- Keys stored only in browser session (not on disk)
- Ideal for prototyping and testing

### **API Configuration**
```bash
# Azure OpenAI (.env file)
AZURE_OPENAI_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Personal OpenAI (via UI - more secure)
# Toggle to "Personal API" and enter key in interface
```

## 📁 Project Structure

```
contract_rag/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example             # Environment template
├── backend/
│   └── rag_engine.py        # Core RAG logic
├── frontend/
│   └── app.py              # Streamlit UI
```

### **Local Development**
Perfect for testing and demos:
```bash
streamlit run frontend/app.py
```



