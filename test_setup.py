#!/usr/bin/env python3
"""
Quick test script to verify the contract_rag setup
Run this from the root directory: python test_setup.py
"""

import os
import sys
from pathlib import Path

def test_file_structure():
    """Test if files are in the right places"""
    print("🔍 Checking file structure...")
    
    required_files = [
        "backend/rag_engine.py",
        "frontend/app.py", 
        "requirements.txt",
        ".env.example"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
    
    print()

def test_environment():
    """Test environment variables"""
    print("🔍 Checking environment setup...")
    
    if os.path.exists(".env"):
        print("✅ .env file exists")
        
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check for at least one API key
        openai_key = os.getenv("OPENAI_API_KEY")
        azure_key = os.getenv("AZURE_OPENAI_KEY")
        
        if openai_key and openai_key != "sk-your-openai-api-key-here":
            print("✅ OpenAI API key configured")
        elif azure_key and azure_key != "your-azure-openai-key-here":
            print("✅ Azure OpenAI API key configured")
        else:
            print("⚠️  No API keys configured (this is OK for testing structure)")
    else:
        print("⚠️  .env file not found (copy from .env.example)")
    
    print()

def test_imports():
    """Test if we can import the RAG engine"""
    print("🔍 Testing imports...")
    
    try:
        # Add backend to path
        sys.path.insert(0, "backend")
        
        from rag_engine import ContractRAGEngine
        print("✅ Can import ContractRAGEngine")
        
        # Test creating instance (without API keys)
        try:
            engine = ContractRAGEngine(use_azure=False)
            print("✅ Can create RAG engine instance")
            
            # Test hello message
            message = engine.get_hello_message("Test User")
            print(f"✅ get_hello_message works: {message}")
            
        except Exception as e:
            print(f"⚠️  RAG engine creation failed (likely missing API keys): {e}")
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    
    print()

def test_dependencies():
    """Test if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "streamlit",
        "openai", 
        "chromadb",
        "PyPDF2",
        "python-docx",
        "langchain",
        "python-dotenv"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - run: pip install {package}")
    
    print()

def main():
    """Run all tests"""
    print("🚀 Contract RAG Setup Test")
    print("=" * 50)
    
    test_file_structure()
    test_environment() 
    test_dependencies()
    test_imports()
    
    print("=" * 50)
    print("🎯 Next Steps:")
    print("1. Fix any ❌ issues above")
    print("2. Copy .env.example to .env and add your API keys")
    print("3. Run: streamlit run frontend/app.py")

if __name__ == "__main__":
    main()