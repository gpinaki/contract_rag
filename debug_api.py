#!/usr/bin/env python3
"""
Debug script to test API key handling - FIXED VERSION
Explicitly loads .env file only
"""

import os
import sys
from dotenv import load_dotenv

# Add backend to path
sys.path.insert(0, "backend")

def load_env_explicitly():
    """Load .env file explicitly, not .env.example"""
    print("🔧 Loading Environment File")
    print("=" * 30)
    
    # Check which files exist
    if os.path.exists('.env'):
        print("✅ Found .env file")
        result = load_dotenv('.env', override=True)  # Explicit loading
        print(f"✅ Loaded .env: {result}")
    else:
        print("❌ No .env file found")
        return False
    
    if os.path.exists('.env.example'):
        print("ℹ️  Found .env.example (this is OK - it's just a template)")
    
    return True

def test_environment():
    """Test environment variables"""
    print("\n🔍 Testing Environment Variables")
    print("=" * 50)
    
    # Explicitly load .env
    if not load_env_explicitly():
        print("❌ Cannot proceed without .env file")
        return
    
    # Check OpenAI key
    openai_key = os.getenv("OPENAI_API_KEY", "")
    print(f"OPENAI_API_KEY: {'✅ Found' if openai_key else '❌ Not found'}")
    
    if openai_key:
        print(f"📏 Key Length: {len(openai_key)} characters")
        print(f"🔑 Key Preview: {openai_key[:8]}...{openai_key[-4:]}")
        
        # Check for placeholder values - SIMPLIFIED
        placeholders = [
            "your-openai-api-key-here",
            "sk-your-openai-api-key-here"
        ]
        is_placeholder = openai_key.lower() in [p.lower() for p in placeholders]
        
        if is_placeholder:
            print(f"  ⚠️  WARNING: API key is a placeholder")
            print(f"  💡 SOLUTION: Replace with real API key from https://platform.openai.com/account/api-keys")
        else:
            print(f"  ✅ API key looks real (not a placeholder)")
    
    # Check Azure keys
    azure_key = os.getenv("AZURE_OPENAI_KEY", "")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    
    print(f"AZURE_OPENAI_KEY: {'✅ Found' if azure_key else '❌ Not found'}")
    print(f"AZURE_OPENAI_ENDPOINT: {'✅ Found' if azure_endpoint else '❌ Not found'}")
    
    if azure_key:
        azure_placeholders = ["your-azure-openai-key-here"]
        is_placeholder = any(placeholder in azure_key.lower() for placeholder in azure_placeholders)
        if is_placeholder:
            print(f"  ℹ️  Azure key is placeholder (this is OK if using Personal API)")
    
    print()

def test_api_connection():
    """Test API connection"""
    print("🔍 Testing API Connection")
    print("=" * 50)
    
    try:
        from rag_engine import ContractRAGEngine
        
        # Test with environment variables
        openai_key = os.getenv("OPENAI_API_KEY", "")
        
        if openai_key and not any(placeholder in openai_key.lower() for placeholder in [
            "your-openai-api-key-here",
            "sk-your-openai-api-key-here"
        ]):
            print("🔄 Testing with OpenAI API key from .env...")
            try:
                engine = ContractRAGEngine(api_key=openai_key, use_azure=False)
                success, message = engine.test_api_connection()
                
                if success:
                    print("✅ OpenAI API connection successful!")
                    print(f"   {message}")
                else:
                    print("❌ OpenAI API connection failed!")
                    print(f"   {message}")
            except Exception as e:
                print(f"❌ Error creating engine: {e}")
        else:
            print("❌ No valid OpenAI API key found in .env")
            print("💡 You'll need to enter it manually in the Streamlit app")
        
        # Test Azure if available
        azure_key = os.getenv("AZURE_OPENAI_KEY", "")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        
        if (azure_key and azure_endpoint and 
            not any(placeholder in azure_key.lower() for placeholder in [
                "your-azure-openai-key-here"
            ]) and 
            "your-resource" not in azure_endpoint.lower()):
            
            print("\n🔄 Testing Azure OpenAI...")
            try:
                engine = ContractRAGEngine(use_azure=True)
                success, message = engine.test_api_connection()
                
                if success:
                    print("✅ Azure OpenAI connection successful!")
                    print(f"   {message}")
                else:
                    print("❌ Azure OpenAI connection failed!")
                    print(f"   {message}")
            except Exception as e:
                print(f"❌ Error with Azure: {e}")
        else:
            print("ℹ️  Azure OpenAI not configured (this is OK if using Personal API)")
        
    except ImportError as e:
        print(f"❌ Cannot import RAG engine: {e}")
        print("💡 Make sure you're running from the project root directory")
    
    print()

def test_file_structure():
    """Test file structure"""
    print("🔍 Testing File Structure")
    print("=" * 50)
    
    required_files = [
        "backend/rag_engine.py",
        "frontend/app.py", 
        "requirements.txt",
        ".env"  # This is required
    ]
    
    optional_files = [
        ".env.example"  # This is optional
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
    
    for file_path in optional_files:
        if os.path.exists(file_path):
            print(f"ℹ️  {file_path} - (template file)")
    
    # Check data directories
    data_dirs = ["data", "embeddings"]
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/ directory exists")
        else:
            print(f"ℹ️  {dir_path}/ directory will be created automatically")
    
    print()

def main():
    """Run all tests"""
    print("🚀 Contract RAG Debug Tool - FIXED VERSION")
    print("=" * 50)
    print("This tool ensures we're using .env file (not .env.example)")
    print("=" * 50)
    print()
    
    test_file_structure()
    test_environment()
    test_api_connection()
    
    print("🎯 Next Steps:")
    print("1. Fix any ❌ issues shown above")
    print("2. Ensure .env has real API key (not placeholder)")
    print("3. Run: streamlit run frontend/app.py")
    print("4. Use 'Personal API' mode in the app")
    print()
    print("💡 File Usage:")
    print("   - .env = YOUR REAL SECRETS (used by app)")
    print("   - .env.example = TEMPLATE ONLY (not used by app)")

if __name__ == "__main__":
    main()