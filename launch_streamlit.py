#!/usr/bin/env python3
"""
Simple launcher script for the RAG Streamlit app with llama.cpp support
"""

import subprocess
import sys
import os
from pathlib import Path


def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'sentence-transformers',
        'numpy',
        'requests',
        'llama-cpp-python'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\n💡 For GPU support (NVIDIA), use:")
        print('CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python')
        return False

    return True


def setup_directories():
    """Create necessary directories"""
    directories = ['documents', 'data', 'cache', 'models']

    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {dir_path}")


def check_llama_model():
    """Check if llama.cpp model is available"""
    models_path = Path("models")
    
    if not models_path.exists():
        print("❌ Models directory not found")
        print("📁 Creating models directory...")
        models_path.mkdir(parents=True, exist_ok=True)
    
    # Check for GGUF files
    gguf_files = list(models_path.glob("*.gguf"))
    
    if not gguf_files:
        print("❌ No GGUF model files found in models/")
        print("📥 Please download a GGUF model file, for example:")
        print("   - Llama-3.2-3B-Instruct-Q4_K_M.gguf")
        print("   - Place it in the models/ directory")
        print("\n💡 You can download from HuggingFace:")
        print("   https://huggingface.co/models?search=gguf")
        return False
    else:
        print(f"✅ Found {len(gguf_files)} GGUF model(s):")
        for model_file in gguf_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"   - {model_file.name} ({size_mb:.1f} MB)")
        return True


def main():
    """Main launcher function"""
    print("🚀 RAG System Streamlit App Launcher (llama.cpp)")
    print("=" * 60)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)

    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")

    # Check requirements
    print("\n📦 Checking package requirements...")
    if not check_requirements():
        sys.exit(1)
    print("✅ All packages available")

    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()

    # Check for llama.cpp model
    print("\n🦙 Checking for llama.cpp models...")
    model_available = check_llama_model()

    if not model_available:
        print("\n⚠️  No models found. The app will start but LLM queries will fail.")
        choice = input("Continue anyway? (y/N): ").lower().strip()
        if choice not in ['y', 'yes']:
            print("Please download a GGUF model and try again.")
            sys.exit(1)

    # Check for documents
    docs_path = Path("documents")
    txt_files = list(docs_path.glob("*.txt"))
    md_files = list(docs_path.glob("*.md"))
    total_files = len(txt_files) + len(md_files)

    print(f"\n📚 Document check: {total_files} files found")
    if total_files == 0:
        print("⚠️  No documents found in documents/ folder")
        print("📄 Add .txt or .md files to documents/ for the system to work")

    # Launch Streamlit
    print("\n🌐 Launching Streamlit app...")
    print("📱 App will open in your browser at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 60)

    try:
        # Get the directory of this script
        script_dir = Path(__file__).parent
        app_file = script_dir / "streamlit_rag_app.py"

        if not app_file.exists():
            print(f"❌ Streamlit app file not found: {app_file}")
            print("Make sure streamlit_rag_app.py is in the same directory")
            sys.exit(1)

        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_file),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--server.headless", "false"
        ])

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()