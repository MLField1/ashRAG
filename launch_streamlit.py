#!/usr/bin/env python3
"""
Simple launcher script for the RAG Streamlit app
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
        'requests'
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
        return False

    return True


def setup_directories():
    """Create necessary directories"""
    directories = ['documents', 'data', 'cache']

    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {dir_path}")


def check_ollama():
    """Check if Ollama is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"🦙 Ollama is running with {len(models)} models")

            # Check for required model
            model_names = [model['name'] for model in models]
            if 'llama3.2:3b' not in model_names:
                print("⚠️  Required model 'llama3.2:3b' not found")
                print("📥 Run: ollama pull llama3.2:3b")
            else:
                print("✅ Required model 'llama3.2:3b' is available")
            return True
        else:
            print("❌ Ollama server responded with error")
            return False
    except Exception as e:
        print("❌ Ollama is not running or not accessible")
        print("🚀 Start Ollama with: ollama serve")
        return False


def main():
    """Main launcher function"""
    print("🚀 RAG System Streamlit App Launcher")
    print("=" * 50)

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

    # Check Ollama
    print("\n🦙 Checking Ollama...")
    ollama_running = check_ollama()

    if not ollama_running:
        print("\n⚠️  Ollama not running. The app will work but LLM queries will fail.")
        choice = input("Continue anyway? (y/N): ").lower().strip()
        if choice not in ['y', 'yes']:
            print("Please start Ollama and try again.")
            sys.exit(1)

    # Check for documents
    docs_path = Path(".venv/documents")
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
    print("-" * 50)

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