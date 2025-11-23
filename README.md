# Educational RAG System

A hardware-adaptive Retrieval-Augmented Generation (RAG) system designed for educational use, particularly focused on electrical utility safety documentation. The system automatically optimizes performance based on available hardware and provides safety-critical content detection and prioritization.

## Features

- **Hardware-Adaptive Performance**: Automatically detects and optimizes for available CPU, GPU, and memory resources
- **Safety Content Detection**: Identifies and prioritizes safety-critical information in electrical utility documents
- **Enhanced Chunking**: Uses recursive text splitting with intelligent overlap for better context preservation
- **Web Interface**: Clean Streamlit-based interface for easy interaction
- **Evaluation Pipeline**: Built-in quality assessment with transparent metrics
- **Caching System**: Smart caching to improve response times
- **Multiple Document Formats**: Supports .txt and .md files

## Quick Start (llama.cpp version)

### Prerequisites

- Python 3.8+
- 4GB+ RAM recommended
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) installed
- [GGUF model file](https://huggingface.co/models?search=gguf) downloaded (recommended: Llama-3.2-3B-Instruct-Q4_K_M.gguf)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd educational-rag-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# For NVIDIA GPU acceleration:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

3. Download a GGUF model and place it in the `models/` directory:
```bash
mkdir -p models
# Download your preferred .gguf model (e.g., from HuggingFace)
# Example (update as needed):
# wget <huggingface-gguf-model-link> -O models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

4. Add your documents:
```bash
mkdir documents
# Place your .txt or .md files in the documents/ directory
```

5. Run the system:
```bash
# Option 1: Web interface
python launch_streamlit.py

# Option 2: Command line interface
python main.py
```

## File Structure

```
├── main.py                    # Core RAG system implementation
├── streamlit_rag_app.py      # Web interface
├── launch_streamlit.py       # Streamlit launcher script
├── rag_config.json          # Configuration file (auto-generated)
├── documents/               # Directory for source documents
├── data/                    # Generated embeddings and chunks
└── cache/                   # Query cache directory
```

## Configuration

The system automatically creates and optimizes a configuration file (`rag_config.json`) based on your hardware. Key settings include:
- `llm_provider`: "llamacpp"
- `model_path`: Path to your downloaded GGUF file
- Hardware and retrieval parameters

Edit `rag_config.json` to customize:
- Chunking parameters
- Safety detection settings
- Retrieval parameters
- Cache settings
- Evaluation thresholds

## Usage

### Web Interface
Access the Streamlit interface at `http://localhost:8501` after running the launcher script. The interface provides:

- System status and hardware information
- Document upload and index building
- Interactive query interface
- Performance statistics and evaluation results

### Command Line
Run `python main.py` for a console-based interface with the same functionality.

### Example Queries

The system is optimized for electrical safety documentation but works with any technical content:

- "What are the lockout tagout procedures?"
- "What PPE is required for arc flash protection?"
- "Explain electrical grounding safety requirements"

## System Requirements

### Minimum
- 4GB RAM
- 2 CPU cores
- 1GB free disk space

### Recommended
- 8GB+ RAM
- 4+ CPU cores
- GPU with 4GB+ VRAM (optional, improves performance)
- SSD storage

## Hardware Optimization

The system automatically detects and optimizes for:

- **Performance Tiers**: LOW, MEDIUM, HIGH, ULTRA based on hardware capabilities
- **GPU Acceleration**: Automatic GPU detection and utilization when available
- **Memory Management**: Adaptive caching and batch sizes based on available RAM
- **Parallel Processing**: Utilizes multiple CPU cores for document processing

## Development

### Key Components

- **HardwareDetector**: Analyzes system capabilities and suggests optimal parameters
- **SafetyKeywordDetector**: Identifies safety-critical content in documents
- **RecursiveTextSplitter**: Intelligent document chunking with overlap
- **HardwareAdaptiveRAGSystem**: Main orchestration class
- **RAGEvaluator**: Quality assessment and benchmarking

## Troubleshooting

### Common Issues

1. **Model file not found**: Ensure a `.gguf` model file is in the `models/` directory and `rag_config.json` points to its correct path.
2. **llama-cpp-python not installed**: Run `pip install llama-cpp-python`, and for GPU users: `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`
3. **Out of Memory**: Reduce the `embedding_batch_size` in your config or use a smaller model variant.
4. **No Documents Found**: Make sure `.txt` or `.md` files are inside the `documents/` directory.

### Performance Tips

- Use SSD storage for better I/O performance
- Ensure adequate RAM for your document collection size
- GPU acceleration significantly improves embedding generation
- Clear cache periodically if experiencing memory issues

## License

This project is provided for educational use. Please ensure compliance with your institution's policies when using with copyrighted educational materials.

## Contributing

Contributions are welcome. Please focus on:

- Educational use case improvements
- Hardware compatibility enhancements
- Documentation quality
- Performance optimizations

## Support

For issues related to setup or usage, please check the troubleshooting section or create an issue with your system specifications and error details.
