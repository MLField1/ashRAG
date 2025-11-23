# llama.cpp Migration Guide

This guide explains the changes needed to migrate from Ollama to llama.cpp.

## Summary of Changes

This migration removes the external Ollama dependency and replaces it with embedded llama.cpp for direct model loading and inference.

## Files Updated

- ✅ `requirements.txt` - Added llama-cpp-python
- ✅ `rag_config.json` - Replaced Ollama settings with llama.cpp settings
- ✅ `launch_streamlit.py` - Updated checks and model detection
- ⚠️ `main.py` - Requires manual updates (see below)
- ⚠️ `streamlit_rag_app.py` - Requires minor updates (see below)

## Manual Changes Required

### 1. main.py Changes

#### A. Add import at top of file (after line 15)

```python
# Add this import block after the existing imports
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None
    logger.error("llama-cpp-python not installed. Inference will fail.")
```

#### B. Update RAGConfig class (around line 700)

Replace these lines in the RAGConfig dataclass:

```python
# OLD - DELETE THESE:
ollama_model: str = 'Llama-3.2-3B-Finetune-q4k.gguf'
ollama_host: str = 'http://localhost:11434'
warm_ollama_on_startup: bool = True
ollama_keep_alive: str = "24h"

# NEW - ADD THESE:
llm_provider: str = "llamacpp"
model_path: str = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
max_context_length: int = 3000
temperature: float = 0.1
warm_llm_on_startup: bool = True
```

#### C. Update RAGConfig.ensure_new_parameters() method

Add these to the defaults dict in ensure_new_parameters():

```python
'llm_provider': getattr(self, 'llm_provider', 'llamacpp'),
'model_path': getattr(self, 'model_path', 'models/Llama-3.2-3B-Instruct-Q4_K_M.gguf'),
'warm_llm_on_startup': getattr(self, 'warm_llm_on_startup', True),
```

#### D. Replace entire OllamaClient class (around line 1800)

Delete the entire `OllamaClient` class and replace with:

```python
class LlamaCppClient:
    """Hardware-Adaptive Client for Llama.cpp"""
    
    def __init__(self, config: RAGConfig, hardware_detector: HardwareDetector):
        if Llama is None:
            raise ImportError("llama-cpp-python not installed. pip install llama-cpp-python")
            
        self.config = config
        self.hw_info = hardware_detector.hardware_info
        self.model_path = config.model_path
        
        if not os.path.exists(self.model_path):
             logger.error(f"❌ Model not found at {self.model_path}")
             raise FileNotFoundError(f"Model file missing: {self.model_path}")

        # Hardware Adaptive GPU Offloading
        n_gpu_layers = 0
        if self.hw_info.has_gpu:
            if self.hw_info.gpu_vram_gb >= 4: 
                n_gpu_layers = -1  # -1 means "all layers"
            elif self.hw_info.gpu_vram_gb >= 2:
                n_gpu_layers = 15  # Partial offload for weaker GPUs
        
        logger.info(f"🦙 Loading Llama.cpp (GPU Layers: {n_gpu_layers}, Threads: {max(1, self.hw_info.cpu_count - 2)})")
        
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=config.max_context_length,
                n_gpu_layers=n_gpu_layers,
                n_threads=max(1, self.hw_info.cpu_count - 2),
                verbose=False
            )
            
            if config.warm_llm_on_startup:
                self._warm_model()
                
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            raise e

    def _warm_model(self):
        logger.info("🔥 Warming Llama model...")
        try:
            self.llm.create_completion("Hello", max_tokens=1)
            logger.info("🔥 Model warmed successfully")
        except Exception as e:
            logger.warning(f"Model warming failed: {e}")

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            output = self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=self.config.temperature,
                stop=["Question:", "\n\n", "User:"]
            )
            return output['choices'][0]['text']
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {e}"
```

#### E. Update HardwareAdaptiveRAGSystem.__init__ (around line 2150)

Change this line:

```python
# OLD:
# self.llm_client = OllamaClient(config, hardware_detector)

# NEW:
self.llm_client = LlamaCppClient(config, hardware_detector)
```

#### F. Update main() function

Replace references to `ollama_model` with `model_path`:

```python
# OLD:
print(f"   🤖 Model: {config.ollama_model}")

# NEW:
print(f"   🤖 Model: {config.model_path}")
```

### 2. streamlit_rag_app.py Changes

#### A. Update import statement (around line 30)

```python
# OLD:
from main import (
    ..., OllamaClient, ...
)

# NEW:
from main import (
    ..., LlamaCppClient, ...
)
```

#### B. Update display_professional_configuration() function (around line 1200)

```python
# OLD:
st.text(f"LLM Model: {config.ollama_model}")

# NEW:
st.text(f"LLM Path: {config.model_path}")
```

## Installation Steps

### 1. Install llama-cpp-python

**For CPU:**
```bash
pip install llama-cpp-python>=0.2.70
```

**For NVIDIA GPU (CUDA):**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

### 2. Download Model

Create models directory and download a GGUF model:

```bash
mkdir -p models
# Download your preferred GGUF model from HuggingFace
# Example: https://huggingface.co/models?search=gguf
```

### 3. Update Configuration

If you have an existing `rag_config.json`, either delete it to let the system recreate it, or manually update it with the new parameters shown above.

### 4. Test the System

```bash
# Test CLI
python main.py

# Or test Streamlit app
python launch_streamlit.py
```

## Benefits

- ✅ No external service dependency (no Ollama server needed)
- ✅ Direct model loading in Python process
- ✅ Hardware-adaptive GPU layer offloading
- ✅ Faster TTFT (no network overhead)
- ✅ Better memory efficiency
- ✅ Cross-platform compatibility

## Troubleshooting

### "Model not found"
Ensure your GGUF model file is in the `models/` directory and update `model_path` in config.

### "llama-cpp-python not installed"
Install with: `pip install llama-cpp-python`

For GPU support: `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`

### "Out of memory"
The system will automatically adjust GPU layer offloading based on available VRAM.

### Slow performance
Enable GPU support if available, or reduce `max_context_length` in config for CPU-only systems.

## Verification

After migration, verify the system is working:

1. Check startup logs for "🦙 Loading Llama.cpp" message
2. Verify GPU layers are being offloaded (if GPU available)
3. Test a query and check response times
4. Run evaluation to ensure quality is maintained

## Rollback

If you need to rollback to Ollama:

```bash
git checkout main
pip uninstall llama-cpp-python
# Restart Ollama server
```