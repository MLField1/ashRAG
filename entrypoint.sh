#!/bin/bash
set -euo pipefail

# Activate virtual environment
source /opt/venv/bin/activate

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Error handling function
handle_error() {
    log "❌ Error occurred in startup script at line $1"
    cleanup
    exit 1
}

# Cleanup function
cleanup() {
    log "🧹 Cleaning up processes..."
    if [[ -n "${OLLAMA_PID:-}" ]]; then
        kill "$OLLAMA_PID" 2>/dev/null || true
        wait "$OLLAMA_PID" 2>/dev/null || true
    fi
}

# Service health check
check_service() {
    local service_name="$1"
    local url="$2"
    local max_attempts="${3:-30}"
    
    log "⏳ Waiting for $service_name to be ready..."
    for i in $(seq 1 "$max_attempts"); do
        if curl -sf "$url" >/dev/null 2>&1; then
            log "✅ $service_name is ready!"
            return 0
        fi
        if [[ $i -eq $max_attempts ]]; then
            log "❌ $service_name failed to start within $max_attempts attempts"
            return 1
        fi
        sleep 2
    done
}

# Set trap for cleanup
trap 'handle_error $LINENO' ERR
trap cleanup SIGTERM SIGINT EXIT

log "🚀 Starting Educational RAG System (masonlf/ashrag)..."

# Verify required files
if [[ ! -f "streamlit_rag_app.py" ]]; then
    log "❌ streamlit_rag_app.py not found!"
    exit 1
fi

# Start Ollama service
log "🦙 Starting Ollama service..."
ollama serve > logs/ollama.log 2>&1 &
OLLAMA_PID=$!

# Wait for Ollama to be ready
if ! check_service "Ollama" "http://localhost:11434/api/tags" 45; then
    log "📋 Ollama startup logs:"
    tail -20 logs/ollama.log 2>/dev/null || echo "No logs available"
    exit 1
fi

# NOW check Ollama version (after service is started)
log "🔧 Checking Ollama version compatibility..."
OLLAMA_VERSION=$(ollama --version 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -1 || echo "unknown")
if [[ "$OLLAMA_VERSION" != "unknown" ]]; then
    log "✅ Ollama version: $OLLAMA_VERSION"
else
    log "⚠️ Could not detect Ollama version"
fi

# Create custom model from downloaded weights
log "🔧 Setting up custom electrical engineering model..."
CUSTOM_MODEL_AVAILABLE=false

if [[ -f "/app/models/Llama-3.2-3B-Finetune-q4k.gguf" && -f "/app/models/Modelfile" ]]; then
    log "📦 Creating custom model from local weights..."
    
    # Create the model in Ollama
    if ollama create masonlf/llama3.2-3b-elecfinetune -f /app/models/Modelfile 2>/dev/null; then
        log "✅ Custom electrical engineering model created successfully"
        CUSTOM_MODEL_AVAILABLE=true
    else
        log "⚠️ Failed to create custom model, will use fallback"
        CUSTOM_MODEL_AVAILABLE=false
    fi
else
    log "⚠️ Custom model files not found, using fallback"
    CUSTOM_MODEL_AVAILABLE=false
fi

# Load models
log "📥 Setting up models..."
MODEL_NAME="masonlf/llama3.2-3b-elecfinetune"
FALLBACK_MODEL="llama3.2:3b"

if [[ "$CUSTOM_MODEL_AVAILABLE" == "true" ]]; then
    if ollama list | grep -q "$MODEL_NAME"; then
        log "✅ Custom electrical engineering model ready: $MODEL_NAME"
    else
        log "⚠️ Custom model verification failed, using fallback: $FALLBACK_MODEL"
        CUSTOM_MODEL_AVAILABLE=false
    fi
fi

if [[ "$CUSTOM_MODEL_AVAILABLE" == "false" ]]; then
    log "⬇️ Loading fallback model: $FALLBACK_MODEL"
    if timeout 600 ollama pull "$FALLBACK_MODEL"; then
        log "✅ Fallback model loaded successfully"
        # Update config to use fallback model
        if [[ -f "rag_config.json" ]]; then
            sed -i "s|\"ollama_model\": \"$MODEL_NAME\"|\"ollama_model\": \"$FALLBACK_MODEL\"|" rag_config.json
            log "🔧 Updated configuration to use fallback model"
        fi
    else
        log "❌ Failed to load any model"
        exit 1
    fi
fi

# Final system check
log "🔍 Running final system checks..."
if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
    log "❌ Ollama API not responding"
    exit 1
fi

# Start Streamlit
log "🌐 Starting Streamlit application on port 8501..."
log "📚 Educational RAG System ready!"
log "🔗 Access the application via your container's public URL"

# Remove EXIT trap since we want streamlit to run
trap 'cleanup' SIGTERM SIGINT

exec streamlit run streamlit_rag_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.maxUploadSize=1000 \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false