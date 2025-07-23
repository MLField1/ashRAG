# Educational RAG System - Optimized Dockerfile
# Build: docker build -t masonlf/ashrag:latest .

FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04

# Build arguments for flexibility
ARG PYTHON_VERSION=3.12
ARG OLLAMA_VERSION=latest

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    OLLAMA_HOST=0.0.0.0 \
    OLLAMA_HOME=/app/.ollama \
    CUDA_VISIBLE_DEVICES=all \
    LOG_DIR=/app/logs \
    RAG_LOG_FILE=/app/logs/rag_system.log

# Create app user for security (FIXED - added home directory)
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Install system dependencies with retry mechanism
RUN set -e && \
    for i in 1 2 3; do \
        echo "System packages install attempt $i/3" && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            python3-full \
            python3-pip \
            python3-dev \
            python3-venv \
            build-essential \
            curl \
            wget \
            git \
            ca-certificates \
            software-properties-common \
            apt-transport-https \
            gnupg \
            lsb-release && \
        break || \
        (echo "Attempt $i failed, retrying..." && sleep 5); \
    done && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install build tools in virtual environment
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt ./

# Install Python dependencies in virtual environment
RUN set -e && \
    echo "Installing Python packages in virtual environment..." && \
    pip install --no-cache-dir \
        --timeout=300 \
        --retries=3 \
        -r requirements.txt && \
    echo "Python packages installed successfully"

# Download custom model weights from Hugging Face
RUN set -e && \
    echo "📥 Downloading custom electrical engineering model..." && \
    mkdir -p /app/models && \
    cd /app/models && \
    wget --progress=bar:force:noscroll -O Llama-3.2-3B-Finetune-q4k.gguf \
        "https://huggingface.co/masonlf/llama3.2-3b-elecfinetune/resolve/main/Llama-3.2-3B-Finetune-q4k.gguf" && \
    echo "✅ Custom model weights downloaded: $(ls -lh Llama-3.2-3B-Finetune-q4k.gguf | awk '{print $5}')"

# Install Ollama using official download method with proper permissions
RUN set -e && \
    echo "🦙 Installing Ollama using official download..." && \
    curl -LO https://ollama.com/download/ollama-linux-amd64.tgz && \
    tar -C /usr -xzf ollama-linux-amd64.tgz && \
    rm -f ollama-linux-amd64.tgz && \
    \
    # Ensure proper permissions and accessibility
    chmod +x /usr/bin/ollama 2>/dev/null || chmod +x /usr/local/bin/ollama 2>/dev/null || true && \
    \
    # Verify installation and location
    which ollama && \
    ls -la $(which ollama) && \
    ollama --version && \
    \
    # Test accessibility for non-root users
    su appuser -c "ollama --version" && \
    echo "✅ Ollama installed successfully with proper permissions"

# Copy application files
COPY main.py streamlit_rag_app.py ./
COPY rag_config.json* ./

# Copy Modelfile for custom model
COPY Modelfile /app/models/Modelfile

# Copy documents for demo purposes
COPY documents/ ./documents/

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh

# Create necessary directories and files (FIXED - added all missing directories and files)
RUN mkdir -p documents data cache logs temp models && \
    mkdir -p /home/appuser/.ollama /app/.venv /app/.ollama && \
    touch /app/logs/rag_system.log /app/.venv/rag_system.log && \
    chmod +x /app/entrypoint.sh && \
    sed -i 's/\r$//' /app/entrypoint.sh && \
    chown -R appuser:appuser /app /opt/venv /home/appuser && \
    chmod -R 755 /app


# Create documents directory (users can mount their own documents at runtime)
RUN mkdir -p documents && chown -R appuser:appuser documents

# Switch to non-root user for security
USER appuser

# Expose ports
EXPOSE 8501 11434

# Health check with proper timeouts
HEALTHCHECK --interval=30s --timeout=20s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health && \
        curl -f http://localhost:11434/api/tags >/dev/null 2>&1 || exit 1

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]