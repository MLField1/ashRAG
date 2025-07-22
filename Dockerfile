# Educational RAG System - Optimized Dockerfile

# Build: docker build -t masonlf/ashrag:latest .

FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04

# Build arguments for flexibility

ARG PYTHON_VERSION=3.12
ARG OLLAMA_VERSION=latest

# Environment variables

ENV DEBIAN_FRONTEND=noninteractive   
PYTHONUNBUFFERED=1   
PYTHONDONTWRITEBYTECODE=1   
PIP_NO_CACHE_DIR=1   
PIP_DISABLE_PIP_VERSION_CHECK=1   
PIP_DEFAULT_TIMEOUT=100   
OLLAMA_HOST=0.0.0.0   
OLLAMA_HOME=/app/.ollama   
CUDA_VISIBLE_DEVICES=all   
LOG_DIR=/app/logs   
RAG_LOG_FILE=/app/logs/rag_system.log

# Create app user for security (FIXED - added home directory)

RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Install system dependencies with retry mechanism

RUN set -e &&   
for i in 1 2 3; do   
echo “System packages install attempt $i/3” &&   
apt-get update &&   
apt-get install -y –no-install-recommends   
python3-full   
python3-pip   
python3-dev   
python3-venv   
build-essential   
curl   
wget   
git   
ca-certificates   
software-properties-common   
apt-transport-https   
gnupg   
lsb-release   
jq &&   
break ||   
(echo “Attempt $i failed, retrying…” && sleep 5);   
done &&   
ln -sf /usr/bin/python3 /usr/bin/python &&   
rm -rf /var/lib/apt/lists/* &&   
apt-get clean

# Create and activate virtual environment

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH=”$VIRTUAL_ENV/bin:$PATH”

# Upgrade pip and install build tools in virtual environment

RUN pip install –upgrade pip setuptools wheel

# Set working directory

WORKDIR /app

# Copy requirements first for better Docker layer caching

COPY requirements.txt ./

# Install Python dependencies in virtual environment

RUN set -e &&   
echo “Installing Python packages in virtual environment…” &&   
pip install –no-cache-dir   
–timeout=300   
–retries=3   
-r requirements.txt &&   
echo “Python packages installed successfully”

# Download custom model weights from Hugging Face

RUN set -e &&   
echo “📥 Downloading custom electrical engineering model…” &&   
mkdir -p /app/models &&   
cd /app/models &&   
wget –progress=bar:force:noscroll -O Llama-3.2-3B-Finetune-q4k.gguf   
“https://huggingface.co/masonlf/llama3.2-3b-elecfinetune/resolve/main/Llama-3.2-3B-Finetune-q4k.gguf” &&   
echo “✅ Custom model weights downloaded: $(ls -lh Llama-3.2-3B-Finetune-q4k.gguf | awk ‘{print $5}’)”

# Install stable Ollama v0.9.6 with fallbacks

RUN set -e &&   
echo “🦙 Installing stable Ollama v0.9.6…” &&   
OLLAMA_INSTALLED=false &&   
  
# Method 1: Try specific stable version v0.9.6
for version in “v0.9.6” “v0.9.5” “v0.9.4” “v0.9.3”; do   
echo “📦 Attempting Ollama $version…” &&   
if curl -L –connect-timeout 15 –max-time 60   
-o /tmp/ollama “https://github.com/ollama/ollama/releases/download/$version/ollama-linux-amd64” 2>/dev/null; then   
chmod +x /tmp/ollama &&   
mv /tmp/ollama /usr/local/bin/ollama &&   
echo “✅ Ollama $version installed successfully” &&   
OLLAMA_INSTALLED=true &&   
break;   
else   
echo “⚠️ Version $version failed, trying next…” &&   
sleep 2;   
fi;   
done &&   
  
# Method 2: Fallback to install script if direct download fails
if [ “$OLLAMA_INSTALLED” = “false” ]; then   
echo “🔄 Fallback: using official install script…” &&   
curl -fsSL https://ollama.com/install.sh | sh &&   
OLLAMA_INSTALLED=true;   
fi &&   
  
# Verify installation
if [ “$OLLAMA_INSTALLED” = “true” ]; then   
ollama –version &&   
echo “🎉 Ollama installation completed successfully”;   
else   
echo “❌ All Ollama installation methods failed”;   
exit 1;   
fi

# Copy application files

COPY main.py streamlit_rag_app.py ./
COPY rag_config.json* ./

# Copy Modelfile for custom model

COPY Modelfile /app/models/Modelfile

# Copy entrypoint script

COPY entrypoint.sh /app/entrypoint.sh

# Create necessary directories and files (FIXED - added all missing directories and files)

RUN mkdir -p documents data cache logs temp models &&   
mkdir -p /home/appuser/.ollama /app/.venv /app/.ollama &&   
touch /app/logs/rag_system.log /app/.venv/rag_system.log &&   
chmod +x /app/entrypoint.sh &&   
sed -i ‘s/\r$//’ /app/entrypoint.sh &&   
chown -R appuser:appuser /app /opt/venv /home/appuser &&   
chmod -R 755 /app

# Copy documents if they exist

COPY –chown=appuser:appuser documents/ ./documents/ 2>/dev/null || true

# Switch to non-root user for security

USER appuser

# Expose ports

EXPOSE 8501 11434

# Health check with proper timeouts

HEALTHCHECK –interval=30s –timeout=20s –start-period=180s –retries=3   
CMD curl -f http://localhost:8501/_stcore/health &&   
curl -f http://localhost:11434/api/tags >/dev/null 2>&1 || exit 1

# Set entrypoint

ENTRYPOINT [”/app/entrypoint.sh”]
