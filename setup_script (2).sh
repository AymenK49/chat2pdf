#!/bin/bash

# PDF Chatbot Setup Script for Ubuntu Server
# Supports CPU-only deployment with optimized models

set -e

echo "🤖 PDF Chatbot Setup Script"
echo "=========================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on Ubuntu
if ! command -v apt-get &> /dev/null; then
    print_error "This script is designed for Ubuntu/Debian systems"
    exit 1
fi

print_header "Checking system requirements..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Python $PYTHON_VERSION found"
    
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
        print_status "Python version is compatible"
    else
        print_error "Python 3.8+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found"
    exit 1
fi

# Check available memory
TOTAL_MEM=$(free -m | awk 'NR==2{printf "%.0f", $2}')
print_status "Total memory: ${TOTAL_MEM}MB"

if [ "$TOTAL_MEM" -lt 4000 ]; then
    print_warning "Low memory detected (${TOTAL_MEM}MB). Recommended: 4GB+"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check available disk space
DISK_SPACE=$(df -m . | awk 'NR==2{print $4}')
print_status "Available disk space: ${DISK_SPACE}MB"

if [ "$DISK_SPACE" -lt 5000 ]; then
    print_warning "Low disk space detected (${DISK_SPACE}MB). Recommended: 5GB+"
fi

print_header "Installing system dependencies..."

# Update package list
sudo apt-get update -qq

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    cmake \
    git \
    curl \
    wget \
    unzip

print_status "System dependencies installed"

print_header "Setting up Python virtual environment..."

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

print_status "Virtual environment activated and pip upgraded"

print_header "Installing Python dependencies..."

# Install PyTorch CPU version first (lighter)
print_status "Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other ML dependencies
print_status "Installing transformers and related packages..."
pip install transformers sentence-transformers

# Install FAISS CPU version
print_status "Installing FAISS (CPU version)..."
pip install faiss-cpu

# Install PDF processing
print_status "Installing PDF processing libraries..."
pip install PyMuPDF pdfminer.six

# Install web interface dependencies
print_status "Installing web interface libraries..."
pip install gradio streamlit

# Install other dependencies
print_status "Installing remaining dependencies..."
pip install -r requirements.txt

print_status "Python dependencies installed successfully"

print_header "Downloading and caching models..."

# Create models directory
mkdir -p models/cache
export TRANSFORMERS_CACHE=./models/cache
export SENTENCE_TRANSFORMERS_HOME=./models/cache

# Download embedding model
python3 -c "
from sentence_transformers import SentenceTransformer
print('Downloading multilingual embedding model...')
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
print('Embedding model cached successfully')
"

# Download QA models
python3 -c "
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

models = [
    'distilbert-base-cased-distilled-squad',
    'etalab-ia/camembert-base-squadFR-fquad-piaf'
]

for model_name in models:
    try:
        print(f'Downloading {model_name}...')
        pipeline('question-answering', model=model_name, tokenizer=model_name)
        print(f'{model_name} cached successfully')
    except Exception as e:
        print(f'Warning: Could not download {model_name}: {e}')
"

print_status "Models downloaded and cached"

print_header "Creating configuration files..."

# Create config file
cat > config.json << EOF
{
    "models": {
        "embedding": "sentence-transformers/distiluse-base-multilingual-cased-v1",
        "qa": {
            "en": "distilbert-base-cased-distilled-squad",
            "fr": "etalab-ia/camembert-base-squadFR-fquad-piaf",
            "ar": "aubmindlab/bert-base-arabertv02-squad2"
        },
        "generative": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    },
    "performance": {
        "max_memory_mb": 3000,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "max_context_length": 400,
        "batch_size": 16
    },
    "interface": {
        "default_port": 7860,
        "default_host": "0.0.0.0"
    }
}
EOF

print_status "Configuration file created"

# Create startup script
cat > start_chatbot.sh << 'EOF'
#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export TRANSFORMERS_CACHE=./models/cache
export SENTENCE_TRANSFORMERS_HOME=./models/cache
export TOKENIZERS_PARALLELISM=false

# Set CPU optimizations
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "🤖 Starting PDF Chatbot..."
echo "Interface will be available at: http://localhost:7860"
echo "Press Ctrl+C to stop"

# Start the application
python3 main.py --interface gradio --port 7860 --host 0.0.0.0
EOF

chmod +x start_chatbot.sh

# Create CLI startup script
cat > start_cli.sh << 'EOF'
#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export TRANSFORMERS_CACHE=./models/cache
export SENTENCE_TRANSFORMERS_HOME=./models/cache
export TOKENIZERS_PARALLELISM=false

# Set CPU optimizations
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "🤖 Starting PDF Chatbot CLI..."

# Start CLI interface
python3 main.py --interface cli
EOF

chmod +x start_cli.sh

print_status "Startup scripts created"

print_header "Creating systemd service (optional)..."

# Create systemd service file
sudo tee /etc/systemd/system/pdf-chatbot.service > /dev/null << EOF
[Unit]
Description=PDF Chatbot Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=TRANSFORMERS_CACHE=$(pwd)/models/cache
Environment=SENTENCE_TRANSFORMERS_HOME=$(pwd)/models/cache
Environment=TOKENIZERS_PARALLELISM=false
Environment=OMP_NUM_THREADS=4
Environment=MKL_NUM_THREADS=4
ExecStart=$(pwd)/venv/bin/python3 $(pwd)/main.py --interface gradio --port 7860 --host 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

print_status "Systemd service created (use 'sudo systemctl enable pdf-chatbot' to enable)"

print_header "Running tests..."

# Test basic functionality
python3 -c "
import torch
import transformers
import sentence_transformers
import faiss
import fitz
import gradio

print('✅ All core libraries imported successfully')

# Test model loading
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    test_embedding = model.encode(['test'])
    print(f'✅ Embedding model test passed - shape: {test_embedding.shape}')
except Exception as e:
    print(f'❌ Embedding model test failed: {e}')

# Test FAISS
try:
    import faiss
    import numpy as np
    d = 512
    index = faiss.IndexFlatIP(d)
    test_vectors = np.random.random((10, d)).astype('float32')
    faiss.normalize_L2(test_vectors)
    index.add(test_vectors)
    print(f'✅ FAISS test passed - {index.ntotal} vectors indexed')
except Exception as e:
    print(f'❌ FAISS test failed: {e}')

print('🎉 Setup validation completed!')
"

print_header "Creating example usage files..."

# Create example PDF (placeholder)
cat > example_usage.md << 'EOF'
# PDF Chatbot Usage Guide

## Quick Start

### 1. Web Interface (Recommended)
```bash
./start_chatbot.sh
```
Then open http://localhost:7860 in your browser.

### 2. Command Line Interface
```bash
./start_cli.sh
```

## Usage Examples

### Web Interface
1. Upload a PDF file
2. Wait for processing to complete
3. Ask questions in English, French, or Arabic
4. Choose between extractive and generative answers

### CLI Interface
```bash
chatbot> load /path/to/document.pdf
chatbot> ask What is the main topic of this document?
chatbot> ask Quel est le sujet principal?
chatbot> ask ما هو الموضوع الرئيسي؟
```

## Performance Tips

1. **Memory Management**: The system automatically manages memory usage
2. **PDF Size**: Large PDFs are automatically chunked for processing
3. **Language Detection**: Questions are automatically detected for language
4. **Model Switching**: Switch between extractive/generative modes as needed

## Troubleshooting

### Low Memory Issues
- Reduce chunk_size in config.json
- Use extractive mode only
- Process smaller PDFs

### Slow Performance
- Increase OMP_NUM_THREADS in startup scripts
- Use SSD storage for model cache
- Ensure adequate RAM (4GB+ recommended)

### Model Download Issues
- Check internet connection
- Clear model cache: `rm -rf models/cache`
- Re-run setup script

## System Service

To run as a system service:
```bash
sudo systemctl enable pdf-chatbot
sudo systemctl start pdf-chatbot
sudo systemctl status pdf-chatbot
```

## Configuration

Edit `config.json` to customize:
- Model selections
- Performance parameters
- Interface settings
EOF

print_status "Example usage guide created"

print_header "Setup completed successfully! 🎉"

echo
echo "==============================================="
echo "🤖 PDF Chatbot Setup Complete!"
echo "==============================================="
echo
echo "Next steps:"
echo "1. Start web interface: ./start_chatbot.sh"
echo "2. Or start CLI: ./start_cli.sh"
echo "3. Read usage guide: cat example_usage.md"
echo
echo "System info:"
echo "- Memory usage: ~2-3GB when running"
echo "- Models cached in: ./models/cache"
echo "- Configuration: ./config.json"
echo "- Logs: Check console output"
echo
echo "Support:"
echo "- Supports English, French, and Arabic"
echo "- PDF processing with PyMuPDF"
echo "- Vector search with FAISS"
echo "- Web UI with Gradio"
echo
print_status "Setup completed successfully!"
