#!/bin/bash
# Quick setup script to create a virtual environment with PyTorch for the experiment

set -e

echo "=============================================="
echo "PYTORCH ENVIRONMENT SETUP"
echo "=============================================="
echo ""

VENV_DIR="$HOME/ttt-pytorch-env"

# Check if venv already exists
if [ -d "$VENV_DIR" ]; then
    echo "✅ Virtual environment already exists at: $VENV_DIR"
    echo ""
    read -p "Do you want to use the existing environment? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please specify a different location or remove the existing one."
        exit 1
    fi
else
    echo "📦 Creating virtual environment at: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate the environment
echo "🔧 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Check if PyTorch is already installed
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "✅ PyTorch is already installed: $TORCH_VERSION"
    
    # Check CUDA
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "✅ CUDA is available"
    else
        echo "⚠️  CUDA is not available (CPU-only PyTorch)"
    fi
else
    echo "📥 Installing PyTorch and dependencies..."
    echo "   This may take several minutes..."
    echo ""
    
    # Install PyTorch with CUDA support
    # Using pip for latest version
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    echo ""
    echo "✅ PyTorch installed"
fi

# Install other required packages
echo ""
echo "📥 Installing other required packages..."
pip install transformers datasets accelerate numpy scipy tensorboard wandb matplotlib seaborn

echo ""
echo "=============================================="
echo "✅ SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "To activate this environment in the future, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To launch the experiment now:"
echo "  source $VENV_DIR/bin/activate"
echo "  bash launch_length_gen_study_gpu7.sh"
echo ""
echo "=============================================="

# Deactivate for now
deactivate
