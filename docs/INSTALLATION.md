# Installation Guide

This guide covers different installation methods for the Sea Turtle Re-Identification system.

## 🚀 Quick Start (Recommended)

### Prerequisites

- Python 3.7 or higher
- pip package manager
- (Optional) CUDA-capable GPU for training

### Standard Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sea-turtle-reid.git
cd sea-turtle-reid

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

This installs the package in editable mode with all required dependencies.

## 📦 Installation Options

### Option 1: Basic Installation

For inference and evaluation only:

```bash
pip install -e .
```

### Option 2: Development Installation

For contributing code:

```bash
pip install -e ".[dev]"
```

Includes: pytest, black, flake8, mypy, pre-commit

### Option 3: Full Installation

With all optional dependencies:

```bash
pip install -e ".[dev,notebooks,monitoring,api]"
```

Includes:
- `dev`: Development tools
- `notebooks`: Jupyter notebook support
- `monitoring`: TensorBoard, Weights & Biases
- `api`: FastAPI for model serving

### Option 4: From Requirements Only

If you prefer manual control:

```bash
pip install -r requirements.txt
```

## 🖥️ Platform-Specific Instructions

### Ubuntu / Debian Linux

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# Clone and install
git clone https://github.com/yourusername/sea-turtle-reid.git
cd sea-turtle-reid
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.9

# Clone and install
git clone https://github.com/yourusername/sea-turtle-reid.git
cd sea-turtle-reid
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Windows

```powershell
# Install Python from python.org if not already installed

# Clone and install
git clone https://github.com/yourusername/sea-turtle-reid.git
cd sea-turtle-reid
python -m venv venv
venv\Scripts\activate
pip install -e .
```

## 🎮 GPU Support

### CUDA Installation

For GPU acceleration, install PyTorch with CUDA support:

#### CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CPU Only
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Verify GPU Installation

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 🐳 Docker Installation

### Using Pre-built Image

```bash
# Pull image (when available)
docker pull yourusername/sea-turtle-reid:latest

# Run container
docker run -it --gpus all -v $(pwd)/data:/workspace/data \
    yourusername/sea-turtle-reid:latest
```

### Build from Dockerfile

```bash
# Build image
docker build -t sea-turtle-reid .

# Run container
docker run -it --gpus all -v $(pwd)/data:/workspace/data \
    sea-turtle-reid
```

## ☁️ Google Colab Installation

Perfect for quick experiments without local setup:

```python
# In Colab notebook
!git clone https://github.com/yourusername/sea-turtle-reid.git
%cd sea-turtle-reid
!pip install -e .

# Verify installation
import torch
from src.models.model_factory import create_model

model = create_model('resnet50', num_classes=438, pretrained=False)
print("✓ Installation successful!")
```

## 📊 Dataset Setup

### Automatic Download

```bash
# Download SeaTurtleID2022 dataset
python scripts/download_data.py --output data/SeaTurtleID2022
```

### Manual Download

1. Visit: https://wildlife-datasets.github.io/SeaTurtleID2022/
2. Download dataset
3. Extract to `data/SeaTurtleID2022/`

### Directory Structure

After download, you should have:

```
data/
└── SeaTurtleID2022/
    ├── images/
    │   ├── individual_001/
    │   ├── individual_002/
    │   └── ...
    └── metadata.json
```

## ✅ Verify Installation

### Quick Test

```bash
# Run verification script
python -c "
from src.models.model_factory import create_model
from src.data.temporal_split import TemporalSplitter
from src.evaluation.metrics import evaluate

print('✓ Core modules imported successfully')

# Create test model
model = create_model('resnet18', num_classes=10, pretrained=False)
print(f'✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters')

print('\nInstallation successful! 🎉')
"
```

### Run Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/ -v

# Should see: "X passed in Y seconds"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution**: Reduce batch size in config file

```yaml
training:
  batch_size: 16  # Reduce from 32
```

#### 2. Module Not Found

**Solution**: Install in editable mode

```bash
pip install -e .
```

#### 3. Permission Denied (Linux/Mac)

**Solution**: Use virtual environment or user install

```bash
pip install --user -e .
```

#### 4. Slow Installation

**Solution**: Use faster package resolver

```bash
pip install --upgrade pip
pip install -e . --use-deprecated=legacy-resolver
```

#### 5. Import Errors

**Solution**: Check Python path

```python
import sys
print(sys.path)

# Add package to path
sys.path.insert(0, '/path/to/sea-turtle-reid')
```

### Getting Help

If you encounter issues:

1. Check [existing issues](https://github.com/yourusername/sea-turtle-reid/issues)
2. Search [discussions](https://github.com/yourusername/sea-turtle-reid/discussions)
3. Create new issue with:
   - Python version (`python --version`)
   - PyTorch version
   - Operating system
   - Full error message

## 🔄 Updating

### Update from Git

```bash
# Pull latest changes
git pull origin main

# Reinstall (in case dependencies changed)
pip install -e . --upgrade
```

### Update Dependencies

```bash
# Update all packages
pip install -r requirements.txt --upgrade

# Or update specific package
pip install torch --upgrade
```

## 🗑️ Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv/

# Remove package (if installed system-wide)
pip uninstall sea-turtle-reid
```

## 📦 Dependencies Overview

### Core Dependencies

- **torch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **numpy**: Numerical computing
- **opencv-python**: Image processing
- **scikit-learn**: Machine learning utilities

### Optional Dependencies

- **tensorboard**: Training visualization
- **wandb**: Experiment tracking
- **jupyter**: Interactive notebooks
- **fastapi**: Model serving

## 🌐 Offline Installation

For systems without internet access:

```bash
# On connected machine:
pip download -r requirements.txt -d packages/

# Transfer packages/ folder to offline machine

# On offline machine:
pip install --no-index --find-links=packages/ -r requirements.txt
```

## 🎓 Next Steps

After installation:

1. **Quick Start**: Follow [README Quick Start](README.md#quick-start)
2. **Training**: See [Training Guide](docs/TRAINING.md)
3. **Notebooks**: Explore [example notebooks](notebooks/)
4. **API**: Check [API documentation](docs/API.md)

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/sea-turtle-reid/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/sea-turtle-reid/discussions)
- **Email**: your.email@example.com

---

**System Requirements**

- **Minimum**: 8GB RAM, 10GB disk space
- **Recommended**: 16GB RAM, NVIDIA GPU (6GB+ VRAM), 50GB disk space
- **Optimal**: 32GB RAM, NVIDIA GPU (12GB+ VRAM), 100GB SSD

---

*Last updated: November 2025*
