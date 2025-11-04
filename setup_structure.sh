#!/bin/bash

# Create all directories
mkdir -p src/models
mkdir -p src/data
mkdir -p src/training
mkdir -p src/evaluation
mkdir -p src/interpretability
mkdir -p src/utils
mkdir -p notebooks
mkdir -p configs
mkdir -p results/{figures,models,metrics}
mkdir -p docs
mkdir -p scripts
mkdir -p tests
mkdir -p data/sample_images
mkdir -p .github/workflows

# Create __init__.py files
touch src/__init__.py
touch src/models/__init__.py
touch src/data/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py
touch src/interpretability/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

echo "Directory structure created successfully!"
