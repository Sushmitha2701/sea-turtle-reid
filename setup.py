"""
Sea Turtle Re-Identification Package Setup
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            # Extract package name without version constraints
            requirements.append(line.split('>=')[0].split('==')[0])

setup(
    name="sea-turtle-reid",
    version="1.0.0",
    author="Sushmitha Shivashankar Singh",
    author_email="your.email@example.com",
    description="Advanced Deep Learning for Sea Turtle Re-Identification with Temporal-Aware Evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sea-turtle-reid",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/sea-turtle-reid/issues",
        "Documentation": "https://github.com/yourusername/sea-turtle-reid/docs",
        "Source Code": "https://github.com/yourusername/sea-turtle-reid",
    },
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "pillow>=8.3.0",
        "opencv-python>=4.5.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
        "wildlife-datasets>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.7b0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
            "pylint>=2.10.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "ipywidgets>=7.6.0",
        ],
        "monitoring": [
            "tensorboard>=2.6.0",
            "wandb>=0.12.0",
        ],
        "api": [
            "fastapi>=0.70.0",
            "uvicorn>=0.15.0",
            "python-multipart>=0.0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "turtle-reid-train=scripts.train:main",
            "turtle-reid-eval=scripts.evaluate:main",
            "turtle-reid-infer=scripts.inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml"],
    },
    keywords=[
        "deep-learning",
        "computer-vision",
        "wildlife-monitoring",
        "re-identification",
        "conservation",
        "marine-biology",
        "sea-turtles",
        "pytorch",
    ],
    license="MIT",
)
