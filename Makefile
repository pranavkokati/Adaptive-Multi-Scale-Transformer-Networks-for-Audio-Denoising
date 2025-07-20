# Adaptive Multi-Scale Transformer Networks for Audio Denoising
# Professional Makefile for development automation

.PHONY: help install install-dev clean test test-cov lint format type-check docs download-data verify-data train evaluate visualize all

# Default target
help:
	@echo "Adaptive Audio Denoising - Development Commands"
	@echo "=============================================="
	@echo ""
	@echo "Installation:"
	@echo "  install      - Install package in development mode"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  clean        - Remove build artifacts and cache"
	@echo ""
	@echo "Data Management:"
	@echo "  download-data - Download all real audio datasets"
	@echo "  verify-data   - Verify dataset integrity and quality"
	@echo ""
	@echo "Development:"
	@echo "  lint         - Run code linting (flake8)"
	@echo "  format       - Format code with black and isort"
	@echo "  type-check   - Run type checking with mypy"
	@echo "  test         - Run test suite"
	@echo "  test-cov     - Run tests with coverage report"
	@echo ""
	@echo "Training and Evaluation:"
	@echo "  train        - Train model with default settings"
	@echo "  evaluate     - Evaluate model performance"
	@echo "  visualize    - Generate research visualizations"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         - Build documentation"
	@echo "  docs-serve   - Serve documentation locally"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  all          - Run full quality check (lint, test, type-check)"
	@echo "  pre-commit   - Run pre-commit hooks"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,notebooks,audio]"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf docs/_build/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +

# Data Management
download-data:
	@echo "📥 Downloading real audio datasets..."
	python scripts/download_datasets.py --all

verify-data:
	@echo "🔍 Verifying dataset integrity..."
	python scripts/verify_data.py --all --save-report

# Code Quality
lint:
	@echo "🔍 Running code linting..."
	flake8 src/ scripts/ tests/ --max-line-length=88 --extend-ignore=E203,W503

format:
	@echo "🎨 Formatting code..."
	black src/ scripts/ tests/ --line-length=88
	isort src/ scripts/ tests/ --profile=black

type-check:
	@echo "🔍 Running type checking..."
	mypy src/ --ignore-missing-imports --disallow-untyped-defs

# Testing
test:
	@echo "🧪 Running test suite..."
	python -m pytest tests/ -v

test-cov:
	@echo "🧪 Running tests with coverage..."
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Training and Evaluation
train:
	@echo "🚀 Training model..."
	python scripts/train_real_data.py --enhancement_method transformer

evaluate:
	@echo "📊 Evaluating model performance..."
	python scripts/evaluate_real.py --enhancement_methods transformer diffusion hybrid multitask lightweight

visualize:
	@echo "📈 Generating research visualizations..."
	python src/research_visualizations.py

# Documentation
docs:
	@echo "📚 Building documentation..."
	cd docs && make html

docs-serve:
	@echo "🌐 Serving documentation..."
	cd docs/_build/html && python -m http.server 8000

# Quality Assurance
all: lint test type-check
	@echo "✅ All quality checks passed!"

pre-commit: format lint type-check test
	@echo "✅ Pre-commit checks passed!"

# Development Workflow
dev-setup: install-dev download-data verify-data
	@echo "✅ Development environment setup complete!"

# Quick Start
quick-start: dev-setup train evaluate visualize
	@echo "🎉 Quick start workflow completed!"

# Docker Commands
docker-build:
	docker build -t adaptive-audio-denoising .

docker-run:
	docker run -it --gpus all adaptive-audio-denoising

# Performance Testing
benchmark:
	@echo "⚡ Running performance benchmarks..."
	python scripts/benchmark.py

# Model Comparison
compare-methods:
	@echo "🔬 Comparing enhancement methods..."
	python scripts/evaluate_real.py --enhancement_methods transformer diffusion hybrid multitask lightweight --save-results

# Data Analysis
analyze-data:
	@echo "📊 Analyzing dataset characteristics..."
	python scripts/analyze_datasets.py

# Export Results
export-results:
	@echo "📤 Exporting results..."
	python scripts/export_results.py

# Clean Results
clean-results:
	rm -rf outputs/
	rm -rf checkpoints/
	rm -rf logs/
	rm -rf results/

# Full Pipeline
pipeline: clean dev-setup train evaluate visualize export-results
	@echo "🎉 Full pipeline completed!"

# Help for specific targets
help-install:
	@echo "Installation Commands:"
	@echo "  make install      - Install package in development mode"
	@echo "  make install-dev  - Install with all development dependencies"
	@echo "  make clean        - Remove all build artifacts"

help-data:
	@echo "Data Management Commands:"
	@echo "  make download-data - Download all real audio datasets (~3.8GB)"
	@echo "  make verify-data   - Verify dataset integrity and quality"
	@echo "  make analyze-data  - Analyze dataset characteristics"

help-dev:
	@echo "Development Commands:"
	@echo "  make format       - Format code with black and isort"
	@echo "  make lint         - Run code linting with flake8"
	@echo "  make type-check   - Run type checking with mypy"
	@echo "  make test         - Run test suite"
	@echo "  make test-cov     - Run tests with coverage report"

help-train:
	@echo "Training Commands:"
	@echo "  make train        - Train model with transformer enhancement"
	@echo "  make evaluate     - Evaluate all enhancement methods"
	@echo "  make visualize    - Generate research visualizations"
	@echo "  make benchmark    - Run performance benchmarks"

# Environment checks
check-env:
	@echo "🔍 Checking environment..."
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import torchaudio; print(f'TorchAudio: {torchaudio.__version__}')"
	@python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
	@python -c "import librosa; print(f'Librosa: {librosa.__version__}')"

# Version management
version:
	@python -c "import re; print(re.search(r'__version__ = [\'\"]([^\'\"]*)[\'\"]', open('src/__init__.py').read()).group(1))"

bump-version:
	@echo "📝 Bumping version..."
	@read -p "Enter new version (e.g., 1.0.1): " version; \
	sed -i "s/__version__ = .*/__version__ = \"$$version\"/" src/__init__.py; \
	echo "Version bumped to $$version"

# Git helpers
git-status:
	@echo "📊 Git status:"
	@git status --short

git-commit:
	@echo "💾 Committing changes..."
	@git add .
	@git commit -m "Update: $(shell date)"

# Backup and restore
backup:
	@echo "💾 Creating backup..."
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz src/ scripts/ tests/ configs/ requirements.txt setup.py README.md

restore:
	@echo "📦 Restoring from backup..."
	@read -p "Enter backup filename: " backup_file; \
	tar -xzf $$backup_file

# System information
sys-info:
	@echo "💻 System Information:"
	@python -c "import platform; print(f'OS: {platform.system()} {platform.release()}')"
	@python -c "import psutil; print(f'CPU: {psutil.cpu_count()} cores')"
	@python -c "import psutil; print(f'Memory: {psutil.virtual_memory().total // (1024**3)} GB')"
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
