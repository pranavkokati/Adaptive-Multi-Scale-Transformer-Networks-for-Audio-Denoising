# Adaptive Multi-Scale Transformer Networks for Audio Denoising
# Professional Dockerfile with multi-stage build

# Use official Python runtime as base image
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data outputs checkpoints logs

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for documentation (if needed)
EXPOSE 8000

# Default command
CMD ["python", "scripts/train_real_data.py", "--help"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir -e ".[dev,docs,notebooks,audio]"

# Install additional development tools
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    ipywidgets \
    pre-commit

# Set environment for development
ENV PYTHONPATH=/app/src

# Development command
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Production stage
FROM base as production

# Install only production dependencies
RUN pip install --no-cache-dir -e ".[audio]"

# Copy trained models (if available)
COPY --chown=appuser:appuser checkpoints/ ./checkpoints/

# Set production environment
ENV PYTHONPATH=/app/src \
    ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1

# Production command
CMD ["python", "scripts/inference_real.py", "--help"]

# GPU stage
FROM base as gpu

# Install CUDA dependencies
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set CUDA environment
ENV CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all

# GPU command
CMD ["python", "scripts/train_real_data.py", "--enhancement_method", "transformer"]

# Testing stage
FROM base as testing

# Install testing dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy tests
COPY --chown=appuser:appuser tests/ ./tests/

# Set testing environment
ENV PYTHONPATH=/app/src \
    TESTING=true

# Run tests
CMD ["python", "-m", "pytest", "tests/", "--cov=src", "--cov-report=html"]

# Documentation stage
FROM base as docs

# Install documentation dependencies
RUN pip install --no-cache-dir -e ".[docs]"

# Copy documentation
COPY --chown=appuser:appuser docs/ ./docs/

# Build documentation
RUN cd docs && make html

# Serve documentation
CMD ["python", "-m", "http.server", "8000", "--directory", "docs/_build/html"]

# Jupyter stage
FROM base as jupyter

# Install Jupyter dependencies
RUN pip install --no-cache-dir -e ".[notebooks]"

# Create Jupyter configuration
RUN jupyter notebook --generate-config

# Set Jupyter password
RUN echo "c.NotebookApp.password = 'sha1:$(echo -n 'password' | sha1sum | cut -d' ' -f1)'" >> ~/.jupyter/jupyter_notebook_config.py

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
