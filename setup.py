#!/usr/bin/env python3
"""
Setup script for Adaptive Multi-Scale Transformer Networks for Audio Denoising.

This package provides state-of-the-art audio denoising capabilities using
real-world datasets and cutting-edge transformer architectures.
"""

import os
import re
from pathlib import Path

from setuptools import find_packages, setup

# Read the README file
def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "Adaptive Multi-Scale Transformer Networks for Audio Denoising"

# Read version from __init__.py
def get_version():
    init_path = Path(__file__).parent / "src" / "__init__.py"
    if init_path.exists():
        with open(init_path, "r", encoding="utf-8") as f:
            content = f.read()
            version_match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', content, re.M)
            if version_match:
                return version_match.group(1)
    return "0.1.0"

# Read requirements
def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

# Project metadata
PROJECT_NAME = "Adaptive Multi Scale Transformer Networks for Audio Denoising"
PROJECT_VERSION = get_version()
PROJECT_DESCRIPTION = "State-of-the-art audio denoising using real-world datasets and transformer architectures"
PROJECT_LONG_DESCRIPTION = read_readme()
PROJECT_AUTHOR = "Your Name"
PROJECT_AUTHOR_EMAIL = "your.email@example.com"
PROJECT_URL = "https://github.com/your-username/adaptive-audio-denoising"
PROJECT_DOWNLOAD_URL = f"{PROJECT_URL}/archive/v{PROJECT_VERSION}.tar.gz"
PROJECT_LICENSE = "MIT"
PROJECT_CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Multimedia :: Sound/Audio :: Analysis",
    "Topic :: Multimedia :: Sound/Audio :: Conversion",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

# Package configuration
PACKAGES = find_packages(where="src")
PACKAGE_DIR = {"": "src"}

# Dependencies
INSTALL_REQUIRES = read_requirements()

# Development dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.11.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.4.0",
        "isort>=5.12.0",
        "pre-commit>=3.3.0",
    ],
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "sphinx-autodoc-typehints>=1.23.0",
        "myst-parser>=1.0.0",
    ],
    "notebooks": [
        "jupyter>=1.0.0",
        "ipython>=8.14.0",
        "ipywidgets>=8.0.0",
    ],
    "audio": [
        "pesq>=0.0.3",
        "pystoi>=0.3.3",
        "webrtcvad>=2.0.10",
    ],
    "gpu": [
        "nvidia-ml-py>=11.0.0",
    ],
}

# Entry points
ENTRY_POINTS = {
    "console_scripts": [
        "adaptive-denoise=scripts.train_real_data:main",
        "adaptive-evaluate=scripts.evaluate_real:main",
        "adaptive-inference=scripts.inference_real:main",
        "adaptive-download=scripts.download_datasets:main",
        "adaptive-verify=scripts.verify_data:main",
    ],
}

# Package data
PACKAGE_DATA = {
    "": ["*.yaml", "*.yml", "*.json"],
}

# Python version requirement
PYTHON_REQUIRES = ">=3.8"

# Setup configuration
setup(
    name=PROJECT_NAME,
    version=PROJECT_VERSION,
    description=PROJECT_DESCRIPTION,
    long_description=PROJECT_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=PROJECT_AUTHOR,
    author_email=PROJECT_AUTHOR_EMAIL,
    url=PROJECT_URL,
    download_url=PROJECT_DOWNLOAD_URL,
    license=PROJECT_LICENSE,
    classifiers=PROJECT_CLASSIFIERS,
    packages=PACKAGES,
    package_dir=PACKAGE_DIR,
    package_data=PACKAGE_DATA,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points=ENTRY_POINTS,
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "audio",
        "denoising",
        "transformer",
        "deep-learning",
        "signal-processing",
        "machine-learning",
        "neural-networks",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": f"{PROJECT_URL}/issues",
        "Source": PROJECT_URL,
        "Documentation": f"{PROJECT_URL}/blob/main/README.md",
        "Changelog": f"{PROJECT_URL}/blob/main/CHANGELOG.md",
    },
)
