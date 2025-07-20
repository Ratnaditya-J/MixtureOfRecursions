#!/usr/bin/env python3
"""
Setup script for Mixture-of-Recursions (MoR) Research Project
"""

import os

from setuptools import find_packages, setup


# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]


setup(
    name="mixture-of-recursions",
    version="1.0.0",
    author="Your Name",
    author_email="your-email@example.com",
    description="Mixture-of-Recursions: Adaptive Token-Level Computation for Efficient Transformers",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mixture-of-recursions",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/mixture-of-recursions/issues",
        "Documentation": "https://github.com/yourusername/mixture-of-recursions#readme",
        "Source Code": "https://github.com/yourusername/mixture-of-recursions",
        "Research Paper": "https://arxiv.org/abs/2024.XXXXX",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
        ],
        "training": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
            "accelerate>=0.20.0",
            "evaluate>=0.4.0",
        ],
        "efficiency": [
            "flash-attn>=2.0.0",
            "triton>=2.0.0",
            "xformers>=0.0.20",
        ],
    },
    entry_points={
        "console_scripts": [
            "mor-train=src.experiments.train_mor:main",
            "mor-evaluate=src.experiments.evaluate_mor:main",
            "mor-demo=simple_mor_demo:main",
            "mor-analyze=src.analysis.mor_analyzer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    zip_safe=False,
    keywords=[
        "mixture-of-recursions",
        "adaptive-computation",
        "transformers",
        "efficiency",
        "deep-learning",
        "pytorch",
        "nlp",
        "machine-learning",
        "artificial-intelligence",
    ],
)
