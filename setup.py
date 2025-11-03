"""Setup configuration for MolEnc package."""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read version from __init__.py
def get_version():
    """Get version from __init__.py file."""
    version_file = this_directory / "molenc" / "__init__.py"
    if version_file.exists():
        with open(version_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return "0.1.0"

# Core dependencies
core_requirements = [
    "numpy>=1.19.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "packaging>=20.0",
]

# Optional dependencies for different features
optional_requirements = {
    "chemistry": [
        "rdkit>=2022.03.1",
    ],
    "deep_learning": [
        "torch>=1.12.0",
        "transformers>=4.20.0",
    ],
    "graph": [
        "torch>=1.12.0",
        "torch-geometric>=2.1.0",
    ],
    'nlp': [
            'gensim>=4.0.0',
            'transformers>=4.20.0',
        ],
    "molbert": [
        "torch>=1.12.0",
        "transformers>=4.20.0",
    ],
    "visualization": [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.9.0",
    ],
    "system": [
        "psutil>=5.9.0",
    ],
    "environment": [
        "virtualenv>=20.0.0",
    ],
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
        "black>=21.0.0",
        "flake8>=3.9.0",
        "mypy>=0.910",
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=0.5.0",
    ]
}

# All optional dependencies
optional_requirements["all"] = [
    req for reqs in optional_requirements.values() 
    for req in reqs if req not in optional_requirements["dev"]
]

# Complete dependencies (core + all optional)
optional_requirements["complete"] = (
    core_requirements + optional_requirements["all"]
)

setup(
    name="molenc",
    version=get_version(),
    author="MolEnc Development Team",
    author_email="molenc@example.com",
    description="A unified molecular encoding library for machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/molenc/molenc",
    project_urls={
        "Bug Tracker": "https://github.com/molenc/molenc/issues",
        "Documentation": "https://molenc.readthedocs.io/",
        "Source Code": "https://github.com/molenc/molenc",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require=optional_requirements,
    include_package_data=True,
    package_data={
        "molenc": [
            "data/*",
            "configs/*",
        ],
    },
    entry_points={
        "console_scripts": [
            "molenc=molenc.cli:main",
        ],
    },
    keywords=[
        "molecular encoding",
        "machine learning",
        "cheminformatics",
        "molecular descriptors",
        "molecular fingerprints",
        "deep learning",
        "graph neural networks",
        "transformers",
    ],
    zip_safe=False,
)