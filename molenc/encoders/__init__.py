"""Molecular encoders module.

This module contains different types of molecular encoders organized into:
- descriptors: Classical chemoinformatics methods
- representations: Deep learning-based methods
"""

# Import encoder categories
from . import descriptors
from . import representations

__all__ = [
    "descriptors",
    "representations",
]