"""Multimodal molecular encoders.

This module contains encoders that combine multiple modalities
(e.g., 2D structure, 3D conformation, text descriptions) to learn
comprehensive molecular representations.
"""

# Import multimodal implementations
try:
    from .unimol import UniMolEncoder
except ImportError:
    UniMolEncoder = None

__all__ = [
    "UniMolEncoder",
]