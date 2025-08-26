"""Sequence-based molecular encoders.

This module contains encoders that treat molecules as sequences (SMILES)
and use transformer-based models to learn representations.
"""

# Import sequence-based implementations
try:
    from .chemberta import ChemBERTaEncoder
except ImportError:
    ChemBERTaEncoder = None

__all__ = [
    "ChemBERTaEncoder",
]