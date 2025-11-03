"""Graph-based molecular encoders.

This module contains encoders that represent molecules as graphs
and use graph neural networks to learn molecular representations.
"""

# Import graph-based implementations
try:
    from .gcn import GCNEncoder
except ImportError:
    GCNEncoder = None

__all__ = [
    "GCNEncoder",
]