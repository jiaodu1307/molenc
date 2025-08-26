"""Molecular fingerprint encoders.

This module contains various molecular fingerprint implementations
including Morgan, MACCS, ECFP, and other traditional fingerprints.
"""

# Import fingerprint implementations
try:
    from .morgan import MorganEncoder
except ImportError:
    MorganEncoder = None

try:
    from .maccs import MACCSEncoder
except ImportError:
    MACCSEncoder = None

try:
    from .ecfp import ECFPEncoder
except ImportError:
    ECFPEncoder = None

try:
    from .atom_pair import AtomPairEncoder
except ImportError:
    AtomPairEncoder = None

__all__ = [
    "MorganEncoder",
    "MACCSEncoder", 
    "ECFPEncoder",
    "AtomPairEncoder",
]