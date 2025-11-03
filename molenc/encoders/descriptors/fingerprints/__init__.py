"""Molecular fingerprint encoders.

This module contains various molecular fingerprint implementations
including Morgan and MACCS fingerprints.
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

# Import base class for extension
try:
    from .base import BaseFingerprintEncoder
except ImportError:
    BaseFingerprintEncoder = None

__all__ = [
    "MorganEncoder",
    "MACCSEncoder",
    "BaseFingerprintEncoder",
]