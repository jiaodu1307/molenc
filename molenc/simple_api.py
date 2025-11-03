"""Simplified API for MolEnc library.

This module provides a clean, simple interface for molecular encoding
that hides the complexity of the underlying system.
"""

from typing import List, Union, Optional, Dict, Any
import numpy as np

from molenc.core.simple_encoder_factory import get_simple_encoder_factory
from molenc.core.base import BaseEncoder
from molenc.core.exceptions import EncoderNotFoundError, InvalidSMILESError


class MolEncoder:
    """
    Simplified molecular encoder interface.
    
    This class provides a clean, easy-to-use interface for molecular encoding
    that automatically handles encoder creation, dependency management, and
    error handling.
    
    Examples:
        >>> encoder = MolEncoder('morgan')
        >>> vector = encoder.encode('CCO')
        >>> vectors = encoder.encode_batch(['CCO', 'CC(=O)O'])
    """
    
    def __init__(self, encoder_type: str, **kwargs):
        """
        Initialize molecular encoder.
        
        Args:
            encoder_type: Type of encoder ('morgan', 'maccs', 'chemberta', 'gcn', 'unimol')
            **kwargs: Additional parameters for the encoder
        """
        self.encoder_type = encoder_type
        self.factory = get_simple_encoder_factory()
        
        # Create the encoder
        try:
            self._encoder = self.factory.create_encoder(encoder_type, **kwargs)
        except Exception as e:
            # Provide helpful error message
            available = self.factory.get_available_encoders()
            available_names = [name for name, is_available in available.items() if is_available]
            
            raise EncoderNotFoundError(
                encoder_type,
                available_names,
                f"Failed to create {encoder_type} encoder: {str(e)}"
            )
    
    def encode(self, smiles: str) -> np.ndarray:
        """
        Encode a single SMILES string.
        
        Args:
            smiles: SMILES string to encode
            
        Returns:
            Molecular representation as numpy array
            
        Raises:
            InvalidSMILESError: If SMILES string is invalid
        """
        return self._encoder.encode(smiles)
    
    def encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Encode a batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings to encode
            
        Returns:
            Batch of molecular representations as numpy array
        """
        return self._encoder.encode_batch(smiles_list)
    
    def get_output_dim(self) -> int:
        """
        Get the output dimension of the encoder.
        
        Returns:
            Output dimension
        """
        return self._encoder.get_output_dim()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration.
        
        Returns:
            Configuration dictionary
        """
        return self._encoder.get_config()
    
    @property
    def encoder(self) -> BaseEncoder:
        """Get the underlying encoder instance."""
        return self._encoder
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MolEncoder(type={self.encoder_type}, output_dim={self.get_output_dim()})"


def list_available_encoders() -> Dict[str, bool]:
    """
    List all available encoders and their status.
    
    Returns:
        Dictionary mapping encoder names to availability status
    """
    factory = get_simple_encoder_factory()
    return factory.get_available_encoders()


def get_encoder_info(encoder_type: str) -> Dict[str, Any]:
    """
    Get detailed information about an encoder.
    
    Args:
        encoder_type: Type of encoder
        
    Returns:
        Dictionary with encoder information
    """
    factory = get_simple_encoder_factory()
    return factory.get_encoder_info(encoder_type)


def create_encoder(encoder_type: str, **kwargs) -> MolEncoder:
    """
    Create a molecular encoder.
    
    Args:
        encoder_type: Type of encoder to create
        **kwargs: Additional parameters for the encoder
        
    Returns:
        MolEncoder instance
    """
    return MolEncoder(encoder_type, **kwargs)


# Convenience functions for common encoders
def create_morgan_encoder(radius: int = 2, n_bits: int = 2048, **kwargs) -> MolEncoder:
    """Create a Morgan fingerprint encoder."""
    return MolEncoder('morgan', radius=radius, n_bits=n_bits, **kwargs)


def create_maccs_encoder(**kwargs) -> MolEncoder:
    """Create a MACCS keys encoder."""
    return MolEncoder('maccs', **kwargs)


def create_chemberta_encoder(model_name: str = "seyonec/ChemBERTa-zinc-base-v1", **kwargs) -> MolEncoder:
    """Create a ChemBERTa encoder."""
    return MolEncoder('chemberta', model_name=model_name, **kwargs)


def create_gcn_encoder(hidden_dims: List[int] = None, **kwargs) -> MolEncoder:
    """Create a GCN encoder."""
    if hidden_dims is None:
        hidden_dims = [128, 128]
    return MolEncoder('gcn', hidden_dims=hidden_dims, **kwargs)


def create_unimol_encoder(output_dim: int = 512, **kwargs) -> MolEncoder:
    """Create a UniMol encoder."""
    return MolEncoder('unimol', output_dim=output_dim, **kwargs)


# For backward compatibility
def get_encoder(encoder_type: str, **kwargs) -> MolEncoder:
    """
    Get an encoder instance (backward compatibility).
    
    Args:
        encoder_type: Type of encoder
        **kwargs: Additional parameters
        
    Returns:
        MolEncoder instance
    """
    return MolEncoder(encoder_type, **kwargs)