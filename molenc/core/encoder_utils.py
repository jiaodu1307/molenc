"""Common utilities for encoder implementations."""

import torch
from typing import Optional, Any, Dict, List
import logging


class EncoderUtils:
    """Common utilities for encoder implementations."""
    
    @staticmethod
    def setup_device(device: Optional[str] = None) -> torch.device:
        """
        Setup device for PyTorch-based encoders.
        
        Args:
            device: Device specification ("cpu", "cuda", or None for auto)
            
        Returns:
            torch.device object
        """
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device)
    
    @staticmethod
    def validate_positive_int(value: int, name: str) -> None:
        """
        Validate that a value is a positive integer.
        
        Args:
            value: Value to validate
            name: Name of the parameter for error messages
            
        Raises:
            ValueError: If value is not positive
        """
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    
    @staticmethod
    def validate_non_negative_int(value: int, name: str) -> None:
        """
        Validate that a value is a non-negative integer.
        
        Args:
            value: Value to validate
            name: Name of the parameter for error messages
            
        Raises:
            ValueError: If value is negative
        """
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    
    @staticmethod
    def validate_power_of_two(value: int, name: str) -> None:
        """
        Validate that a value is a positive power of 2.
        
        Args:
            value: Value to validate
            name: Name of the parameter for error messages
            
        Raises:
            ValueError: If value is not a positive power of 2
        """
        if value <= 0 or (value & (value - 1)) != 0:
            raise ValueError(f"{name} must be a positive power of 2, got {value}")
    
    @staticmethod
    def validate_choice(value: Any, choices: List[Any], name: str) -> None:
        """
        Validate that a value is in a list of valid choices.
        
        Args:
            value: Value to validate
            choices: List of valid choices
            name: Name of the parameter for error messages
            
        Raises:
            ValueError: If value is not in choices
        """
        if value not in choices:
            raise ValueError(f"{name} must be one of {choices}, got {value}")
    
    @staticmethod
    def setup_logger(name: str) -> logging.Logger:
        """
        Setup a logger for an encoder.
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


class MoleculeUtils:
    """Utilities for molecular processing."""
    
    @staticmethod
    def smiles_to_mol(smiles: str, sanitize: bool = True) -> Any:
        """
        Convert SMILES string to RDKit molecule object.
        
        Args:
            smiles: SMILES string
            sanitize: Whether to sanitize the molecule
            
        Returns:
            RDKit molecule object
            
        Raises:
            InvalidSMILESError: If SMILES is invalid
        """
        from rdkit import Chem
        from molenc.core.exceptions import InvalidSMILESError
        
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        if mol is None:
            raise InvalidSMILESError(f"Invalid SMILES: {smiles}")
        return mol
    
    @staticmethod
    def add_hydrogens(mol: Any) -> Any:
        """
        Add explicit hydrogens to a molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Molecule with explicit hydrogens
        """
        from rdkit import Chem
        return Chem.AddHs(mol)
    
    @staticmethod
    def generate_conformer(mol: Any, num_confs: int = 1, random_seed: int = 42) -> Any:
        """
        Generate 3D conformer for a molecule.
        
        Args:
            mol: RDKit molecule object
            num_confs: Number of conformers to generate
            random_seed: Random seed for reproducibility
            
        Returns:
            Molecule with 3D conformer
        """
        from rdkit.Chem import AllChem
        
        mol_with_h = MoleculeUtils.add_hydrogens(mol)
        AllChem.EmbedMolecule(mol_with_h, randomSeed=random_seed)
        AllChem.MMFFOptimizeMolecule(mol_with_h)
        return mol_with_h


class TensorUtils:
    """Utilities for tensor operations."""
    
    @staticmethod
    def pad_tensor(tensor: torch.Tensor, target_length: int, pad_value: float = 0.0) -> torch.Tensor:
        """
        Pad tensor to target length.
        
        Args:
            tensor: Input tensor
            target_length: Target length
            pad_value: Value to use for padding
            
        Returns:
            Padded tensor
        """
        current_length = tensor.size(0)
        if current_length >= target_length:
            return tensor[:target_length]
        
        pad_size = target_length - current_length
        padding = torch.full((pad_size,) + tensor.shape[1:], pad_value, dtype=tensor.dtype)
        return torch.cat([tensor, padding], dim=0)
    
    @staticmethod
    def pool_tensor(tensor: torch.Tensor, method: str = "mean") -> torch.Tensor:
        """
        Pool tensor along the first dimension.
        
        Args:
            tensor: Input tensor [seq_len, features]
            method: Pooling method ("mean", "max", "sum")
            
        Returns:
            Pooled tensor [features]
        """
        if method == "mean":
            return torch.mean(tensor, dim=0)
        elif method == "max":
            return torch.max(tensor, dim=0)[0]
        elif method == "sum":
            return torch.sum(tensor, dim=0)
        else:
            raise ValueError(f"Unknown pooling method: {method}")