"""Utility functions for MolEnc library."""

import numpy as np
from typing import List, Union, Optional, Tuple, Any, Dict
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import logging
from pathlib import Path


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging for MolEnc.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    logger = logging.getLogger('molenc')
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def normalize_smiles(smiles: str, canonical: bool = True) -> Optional[str]:
    """
    Normalize a SMILES string using RDKit.

    Args:
        smiles: Input SMILES string
        canonical: Whether to return canonical SMILES

    Returns:
        Normalized SMILES string or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if canonical:
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            return Chem.MolToSmiles(mol)
    except Exception:
        return None


def calculate_molecular_properties(smiles: str) -> Optional[Dict[str, float]]:
    """
    Calculate basic molecular properties.

    Args:
        smiles: SMILES string

    Returns:
        Dictionary of molecular properties or None if invalid SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        properties: Dict[str, float] = {
            'molecular_weight': float(Descriptors.MolWt(mol)),
            'logp': float(Descriptors.MolLogP(mol)),
            'num_atoms': float(mol.GetNumAtoms()),
            'num_bonds': float(mol.GetNumBonds()),
            'num_rings': float(rdMolDescriptors.CalcNumRings(mol)),
            'tpsa': float(Descriptors.TPSA(mol)),
            'num_rotatable_bonds': float(Descriptors.NumRotatableBonds(mol)),
            'num_hbd': float(Descriptors.NumHDonors(mol)),
            'num_hba': float(Descriptors.NumHAcceptors(mol))
        }

        return properties
    except Exception:
        return None


# Import unified batch processing utilities
from .batch_utils import batch_normalize_smiles


# Import unified similarity functions from similarity_utils
from .similarity_utils import (
    cosine_similarity,
    euclidean_distance,
    tanimoto_similarity,
    find_similar_molecules
)


def save_vectors(vectors: np.ndarray,
                 file_path: str,
                 format: str = 'npy',
                 metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save molecular vectors to file.

    Args:
        vectors: Array of molecular vectors
        file_path: Path to save file
        format: File format ('npy', 'npz', 'csv')
        metadata: Optional metadata to save with vectors
    """
    file_path_obj = Path(file_path)
    file_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if format == 'npy':
        np.save(file_path_obj, vectors, allow_pickle=True)
    elif format == 'npz':
        if metadata:
            np.savez(file_path_obj, vectors=vectors, **metadata)
        else:
            np.savez(file_path_obj, vectors=vectors)
    elif format == 'csv':
        np.savetxt(file_path_obj, vectors, delimiter=',')
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_vectors(file_path: str, format: str = 'npy') -> Union[np.ndarray, Dict[str, Any]]:
    """
    Load molecular vectors from file.

    Args:
        file_path: Path to vector file
        format: File format ('npy', 'npz', 'csv')

    Returns:
        Loaded vectors (and metadata if npz format)
    """
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        raise FileNotFoundError(f"Vector file not found: {file_path}")

    if format == 'npy':
        return np.load(file_path_obj, allow_pickle=True)
    elif format == 'npz':
        return dict(np.load(file_path_obj, allow_pickle=True))
    elif format == 'csv':
        return np.loadtxt(file_path_obj, delimiter=',')
    else:
        raise ValueError(f"Unsupported format: {format}")


# Import unified dependency checking from dependency_utils
from .dependency_utils import check_dependencies


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.

    Returns:
        Dictionary containing system information
    """
    import platform
    import sys

    info: Dict[str, Any] = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'architecture': platform.architecture(),
    }

    # Check for GPU availability
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
    except ImportError:
        info['torch_available'] = False

    return info
