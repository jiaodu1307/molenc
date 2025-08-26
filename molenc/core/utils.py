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


def batch_normalize_smiles(smiles_list: List[str],
                           canonical: bool = True,
                           remove_invalid: bool = True) -> List[str]:
    """
    Normalize a batch of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        canonical: Whether to return canonical SMILES
        remove_invalid: Whether to remove invalid SMILES

    Returns:
        List of normalized SMILES strings
    """
    normalized: List[str] = []

    for smiles in smiles_list:
        norm_smiles = normalize_smiles(smiles, canonical)
        if norm_smiles is not None:
            normalized.append(norm_smiles)
        elif not remove_invalid:
            normalized.append(smiles)  # Keep original if normalization fails

    return normalized


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(vec1 - vec2))


def tanimoto_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Tanimoto similarity for binary vectors.

    Args:
        vec1: First binary vector
        vec2: Second binary vector

    Returns:
        Tanimoto similarity score
    """
    intersection: np.ndarray = np.sum(vec1 * vec2)
    union: np.ndarray = np.sum((vec1 + vec2) > 0)

    if union == 0:
        return 0.0

    return float(intersection / union)


def find_similar_molecules(query_vector: np.ndarray,
                           database_vectors: np.ndarray,
                           similarity_metric: str = 'cosine',
                           top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find similar molecules based on vector similarity.

    Args:
        query_vector: Query molecule vector
        database_vectors: Database of molecule vectors (n_molecules x n_features)
        similarity_metric: Similarity metric ('cosine', 'euclidean', 'tanimoto')
        top_k: Number of top similar molecules to return

    Returns:
        Tuple of (indices, similarity_scores) for top-k similar molecules
    """
    if similarity_metric == 'cosine':
        similarities = np.array([
            cosine_similarity(query_vector, db_vec)
            for db_vec in database_vectors
        ], dtype=np.float32)
        # Higher is better for cosine similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]

    elif similarity_metric == 'euclidean':
        distances = np.array([
            euclidean_distance(query_vector, db_vec)
            for db_vec in database_vectors
        ], dtype=np.float32)
        # Lower is better for distance
        top_indices = np.argsort(distances)[:top_k]
        similarities = 1.0 / (1.0 + distances)  # Convert to similarity

    elif similarity_metric == 'tanimoto':
        similarities = np.array([
            tanimoto_similarity(query_vector, db_vec)
            for db_vec in database_vectors
        ], dtype=np.float32)
        # Higher is better for Tanimoto similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]

    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

    top_similarities = similarities[top_indices]

    return top_indices, top_similarities


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


def check_dependencies(dependencies: List[str]) -> Dict[str, bool]:
    """
    Check if required dependencies are installed.

    Args:
        dependencies: List of package names to check

    Returns:
        Dictionary mapping package names to availability status
    """
    import importlib

    status: Dict[str, bool] = {}
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            status[dep] = True
        except ImportError:
            status[dep] = False

    return status


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
