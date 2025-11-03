"""Unified similarity calculation utilities.

This module provides standardized similarity and distance calculation
functions that can be used across the entire MolEnc package.
"""

import numpy as np
from typing import Union, Tuple, List
from scipy.spatial.distance import cosine as scipy_cosine
from scipy.spatial.distance import euclidean as scipy_euclidean


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0-1, higher is more similar)
    """
    # Handle zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Use scipy for numerical stability
    return 1.0 - scipy_cosine(vec1, vec2)


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance (lower is more similar)
    """
    return float(scipy_euclidean(vec1, vec2))


def tanimoto_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Tanimoto similarity between two binary vectors.
    
    Args:
        vec1: First binary vector
        vec2: Second binary vector
        
    Returns:
        Tanimoto similarity score (0-1, higher is more similar)
    """
    # Convert to binary if needed
    if vec1.dtype != bool:
        vec1 = vec1.astype(bool)
    if vec2.dtype != bool:
        vec2 = vec2.astype(bool)
    
    intersection = np.sum(vec1 & vec2)
    union = np.sum(vec1 | vec2)
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)


def jaccard_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Jaccard similarity (alias for Tanimoto).
    
    Args:
        vec1: First binary vector
        vec2: Second binary vector
        
    Returns:
        Jaccard similarity score (0-1, higher is more similar)
    """
    return tanimoto_similarity(vec1, vec2)


def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Manhattan (L1) distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Manhattan distance (lower is more similar)
    """
    return float(np.sum(np.abs(vec1 - vec2)))


def dice_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Dice similarity coefficient between two binary vectors.
    
    Args:
        vec1: First binary vector
        vec2: Second binary vector
        
    Returns:
        Dice similarity score (0-1, higher is more similar)
    """
    # Convert to binary if needed
    if vec1.dtype != bool:
        vec1 = vec1.astype(bool)
    if vec2.dtype != bool:
        vec2 = vec2.astype(bool)
    
    intersection = np.sum(vec1 & vec2)
    total = np.sum(vec1) + np.sum(vec2)
    
    if total == 0:
        return 0.0
    
    return float(2.0 * intersection / total)


def find_similar_molecules(query_vector: np.ndarray,
                          database_vectors: np.ndarray,
                          similarity_metric: str = 'cosine',
                          top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find similar molecules based on vector similarity.

    Args:
        query_vector: Query molecule vector
        database_vectors: Database of molecule vectors (n_molecules x n_features)
        similarity_metric: Similarity metric ('cosine', 'euclidean', 'tanimoto', 'manhattan', 'dice')
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

    elif similarity_metric == 'manhattan':
        distances = np.array([
            manhattan_distance(query_vector, db_vec)
            for db_vec in database_vectors
        ], dtype=np.float32)
        # Lower is better for distance
        top_indices = np.argsort(distances)[:top_k]
        similarities = 1.0 / (1.0 + distances)  # Convert to similarity

    elif similarity_metric in ['tanimoto', 'jaccard']:
        similarities = np.array([
            tanimoto_similarity(query_vector, db_vec)
            for db_vec in database_vectors
        ], dtype=np.float32)
        # Higher is better for Tanimoto similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]

    elif similarity_metric == 'dice':
        similarities = np.array([
            dice_similarity(query_vector, db_vec)
            for db_vec in database_vectors
        ], dtype=np.float32)
        # Higher is better for Dice similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]

    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

    top_similarities = similarities[top_indices]
    return top_indices, top_similarities


def batch_similarity_matrix(vectors: np.ndarray, 
                           similarity_metric: str = 'cosine') -> np.ndarray:
    """
    Calculate pairwise similarity matrix for a batch of vectors.
    
    Args:
        vectors: Array of vectors (n_vectors x n_features)
        similarity_metric: Similarity metric to use
        
    Returns:
        Similarity matrix (n_vectors x n_vectors)
    """
    n_vectors = vectors.shape[0]
    similarity_matrix = np.zeros((n_vectors, n_vectors), dtype=np.float32)
    
    for i in range(n_vectors):
        for j in range(i, n_vectors):
            if similarity_metric == 'cosine':
                sim = cosine_similarity(vectors[i], vectors[j])
            elif similarity_metric == 'tanimoto':
                sim = tanimoto_similarity(vectors[i], vectors[j])
            elif similarity_metric == 'dice':
                sim = dice_similarity(vectors[i], vectors[j])
            elif similarity_metric == 'euclidean':
                # Convert distance to similarity
                dist = euclidean_distance(vectors[i], vectors[j])
                sim = 1.0 / (1.0 + dist)
            elif similarity_metric == 'manhattan':
                # Convert distance to similarity
                dist = manhattan_distance(vectors[i], vectors[j])
                sim = 1.0 / (1.0 + dist)
            else:
                raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
            
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Symmetric matrix
    
    return similarity_matrix


# Convenience aliases
def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine distance (1 - cosine_similarity)."""
    return 1.0 - cosine_similarity(vec1, vec2)


def tanimoto_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate Tanimoto distance (1 - tanimoto_similarity)."""
    return 1.0 - tanimoto_similarity(vec1, vec2)