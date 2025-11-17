#!/usr/bin/env python3
"""
Example usage of SchNet encoder for molecular encoding.
Demonstrates various use cases and best practices.
"""

import sys
import os
import numpy as np
from typing import List, Dict, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from schnet_encoder import SchNetEncoder
except ImportError as e:
    print(f"Error importing SchNetEncoder: {e}")
    print("Please ensure you have all required dependencies installed.")
    sys.exit(1)


def example_basic_encoding():
    """Basic single molecule encoding example."""
    print("=== Basic Single Molecule Encoding ===")
    
    # Initialize encoder
    encoder = SchNetEncoder()
    
    # Example molecules
    molecules = [
        ("CCO", "Ethanol"),
        ("c1ccccc1", "Benzene"),
        ("CC(=O)O", "Acetic acid"),
        ("CCN", "Ethylamine"),
    ]
    
    print("Encoding various molecules:")
    for smiles, name in molecules:
        try:
            embedding = encoder.encode_smiles(smiles)
            print(f"  {name} ({smiles}):")
            print(f"    Shape: {embedding.shape}")
            print(f"    Norm: {np.linalg.norm(embedding):.3f}")
            print(f"    Mean: {np.mean(embedding):.3f}")
            print(f"    Std: {np.std(embedding):.3f}")
            print()
        except Exception as e:
            print(f"  Error encoding {name}: {e}")


def example_batch_processing():
    """Batch processing example for efficiency."""
    print("=== Batch Processing ===")
    
    encoder = SchNetEncoder()
    
    # Drug-like molecules dataset
    drug_molecules = [
        "CC1=CC=C(C=C1)C(=O)O",  # Benzoic acid
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen-like
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(=O)N",  # Complex structure
        "C1=CC=C(C=C1)CN",  # Benzylamine
        "CC(C)COC(=O)C",  # Isobutyl acetate
        "C1CC1C2=CC=CC=C2",  # Cyclopropylbenzene
        "CCOCCOCC",  # Diethylene glycol dimethyl ether
    ]
    
    print(f"Processing {len(drug_molecules)} drug-like molecules...")
    
    # Single molecule processing
    import time
    start_time = time.time()
    single_embeddings = []
    for smiles in drug_molecules:
        embedding = encoder.encode_smiles(smiles)
        single_embeddings.append(embedding)
    single_time = time.time() - start_time
    
    # Batch processing
    start_time = time.time()
    batch_embeddings = encoder.encode_batch(drug_molecules)
    batch_time = time.time() - start_time
    
    print(f"Single processing time: {single_time:.3f}s")
    print(f"Batch processing time: {batch_time:.3f}s")
    print(f"Speedup: {single_time/batch_time:.2f}x")
    print(f"Results shape: {batch_embeddings.shape}")
    
    # Verify results are similar (within numerical precision)
    single_array = np.array(single_embeddings)
    diff = np.mean(np.abs(single_array - batch_embeddings))
    print(f"Mean absolute difference: {diff:.6f}")


def example_similarity_search():
    """Molecular similarity search example."""
    print("=== Molecular Similarity Search ===")
    
    encoder = SchNetEncoder()
    
    # Reference molecules
    reference_mols = {
        "CCO": "Ethanol",
        "CCCO": "Propanol",
        "CCCCO": "Butanol",
        "c1ccccc1": "Benzene",
        "c1ccc(C)cc1": "Toluene",
        "c1ccc(CC)cc1": "Ethylbenzene",
    }
    
    # Query molecules
    query_mols = {
        "CC(C)O": "Isopropanol",
        "c1ccc(CCC)cc1": "Propylbenzene",
        "CC(=O)O": "Acetic acid",
    }
    
    print("Computing embeddings...")
    
    # Compute embeddings for all molecules
    all_smiles = list(reference_mols.keys()) + list(query_mols.keys())
    all_names = list(reference_mols.values()) + list(query_mols.values())
    
    embeddings = encoder.encode_batch(all_smiles)
    
    print(f"Computed embeddings for {len(all_smiles)} molecules")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Split embeddings
    n_ref = len(reference_mols)
    ref_embeddings = embeddings[:n_ref]
    query_embeddings = embeddings[n_ref:]
    
    print("\nSimilarity search results:")
    print("Query Molecule → Most Similar Reference")
    print("-" * 50)
    
    for i, (query_smiles, query_name) in enumerate(query_mols.items()):
        query_embedding = query_embeddings[i]
        
        # Compute similarities with all reference molecules
        similarities = []
        for j, (ref_smiles, ref_name) in enumerate(reference_mols.items()):
            ref_embedding = ref_embeddings[j]
            
            # Cosine similarity
            similarity = np.dot(query_embedding, ref_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(ref_embedding)
            )
            similarities.append((similarity, ref_name, ref_smiles))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        
        print(f"{query_name} ({query_smiles}):")
        for sim, ref_name, ref_smiles in similarities[:3]:  # Top 3
            print(f"  → {ref_name} ({ref_smiles}): {sim:.3f}")
        print()


def example_custom_configuration():
    """Custom encoder configuration example."""
    print("=== Custom Configuration ===")
    
    # Create encoder with custom configuration
    encoder = SchNetEncoder(
        hidden_channels=256,      # Larger embedding dimension
        num_interactions=8,         # More interaction blocks
        cutoff=15.0,                # Larger cutoff distance
        num_gaussians=100,          # More Gaussian basis functions
    )
    
    print(f"Custom encoder configuration:")
    print(f"  Hidden channels: {encoder.hidden_channels}")
    print(f"  Interactions: {len(encoder.interactions)}")
    print(f"  Cutoff: {encoder.cutoff}")
    print(f"  Gaussian basis: {encoder.num_gaussians}")
    
    # Test with a complex molecule
    complex_smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"  # Ibuprofen
    
    embedding = encoder.encode_smiles(complex_smiles)
    
    print(f"\nComplex molecule: {complex_smiles}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.3f}")
    print(f"Memory usage: {embedding.nbytes} bytes")


def example_dimensionality_reduction():
    """Dimensionality reduction for visualization."""
    print("=== Dimensionality Reduction ===")
    
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn not available. Install with: pip install scikit-learn")
        return
    
    encoder = SchNetEncoder()
    
    # Diverse molecular dataset
    molecules = [
        "CCO", "CCCO", "CCCCO", "CC(C)O", "C(C)(C)O",  # Alcohols
        "c1ccccc1", "c1ccc(C)cc1", "c1ccc(CC)cc1", "c1ccc(CCC)cc1",  # Aromatics
        "CC(=O)O", "CCCO(=O)O", "c1ccccc1C(=O)O",  # Acids
        "CCN", "CCCN", "CCCN(C)C",  # Amines
        "C=C", "C#C", "C=CC", "C#CC",  # Unsaturated
        "C1CCCCC1", "C1CCCCCC1", "C1CCCCCCC1",  # Cyclic
    ]
    
    print(f"Computing embeddings for {len(molecules)} molecules...")
    embeddings = encoder.encode_batch(molecules)
    
    print(f"Original dimension: {embeddings.shape[1]}")
    
    # PCA to 10 dimensions
    pca = PCA(n_components=10)
    embeddings_pca = pca.fit_transform(embeddings)
    print(f"PCA dimension: {embeddings_pca.shape[1]}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_[:3]}")
    
    # t-SNE to 2 dimensions for visualization
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    print(f"t-SNE dimension: {embeddings_2d.shape[1]}")
    
    # Print first few 2D coordinates
    print("\nFirst 5 molecules in 2D space:")
    for i, (mol, coord) in enumerate(zip(molecules[:5], embeddings_2d[:5])):
        print(f"  {mol}: ({coord[0]:.3f}, {coord[1]:.3f})")


def example_error_handling():
    """Error handling examples."""
    print("=== Error Handling ===")
    
    encoder = SchNetEncoder()
    
    # Test various problematic inputs
    test_cases = [
        ("invalid_smiles", "Invalid SMILES"),
        ("", "Empty string"),
        ("C(C)(C)(C)(C)C", "Overvalent carbon"),
        ("XYZ123", "Gibberish"),
        ("C", "Single carbon (may have issues with 3D)"),
    ]
    
    print("Testing error handling:")
    for smiles, description in test_cases:
        try:
            result = encoder.encode_smiles(smiles)
            print(f"  ✓ {description}: Success (shape: {result.shape})")
        except Exception as e:
            print(f"  ✗ {description}: {type(e).__name__}: {str(e)[:50]}...")


def main():
    """Run all examples."""
    print("SchNet Molecular Encoder - Usage Examples")
    print("=" * 50)
    
    examples = [
        ("Basic Encoding", example_basic_encoding),
        ("Batch Processing", example_batch_processing),
        ("Similarity Search", example_similarity_search),
        ("Custom Configuration", example_custom_configuration),
        ("Dimensionality Reduction", example_dimensionality_reduction),
        ("Error Handling", example_error_handling),
    ]
    
    for name, func in examples:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            func()
        except Exception as e:
            print(f"Error in {name}: {e}")
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()