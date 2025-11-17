#!/usr/bin/env python3
"""
D-MPNN Encoder Usage Examples
Demonstrates how to use the D-MPNN molecular encoder
"""

import numpy as np
from dmpnn_encoder import DMPNNEncoder

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    
    # Initialize encoder with default parameters
    encoder = DMPNNEncoder(
        node_dim=64,
        edge_dim=64,
        depth=3,
        dropout=0.0,
        aggregation='mean'
    )
    
    # Encode a single molecule
    smiles = "CCO"  # Ethanol
    embedding = encoder.encode_smiles(smiles)
    
    print(f"SMILES: {smiles}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding type: {type(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    print()

def example_batch_encoding():
    """Batch encoding example"""
    print("=== Batch Encoding Example ===")
    
    encoder = DMPNNEncoder(
        node_dim=64,
        edge_dim=64,
        depth=3,
        dropout=0.0,
        aggregation='mean'
    )
    
    # List of molecules to encode
    molecules = [
        "CCO",           # Ethanol
        "CC(=O)O",       # Acetic acid
        "c1ccccc1",      # Benzene
        "CCN(CC)CC",     # Triethylamine
        "CC(C)CCO",      # Isobutanol
    ]
    
    # Encode all molecules at once
    embeddings = encoder.encode_batch(molecules)
    
    print(f"Encoded {len(molecules)} molecules")
    print(f"Embeddings shape: {embeddings.shape}")
    
    for i, (smiles, embedding) in enumerate(zip(molecules, embeddings)):
        print(f"  {smiles}: {embedding[:5]}...")
    print()

def example_different_parameters():
    """Example with different encoder parameters"""
    print("=== Different Parameters Example ===")
    
    # Test different depths
    test_molecule = "CCO"
    
    for depth in [1, 2, 3, 4]:
        encoder = DMPNNEncoder(
            node_dim=64,
            edge_dim=64,
            depth=depth,
            dropout=0.0,
            aggregation='mean'
        )
        
        embedding = encoder.encode_smiles(test_molecule)
        print(f"Depth {depth}: {embedding[:5]}...")
    
    print()
    
    # Test different aggregation methods
    for aggregation in ['mean', 'sum', 'max']:
        encoder = DMPNNEncoder(
            node_dim=64,
            edge_dim=64,
            depth=3,
            dropout=0.0,
            aggregation=aggregation
        )
        
        embedding = encoder.encode_smiles(test_molecule)
        print(f"Aggregation '{aggregation}': {embedding[:5]}...")
    print()

def example_similarity_search():
    """Example of computing molecular similarity"""
    print("=== Similarity Search Example ===")
    
    encoder = DMPNNEncoder(
        node_dim=64,
        edge_dim=64,
        depth=3,
        dropout=0.0,
        aggregation='mean'
    )
    
    # Reference molecule
    reference_smiles = "CCO"  # Ethanol
    reference_embedding = encoder.encode_smiles(reference_smiles)
    
    # Candidate molecules
    candidates = [
        "CCO",           # Ethanol (same)
        "CCCO",          # Propanol
        "CC(C)O",        # Isopropanol
        "CC(=O)O",       # Acetic acid
        "c1ccccc1",      # Benzene (very different)
    ]
    
    # Compute similarities
    similarities = []
    for smiles in candidates:
        embedding = encoder.encode_smiles(smiles)
        # Compute cosine similarity
        similarity = np.dot(reference_embedding, embedding) / (
            np.linalg.norm(reference_embedding) * np.linalg.norm(embedding)
        )
        similarities.append((smiles, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Reference: {reference_smiles}")
    print("Similarity ranking:")
    for smiles, similarity in similarities:
        print(f"  {smiles:<10}: {similarity:.3f}")
    print()

def example_molecular_properties():
    """Example of using embeddings for property prediction"""
    print("=== Molecular Properties Example ===")
    
    encoder = DMPNNEncoder(
        node_dim=64,
        edge_dim=64,
        depth=3,
        dropout=0.0,
        aggregation='mean'
    )
    
    # Simple mock property prediction
    # In practice, you would train a model on top of these embeddings
    
    molecules = [
        ("CCO", "Alcohol"),
        ("CC(=O)O", "Acid"),
        ("c1ccccc1", "Aromatic"),
        ("CCN(CC)CC", "Amine"),
        ("CC(C)CCO", "Branched alcohol"),
    ]
    
    print("Molecular embeddings (first 10 dimensions):")
    for smiles, description in molecules:
        embedding = encoder.encode_smiles(smiles)
        print(f"  {smiles:<10} ({description}): {embedding[:10]}")
    
    print("\nNote: These embeddings capture molecular structure and can be used")
    print("for downstream tasks like property prediction, similarity search,")
    print("or classification when combined with appropriate models.")
    print()

def example_visualization_data():
    """Example of preparing data for visualization"""
    print("=== Visualization Data Example ===")
    
    encoder = DMPNNEncoder(
        node_dim=64,
        edge_dim=64,
        depth=3,
        dropout=0.0,
        aggregation='mean'
    )
    
    # Generate embeddings for visualization
    molecules = [
        "CCO", "CCCO", "CC(C)O", "CC(C)(C)O",  # Alcohols
        "CC(=O)O", "CCCOO", "CC(C)OO",          # Acids
        "c1ccccc1", "c1ccc(C)cc1", "c1ccc(O)cc1", # Aromatics
    ]
    
    embeddings = encoder.encode_batch(molecules)
    
    print(f"Generated embeddings for {len(molecules)} molecules")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    print(f"Embeddings mean: {embeddings.mean():.3f}")
    print(f"Embeddings std: {embeddings.std():.3f}")
    
    print("\nThese embeddings can be used with dimensionality reduction")
    print("techniques like PCA, t-SNE, or UMAP for molecular visualization.")
    print()

if __name__ == "__main__":
    print("D-MPNN Encoder Usage Examples")
    print("=" * 50)
    print()
    
    # Run all examples
    examples = [
        example_basic_usage,
        example_batch_encoding,
        example_different_parameters,
        example_similarity_search,
        example_molecular_properties,
        example_visualization_data,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            print()
    
    print("All examples completed!")
    print("\nFor more information, see the API documentation or")
    print("check the test files for additional usage patterns.")