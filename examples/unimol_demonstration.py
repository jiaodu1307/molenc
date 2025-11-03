"""Demonstration of using the UniMol pre-trained model for molecular encoding.

This script shows how to use the UniMol encoder to convert SMILES strings 
into molecular embeddings that capture 3D structural information.
"""

import sys
from pathlib import Path
import numpy as np

# Add the project root to the path so we can import molenc
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from molenc import MolEncoder


def main():
    """Run the UniMol demonstration."""
    print("=== UniMol Pre-trained Model Demonstration ===\n")
    
    # Example molecules with diverse structures
    test_molecules = [
        ("Ethanol", "CCO"),
        ("Acetic Acid", "CC(=O)O"),
        ("Benzene", "c1ccccc1"),
        ("Triethylamine", "CCN(CC)CC"),
        ("Cyclohexane", "C1CCCCC1"),
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),  # More complex molecule
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  # Another complex molecule
    ]
    
    print("1. Creating UniMol Encoder")
    print("-" * 30)
    
    try:
        # Create UniMol encoder - this will automatically handle environment setup
        encoder = MolEncoder('unimol')
        print(f"✓ Encoder created successfully: {encoder}")
        print(f"✓ Output dimension: {encoder.get_output_dim()}")
        print(f"✓ Device: {encoder.encoder.device}")
        print()
        
    except Exception as e:
        print(f"✗ Failed to create encoder: {e}")
        return False
    
    print("2. Encoding Individual Molecules")
    print("-" * 30)
    
    embeddings = {}
    
    for name, smiles in test_molecules:
        try:
            print(f"Encoding {name} (SMILES: {smiles})")
            embedding = encoder.encode(smiles)
            embeddings[name] = embedding
            print(f"  ✓ Success! Embedding shape: {embedding.shape}")
            print(f"  ✓ First 5 elements: {embedding[:5]}")
            print()
        except Exception as e:
            print(f"  ✗ Failed to encode {name}: {e}")
            print()
    
    print("3. Batch Encoding")
    print("-" * 30)
    
    try:
        # Extract just the SMILES strings for batch processing
        smiles_list = [smiles for _, smiles in test_molecules]
        names_list = [name for name, _ in test_molecules]
        
        print(f"Encoding batch of {len(smiles_list)} molecules:")
        for name, smiles in zip(names_list, smiles_list):
            print(f"  - {name}: {smiles}")
        
        batch_embeddings = encoder.encode_batch(smiles_list)
        print(f"✓ Batch encoding successful!")
        print(f"✓ Result shape: {batch_embeddings.shape}")
        print(f"✓ Data type: {batch_embeddings.dtype}")
        print()
        
    except Exception as e:
        print(f"✗ Batch encoding failed: {e}")
        print()
    
    print("4. Comparing Molecular Similarity")
    print("-" * 30)
    
    if len(embeddings) >= 2:
        # Compare the first two molecules
        names = list(embeddings.keys())[:2]
        emb1, emb2 = embeddings[names[0]], embeddings[names[1]]
        
        # Import from unified similarity utilities
        from molenc.core.similarity_utils import cosine_similarity
        
        similarity = cosine_similarity(emb1, emb2)
        print(f"Similarity between {names[0]} and {names[1]}: {similarity:.4f}")
        print()
    
    print("5. Encoder Configuration")
    print("-" * 30)
    
    try:
        config = encoder.get_config()
        print("Encoder configuration:")
        for key, value in config.items():
            if key != 'model':  # Skip the model object
                print(f"  {key}: {value}")
        print()
    except Exception as e:
        print(f"✗ Failed to get configuration: {e}")
        print()
    
    print("=== UniMol Demonstration Completed ===")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)