"""Demonstration of using Morgan fingerprints for molecular encoding.

This script shows how to use the Morgan fingerprint encoder to convert 
SMILES strings into binary fingerprint vectors.
"""

import sys
from pathlib import Path
import numpy as np

# Add the project root to the path so we can import molenc
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from molenc import MolEncoder


def main():
    """Run the Morgan fingerprint demonstration."""
    print("=== Morgan Fingerprint Demonstration ===\n")
    
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
    
    print("1. Creating Morgan Encoder")
    print("-" * 30)
    
    try:
        # Create Morgan fingerprint encoder with default parameters
        encoder = MolEncoder('morgan')
        print(f"✓ Encoder created successfully: {encoder}")
        print(f"✓ Output dimension: {encoder.get_output_dim()}")
        print()
        
    except Exception as e:
        print(f"✗ Failed to create encoder: {e}")
        return False
    
    print("2. Creating Custom Morgan Encoder")
    print("-" * 30)
    
    try:
        # Create Morgan fingerprint encoder with custom parameters
        custom_encoder = MolEncoder('morgan', radius=3, n_bits=2048)
        print(f"✓ Custom encoder created successfully: {custom_encoder}")
        print(f"✓ Output dimension: {custom_encoder.get_output_dim()}")
        print()
        
    except Exception as e:
        print(f"✗ Failed to create custom encoder: {e}")
        return False
    
    print("3. Encoding Individual Molecules")
    print("-" * 30)
    
    fingerprints = {}
    
    for name, smiles in test_molecules:
        try:
            print(f"Encoding {name} (SMILES: {smiles})")
            fingerprint = encoder.encode(smiles)
            fingerprints[name] = fingerprint
            print(f"  ✓ Success! Fingerprint shape: {fingerprint.shape}")
            print(f"  ✓ Data type: {fingerprint.dtype}")
            print(f"  ✓ Number of bits set: {np.sum(fingerprint)}")
            print(f"  ✓ First 10 elements: {fingerprint[:10]}")
            print()
        except Exception as e:
            print(f"  ✗ Failed to encode {name}: {e}")
            print()
    
    print("4. Batch Encoding")
    print("-" * 30)
    
    try:
        # Extract just the SMILES strings for batch processing
        smiles_list = [smiles for _, smiles in test_molecules]
        names_list = [name for name, _ in test_molecules]
        
        print(f"Encoding batch of {len(smiles_list)} molecules:")
        for name, smiles in zip(names_list, smiles_list):
            print(f"  - {name}: {smiles}")
        
        batch_fingerprints = encoder.encode_batch(smiles_list)
        print(f"✓ Batch encoding successful!")
        print(f"✓ Result shape: {batch_fingerprints.shape}")
        print(f"✓ Data type: {batch_fingerprints.dtype}")
        print(f"✓ Number of bits set in first molecule: {np.sum(batch_fingerprints[0])}")
        print()
        
    except Exception as e:
        print(f"✗ Batch encoding failed: {e}")
        print()
    
    print("5. Comparing Molecular Similarity")
    print("-" * 30)
    
    if len(fingerprints) >= 2:
        # Compare the first two molecules
        names = list(fingerprints.keys())[:2]
        fp1, fp2 = fingerprints[names[0]], fingerprints[names[1]]
        
        # Calculate Tanimoto similarity
        def tanimoto_similarity(fp_a, fp_b):
            """Calculate Tanimoto similarity between two fingerprints."""
            intersection = np.sum(np.logical_and(fp_a, fp_b))
            union = np.sum(np.logical_or(fp_a, fp_b))
            if union == 0:
                return 0.0
            return intersection / union
        
        similarity = tanimoto_similarity(fp1, fp2)
        print(f"Tanimoto similarity between {names[0]} and {names[1]}: {similarity:.4f}")
        print()
    
    print("6. Encoder Configuration")
    print("-" * 30)
    
    try:
        config = encoder.get_config()
        print("Encoder configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()
    except Exception as e:
        print(f"✗ Failed to get configuration: {e}")
        print()
    
    print("=== Morgan Fingerprint Demonstration Completed ===")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)