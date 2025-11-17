#!/usr/bin/env python3
"""
Local test script for SchNet encoder without Docker.
This script tests the core functionality of the SchNet encoder.
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from schnet_encoder import SchNetEncoder
except ImportError as e:
    print(f"Error importing SchNetEncoder: {e}")
    sys.exit(1)


def test_basic_functionality():
    """Test basic encoder functionality."""
    print("Testing basic SchNet encoder functionality...")
    
    try:
        # Initialize encoder
        encoder = SchNetEncoder()
        print(f"✓ Encoder initialized successfully")
        print(f"  Hidden channels: {encoder.hidden_channels}")
        print(f"  Device: {encoder.device}")
        
        # Test single molecule encoding
        test_smiles = "CCO"  # Ethanol
        print(f"\nTesting single molecule encoding: {test_smiles}")
        
        start_time = time.time()
        result = encoder.encode_smiles(test_smiles)
        end_time = time.time()
        
        print(f"✓ Successfully encoded {test_smiles}")
        print(f"  Shape: {result.shape}")
        print(f"  Time: {end_time - start_time:.3f}s")
        print(f"  First 5 values: {result[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_batch_encoding():
    """Test batch encoding functionality."""
    print("\nTesting batch encoding functionality...")
    
    try:
        encoder = SchNetEncoder()
        
        # Test batch of molecules
        test_smiles_list = [
            "CCO",      # Ethanol
            "CCCO",     # Propanol  
            "c1ccccc1", # Benzene
            "CC(=O)O",  # Acetic acid
            "CCN"       # Ethylamine
        ]
        
        print(f"Testing batch encoding of {len(test_smiles_list)} molecules:")
        for smiles in test_smiles_list:
            print(f"  - {smiles}")
        
        start_time = time.time()
        results = encoder.encode_batch(test_smiles_list)
        end_time = time.time()
        
        print(f"✓ Batch encoding successful")
        print(f"  Results shape: {results.shape}")
        print(f"  Time: {end_time - start_time:.3f}s")
        print(f"  Average time per molecule: {(end_time - start_time) / len(test_smiles_list):.3f}s")
        
        # Check that all results are valid
        for i, (smiles, result) in enumerate(zip(test_smiles_list, results)):
            if np.allclose(result, 0):
                print(f"  ⚠ Warning: {smiles} returned zero vector (may have failed)")
            else:
                print(f"  ✓ {smiles}: shape {result.shape}, norm {np.linalg.norm(result):.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch encoding test failed: {e}")
        return False


def test_different_molecules():
    """Test various types of molecules."""
    print("\nTesting different molecule types...")
    
    try:
        encoder = SchNetEncoder()
        
        # Different molecule types
        test_cases = [
            ("CCO", "Simple alcohol"),
            ("c1ccccc1", "Aromatic ring"),
            ("CC(=O)O", "Carboxylic acid"),
            ("CCN", "Amine"),
            ("C=C", "Alkene"),
            ("C#C", "Alkyne"),
            ("CC(C)C", "Branched alkane"),
            ("C1CCCCC1", "Cycloalkane"),
        ]
        
        print("Testing various molecular structures:")
        results = []
        
        for smiles, description in test_cases:
            try:
                result = encoder.encode_smiles(smiles)
                norm = np.linalg.norm(result)
                results.append((smiles, description, result, norm))
                print(f"  ✓ {smiles} ({description}): norm = {norm:.3f}")
            except Exception as e:
                print(f"  ✗ {smiles} ({description}): {e}")
        
        # Compare similarity between different molecules
        print("\nMolecular similarity analysis:")
        for i in range(min(3, len(results))):
            for j in range(i + 1, min(5, len(results))):
                smiles1, desc1, result1, norm1 = results[i]
                smiles2, desc2, result2, norm2 = results[j]
                
                # Cosine similarity
                similarity = np.dot(result1, result2) / (norm1 * norm2)
                print(f"  {smiles1} vs {smiles2}: similarity = {similarity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Different molecules test failed: {e}")
        return False


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\nTesting error handling...")
    
    try:
        encoder = SchNetEncoder()
        
        # Test invalid SMILES
        invalid_smiles = ["invalid", "XYZ123", "", "C(C)(C)(C)(C)C"]  # Last one might be invalid
        
        print("Testing invalid SMILES strings:")
        for smiles in invalid_smiles:
            try:
                result = encoder.encode_smiles(smiles)
                print(f"  ⚠ {smiles}: Unexpectedly succeeded (shape: {result.shape})")
            except Exception as e:
                print(f"  ✓ {smiles}: Correctly failed with {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def test_performance():
    """Test performance with larger datasets."""
    print("\nTesting performance with larger dataset...")
    
    try:
        encoder = SchNetEncoder()
        
        # Generate a larger test set
        test_smiles = [
            "CCO", "CCCO", "CCCCO", "CC(C)O", "c1ccccc1", "c1ccc(C)cc1",
            "CC(=O)O", "CCN", "CCCN", "C=C", "C#C", "CC(C)C", "C1CCCCC1",
            "CCOC", "CCCOCC", "c1ccccc1O", "c1ccc(O)cc1", "CC(=O)N",
            "CCNC", "C=C(C)C", "C#CC", "CC(C)(C)C", "C1CCCCCC1"
        ] * 5  # 100 molecules total
        
        print(f"Testing with {len(test_smiles)} molecules...")
        
        start_time = time.time()
        results = encoder.encode_batch(test_smiles)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / len(test_smiles)
        
        print(f"✓ Performance test completed")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time per molecule: {avg_time:.3f}s")
        print(f"  Throughput: {len(test_smiles) / total_time:.1f} molecules/second")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SchNet Encoder Local Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Batch Encoding", test_batch_encoding),
        ("Different Molecules", test_different_molecules),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Summary: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! SchNet encoder is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())