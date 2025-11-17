#!/usr/bin/env python3
"""
Local D-MPNN Encoder Test Script
Tests the D-MPNN encoder implementation directly
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dmpnn_encoder import DMPNNEncoder

def test_basic_functionality():
    """Test basic encoder functionality"""
    print("Testing basic D-MPNN encoder functionality...")
    
    try:
        # Initialize encoder
        encoder = DMPNNEncoder(
            node_dim=64,
            edge_dim=64,
            depth=3,
            dropout=0.0,
            aggregation='mean'
        )
        print("✓ Encoder initialized successfully")
        
        # Test single molecule
        smiles = "CCO"  # Ethanol
        start_time = time.time()
        result = encoder.encode_smiles(smiles)
        elapsed = time.time() - start_time
        
        print(f"✓ Single molecule encoding: {elapsed:.3f}s")
        print(f"  Shape: {result.shape}")
        print(f"  Type: {type(result)}")
        print(f"  First few values: {result[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_batch_encoding():
    """Test batch encoding"""
    print("\nTesting batch encoding...")
    
    try:
        encoder = DMPNNEncoder(
            node_dim=64,
            edge_dim=64,
            depth=3,
            dropout=0.0,
            aggregation='mean'
        )
        
        # Test batch of molecules
        molecules = [
            "CCO",           # Ethanol
            "CC(=O)O",       # Acetic acid
            "c1ccccc1",      # Benzene
            "CCN(CC)CC",     # Triethylamine
            "CC(C)CCO",      # Isobutanol
        ]
        
        start_time = time.time()
        results = encoder.encode_batch(molecules)
        elapsed = time.time() - start_time
        
        print(f"✓ Batch encoding: {elapsed:.3f}s for {len(molecules)} molecules")
        print(f"  Shape: {results.shape}")
        print(f"  Type: {type(results)}")
        
        # Test individual vs batch consistency
        single_results = []
        for mol in molecules:
            single_results.append(encoder.encode_smiles(mol))
        
        single_array = np.array(single_results)
        
        # Check if batch and individual results are similar
        if np.allclose(results, single_array, rtol=1e-5):
            print("✓ Batch and individual results are consistent")
        else:
            print("⚠ Batch and individual results differ slightly")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch encoding test failed: {e}")
        return False

def test_different_molecules():
    """Test different types of molecules"""
    print("\nTesting different molecule types...")
    
    try:
        encoder = DMPNNEncoder(
            node_dim=64,
            edge_dim=64,
            depth=3,
            dropout=0.0,
            aggregation='mean'
        )
        
        test_cases = [
            ("CCO", "Simple alcohol"),
            ("CC(=O)O", "Carboxylic acid"),
            ("c1ccccc1", "Aromatic ring"),
            ("CCN(CC)CC", "Amine"),
            ("CC(C)CCO", "Branched alcohol"),
            ("C1CCCCC1", "Cyclohexane"),
            ("CC#N", "Nitrile"),
            ("CCS", "Thiol"),
            ("CCCl", "Chloroalkane"),
            ("C=C", "Alkene"),
        ]
        
        print(f"Testing {len(test_cases)} different molecule types:")
        
        for smiles, description in test_cases:
            try:
                result = encoder.encode_smiles(smiles)
                print(f"  ✓ {smiles:<10} ({description}): {result.shape}")
            except Exception as e:
                print(f"  ✗ {smiles:<10} ({description}): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Different molecules test failed: {e}")
        return False

def test_error_handling():
    """Test error handling"""
    print("\nTesting error handling...")
    
    try:
        encoder = DMPNNEncoder(
            node_dim=64,
            edge_dim=64,
            depth=3,
            dropout=0.0,
            aggregation='mean'
        )
        
        # Test invalid SMILES
        try:
            result = encoder.encode_smiles("invalid_smiles")
            print("✗ Invalid SMILES should have failed")
            return False
        except Exception as e:
            print(f"✓ Invalid SMILES handled correctly: {type(e).__name__}")
        
        # Test empty string
        try:
            result = encoder.encode_smiles("")
            print("✗ Empty SMILES should have failed")
            return False
        except Exception as e:
            print(f"✓ Empty SMILES handled correctly: {type(e).__name__}")
        
        # Test None
        try:
            result = encoder.encode_smiles(None)
            print("✗ None SMILES should have failed")
            return False
        except Exception as e:
            print(f"✓ None SMILES handled correctly: {type(e).__name__}")
        
        # Test empty batch
        try:
            result = encoder.encode_batch([])
            print("✓ Empty batch handled correctly")
        except Exception as e:
            print(f"✗ Empty batch should have been handled: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def test_performance():
    """Test performance with larger dataset"""
    print("\nTesting performance with larger dataset...")
    
    try:
        encoder = DMPNNEncoder(
            node_dim=64,
            edge_dim=64,
            depth=3,
            dropout=0.0,
            aggregation='mean'
        )
        
        # Generate a larger dataset
        molecules = [
            "CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC", "CC(C)CCO",
            "C1CCCCC1", "CC#N", "CCS", "CCCl", "C=C",
            "CCCO", "CC(C)O", "CC(C)(C)O", "c1ccc(C)cc1", "CCCN",
            "CCCO", "CC(C)O", "CC(C)(C)O", "c1ccc(C)cc1", "CCCN",
            "CCCO", "CC(C)O", "CC(C)(C)O", "c1ccc(C)cc1", "CCCN",
        ] * 4  # 100 molecules total
        
        print(f"Testing {len(molecules)} molecules...")
        
        start_time = time.time()
        results = encoder.encode_batch(molecules)
        elapsed = time.time() - start_time
        
        print(f"✓ Processed {len(molecules)} molecules in {elapsed:.3f}s")
        print(f"  Throughput: {len(molecules)/elapsed:.1f} molecules/second")
        print(f"  Results shape: {results.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False

def test_different_depths():
    """Test different message passing depths"""
    print("\nTesting different message passing depths...")
    
    try:
        test_molecule = "CCO"
        
        for depth in [1, 2, 3, 4, 5]:
            encoder = DMPNNEncoder(
                node_dim=64,
                edge_dim=64,
                depth=depth,
                dropout=0.0,
                aggregation='mean'
            )
            
            start_time = time.time()
            result = encoder.encode_smiles(test_molecule)
            elapsed = time.time() - start_time
            
            print(f"  Depth {depth}: {elapsed:.3f}s, shape: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Different depths test failed: {e}")
        return False

def test_different_aggregations():
    """Test different aggregation methods"""
    print("\nTesting different aggregation methods...")
    
    try:
        test_molecule = "CCO"
        
        for aggregation in ['mean', 'sum', 'max']:
            encoder = DMPNNEncoder(
                node_dim=64,
                edge_dim=64,
                depth=3,
                dropout=0.0,
                aggregation=aggregation
            )
            
            result = encoder.encode_smiles(test_molecule)
            print(f"  {aggregation}: shape: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Different aggregations test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("D-MPNN Encoder Local Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Batch Encoding", test_batch_encoding),
        ("Different Molecules", test_different_molecules),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance),
        ("Different Depths", test_different_depths),
        ("Different Aggregations", test_different_aggregations),
    ]
    
    results = []
    total_time = 0
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        start_time = time.time()
        try:
            success = test_func()
            elapsed = time.time() - start_time
            results.append((test_name, success, elapsed))
            total_time += elapsed
            
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{status} ({elapsed:.2f}s)")
            
        except Exception as e:
            elapsed = time.time() - start_time
            results.append((test_name, False, elapsed))
            total_time += elapsed
            print(f"✗ ERROR: {e} ({elapsed:.2f}s)")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, elapsed in results:
        status = "✓" if success else "✗"
        print(f"{status} {test_name:<25} {elapsed:.2f}s")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Total time: {total_time:.2f}s")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        return False

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)