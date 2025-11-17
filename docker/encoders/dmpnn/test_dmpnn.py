#!/usr/bin/env python3
"""
D-MPNN Encoder Test Script
Tests the D-MPNN molecular encoder API endpoints
"""

import requests
import json
import time
import sys
from typing import Dict, List, Any

# Configuration
BASE_URL = "http://localhost:8005"  # D-MPNN service port
API_BASE = f"{BASE_URL}"
NGINX_URL = "http://localhost/api/dmpnn"  # Nginx proxy URL

# Test molecules
TEST_MOLECULES = [
    "CCO",  # Ethanol
    "CC(=O)O",  # Acetic acid
    "c1ccccc1",  # Benzene
    "CCN(CC)CC",  # Triethylamine
    "CC(C)CCO",  # Isobutanol
]

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        print(f"Health check status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Service status: {data.get('status', 'unknown')}")
            print(f"Encoder loaded: {data.get('encoder_loaded', False)}")
            print(f"Version: {data.get('version', 'unknown')}")
            print(f"Uptime: {data.get('uptime', 0):.2f} seconds")
            return True
        else:
            print(f"Health check failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"Health check error: {e}")
        return False

def test_info_endpoint():
    """Test info endpoint"""
    print("\nTesting info endpoint...")
    
    try:
        response = requests.get(f"{API_BASE}/info", timeout=10)
        print(f"Info endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Encoder name: {data.get('name', 'unknown')}")
            print(f"Description: {data.get('description', 'unknown')}")
            print(f"Dimensions: {data.get('dimensions', 0)}")
            print(f"Max sequence length: {data.get('max_sequence_length', 0)}")
            print(f"Supported features: {data.get('supported_features', [])}")
            return True
        else:
            print(f"Info endpoint failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"Info endpoint error: {e}")
        return False

def test_single_encoding():
    """Test single molecule encoding"""
    print(f"\nTesting single molecule encoding...")
    
    test_smiles = TEST_MOLECULES[0]  # Ethanol
    
    try:
        payload = {
            "smiles": test_smiles,
            "depth": 3
        }
        
        response = requests.post(
            f"{API_BASE}/encode",
            json=payload,
            timeout=30
        )
        
        print(f"Single encoding status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Processing time: {data.get('processing_time', 0):.3f} seconds")
            print(f"Dimensions: {data.get('dimensions', 0)}")
            print(f"Molecule count: {data.get('molecule_count', 0)}")
            
            embeddings = data.get('embeddings', [])
            if embeddings:
                print(f"Number of embeddings: {len(embeddings)}")
                print(f"Embedding dimension: {len(embeddings[0]) if embeddings[0] else 0}")
                print(f"First few values: {embeddings[0][:5] if embeddings[0] else 'N/A'}")
            return True
        else:
            print(f"Single encoding failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"Single encoding error: {e}")
        return False

def test_batch_encoding():
    """Test batch molecule encoding"""
    print(f"\nTesting batch molecule encoding...")
    
    try:
        payload = {
            "molecules": TEST_MOLECULES,
            "depth": 3
        }
        
        response = requests.post(
            f"{API_BASE}/encode_batch",
            json=payload,
            timeout=60
        )
        
        print(f"Batch encoding status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Processing time: {data.get('processing_time', 0):.3f} seconds")
            print(f"Dimensions: {data.get('dimensions', 0)}")
            print(f"Molecule count: {data.get('molecule_count', 0)}")
            
            embeddings = data.get('embeddings', [])
            if embeddings:
                print(f"Total embeddings: {len(embeddings)}")
                if len(embeddings) > 0:
                    print(f"First embedding dimension: {len(embeddings[0])}")
                    print(f"First few values: {embeddings[0][:5]}")
            return True
        else:
            print(f"Batch encoding failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"Batch encoding error: {e}")
        return False

def test_stream_encoding():
    """Test stream encoding"""
    print(f"\nTesting stream encoding...")
    
    try:
        payload = {
            "molecules": TEST_MOLECULES,
            "depth": 3
        }
        
        response = requests.post(
            f"{API_BASE}/encode_stream",
            json=payload,
            timeout=60
        )
        
        print(f"Stream encoding status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            print(f"Total results: {len(results)}")
            print(f"Successful: {data.get('successful', 0)}")
            print(f"Failed: {data.get('failed', 0)}")
            
            # Show first few results
            for i, result in enumerate(results[:3]):
                status = "✓" if result.get('success') else "✗"
                print(f"  {status} {result.get('smiles', 'N/A')}")
                if result.get('success') and result.get('embeddings'):
                    print(f"    Embedding shape: {len(result['embeddings'])}")
            
            return True
        else:
            print(f"Stream encoding failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"Stream encoding error: {e}")
        return False

def test_nginx_proxy():
    """Test nginx proxy endpoint"""
    print(f"\nTesting nginx proxy endpoint...")

    # Skip if nginx is not running locally
    try:
        # quick probe to root to detect gateway
        root = requests.get("http://localhost/health", timeout=2)
        if root.status_code != 200:
            print("Nginx gateway not responding; skipping proxy test")
            return True
    except Exception:
        print("Nginx gateway not available; skipping proxy test")
        return True
    
    try:
        # Test health check through nginx
        response = requests.get(f"{NGINX_URL}/health", timeout=10)
        print(f"Nginx health check status: {response.status_code}")
        
        if response.status_code == 200:
            print("Nginx proxy is working correctly")
            return True
        else:
            print(f"Nginx proxy failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"Nginx proxy error: {e}")
        return False

def test_error_handling():
    """Test error handling"""
    print(f"\nTesting error handling...")
    
    ok = True
    
    # Test invalid SMILES
    try:
        payload = {"smiles": "invalid_smiles_string"}
        response = requests.post(f"{API_BASE}/encode", json=payload, timeout=10)
        print(f"Invalid SMILES status: {response.status_code}")
    
        if response.status_code == 400:
            print("✓ Invalid SMILES handled correctly")
        else:
            print("✗ Invalid SMILES not handled properly")
            ok = False
    
    except Exception as e:
        print(f"Invalid SMILES test error: {e}")
        ok = False
    
    # Test empty batch
    try:
        payload = {"molecules": []}
        response = requests.post(f"{API_BASE}/encode_batch", json=payload, timeout=10)
        print(f"Empty batch status: {response.status_code}")
    
        if response.status_code == 400:
            print("✓ Empty batch handled correctly")
        else:
            print("✗ Empty batch not handled properly")
            ok = False
    
    except Exception as e:
        print(f"Empty batch test error: {e}")
        ok = False
    
    return ok

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("D-MPNN Encoder API Test Suite")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Info Endpoint", test_info_endpoint),
        ("Single Encoding", test_single_encoding),
        ("Batch Encoding", test_batch_encoding),
        ("Stream Encoding", test_stream_encoding),
        ("Nginx Proxy", test_nginx_proxy),
        ("Error Handling", test_error_handling),
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
        print(f"{status} {test_name:<20} {elapsed:.2f}s")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Total time: {total_time:.2f}s")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        return False

if __name__ == "__main__":
    print("Make sure the D-MPNN service is running before testing!")
    print("You can start it with: docker-compose up -d dmpnn")
    print()
    
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)