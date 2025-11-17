#!/usr/bin/env python3
"""
Test script for MACCS encoder Docker service.
"""
import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8003"
API_URL = "http://localhost/api/maccs"

# Test molecules
test_smiles = [
    "CCO",  # ethanol
    "CC(=O)O",  # acetic acid
    "c1ccccc1",  # benzene
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # ibuprofen
]

def test_health():
    """Test health check endpoint."""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Health response: {response.json()}")
            return True
        else:
            print(f"Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

def test_info():
    """Test info endpoint."""
    print("\nTesting info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/info")
        print(f"Info check: {response.status_code}")
        if response.status_code == 200:
            print(f"Info response: {response.json()}")
            return True
        else:
            print(f"Info check failed: {response.text}")
            return False
    except Exception as e:
        print(f"Info check error: {e}")
        return False

def test_single_encoding():
    """Test single molecule encoding."""
    print("\nTesting single molecule encoding...")
    try:
        data = {
            "smiles": test_smiles[0]
        }
        response = requests.post(f"{BASE_URL}/encode", json=data)
        print(f"Single encoding: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Fingerprint shape: {result['shape']}")
            print(f"Metadata: {result['metadata']}")
            print(f"First 10 bits: {result['fingerprints'][0][:10]}")
            return True
        else:
            print(f"Single encoding failed: {response.text}")
            return False
    except Exception as e:
        print(f"Single encoding error: {e}")
        return False

def test_batch_encoding():
    """Test batch encoding."""
    print("\nTesting batch encoding...")
    try:
        data = {
            "smiles": test_smiles
        }
        response = requests.post(f"{BASE_URL}/encode/batch", json=data)
        print(f"Batch encoding: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Batch fingerprint shape: {result['shape']}")
            print(f"Batch metadata: {result['metadata']}")
            print(f"Number of molecules processed: {len(result['fingerprints'])}")
            return True
        else:
            print(f"Batch encoding failed: {response.text}")
            return False
    except Exception as e:
        print(f"Batch encoding error: {e}")
        return False

def test_gateway_routing():
    """Test nginx gateway routing."""
    print("\nTesting gateway routing...")
    try:
        # Test through gateway
        response = requests.get(f"{API_URL}/health")
        print(f"Gateway health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Gateway response: {response.json()}")
            
            # Test encoding through gateway
            data = {"smiles": "CCO"}
            response = requests.post(f"{API_URL}/encode", json=data)
            print(f"Gateway encoding: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Gateway result shape: {result['shape']}")
                return True
            else:
                print(f"Gateway encoding failed: {response.text}")
                return False
        else:
            print(f"Gateway health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"Gateway routing error: {e}")
        return False

def test_invalid_smiles():
    """Test error handling with invalid SMILES."""
    print("\nTesting invalid SMILES handling...")
    try:
        data = {
            "smiles": "invalid_smiles_string"
        }
        response = requests.post(f"{BASE_URL}/encode", json=data)
        print(f"Invalid SMILES response: {response.status_code}")
        if response.status_code == 400:
            print(f"Error response: {response.json()}")
            return True
        else:
            print(f"Unexpected response for invalid SMILES: {response.text}")
            return False
    except Exception as e:
        print(f"Invalid SMILES test error: {e}")
        return False

def main():
    """Run all tests."""
    print("=== MACCS Encoder Docker Test ===")
    print(f"Testing MACCS encoder at {BASE_URL}")
    print(f"Gateway API at {API_URL}")
    
    # Wait a bit for service to be ready
    print("Waiting for service to be ready...")
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health),
        ("Info Endpoint", test_info),
        ("Single Encoding", test_single_encoding),
        ("Batch Encoding", test_batch_encoding),
        ("Invalid SMILES", test_invalid_smiles),
        ("Gateway Routing", test_gateway_routing),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{test_name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"{test_name}: ERROR - {e}")
            results.append((test_name, False))
        time.sleep(1)  # Small delay between tests
    
    # Summary
    print(f"\n{'='*50}")
    print("=== Test Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_name}")
    
    if passed == total:
        print("\n🎉 All tests passed! MACCS encoder is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the service.")

if __name__ == "__main__":
    main()