#!/usr/bin/env python3
"""
Example usage of MACCS encoder Docker service.
"""
import requests
import json
import time

# API endpoints
GATEWAY_URL = "http://localhost/api/maccs"
DIRECT_URL = "http://localhost:8003"

def example_basic_usage():
    """Basic usage example."""
    print("=== Basic MACCS Encoding Example ===")
    
    # Single molecule
    smiles = "CC(=O)O"  # Acetic acid
    
    response = requests.post(f"{GATEWAY_URL}/encode", json={"smiles": smiles})
    
    if response.status_code == 200:
        result = response.json()
        print(f"SMILES: {smiles}")
        print(f"Fingerprint shape: {result['shape']}")
        print(f"Number of bits: {result['metadata']['n_bits']}")
        print(f"First 20 bits: {result['fingerprints'][0][:20]}")
        print(f"Number of set bits: {sum(result['fingerprints'][0])}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def example_drug_molecules():
    """Example with common drug molecules."""
    print("\n=== Drug Molecules MACCS Encoding ===")
    
    drugs = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    }
    
    smiles_list = list(drugs.values())
    
    response = requests.post(f"{GATEWAY_URL}/encode/batch", json={"smiles": smiles_list})
    
    if response.status_code == 200:
        result = response.json()
        fingerprints = result['fingerprints']
        
        print(f"Encoded {len(drugs)} drug molecules")
        print(f"Fingerprint shape: {result['shape']}")
        
        for i, (name, smiles) in enumerate(drugs.items()):
            fp = fingerprints[i]
            n_set_bits = sum(fp)
            print(f"{name:12} - {n_set_bits:3d} set bits, SMILES: {smiles}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def example_similarity_analysis():
    """Example of similarity analysis using MACCS fingerprints."""
    print("\n=== MACCS-based Similarity Analysis ===")
    
    # Similar molecules
    molecules = {
        "Ethanol": "CCO",
        "Propanol": "CCCO", 
        "Butanol": "CCCCO",
        "Methanol": "CO",
        "Acetic Acid": "CC(=O)O",
        "Propionic Acid": "CCC(=O)O",
    }
    
    smiles_list = list(molecules.values())
    
    response = requests.post(f"{GATEWAY_URL}/encode", json={"smiles": smiles_list})
    
    if response.status_code == 200:
        result = response.json()
        fingerprints = np.array(result['fingerprints'])
        
        print(f"Encoded {len(molecules)} molecules")
        
        # Calculate pairwise similarities
        from itertools import combinations
        
        print("\nPairwise similarities (Tanimoto coefficient):")
        names = list(molecules.keys())
        
        for i, j in combinations(range(len(names)), 2):
            name1, name2 = names[i], names[j]
            fp1, fp2 = fingerprints[i], fingerprints[j]
            
            # Tanimoto similarity
            intersection = np.sum(fp1 * fp2)
            union = np.sum(fp1) + np.sum(fp2) - intersection
            similarity = intersection / union if union > 0 else 0.0
            
            print(f"{name1:12} vs {name2:12}: {similarity:.3f}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def example_performance_test():
    """Performance test with larger dataset."""
    print("\n=== MACCS Performance Test ===")
    
    # Generate a larger set of molecules
    test_smiles = [
        "CCO", "CCCO", "CCCCO", "CC(=O)O", "c1ccccc1", "CC(C)C",
        "CC(C)(C)C", "CCCN", "CCNC", "CCCOCC", "CCOC(C)C",
        "c1ccc(C)cc1", "c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(Cl)cc1",
    ] * 10  # 140 molecules
    
    print(f"Testing with {len(test_smiles)} molecules...")
    
    start_time = time.time()
    response = requests.post(f"{GATEWAY_URL}/encode/batch", json={"smiles": test_smiles})
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        processing_time = end_time - start_time
        
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Molecules per second: {len(test_smiles) / processing_time:.1f}")
        print(f"Fingerprint shape: {result['shape']}")
        print(f"Batch processing: {result['metadata']['n_batches']} batches")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def main():
    """Run all examples."""
    print("=== MACCS Encoder Usage Examples ===")
    print(f"Using gateway: {GATEWAY_URL}")
    
    # Check if service is available
    try:
        response = requests.get(f"{GATEWAY_URL}/health")
        if response.status_code != 200:
            print(f"Service not available. Status: {response.status_code}")
            return
    except Exception as e:
        print(f"Cannot connect to service: {e}")
        return
    
    # Run examples
    example_basic_usage()
    example_drug_molecules()
    
    # Only run similarity analysis if numpy is available
    try:
        import numpy as np
        example_similarity_analysis()
    except ImportError:
        print("\n=== Skipping similarity analysis (numpy not available) ===")
    
    example_performance_test()
    
    print("\n=== Examples completed ===")

if __name__ == "__main__":
    main()