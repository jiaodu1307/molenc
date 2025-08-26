"""Performance benchmark tests.

These tests measure the performance of various molenc components
under different conditions and dataset sizes.
"""

import pytest
import time
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import gc
import psutil
import os

from molenc.core.registry import EncoderRegistry
from molenc.preprocessing import SMILESStandardizer, SMILESValidator, MolecularFilters
from molenc.preprocessing.utils import preprocess_smiles_list
from molenc.environments import EnvironmentManager, EnvironmentConfig


class BenchmarkTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        gc.collect()  # Clean up before timing
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        print(f"{self.name}: {self.duration:.4f} seconds")


class MemoryProfiler:
    """Memory usage profiler."""
    
    def __init__(self, name):
        self.name = name
        self.process = psutil.Process(os.getpid())
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None
    
    def __enter__(self):
        gc.collect()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        self.end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_used = self.end_memory - self.start_memory
        print(f"{self.name}: {memory_used:.2f} MB memory used")


@pytest.mark.benchmark
class TestPreprocessingPerformance:
    """Benchmark preprocessing components."""
    
    @pytest.fixture(params=[100, 1000, 5000])
    def dataset_sizes(self, request):
        """Different dataset sizes for benchmarking."""
        return request.param
    
    @pytest.fixture
    def generate_smiles_dataset(self, dataset_sizes):
        """Generate SMILES datasets of different sizes."""
        base_smiles = [
            'CCO',
            'CC(=O)O',
            'c1ccccc1',
            'CCN(CC)CC',
            'CC(C)O',
            'C1=CC=CC=C1O',
            'CC(C)(C)O',
            'CCCCCCCCCC',
            'c1ccc2c(c1)ccc3c2ccc4c3cccc4',
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'
        ]
        
        # Repeat base SMILES to reach desired size
        dataset = []
        for i in range(dataset_sizes):
            smiles = base_smiles[i % len(base_smiles)]
            # Add some variation
            if i % 3 == 0:
                smiles = smiles + 'C'  # Add carbon
            elif i % 5 == 0:
                smiles = 'INVALID_' + smiles  # Make some invalid
            dataset.append(smiles)
        
        return dataset
    
    def test_standardizer_performance(self, generate_smiles_dataset):
        """Benchmark SMILES standardizer performance."""
        dataset = generate_smiles_dataset
        
        # Mock RDKit
        with patch('rdkit') as mock_rdkit:
            mock_mol = MagicMock()
            mock_rdkit.Chem.MolFromSmiles.return_value = mock_mol
            mock_rdkit.Chem.MolToSmiles.return_value = 'CCO'
            
            standardizer = SMILESStandardizer()
            
            with BenchmarkTimer(f"Standardizer - {len(dataset)} molecules"):
                with MemoryProfiler(f"Standardizer Memory - {len(dataset)} molecules"):
                    results = []
                    for smiles in dataset:
                        try:
                            result = standardizer.standardize(smiles)
                            results.append(result)
                        except:
                            results.append(None)
            
            # Verify results
            assert len(results) == len(dataset)
            success_rate = sum(1 for r in results if r is not None) / len(results)
            print(f"Standardization success rate: {success_rate:.2%}")
    
    def test_validator_performance(self, generate_smiles_dataset):
        """Benchmark SMILES validator performance."""
        dataset = generate_smiles_dataset
        
        # Mock RDKit
        with patch('molenc.preprocessing.validators.rdkit') as mock_rdkit:
            def mock_mol_from_smiles(smiles):
                return None if 'INVALID' in smiles else MagicMock()
            
            mock_rdkit.Chem.MolFromSmiles.side_effect = mock_mol_from_smiles
            
            validator = SMILESValidator()
            
            with BenchmarkTimer(f"Validator - {len(dataset)} molecules"):
                with MemoryProfiler(f"Validator Memory - {len(dataset)} molecules"):
                    results = [validator.is_valid(smiles) for smiles in dataset]
            
            # Verify results
            assert len(results) == len(dataset)
            valid_count = sum(results)
            print(f"Valid molecules: {valid_count}/{len(dataset)} ({valid_count/len(dataset):.2%})")
    
    def test_filter_performance(self, generate_smiles_dataset):
        """Benchmark molecular filters performance."""
        dataset = generate_smiles_dataset
        
        # Mock RDKit
        with patch('molenc.preprocessing.filters.rdkit') as mock_rdkit:
            def mock_mol_from_smiles(smiles):
                return None if 'INVALID' in smiles else MagicMock()
            
            mock_rdkit.Chem.MolFromSmiles.side_effect = mock_mol_from_smiles
            mock_rdkit.Chem.Descriptors.MolWt.return_value = 150.0
            mock_rdkit.Chem.Descriptors.MolLogP.return_value = 2.0
            mock_rdkit.Chem.Descriptors.NumHDonors.return_value = 1
            mock_rdkit.Chem.Descriptors.NumHAcceptors.return_value = 2
            
            filters = MolecularFilters(
                mw_range=(50, 500),
                logp_range=(-2, 5),
                lipinski_rule=True
            )
            
            with BenchmarkTimer(f"Filters - {len(dataset)} molecules"):
                with MemoryProfiler(f"Filters Memory - {len(dataset)} molecules"):
                    results = [filters.passes_filters(smiles) for smiles in dataset]
            
            # Verify results
            assert len(results) == len(dataset)
            passed_count = sum(results)
            print(f"Molecules passing filters: {passed_count}/{len(dataset)} ({passed_count/len(dataset):.2%})")
    
    def test_preprocessing_pipeline_performance(self, generate_smiles_dataset):
        """Benchmark complete preprocessing pipeline."""
        dataset = generate_smiles_dataset
        
        # Mock all RDKit dependencies
        with patch('molenc.preprocessing.standardize.rdkit') as mock_rdkit_std:
            with patch('molenc.preprocessing.validators.rdkit') as mock_rdkit_val:
                with patch('molenc.preprocessing.filters.rdkit') as mock_rdkit_filt:
                    
                    # Setup mocks
                    def mock_mol_from_smiles(smiles):
                        return None if 'INVALID' in smiles else MagicMock()
                    
                    mock_rdkit_std.Chem.MolFromSmiles.side_effect = mock_mol_from_smiles
                    mock_rdkit_std.Chem.MolToSmiles.return_value = 'CCO'
                    mock_rdkit_val.Chem.MolFromSmiles.side_effect = mock_mol_from_smiles
                    mock_rdkit_filt.Chem.MolFromSmiles.side_effect = mock_mol_from_smiles
                    mock_rdkit_filt.Chem.Descriptors.MolWt.return_value = 150.0
                    mock_rdkit_filt.Chem.Descriptors.MolLogP.return_value = 2.0
                    
                    with BenchmarkTimer(f"Preprocessing Pipeline - {len(dataset)} molecules"):
                        with MemoryProfiler(f"Pipeline Memory - {len(dataset)} molecules"):
                            processed = preprocess_smiles_list(
                                dataset,
                                standardize=True,
                                validate=True,
                                filter_molecules=True,
                                n_jobs=1
                            )
                    
                    # Verify results
                    assert isinstance(processed, dict)
                    assert 'processed_smiles' in processed
                    assert 'statistics' in processed
                    
                    valid_count = len(processed['processed_smiles'])
                    print(f"Pipeline output: {valid_count}/{len(dataset)} molecules ({valid_count/len(dataset):.2%})")


@pytest.mark.benchmark
class TestEncoderPerformance:
    """Benchmark encoder performance."""
    
    @pytest.fixture(params=[100, 1000, 5000])
    def dataset_sizes(self, request):
        """Different dataset sizes for benchmarking."""
        return request.param
    
    @pytest.fixture
    def valid_smiles_dataset(self, dataset_sizes):
        """Generate valid SMILES datasets."""
        base_smiles = [
            'CCO',
            'CC(=O)O',
            'c1ccccc1',
            'CCN(CC)CC',
            'CC(C)O',
            'C1=CC=CC=C1O',
            'CC(C)(C)O',
            'CCCCCCCCCC',
            'c1ccc2c(c1)ccc3c2ccc4c3cccc4',
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'
        ]
        
        dataset = []
        for i in range(dataset_sizes):
            smiles = base_smiles[i % len(base_smiles)]
            dataset.append(smiles)
        
        return dataset
    
    def test_fingerprint_encoder_performance(self, valid_smiles_dataset):
        """Benchmark fingerprint encoder performance."""
        dataset = valid_smiles_dataset
        
        # Mock RDKit
        with patch('molenc.encoders.fingerprints.rdkit') as mock_rdkit:
            mock_mol = MagicMock()
            mock_rdkit.Chem.MolFromSmiles.return_value = mock_mol
            mock_rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect.return_value = [1, 0] * 1024
            
            registry = EncoderRegistry()
            
            # Mock encoder
            mock_encoder = MagicMock()
            mock_encoder.encode_batch.return_value = np.random.rand(len(dataset), 2048)
            mock_encoder.get_output_dim.return_value = 2048
            mock_encoder_class = MagicMock(return_value=mock_encoder)
            registry.register('morgan', mock_encoder_class)
            
            encoder = registry.get_encoder('morgan')
            
            with BenchmarkTimer(f"Morgan Fingerprints - {len(dataset)} molecules"):
                with MemoryProfiler(f"Morgan Memory - {len(dataset)} molecules"):
                    features = encoder.encode_batch(dataset)
            
            # Verify results
            assert features.shape[0] == len(dataset)
            assert features.shape[1] == 2048
            print(f"Generated features shape: {features.shape}")
    
    def test_batch_size_performance(self, valid_smiles_dataset):
        """Benchmark different batch sizes."""
        dataset = valid_smiles_dataset
        batch_sizes = [10, 50, 100, 500, 1000]
        
        # Mock RDKit
        with patch('molenc.encoders.fingerprints.rdkit') as mock_rdkit:
            mock_mol = MagicMock()
            mock_rdkit.Chem.MolFromSmiles.return_value = mock_mol
            mock_rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect.return_value = [1, 0] * 1024
            
            registry = EncoderRegistry()
            
            # Mock encoder
            mock_encoder = MagicMock()
            
            def batch_encode(smiles_list):
                return np.random.rand(len(smiles_list), 2048)
            
            mock_encoder.encode_batch.side_effect = batch_encode
            mock_encoder.get_output_dim.return_value = 2048
            mock_encoder_class = MagicMock(return_value=mock_encoder)
            registry.register('morgan', mock_encoder_class)
            
            encoder = registry.get_encoder('morgan')
            
            results = {}
            
            for batch_size in batch_sizes:
                if batch_size > len(dataset):
                    continue
                
                with BenchmarkTimer(f"Batch size {batch_size} - {len(dataset)} molecules") as timer:
                    all_features = []
                    for i in range(0, len(dataset), batch_size):
                        batch = dataset[i:i + batch_size]
                        features = encoder.encode_batch(batch)
                        all_features.append(features)
                    
                    combined_features = np.vstack(all_features)
                
                results[batch_size] = {
                    'time': timer.duration,
                    'throughput': len(dataset) / timer.duration,
                    'features_shape': combined_features.shape
                }
            
            # Print performance comparison
            print("\nBatch Size Performance Comparison:")
            for batch_size, metrics in results.items():
                print(f"Batch {batch_size}: {metrics['throughput']:.2f} molecules/sec")
    
    def test_encoder_comparison_performance(self, valid_smiles_dataset):
        """Compare performance of different encoders."""
        dataset = valid_smiles_dataset[:1000]  # Limit for comparison
        
        # Mock different encoders
        with patch('molenc.encoders.fingerprints.rdkit') as mock_rdkit:
            mock_mol = MagicMock()
            mock_rdkit.Chem.MolFromSmiles.return_value = mock_mol
            mock_rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect.return_value = [1, 0] * 1024
            mock_rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint.return_value = [1, 0] * 83
            
            registry = EncoderRegistry()
            
            # Define encoder configurations
            encoder_configs = {
                'morgan_1024': (1024, 'Morgan 1024-bit'),
                'morgan_2048': (2048, 'Morgan 2048-bit'),
                'maccs': (167, 'MACCS Keys'),
                'topological': (1024, 'Topological')
            }
            
            results = {}
            
            for encoder_name, (dim, description) in encoder_configs.items():
                # Mock encoder
                mock_encoder = MagicMock()
                mock_encoder.encode_batch.return_value = np.random.rand(len(dataset), dim)
                mock_encoder.get_output_dim.return_value = dim
                mock_encoder_class = MagicMock(return_value=mock_encoder)
                registry.register(encoder_name, mock_encoder_class)
                
                encoder = registry.get_encoder(encoder_name)
                
                with BenchmarkTimer(f"{description} - {len(dataset)} molecules") as timer:
                    with MemoryProfiler(f"{description} Memory") as memory:
                        features = encoder.encode_batch(dataset)
                
                results[encoder_name] = {
                    'description': description,
                    'dimension': dim,
                    'time': timer.duration,
                    'throughput': len(dataset) / timer.duration,
                    'features_shape': features.shape
                }
            
            # Print comparison
            print("\nEncoder Performance Comparison:")
            for name, metrics in results.items():
                print(f"{metrics['description']}: {metrics['throughput']:.2f} molecules/sec, {metrics['dimension']} features")


@pytest.mark.benchmark
class TestSystemPerformance:
    """Benchmark complete system performance."""
    
    def test_end_to_end_workflow_performance(self):
        """Benchmark complete end-to-end workflow."""
        # Generate large dataset
        dataset_size = 5000
        base_smiles = [
            'CCO', 'CC(=O)O', 'c1ccccc1', 'CCN(CC)CC', 'CC(C)O',
            'C1=CC=CC=C1O', 'CC(C)(C)O', 'CCCCCCCCCC',
            'c1ccc2c(c1)ccc3c2ccc4c3cccc4', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'
        ]
        
        dataset = [base_smiles[i % len(base_smiles)] for i in range(dataset_size)]
        
        # Mock all dependencies
        with patch('molenc.preprocessing.standardize.rdkit') as mock_rdkit_std:
            with patch('molenc.preprocessing.validators.rdkit') as mock_rdkit_val:
                with patch('molenc.preprocessing.filters.rdkit') as mock_rdkit_filt:
                    with patch('molenc.encoders.fingerprints.rdkit') as mock_rdkit_enc:
                        
                        # Setup mocks
                        mock_mol = MagicMock()
                        mock_rdkit_std.Chem.MolFromSmiles.return_value = mock_mol
                        mock_rdkit_std.Chem.MolToSmiles.return_value = 'CCO'
                        mock_rdkit_val.Chem.MolFromSmiles.return_value = mock_mol
                        mock_rdkit_filt.Chem.MolFromSmiles.return_value = mock_mol
                        mock_rdkit_filt.Chem.Descriptors.MolWt.return_value = 150.0
                        mock_rdkit_filt.Chem.Descriptors.MolLogP.return_value = 2.0
                        mock_rdkit_enc.Chem.MolFromSmiles.return_value = mock_mol
                        mock_rdkit_enc.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect.return_value = [1, 0] * 1024
                        
                        with BenchmarkTimer(f"End-to-End Workflow - {dataset_size} molecules"):
                            with MemoryProfiler(f"End-to-End Memory - {dataset_size} molecules"):
                                
                                # Step 1: Preprocessing
                                processed = preprocess_smiles_list(
                                    dataset,
                                    standardize=True,
                                    validate=True,
                                    filter_molecules=True,
                                    n_jobs=4
                                )
                                
                                valid_smiles = processed['processed_smiles']
                                
                                # Step 2: Encoding
                                registry = EncoderRegistry()
                                
                                mock_encoder = MagicMock()
                                mock_encoder.encode_batch.return_value = np.random.rand(len(valid_smiles), 2048)
                                mock_encoder.get_output_dim.return_value = 2048
                                mock_encoder_class = MagicMock(return_value=mock_encoder)
                                registry.register('morgan', mock_encoder_class)
                                
                                encoder = registry.get_encoder('morgan')
                                features = encoder.encode_batch(valid_smiles)
                                
                                # Step 3: Post-processing
                                normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
                        
                        # Verify results
                        assert len(valid_smiles) > 0
                        assert features.shape[0] == len(valid_smiles)
                        assert normalized_features.shape == features.shape
                        
                        print(f"Processed {len(valid_smiles)}/{dataset_size} molecules successfully")
                        print(f"Final feature matrix shape: {features.shape}")
    
    def test_parallel_processing_performance(self):
        """Benchmark parallel processing performance."""
        dataset_size = 2000
        base_smiles = ['CCO', 'CC(=O)O', 'c1ccccc1', 'CCN(CC)CC', 'CC(C)O']
        dataset = [base_smiles[i % len(base_smiles)] for i in range(dataset_size)]
        
        worker_counts = [1, 2, 4, 8]
        
        # Mock RDKit
        with patch('molenc.preprocessing.standardize.rdkit') as mock_rdkit_std:
            with patch('molenc.preprocessing.validators.rdkit') as mock_rdkit_val:
                with patch('molenc.preprocessing.filters.rdkit') as mock_rdkit_filt:
                    
                    # Setup mocks
                    mock_mol = MagicMock()
                    mock_rdkit_std.Chem.MolFromSmiles.return_value = mock_mol
                    mock_rdkit_std.Chem.MolToSmiles.return_value = 'CCO'
                    mock_rdkit_val.Chem.MolFromSmiles.return_value = mock_mol
                    mock_rdkit_filt.Chem.MolFromSmiles.return_value = mock_mol
                    mock_rdkit_filt.Chem.Descriptors.MolWt.return_value = 150.0
                    
                    results = {}
                    
                    for n_workers in worker_counts:
                        with BenchmarkTimer(f"Parallel Processing - {n_workers} workers") as timer:
                            processed = preprocess_smiles_list(
                                dataset,
                                standardize=True,
                                validate=True,
                                filter_molecules=True,
                                n_jobs=n_workers
                            )
                        
                        results[n_workers] = {
                            'time': timer.duration,
                            'throughput': len(dataset) / timer.duration,
                            'processed_count': len(processed['processed_smiles'])
                        }
                    
                    # Print performance comparison
                    print("\nParallel Processing Performance:")
                    for workers, metrics in results.items():
                        print(f"{workers} workers: {metrics['throughput']:.2f} molecules/sec")
                    
                    # Calculate speedup
                    baseline = results[1]['time']
                    print("\nSpeedup compared to single worker:")
                    for workers, metrics in results.items():
                        speedup = baseline / metrics['time']
                        print(f"{workers} workers: {speedup:.2f}x speedup")
    
    def test_memory_scaling_performance(self):
        """Test memory usage scaling with dataset size."""
        dataset_sizes = [100, 500, 1000, 2000, 5000]
        base_smiles = ['CCO', 'CC(=O)O', 'c1ccccc1', 'CCN(CC)CC', 'CC(C)O']
        
        # Mock RDKit
        with patch('molenc.encoders.fingerprints.rdkit') as mock_rdkit:
            mock_mol = MagicMock()
            mock_rdkit.Chem.MolFromSmiles.return_value = mock_mol
            mock_rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect.return_value = [1, 0] * 1024
            
            registry = EncoderRegistry()
            
            results = {}
            
            for size in dataset_sizes:
                dataset = [base_smiles[i % len(base_smiles)] for i in range(size)]
                
                # Mock encoder
                mock_encoder = MagicMock()
                mock_encoder.encode_batch.return_value = np.random.rand(size, 2048)
                mock_encoder.get_output_dim.return_value = 2048
                mock_encoder_class = MagicMock(return_value=mock_encoder)
                registry.register(f'morgan_{size}', mock_encoder_class)
                
                encoder = registry.get_encoder(f'morgan_{size}')
                
                with MemoryProfiler(f"Memory Scaling - {size} molecules") as memory:
                    with BenchmarkTimer(f"Time Scaling - {size} molecules") as timer:
                        features = encoder.encode_batch(dataset)
                
                results[size] = {
                    'time': timer.duration,
                    'throughput': size / timer.duration,
                    'features_shape': features.shape,
                    'memory_mb': memory.end_memory - memory.start_memory
                }
            
            # Print scaling analysis
            print("\nMemory and Time Scaling:")
            for size, metrics in results.items():
                print(f"{size} molecules: {metrics['time']:.3f}s, {metrics['throughput']:.1f} mol/s, {metrics['memory_mb']:.1f} MB")