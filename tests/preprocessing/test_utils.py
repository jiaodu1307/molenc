"""Tests for preprocessing utility functions."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from molenc.core.exceptions import InvalidSMILESError, DependencyError


class TestPreprocessingUtils:
    """Tests for preprocessing utility functions."""
    
    @pytest.mark.rdkit
    def test_preprocess_smiles_list_basic(self, sample_smiles, invalid_smiles):
        """Test basic SMILES list preprocessing."""
        from molenc.preprocessing.utils import preprocess_smiles_list
        
        # Mix valid and invalid SMILES
        mixed_smiles = sample_smiles + invalid_smiles
        
        result = preprocess_smiles_list(mixed_smiles)
        
        assert isinstance(result, dict)
        
        # Check required keys
        required_keys = ['processed_smiles', 'failed_indices', 'stats']
        for key in required_keys:
            assert key in result
        
        # Check processed SMILES are valid
        processed = result['processed_smiles']
        assert isinstance(processed, list)
        
        for smiles in processed:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
        
        # Check failed indices
        failed_indices = result['failed_indices']
        assert isinstance(failed_indices, list)
        
        for idx in failed_indices:
            assert 0 <= idx < len(mixed_smiles)
        
        # Check stats
        stats = result['stats']
        assert isinstance(stats, dict)
        assert stats['total'] == len(mixed_smiles)
        assert stats['processed'] == len(processed)
        assert stats['failed'] == len(failed_indices)
        assert stats['processed'] + stats['failed'] == stats['total']
    
    @pytest.mark.rdkit
    def test_preprocess_smiles_list_with_standardization(self, sample_smiles):
        """Test SMILES list preprocessing with standardization."""
        from molenc.preprocessing.utils import preprocess_smiles_list
        
        result = preprocess_smiles_list(
            sample_smiles,
            standardize=True,
            standardize_options={
                'remove_stereochemistry': False,
                'neutralize': True,
                'remove_salts': True,
                'canonicalize': True
            }
        )
        
        processed = result['processed_smiles']
        
        # All processed SMILES should be canonical
        for smiles in processed:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            canonical = Chem.MolToSmiles(mol)
            assert smiles == canonical
    
    @pytest.mark.rdkit
    def test_preprocess_smiles_list_with_validation(self, sample_smiles, invalid_smiles):
        """Test SMILES list preprocessing with validation."""
        from molenc.preprocessing.utils import preprocess_smiles_list
        
        mixed_smiles = sample_smiles + invalid_smiles
        
        result = preprocess_smiles_list(
            mixed_smiles,
            validate=True,
            validation_options={
                'strict_mode': True,
                'check_aromaticity': True,
                'check_valence': True,
                'max_atoms': 100
            }
        )
        
        processed = result['processed_smiles']
        failed_indices = result['failed_indices']
        
        # Should have filtered out invalid SMILES
        assert len(failed_indices) >= len(invalid_smiles)
        
        # All processed SMILES should be valid
        for smiles in processed:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
    
    @pytest.mark.rdkit
    def test_preprocess_smiles_list_with_filtering(self, sample_smiles):
        """Test SMILES list preprocessing with molecular filtering."""
        from molenc.preprocessing.utils import preprocess_smiles_list
        
        result = preprocess_smiles_list(
            sample_smiles,
            filter_molecules=True,
            filter_options={
                'mw_range': (30, 400),
                'logp_range': (-3, 8),
                'hbd_range': (0, 10),
                'hba_range': (0, 15),
                'lipinski_rule': False
            }
        )
        
        processed = result['processed_smiles']
        
        # Check that processed molecules meet filter criteria
        from molenc.preprocessing.filters import MolecularFilters
        filters = MolecularFilters(
            mw_range=(30, 400),
            logp_range=(-3, 8),
            hbd_range=(0, 10),
            hba_range=(0, 15)
        )
        
        for smiles in processed:
            assert filters.passes_filters(smiles) is True
    
    @pytest.mark.rdkit
    def test_preprocess_smiles_list_parallel_processing(self, sample_smiles):
        """Test SMILES list preprocessing with parallel processing."""
        from molenc.preprocessing.utils import preprocess_smiles_list
        
        # Create larger dataset for parallel processing
        large_dataset = sample_smiles * 10
        
        result = preprocess_smiles_list(
            large_dataset,
            parallel=True,
            n_jobs=2,
            standardize=True,
            validate=True
        )
        
        processed = result['processed_smiles']
        stats = result['stats']
        
        # Should process most molecules successfully
        success_rate = len(processed) / len(large_dataset)
        assert success_rate > 0.8
        
        # Check that parallel processing worked
        assert stats['total'] == len(large_dataset)
        assert stats['processed'] == len(processed)
    
    @pytest.mark.rdkit
    def test_preprocess_smiles_list_comprehensive_pipeline(self, sample_smiles, invalid_smiles):
        """Test comprehensive preprocessing pipeline."""
        from molenc.preprocessing.utils import preprocess_smiles_list
        
        mixed_smiles = sample_smiles + invalid_smiles
        
        result = preprocess_smiles_list(
            mixed_smiles,
            standardize=True,
            standardize_options={
                'remove_stereochemistry': False,
                'neutralize': True,
                'remove_salts': True,
                'canonicalize': True
            },
            validate=True,
            validation_options={
                'strict_mode': False,
                'check_aromaticity': True,
                'max_atoms': 50
            },
            filter_molecules=True,
            filter_options={
                'mw_range': (50, 400),
                'logp_range': (-3, 5),
                'lipinski_rule': False
            },
            parallel=True,
            n_jobs=2
        )
        
        processed = result['processed_smiles']
        failed_indices = result['failed_indices']
        stats = result['stats']
        
        # Should have comprehensive stats
        assert 'standardization' in stats
        assert 'validation' in stats
        assert 'filtering' in stats
        
        # All processed SMILES should pass all criteria
        for smiles in processed:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
            
            # Should be canonical (preserve stereochemistry if present)
            canonical = Chem.MolToSmiles(mol, canonical=True)
            assert smiles == canonical
    
    @pytest.mark.rdkit
    def test_preprocess_smiles_list_empty_input(self):
        """Test preprocessing with empty input."""
        from molenc.preprocessing.utils import preprocess_smiles_list
        
        result = preprocess_smiles_list([])
        
        assert result['processed_smiles'] == []
        assert result['failed_indices'] == []
        assert result['stats']['total'] == 0
        assert result['stats']['processed'] == 0
        assert result['stats']['failed'] == 0
    
    @pytest.mark.rdkit
    def test_preprocess_smiles_list_error_handling(self):
        """Test error handling in preprocessing pipeline."""
        from molenc.preprocessing.utils import preprocess_smiles_list
        
        # Test with invalid options
        with pytest.raises((ValueError, TypeError)):
            preprocess_smiles_list(
                ["CCO"],
                standardize=True,
                standardize_options={'invalid_option': True}
            )
    
    @pytest.mark.rdkit
    def test_process_single_smiles_valid(self, sample_smiles):
        """Test processing single valid SMILES."""
        from molenc.preprocessing.utils import _process_single_smiles
        
        for smiles in sample_smiles:
            result = _process_single_smiles(
                smiles,
                standardize=True,
                validate=True,
                filter_molecules=True,
                standardize_options={},
                validation_options={},
                filter_options={}
            )
            
            assert isinstance(result, dict)
            assert 'success' in result
            assert 'processed_smiles' in result
            assert 'error_stage' in result
            assert 'error_message' in result
            
            if result['success']:
                assert result['processed_smiles'] is not None
                assert result['error_stage'] is None
                assert result['error_message'] is None
    
    @pytest.mark.rdkit
    def test_process_single_smiles_invalid(self, invalid_smiles):
        """Test processing single invalid SMILES."""
        from molenc.preprocessing.utils import _process_single_smiles
        
        for smiles in invalid_smiles:
            result = _process_single_smiles(
                smiles,
                standardize=True,
                validate=True,
                filter_molecules=True,
                standardize_options={},
                validation_options={},
                filter_options={}
            )
            
            assert isinstance(result, dict)
            assert result['success'] is False
            assert result['processed_smiles'] is None
            assert result['error_stage'] is not None
            assert result['error_message'] is not None
    
    @pytest.mark.rdkit
    def test_create_preprocessing_pipeline(self):
        """Test creating custom preprocessing pipeline."""
        from molenc.preprocessing.utils import create_preprocessing_pipeline
        
        pipeline = create_preprocessing_pipeline(
            standardize=True,
            validate=True,
            filter_molecules=True,
            standardize_options={'canonicalize': True},
            validation_options={'strict_mode': False},
            filter_options={'mw_range': (50, 500)}
        )
        
        assert callable(pipeline)
        
        # Test the pipeline
        test_smiles = "CCO"
        result = pipeline(test_smiles)
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'processed_smiles' in result
    
    @pytest.mark.rdkit
    def test_get_preprocessing_stats(self, sample_smiles, invalid_smiles):
        """Test getting preprocessing statistics."""
        from molenc.preprocessing.utils import get_preprocessing_stats
        
        mixed_smiles = sample_smiles + invalid_smiles
        
        # Test basic stats without any configuration
        stats = get_preprocessing_stats(mixed_smiles)
        
        assert isinstance(stats, dict)
        assert stats['total'] == len(mixed_smiles)
        assert stats['processed'] == len(mixed_smiles)  # Without validation/filtering, all are processed
        assert stats['failed'] == 0  # Without validation/filtering, none fail
        assert stats['rejection_rate'] == 0.0
        
        # Test with validation configuration
        validator_config = {'check_valence': True, 'check_aromaticity': True}
        stats_with_validation = get_preprocessing_stats(mixed_smiles, validator_config=validator_config)
        
        assert isinstance(stats_with_validation, dict)
        assert stats_with_validation['total'] == len(mixed_smiles)
        assert 'validation' in stats_with_validation
        assert 'valid_after_validation' in stats_with_validation
    
    @pytest.mark.rdkit
    def test_batch_standardize(self, sample_smiles):
        """Test batch standardization function."""
        from molenc.preprocessing.utils import batch_standardize
        
        standardized = batch_standardize(
            sample_smiles,
            remove_stereochemistry=False,
            neutralize=True,
            remove_salts=True,
            canonicalize=True
        )
        
        assert isinstance(standardized, list)
        assert len(standardized) <= len(sample_smiles)  # Some might fail
        
        # All standardized SMILES should be canonical
        for smiles in standardized:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
            canonical = Chem.MolToSmiles(mol)
            assert smiles == canonical
    
    @pytest.mark.rdkit
    def test_batch_standardize_with_invalid_smiles(self, invalid_smiles):
        """Test batch standardization with invalid SMILES."""
        from molenc.preprocessing.utils import batch_standardize
        
        standardized = batch_standardize(invalid_smiles)
        
        # Should return empty list or handle gracefully
        assert isinstance(standardized, list)
        assert len(standardized) == 0  # All should fail


class TestPreprocessingIntegration:
    """Integration tests for preprocessing utilities."""
    
    @pytest.mark.rdkit
    def test_end_to_end_preprocessing(self):
        """Test end-to-end preprocessing workflow."""
        from molenc.preprocessing.utils import preprocess_smiles_list
        
        # Real-world dataset simulation
        raw_smiles = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "INVALID_SMILES",  # Invalid
            "",  # Empty
            "C[C@H](N)C(=O)O",  # L-Alanine
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "C(C",  # Invalid parenthesis
            "CC1=CC=C(C=C1)C(=O)O",  # p-Toluic acid
        ]
        
        result = preprocess_smiles_list(
            raw_smiles,
            standardize=True,
            standardize_options={
                'remove_stereochemistry': False,
                'neutralize': True,
                'remove_salts': True,
                'canonicalize': True
            },
            validate=True,
            validation_options={
                'strict_mode': False,
                'check_aromaticity': True,
                'max_atoms': 100
            },
            filter_molecules=True,
            filter_options={
                'mw_range': (30, 600),
                'logp_range': (-5, 8),
                'lipinski_rule': False
            },
            parallel=False
        )
        
        processed = result['processed_smiles']
        failed_indices = result['failed_indices']
        stats = result['stats']
        
        # Should process most valid molecules
        assert len(processed) >= 6  # At least 6 valid molecules
        assert len(failed_indices) >= 3  # At least 3 invalid
        
        # Check comprehensive stats
        assert stats['total'] == len(raw_smiles)
        assert 'standardization' in stats
        assert 'validation' in stats
        assert 'filtering' in stats
        
        # All processed SMILES should be high quality
        for smiles in processed:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
            
            # Should be canonical
            canonical = Chem.MolToSmiles(mol)
            assert smiles == canonical
    
    @pytest.mark.rdkit
    def test_preprocessing_with_large_dataset(self):
        """Test preprocessing with larger dataset."""
        from molenc.preprocessing.utils import preprocess_smiles_list
        import time
        
        # Create larger dataset
        base_smiles = [
            "CCO", "c1ccccc1", "CC(=O)O", "CC(C)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        ]
        large_dataset = base_smiles * 100  # 600 SMILES
        
        start_time = time.time()
        result = preprocess_smiles_list(
            large_dataset,
            standardize=True,
            validate=True,
            filter_molecules=True,
            parallel=True,
            n_jobs=2
        )
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 30.0  # 30 seconds threshold
        
        # Should process most molecules successfully
        processed = result['processed_smiles']
        success_rate = len(processed) / len(large_dataset)
        assert success_rate > 0.9  # At least 90% success rate
    
    @pytest.mark.rdkit
    def test_preprocessing_consistency(self, sample_smiles):
        """Test preprocessing consistency across multiple runs."""
        from molenc.preprocessing.utils import preprocess_smiles_list
        
        # Run preprocessing multiple times
        results = []
        for _ in range(3):
            result = preprocess_smiles_list(
                sample_smiles,
                standardize=True,
                validate=True,
                filter_molecules=False,  # Disable filtering for consistency
                parallel=False  # Disable parallel for deterministic results
            )
            results.append(result)
        
        # Results should be consistent
        first_result = results[0]
        for result in results[1:]:
            assert result['processed_smiles'] == first_result['processed_smiles']
            assert result['failed_indices'] == first_result['failed_indices']
            assert result['stats']['processed'] == first_result['stats']['processed']
    
    @pytest.mark.rdkit
    def test_preprocessing_error_recovery(self):
        """Test preprocessing error recovery and reporting."""
        from molenc.preprocessing.utils import preprocess_smiles_list
        
        # Mix of valid, invalid, and edge case SMILES
        problematic_smiles = [
            "CCO",  # Valid
            "INVALID",  # Invalid
            "",  # Empty
            None,  # None (if handled)
            "C" * 1000,  # Very long (might cause issues)
            "c1ccccc1",  # Valid aromatic
            "C(C",  # Unmatched parenthesis
            "[Xe]",  # Unusual element
        ]
        
        # Filter out None values if not handled
        test_smiles = [s for s in problematic_smiles if s is not None]
        
        result = preprocess_smiles_list(
            test_smiles,
            standardize=True,
            validate=True,
            filter_molecules=True
        )
        
        # Should handle errors gracefully
        assert isinstance(result, dict)
        assert 'processed_smiles' in result
        assert 'failed_indices' in result
        assert 'stats' in result
        
        # Should process at least some valid molecules
        assert len(result['processed_smiles']) >= 2  # CCO and benzene
        
        # Should report failures appropriately
        assert len(result['failed_indices']) >= 3  # Invalid, empty, unmatched
    
    @pytest.mark.rdkit
    @pytest.mark.slow
    def test_comprehensive_preprocessing_benchmark(self):
        """Comprehensive preprocessing benchmark test."""
        from molenc.preprocessing.utils import preprocess_smiles_list
        import time
        
        # Create diverse dataset
        diverse_smiles = [
            # Small molecules
            "C", "CC", "CCO", "CCN", "CCC",
            
            # Aromatic compounds
            "c1ccccc1", "c1ccc2ccccc2c1", "c1ccncc1",
            
            # Drug-like molecules
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            
            # Natural products
            "C[C@H]1CC[C@H]2[C@@H](C1)CC[C@@H]3[C@@H]2CC[C@H]4[C@@H]3CCC(=O)C4",
            
            # Invalid SMILES
            "INVALID", "C(C", "C]C", "",
            
            # Edge cases
            "[H][H]", "[Na+].[Cl-]", "C#C", "C=C=C",
        ]
        
        # Replicate for larger dataset
        large_diverse = diverse_smiles * 50  # 1000+ SMILES
        
        start_time = time.time()
        result = preprocess_smiles_list(
            large_diverse,
            standardize=True,
            standardize_options={
                'remove_stereochemistry': False,
                'neutralize': True,
                'remove_salts': True,
                'canonicalize': True
            },
            validate=True,
            validation_options={
                'strict_mode': False,
                'check_aromaticity': True,
                'check_valence': True,
                'max_atoms': 200
            },
            filter_molecules=True,
            filter_options={
                'mw_range': (12, 1000),  # Very permissive
                'logp_range': (-10, 15),
                'hbd_range': (0, 20),
                'hba_range': (0, 30),
                'lipinski_rule': False,
                'pains_filter': False
            },
            parallel=True,
            n_jobs=2
        )
        end_time = time.time()
        
        # Performance checks
        assert end_time - start_time < 60.0  # 1 minute threshold
        
        # Quality checks
        processed = result['processed_smiles']
        stats = result['stats']
        
        # Should process most valid molecules
        success_rate = len(processed) / stats['total']
        assert success_rate > 0.7  # At least 70% success rate
        
        # Should have comprehensive statistics
        assert 'standardization' in stats
        assert 'validation' in stats
        assert 'filtering' in stats
        
        # Check error breakdown
        if 'error_breakdown' in stats:
            error_breakdown = stats['error_breakdown']
            total_errors = sum(error_breakdown.values())
            assert total_errors == stats['failed']