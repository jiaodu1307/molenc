"""Integration tests for the complete molenc system."""

import os
import tempfile
import yaml
import numpy as np
from unittest.mock import patch, MagicMock

import pytest

from molenc.environments.config import EnvironmentConfig
from molenc.environments.manager import EnvironmentManager
from molenc.environments.dependencies import DependencyChecker
from molenc.preprocessing.utils import preprocess_smiles_list
from molenc.core.registry import EncoderRegistry


class TestSystemIntegration:
    """Integration tests for the complete molenc system."""
    
    def test_environment_manager_integration(self):
        """Test integration between environment manager and configuration."""
        # Create temporary configuration
        config_data = {
            'environment': {
                'check_dependencies': True,
                'auto_install': False,
                'max_workers': 4
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            # Load configuration
            config = EnvironmentConfig.from_file(config_file)
            
            # Create environment manager
            env_manager = EnvironmentManager(config)
            
            # Mock dependency checking
            with patch.object(env_manager.dependency_checker, 'check_all_dependencies') as mock_check:
                mock_check.return_value = {
                    'core_dependencies': {'numpy': True, 'pandas': True, 'sklearn': True},
                    'optional_dependencies': {'rdkit': False, 'torch': False},
                    'core_satisfied': True,
                    'core_available': 3,
                    'core_total': 3,
                    'optional_available': 0,
                    'optional_total': 2,
                    'python_version': '3.8.0',
                    'platform': 'linux'
                }
                
                # Check environment
                env_status = env_manager.check_environment()
                
                # Verify structure
                assert 'system_info' in env_status
                assert 'dependencies' in env_status
                assert 'config' in env_status
                
                # Verify dependencies structure
                deps = env_status['dependencies']
                assert 'core_dependencies' in deps
                assert 'optional_dependencies' in deps
                assert deps['core_satisfied'] is True
                
        finally:
            # Cleanup
            if os.path.exists(config_file):
                os.unlink(config_file)
    
    def test_complete_system_workflow(self):
        """Test complete system workflow from configuration to encoding."""
        # Step 1: Create temporary configuration
        config_data = {
            'environment': {
                'check_dependencies': True,
                'auto_install': False,
                'max_workers': 2
            },
            'preprocessing': {
                'standardize': True,
                'validate': True,
                'filter_molecules': True
            },
            'encoding': {
                'default_encoder': 'morgan',
                'batch_size': 100
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            # Step 2: Load configuration
            config = EnvironmentConfig.from_file(config_file)
            
            # Sample molecules for testing
            sample_molecules = [
                'CCO',  # ethanol
                'CC(=O)O',  # acetic acid
                'c1ccccc1',  # benzene
                'INVALID_SMILES',  # should be filtered out
                'CC(C)O'  # isopropanol
            ]
            
            # Step 3: Mock all RDKit dependencies and preprocessing classes
            with patch('molenc.preprocessing.standardize.Chem') as mock_chem, \
                 patch('molenc.preprocessing.standardize.rdMolStandardize') as mock_mol_std, \
                 patch('molenc.preprocessing.validators.Chem') as mock_val_chem, \
                 patch('molenc.preprocessing.filters.Chem') as mock_filter_chem, \
                 patch('molenc.preprocessing.filters.Descriptors') as mock_descriptors, \
                 patch('molenc.encoders.descriptors.fingerprints.morgan.Chem') as mock_fp_chem, \
                 patch('molenc.encoders.descriptors.fingerprints.morgan.rdMolDescriptors') as mock_mol_descriptors, \
                 patch('molenc.preprocessing.standardize.SMILESStandardizer') as mock_std_class, \
                 patch('molenc.preprocessing.validators.SMILESValidator') as mock_val_class, \
                 patch('molenc.preprocessing.filters.MolecularFilters') as mock_filter_class:
                
                # Setup comprehensive mocks
                mock_mol = MagicMock()
                
                # Setup preprocessing class mocks
                mock_standardizer = MagicMock()
                mock_validator = MagicMock()
                mock_filter = MagicMock()
                
                mock_std_class.return_value = mock_standardizer
                mock_val_class.return_value = mock_validator
                mock_filter_class.return_value = mock_filter
                
                # Configure standardizer behavior
                def mock_standardize(smiles):
                    return smiles if 'INVALID' not in smiles else smiles
                mock_standardizer.standardize.side_effect = mock_standardize
                
                # Configure validator behavior
                def mock_validate(smiles):
                    is_valid = 'INVALID' not in smiles
                    return is_valid, {'errors': [] if is_valid else ['Invalid SMILES']}
                mock_validator.validate.side_effect = mock_validate
                
                # Configure filter behavior
                def mock_filter_molecule(smiles):
                    passes = 'INVALID' not in smiles
                    return passes, {} if passes else {'reason': 'Invalid molecule'}
                mock_filter.filter_molecule.side_effect = mock_filter_molecule
                
                # Standardizer mocks
                def mock_mol_from_smiles_std(smiles):
                    return None if 'INVALID' in smiles else mock_mol
                mock_chem.MolFromSmiles.side_effect = mock_mol_from_smiles_std
                mock_chem.MolToSmiles.return_value = 'CCO'
                
                # Validator mocks
                def mock_mol_from_smiles_val(smiles):
                    return None if 'INVALID' in smiles else mock_mol
                mock_val_chem.MolFromSmiles.side_effect = mock_mol_from_smiles_val
                
                # Filter mocks
                def mock_mol_from_smiles_filt(smiles):
                    return None if 'INVALID' in smiles else mock_mol
                mock_filter_chem.MolFromSmiles.side_effect = mock_mol_from_smiles_filt
                
                # Descriptor mocks
                mock_descriptors.MolWt.return_value = 150.0
                mock_descriptors.MolLogP.return_value = 2.0
                mock_descriptors.NumHDonors.return_value = 1
                mock_descriptors.NumHAcceptors.return_value = 2
                
                # Encoder mocks
                def mock_mol_from_smiles_enc(smiles):
                    return None if 'INVALID' in smiles else mock_mol
                mock_fp_chem.MolFromSmiles.side_effect = mock_mol_from_smiles_enc
                mock_mol_descriptors.GetMorganFingerprintAsBitVect.return_value = [1, 0] * 1024
                
                # Step 4: Preprocess molecules
                processed_data = preprocess_smiles_list(
                    sample_molecules,
                    standardize=True,
                    validate=True,
                    filter_molecules=True,
                    n_jobs=config.max_workers
                )
                
                valid_smiles = processed_data['processed_smiles']
                stats = processed_data['stats']
                
                # Step 5: Setup encoder registry and encode
                registry = EncoderRegistry()
                
                # Mock encoder
                mock_encoder = MagicMock()
                mock_encoder.encode_batch.return_value = np.random.rand(len(valid_smiles), 2048)
                mock_encoder.get_output_dim.return_value = 2048
                mock_encoder_class = MagicMock(return_value=mock_encoder)
                registry.register('morgan', mock_encoder_class)
                
                encoder = registry.get_encoder('morgan')
                features = encoder.encode_batch(valid_smiles)
                
                # Step 6: Verify complete workflow
                assert os.path.exists(config_file)
                assert len(valid_smiles) > 0
                assert len(valid_smiles) < len(sample_molecules)  # Some should be filtered out
                assert isinstance(stats, dict)
                assert 'total' in stats
                assert features.shape[0] == len(valid_smiles)
                assert features.shape[1] == 2048
        
        finally:
            # Cleanup
            if os.path.exists(config_file):
                os.unlink(config_file)
    
    def test_environment_dependency_integration(self):
        """Test integration between environment management and dependencies."""
        # Create dependency checker
        checker = DependencyChecker()
        
        # Mock package checking
        with patch('importlib.import_module') as mock_import:
            with patch('pkg_resources.get_distribution') as mock_get_dist:
                
                # Mock successful imports
                mock_import.return_value = MagicMock()
                mock_get_dist.return_value.version = '1.0.0'
                
                # Check core dependencies
                core_results = checker.check_core_dependencies()
                assert isinstance(core_results, dict)
                
                # Check optional dependencies
                optional_results = checker.check_optional_dependencies()
                assert isinstance(optional_results, dict)
                
                # Check all dependencies
                all_results = checker.check_all_dependencies()
                assert isinstance(all_results, dict)
                assert 'core_dependencies' in all_results
                assert 'optional_dependencies' in all_results
    
    def test_configuration_validation_integration(self):
        """Test integration of configuration validation across components."""
        # Test valid configuration
        valid_config_data = {
            'environment': {
                'max_workers': 4,
                'gpu_enabled': False,
                'log_level': 'DEBUG',
                'memory_limit': '2GB'
            },
            'preprocessing': {
                'standardize': True,
                'validate': True,
                'filter_molecules': True
            },
            'encoding': {
                'default_encoder': 'morgan',
                'batch_size': 100
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config_data, f)
            config_file = f.name
        
        try:
            # Should load without errors
            config = EnvironmentConfig.from_file(config_file)
            assert config.max_workers == 4
            assert config.gpu_enabled is False
            assert config.log_level == 'DEBUG'
            assert config.memory_limit == '2GB'
            
        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)
        
        # Test invalid configuration (unsupported file format)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid config content")
            invalid_config_file = f.name
        
        try:
            # Should raise validation error for unsupported format
            with pytest.raises(ValueError):
                EnvironmentConfig.from_file(invalid_config_file)
                
        finally:
            if os.path.exists(invalid_config_file):
                os.unlink(invalid_config_file)