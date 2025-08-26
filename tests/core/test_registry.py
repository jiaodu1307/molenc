"""Tests for core.registry module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List

from molenc.core.registry import (
    EncoderRegistry,
    MolEncoder,
    register_encoder,
    _registry
)
from molenc.core.base import BaseEncoder
from molenc.core.exceptions import (
    EncoderNotFoundError,
    EncoderInitializationError,
    DependencyError
)


class MockEncoder(BaseEncoder):
    """Mock encoder for testing."""
    
    def __init__(self, output_dim=1024, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
    
    def _encode_single(self, smiles: str) -> np.ndarray:
        return np.random.rand(self.output_dim)
    
    def get_output_dim(self) -> int:
        return self.output_dim


class FailingEncoder(BaseEncoder):
    """Encoder that fails during initialization."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise RuntimeError("Initialization failed")
    
    def _encode_single(self, smiles: str) -> np.ndarray:
        return np.array([1, 2, 3])
    
    def get_output_dim(self) -> int:
        return 3


class TestEncoderRegistry:
    """Test the EncoderRegistry class."""
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = EncoderRegistry()
        assert isinstance(registry._encoders, dict)
        assert isinstance(registry._encoder_modules, dict)
        assert len(registry._encoders) == 0
        assert len(registry._encoder_modules) == 0
    
    def test_register_encoder_class(self):
        """Test registering an encoder class."""
        registry = EncoderRegistry()
        registry.register('test_encoder', MockEncoder)
        
        assert 'test_encoder' in registry._encoders
        assert registry._encoders['test_encoder'] == MockEncoder
        assert registry.is_registered('test_encoder')
    
    def test_register_encoder_with_module_path(self):
        """Test registering an encoder with module path."""
        registry = EncoderRegistry()
        module_path = 'test.module.path'
        registry.register('test_encoder', MockEncoder, module_path)
        
        assert 'test_encoder' in registry._encoders
        assert 'test_encoder' in registry._encoder_modules
        assert registry._encoder_modules['test_encoder'] == module_path
    
    def test_get_encoder_success(self):
        """Test successfully getting an encoder."""
        registry = EncoderRegistry()
        registry.register('test_encoder', MockEncoder)
        
        encoder = registry.get_encoder('test_encoder', output_dim=512)
        
        assert isinstance(encoder, MockEncoder)
        assert encoder.get_output_dim() == 512
    
    def test_get_encoder_not_found(self):
        """Test getting a non-existent encoder."""
        registry = EncoderRegistry()
        registry.register('existing_encoder', MockEncoder)
        
        with pytest.raises(EncoderNotFoundError) as exc_info:
            registry.get_encoder('non_existent')
        
        assert 'non_existent' in str(exc_info.value)
        assert 'existing_encoder' in str(exc_info.value)
    
    def test_get_encoder_initialization_error(self):
        """Test encoder initialization failure."""
        registry = EncoderRegistry()
        registry.register('failing_encoder', FailingEncoder)
        
        with pytest.raises(EncoderInitializationError) as exc_info:
            registry.get_encoder('failing_encoder')
        
        assert 'failing_encoder' in str(exc_info.value)
        assert 'Initialization failed' in str(exc_info.value)
    
    @patch('molenc.core.registry.importlib.import_module')
    def test_lazy_load_encoder_success(self, mock_import):
        """Test successful lazy loading of encoder."""
        registry = EncoderRegistry()
        module_path = 'test.module'
        
        # Register with module path only
        registry._encoder_modules['lazy_encoder'] = module_path
        
        # Mock the module import and registration
        mock_module = Mock()
        mock_import.return_value = mock_module
        
        # Simulate the module registering the encoder when imported
        def side_effect(path):
            if path == module_path:
                registry._encoders['lazy_encoder'] = MockEncoder
            return mock_module
        
        mock_import.side_effect = side_effect
        
        encoder = registry.get_encoder('lazy_encoder')
        
        assert isinstance(encoder, MockEncoder)
        mock_import.assert_called_once_with(module_path)
    
    @patch('molenc.core.registry.importlib.import_module')
    def test_lazy_load_encoder_import_error(self, mock_import):
        """Test lazy loading with import error."""
        registry = EncoderRegistry()
        module_path = 'non.existent.module'
        registry._encoder_modules['lazy_encoder'] = module_path
        
        mock_import.side_effect = ImportError("Module not found")
        
        with pytest.raises(DependencyError) as exc_info:
            registry.get_encoder('lazy_encoder')
        
        assert 'lazy_encoder' in str(exc_info.value)
        assert module_path in str(exc_info.value)
    
    def test_list_encoders(self):
        """Test listing registered encoders."""
        registry = EncoderRegistry()
        registry.register('encoder1', MockEncoder)
        registry.register('encoder2', MockEncoder)
        
        encoders = registry.list_encoders()
        
        assert isinstance(encoders, list)
        assert 'encoder1' in encoders
        assert 'encoder2' in encoders
        assert len(encoders) == 2
    
    def test_is_registered(self):
        """Test checking if encoder is registered."""
        registry = EncoderRegistry()
        registry.register('registered_encoder', MockEncoder)
        registry._encoder_modules['lazy_encoder'] = 'test.module'
        
        assert registry.is_registered('registered_encoder')
        assert registry.is_registered('lazy_encoder')
        assert not registry.is_registered('non_existent')


class TestRegisterEncoderDecorator:
    """Test the register_encoder decorator."""
    
    def test_register_encoder_decorator(self):
        """Test using register_encoder as decorator."""
        # Create a fresh registry for this test
        test_registry = EncoderRegistry()
        
        # Mock the global registry
        with patch('molenc.core.registry._registry', test_registry):
            @register_encoder('decorated_encoder')
            class DecoratedEncoder(BaseEncoder):
                def _encode_single(self, smiles: str) -> np.ndarray:
                    return np.array([1, 2, 3])
                
                def get_output_dim(self) -> int:
                    return 3
            
            assert test_registry.is_registered('decorated_encoder')
            encoder = test_registry.get_encoder('decorated_encoder')
            assert isinstance(encoder, DecoratedEncoder)
    
    def test_register_encoder_decorator_with_module_path(self):
        """Test register_encoder decorator with module path."""
        test_registry = EncoderRegistry()
        
        with patch('molenc.core.registry._registry', test_registry):
            @register_encoder('decorated_encoder', 'test.module.path')
            class DecoratedEncoder(BaseEncoder):
                def _encode_single(self, smiles: str) -> np.ndarray:
                    return np.array([1, 2, 3])
                
                def get_output_dim(self) -> int:
                    return 3
            
            assert test_registry.is_registered('decorated_encoder')
            assert 'decorated_encoder' in test_registry._encoder_modules


class TestMolEncoder:
    """Test the MolEncoder class."""
    
    def test_mol_encoder_initialization(self):
        """Test MolEncoder initialization."""
        with patch('molenc.core.registry._registry') as mock_registry:
            mock_encoder = Mock(spec=BaseEncoder)
            mock_registry.get_encoder.return_value = mock_encoder
            
            mol_encoder = MolEncoder('test_encoder', param1='value1')
            
            assert mol_encoder.encoder_name == 'test_encoder'
            assert mol_encoder.encoder == mock_encoder
            mock_registry.get_encoder.assert_called_once_with('test_encoder', param1='value1')
    
    def test_mol_encoder_encode(self, sample_smiles):
        """Test MolEncoder encode method."""
        with patch('molenc.core.registry._registry') as mock_registry:
            mock_encoder = Mock(spec=BaseEncoder)
            mock_encoder.encode.return_value = np.array([1, 2, 3])
            mock_registry.get_encoder.return_value = mock_encoder
            
            mol_encoder = MolEncoder('test_encoder')
            result = mol_encoder.encode(sample_smiles[0])
            
            mock_encoder.encode.assert_called_once_with(sample_smiles[0])
            np.testing.assert_array_equal(result, np.array([1, 2, 3]))
    
    def test_mol_encoder_encode_batch(self, sample_smiles):
        """Test MolEncoder encode_batch method."""
        with patch('molenc.core.registry._registry') as mock_registry:
            mock_encoder = Mock(spec=BaseEncoder)
            expected_result = [np.array([1, 2, 3]), np.array([4, 5, 6])]
            mock_encoder.encode_batch.return_value = expected_result
            mock_registry.get_encoder.return_value = mock_encoder
            
            mol_encoder = MolEncoder('test_encoder')
            result = mol_encoder.encode_batch(sample_smiles[:2])
            
            mock_encoder.encode_batch.assert_called_once_with(sample_smiles[:2])
            assert result == expected_result
    
    def test_mol_encoder_get_output_dim(self):
        """Test MolEncoder get_output_dim method."""
        with patch('molenc.core.registry._registry') as mock_registry:
            mock_encoder = Mock(spec=BaseEncoder)
            mock_encoder.get_output_dim.return_value = 1024
            mock_registry.get_encoder.return_value = mock_encoder
            
            mol_encoder = MolEncoder('test_encoder')
            result = mol_encoder.get_output_dim()
            
            assert result == 1024
            mock_encoder.get_output_dim.assert_called_once()
    
    def test_mol_encoder_get_config(self):
        """Test MolEncoder get_config method."""
        with patch('molenc.core.registry._registry') as mock_registry:
            mock_encoder = Mock(spec=BaseEncoder)
            expected_config = {'encoder_type': 'MockEncoder', 'param': 'value'}
            mock_encoder.get_config.return_value = expected_config
            mock_registry.get_encoder.return_value = mock_encoder
            
            mol_encoder = MolEncoder('test_encoder')
            result = mol_encoder.get_config()
            
            assert result == expected_config
            mock_encoder.get_config.assert_called_once()
    
    @patch('molenc.core.config.Config')
    def test_mol_encoder_from_config(self, mock_config_class):
        """Test MolEncoder.from_config class method."""
        mock_config = Mock()
        mock_config.to_dict.return_value = {'encoder_name': 'test', 'param': 'value'}
        mock_config_class.from_file.return_value = mock_config
        
        with patch('molenc.core.registry._registry') as mock_registry:
            mock_encoder = Mock(spec=BaseEncoder)
            mock_registry.get_encoder.return_value = mock_encoder
            
            mol_encoder = MolEncoder.from_config('config.yaml')
            
            mock_config_class.from_file.assert_called_once_with('config.yaml')
            mock_config.to_dict.assert_called_once()
            assert isinstance(mol_encoder, MolEncoder)
    
    @patch('molenc.core.config.Config')
    def test_mol_encoder_from_preset(self, mock_config_class):
        """Test MolEncoder.from_preset class method."""
        mock_config = Mock()
        mock_config.to_dict.return_value = {'encoder_name': 'morgan', 'radius': 2}
        mock_config_class.from_preset.return_value = mock_config
        
        with patch('molenc.core.registry._registry') as mock_registry:
            mock_encoder = Mock(spec=BaseEncoder)
            mock_registry.get_encoder.return_value = mock_encoder
            
            mol_encoder = MolEncoder.from_preset('drug_discovery')
            
            mock_config_class.from_preset.assert_called_once_with('drug_discovery')
            mock_config.to_dict.assert_called_once()
            assert isinstance(mol_encoder, MolEncoder)
    
    def test_mol_encoder_list_encoders(self):
        """Test MolEncoder.list_encoders static method."""
        with patch('molenc.core.registry._registry') as mock_registry:
            expected_encoders = ['morgan', 'maccs', 'unimol']
            mock_registry.list_encoders.return_value = expected_encoders
            
            result = MolEncoder.list_encoders()
            
            assert result == expected_encoders
            mock_registry.list_encoders.assert_called_once()
    
    def test_mol_encoder_repr(self):
        """Test MolEncoder string representation."""
        with patch('molenc.core.registry._registry') as mock_registry:
            mock_encoder = Mock(spec=BaseEncoder)
            mock_encoder.get_output_dim.return_value = 1024
            mock_registry.get_encoder.return_value = mock_encoder
            
            mol_encoder = MolEncoder('test_encoder')
            repr_str = repr(mol_encoder)
            
            assert 'MolEncoder' in repr_str
            assert 'test_encoder' in repr_str
            assert '1024' in repr_str


class TestGlobalRegistry:
    """Test the global registry functionality."""
    
    def test_global_registry_exists(self):
        """Test that global registry exists and is properly initialized."""
        assert _registry is not None
        assert isinstance(_registry, EncoderRegistry)
    
    def test_builtin_encoders_registered(self):
        """Test that built-in encoders are registered."""
        # Note: This test checks the registration, not the actual loading
        # since the modules might not be available in the test environment
        
        expected_encoders = [
            'morgan', 'maccs', 'mol2vec',
            'chemberta',
            'gcn', 'gat',
            'unimol'
        ]
        
        for encoder_name in expected_encoders:
            assert _registry.is_registered(encoder_name), f"{encoder_name} should be registered"
    
    def test_builtin_encoders_have_module_paths(self):
        """Test that built-in encoders have module paths for lazy loading."""
        # Check that module paths are set for lazy loading
        assert 'morgan' in _registry._encoder_modules
        assert 'unimol' in _registry._encoder_modules
        
        # Check that the module paths look reasonable
        morgan_path = _registry._encoder_modules['morgan']
        assert 'molenc.encoders' in morgan_path
        assert 'fingerprints' in morgan_path


class TestRegistryIntegration:
    """Test integration between registry components."""
    
    def test_end_to_end_encoder_usage(self):
        """Test complete end-to-end usage of MolEncoder."""
        # Create encoder instance
        encoder = MolEncoder('morgan')
        
        # Test single molecule encoding
        smiles = "CCO"  # Ethanol
        single_result = encoder.encode(smiles)
        assert isinstance(single_result, np.ndarray)
        assert single_result.ndim == 1
        
        # Test batch encoding
        smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
        batch_result = encoder.encode_batch(smiles_list)
        assert isinstance(batch_result, np.ndarray)
        assert batch_result.shape[0] == len(smiles_list)
        
        # Test that all results have consistent dimensions
        dim = single_result.shape[0]
        for i in range(batch_result.shape[0]):
            assert batch_result[i].shape[0] == dim
            
        # Test configuration
        config = encoder.get_config()
        assert isinstance(config, dict)
        assert 'encoder_type' in config
        
        # Test string representation
        repr_str = repr(encoder)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0
    
    def test_error_propagation(self):
        """Test that errors propagate correctly through the system."""
        test_registry = EncoderRegistry()
        
        with patch('molenc.core.registry._registry', test_registry):
            # Test encoder not found error
            with pytest.raises(EncoderNotFoundError):
                MolEncoder('non_existent_encoder')
            
            # Test initialization error
            test_registry.register('failing_encoder', FailingEncoder)
            with pytest.raises(EncoderInitializationError):
                MolEncoder('failing_encoder')