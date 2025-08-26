"""Tests for core.base module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List

from molenc.core.base import BaseEncoder
from molenc.core.exceptions import InvalidSMILESError


class MockEncoder(BaseEncoder):
    """Mock encoder implementation for testing."""
    
    def __init__(self, output_dim=1024, **kwargs):
        super().__init__(output_dim=output_dim, **kwargs)
        self.output_dim = output_dim
    
    def _encode_single(self, smiles: str) -> np.ndarray:
        """Mock encoding that returns random vector."""
        return np.random.rand(self.output_dim)
    
    def get_output_dim(self) -> int:
        """Return the output dimension."""
        return self.output_dim


class FailingEncoder(BaseEncoder):
    """Mock encoder that always fails for testing error handling."""
    
    def _encode_single(self, smiles: str) -> np.ndarray:
        """Always raise an exception."""
        raise RuntimeError("Encoding failed")
    
    def get_output_dim(self) -> int:
        """Return a fixed dimension."""
        return 512


class TestBaseEncoder:
    """Test the BaseEncoder abstract class."""
    
    def test_cannot_instantiate_base_encoder(self):
        """Test that BaseEncoder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEncoder()
    
    def test_mock_encoder_instantiation(self):
        """Test that mock encoder can be instantiated."""
        encoder = MockEncoder()
        assert isinstance(encoder, BaseEncoder)
        assert encoder.handle_errors == 'raise'
        assert encoder.get_output_dim() == 1024
    
    def test_encoder_with_custom_params(self):
        """Test encoder initialization with custom parameters."""
        encoder = MockEncoder(
            output_dim=512,
            handle_errors='skip',
            custom_param='test_value'
        )
        assert encoder.handle_errors == 'skip'
        assert encoder.get_output_dim() == 512
        assert encoder.config['custom_param'] == 'test_value'
        assert encoder.config['output_dim'] == 512


class TestValidateSmiles:
    """Test SMILES validation functionality."""
    
    def test_validate_valid_smiles(self, sample_smiles):
        """Test validation of valid SMILES."""
        encoder = MockEncoder()
        
        for smiles in sample_smiles:
            assert encoder.validate_smiles(smiles) is True
    
    def test_validate_invalid_smiles(self, invalid_smiles):
        """Test validation of invalid SMILES."""
        encoder = MockEncoder()
        
        for smiles in invalid_smiles:
            assert encoder.validate_smiles(smiles) is False
    
    @patch('molenc.core.base.Chem.MolFromSmiles')
    def test_validate_smiles_rdkit_exception(self, mock_mol_from_smiles):
        """Test SMILES validation when RDKit raises an exception."""
        mock_mol_from_smiles.side_effect = Exception("RDKit error")
        encoder = MockEncoder()
        
        result = encoder.validate_smiles("CCO")
        assert result is False
    
    @patch('molenc.core.base.Chem.MolFromSmiles')
    def test_validate_smiles_returns_none(self, mock_mol_from_smiles):
        """Test SMILES validation when RDKit returns None."""
        mock_mol_from_smiles.return_value = None
        encoder = MockEncoder()
        
        result = encoder.validate_smiles("invalid")
        assert result is False


class TestEncodeSingle:
    """Test single molecule encoding."""
    
    def test_encode_single_valid_smiles(self):
        """Test encoding a single valid SMILES."""
        encoder = MockEncoder(output_dim=512)
        result = encoder.encode("CCO")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (512,)
    
    def test_encode_single_invalid_smiles_raise(self):
        """Test encoding invalid SMILES with raise error handling."""
        encoder = MockEncoder(handle_errors='raise')
        
        with pytest.raises(InvalidSMILESError):
            encoder.encode("invalid_smiles")
    
    def test_encode_single_invalid_smiles_skip(self, capsys):
        """Test encoding invalid SMILES with skip error handling."""
        encoder = MockEncoder(handle_errors='skip')
        result = encoder.encode("invalid_smiles")
        
        assert result is None
    
    def test_encode_single_invalid_smiles_warn(self, capsys):
        """Test encoding invalid SMILES with warn error handling."""
        encoder = MockEncoder(handle_errors='warn')
        result = encoder.encode("invalid_smiles")
        
        captured = capsys.readouterr()
        assert result is None
        assert "Warning" in captured.out
        assert "invalid_smiles" in captured.out
    
    def test_encode_wrong_input_type(self):
        """Test encoding with wrong input type."""
        encoder = MockEncoder()
        
        with pytest.raises(TypeError):
            encoder.encode(123)
        
        with pytest.raises(TypeError):
            encoder.encode(None)


class TestEncodeBatch:
    """Test batch encoding functionality."""
    
    def test_encode_batch_valid_smiles(self, sample_smiles):
        """Test batch encoding of valid SMILES."""
        encoder = MockEncoder(output_dim=256)
        results = encoder.encode_batch(sample_smiles)
        
        assert isinstance(results, np.ndarray)
        assert results.shape[0] == len(sample_smiles)
        assert results.shape[1] == 256
        
        for i in range(results.shape[0]):
            assert isinstance(results[i], np.ndarray)
            assert results[i].shape == (256,)
    
    def test_encode_batch_mixed_validity_raise(self, sample_smiles, invalid_smiles):
        """Test batch encoding with mixed valid/invalid SMILES and raise handling."""
        encoder = MockEncoder(handle_errors='raise')
        mixed_smiles = sample_smiles[:2] + [invalid_smiles[0]] + sample_smiles[2:]
        
        with pytest.raises(InvalidSMILESError):
            encoder.encode_batch(mixed_smiles)
    
    def test_encode_batch_mixed_validity_skip(self, sample_smiles, invalid_smiles):
        """Test batch encoding with mixed valid/invalid SMILES and skip handling."""
        encoder = MockEncoder(handle_errors='skip')
        mixed_smiles = sample_smiles[:2] + invalid_smiles[:2] + sample_smiles[2:]
        
        results = encoder.encode_batch(mixed_smiles)
        
        # Should only return results for valid SMILES
        assert len(results) == len(sample_smiles)
        for result in results:
            assert isinstance(result, np.ndarray)
    
    def test_encode_batch_mixed_validity_warn(self, sample_smiles, invalid_smiles, capsys):
        """Test batch encoding with mixed valid/invalid SMILES and warn handling."""
        encoder = MockEncoder(handle_errors='warn')
        mixed_smiles = sample_smiles[:1] + invalid_smiles[:1] + sample_smiles[1:2]
        
        results = encoder.encode_batch(mixed_smiles)
        captured = capsys.readouterr()
        
        # Should return results for valid SMILES only
        assert len(results) == 2  # 2 valid SMILES
        assert "Warning" in captured.out
    
    def test_encode_batch_empty_list(self):
        """Test batch encoding with empty list."""
        encoder = MockEncoder()
        results = encoder.encode_batch([])
        
        assert isinstance(results, np.ndarray)
        assert results.shape == (0, 1024)
    
    def test_encode_batch_encoding_failure(self, sample_smiles):
        """Test batch encoding when encoder fails."""
        encoder = FailingEncoder(handle_errors='skip')
        results = encoder.encode_batch(sample_smiles)
        
        # All should fail, so empty results
        assert len(results) == 0
    
    def test_encode_list_input(self, sample_smiles):
        """Test that encode() method works with list input."""
        encoder = MockEncoder()
        results = encoder.encode(sample_smiles)
        
        assert isinstance(results, np.ndarray)
        assert results.shape[0] == len(sample_smiles)


class TestErrorHandling:
    """Test error handling mechanisms."""
    
    def test_encode_with_error_handling_raise(self):
        """Test _encode_with_error_handling with raise mode."""
        encoder = MockEncoder(handle_errors='raise')
        
        # Valid SMILES should work
        result = encoder._encode_with_error_handling("CCO")
        assert isinstance(result, np.ndarray)
        
        # Invalid SMILES should raise
        with pytest.raises(InvalidSMILESError):
            encoder._encode_with_error_handling("invalid")
    
    def test_encode_with_error_handling_skip(self):
        """Test _encode_with_error_handling with skip mode."""
        encoder = MockEncoder(handle_errors='skip')
        
        # Valid SMILES should work
        result = encoder._encode_with_error_handling("CCO")
        assert isinstance(result, np.ndarray)
        
        # Invalid SMILES should return None
        result = encoder._encode_with_error_handling("invalid")
        assert result is None
    
    def test_encode_with_error_handling_warn(self, capsys):
        """Test _encode_with_error_handling with warn mode."""
        encoder = MockEncoder(handle_errors='warn')
        
        # Valid SMILES should work
        result = encoder._encode_with_error_handling("CCO")
        assert isinstance(result, np.ndarray)
        
        # Invalid SMILES should return None and print warning
        result = encoder._encode_with_error_handling("invalid")
        captured = capsys.readouterr()
        
        assert result is None
        assert "Warning" in captured.out
        assert "invalid" in captured.out


class TestConfiguration:
    """Test configuration and metadata methods."""
    
    def test_get_config(self):
        """Test get_config method."""
        encoder = MockEncoder(
            output_dim=512,
            handle_errors='skip',
            custom_param='test'
        )
        
        config = encoder.get_config()
        
        assert config['encoder_type'] == 'MockEncoder'
        assert config['handle_errors'] == 'skip'
        assert config['output_dim'] == 512
        assert config['custom_param'] == 'test'
    
    def test_repr(self):
        """Test string representation."""
        encoder = MockEncoder(output_dim=256)
        repr_str = repr(encoder)
        
        assert 'MockEncoder' in repr_str
        assert 'output_dim=256' in repr_str
    
    def test_config_immutability(self):
        """Test that config doesn't affect encoder after initialization."""
        encoder = MockEncoder(test_param='original')
        config = encoder.get_config()
        
        # Modifying returned config shouldn't affect encoder
        config['test_param'] = 'modified'
        new_config = encoder.get_config()
        
        assert new_config['test_param'] == 'original'


class TestAbstractMethods:
    """Test that abstract methods are properly defined."""
    
    def test_abstract_methods_exist(self):
        """Test that required abstract methods exist."""
        abstract_methods = BaseEncoder.__abstractmethods__
        
        assert '_encode_single' in abstract_methods
        assert 'get_output_dim' in abstract_methods
    
    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementations cannot be instantiated."""
        
        class IncompleteEncoder(BaseEncoder):
            def _encode_single(self, smiles: str) -> np.ndarray:
                return np.array([1, 2, 3])
            # Missing get_output_dim
        
        with pytest.raises(TypeError):
            IncompleteEncoder()
    
    def test_complete_implementation_works(self):
        """Test that complete implementations work."""
        
        class CompleteEncoder(BaseEncoder):
            def _encode_single(self, smiles: str) -> np.ndarray:
                return np.array([1, 2, 3])
            
            def get_output_dim(self) -> int:
                return 3
        
        encoder = CompleteEncoder()
        assert isinstance(encoder, BaseEncoder)
        assert encoder.get_output_dim() == 3