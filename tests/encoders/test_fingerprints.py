"""Tests for fingerprint encoders."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from molenc.core.exceptions import InvalidSMILESError, DependencyError


class TestMorganEncoder:
    """Tests for MorganEncoder."""
    
    @pytest.mark.rdkit
    def test_morgan_encoder_initialization(self):
        """Test MorganEncoder initialization with default parameters."""
        from molenc.encoders.descriptors.fingerprints.morgan import MorganEncoder
        
        encoder = MorganEncoder()
        assert encoder.radius == 2
        assert encoder.n_bits == 2048
        assert encoder.use_features is False
        assert encoder.use_chirality is False
        assert encoder.use_bond_types is True
    
    @pytest.mark.rdkit
    def test_morgan_encoder_custom_params(self):
        """Test MorganEncoder initialization with custom parameters."""
        from molenc.encoders.descriptors.fingerprints.morgan import MorganEncoder
        
        encoder = MorganEncoder(
            radius=3,
            n_bits=1024,
            use_features=True,
            use_chirality=True,
            use_bond_types=False
        )
        assert encoder.radius == 3
        assert encoder.n_bits == 1024
        assert encoder.use_features is True
        assert encoder.use_chirality is True
        assert encoder.use_bond_types is False
    
    def test_morgan_encoder_missing_rdkit(self):
        """Test MorganEncoder raises error when RDKit is not available."""
        with patch.dict('sys.modules', {'rdkit': None}):
            with pytest.raises(DependencyError) as exc_info:
                from molenc.encoders.descriptors.fingerprints.morgan import MorganEncoder
                MorganEncoder()
            assert "rdkit" in str(exc_info.value)
            assert "morgan" in str(exc_info.value)
    
    @pytest.mark.rdkit
    def test_morgan_encode_single_valid_smiles(self, sample_smiles):
        """Test encoding a single valid SMILES string."""
        from molenc.encoders.descriptors.fingerprints.morgan import MorganEncoder
        
        encoder = MorganEncoder()
        fingerprint = encoder.encode(sample_smiles[0])
        
        assert isinstance(fingerprint, np.ndarray)
        assert fingerprint.shape == (2048,)
        assert fingerprint.dtype == np.uint8
        assert fingerprint.sum() > 0  # Should have some bits set
    
    @pytest.mark.rdkit
    def test_morgan_encode_batch(self, sample_smiles):
        """Test batch encoding of SMILES strings."""
        from molenc.encoders.descriptors.fingerprints.morgan import MorganEncoder
        
        encoder = MorganEncoder()
        fingerprints = encoder.encode_batch(sample_smiles)
        
        assert isinstance(fingerprints, np.ndarray)
        assert fingerprints.shape == (len(sample_smiles), 2048)
        assert fingerprints.dtype == np.uint8
        assert np.all(fingerprints.sum(axis=1) > 0)  # All should have bits set
    
    @pytest.mark.rdkit
    def test_morgan_encode_invalid_smiles(self, invalid_smiles):
        """Test encoding invalid SMILES strings."""
        from molenc.encoders.descriptors.fingerprints.morgan import MorganEncoder
        
        encoder = MorganEncoder()
        
        # Test with error_handling='raise'
        with pytest.raises(InvalidSMILESError):
            encoder.encode(invalid_smiles[0])
        
        # Test with error_handling='skip'
        encoder_skip = MorganEncoder(error_handling='skip')
        result = encoder_skip.encode_batch(invalid_smiles)
        assert result.shape[0] == 0  # Should return empty array
    
    @pytest.mark.rdkit
    def test_morgan_get_output_dim(self):
        """Test getting output dimension."""
        from molenc.encoders.descriptors.fingerprints.morgan import MorganEncoder
        
        encoder = MorganEncoder(n_bits=1024)
        assert encoder.get_output_dim() == 1024
        
        encoder = MorganEncoder(n_bits=4096)
        assert encoder.get_output_dim() == 4096
    
    @pytest.mark.rdkit
    def test_morgan_get_feature_names(self):
        """Test getting feature names."""
        from molenc.encoders.descriptors.fingerprints.morgan import MorganEncoder
        
        encoder = MorganEncoder(n_bits=512)
        feature_names = encoder.get_feature_names()
        
        assert len(feature_names) == 512
        assert all(name.startswith('morgan_bit_') for name in feature_names)
        assert feature_names[0] == 'morgan_bit_0'
        assert feature_names[-1] == 'morgan_bit_511'
    
    @pytest.mark.rdkit
    def test_morgan_repr(self):
        """Test string representation."""
        from molenc.encoders.descriptors.fingerprints.morgan import MorganEncoder
        
        encoder = MorganEncoder(radius=3, n_bits=1024, use_features=True)
        repr_str = repr(encoder)
        
        assert 'MorganEncoder' in repr_str
        assert 'radius=3' in repr_str
        assert 'n_bits=1024' in repr_str
        assert 'use_features=True' in repr_str


class TestMACCSEncoder:
    """Tests for MACCSEncoder."""
    
    @pytest.mark.rdkit
    def test_maccs_encoder_initialization(self):
        """Test MACCSEncoder initialization."""
        from molenc.encoders.descriptors.fingerprints.maccs import MACCSEncoder
        
        encoder = MACCSEncoder()
        assert hasattr(encoder, 'MACCS_SIZE')
        assert encoder.MACCS_SIZE == 166
    
    def test_maccs_encoder_missing_rdkit(self):
        """Test MACCSEncoder raises error when RDKit is not available."""
        with patch.dict('sys.modules', {'rdkit': None}):
            with pytest.raises(DependencyError) as exc_info:
                from molenc.encoders.descriptors.fingerprints.maccs import MACCSEncoder
                MACCSEncoder()
            assert "rdkit" in str(exc_info.value)
            assert "maccs" in str(exc_info.value)
    
    @pytest.mark.rdkit
    def test_maccs_encode_single_valid_smiles(self, sample_smiles):
        """Test encoding a single valid SMILES string."""
        from molenc.encoders.descriptors.fingerprints.maccs import MACCSEncoder
        
        encoder = MACCSEncoder()
        fingerprint = encoder.encode(sample_smiles[0])
        
        assert isinstance(fingerprint, np.ndarray)
        assert fingerprint.shape == (166,)
        assert fingerprint.dtype == np.uint8
        assert fingerprint.sum() > 0  # Should have some keys set
    
    @pytest.mark.rdkit
    def test_maccs_encode_batch(self, sample_smiles):
        """Test batch encoding of SMILES strings."""
        from molenc.encoders.descriptors.fingerprints.maccs import MACCSEncoder
        
        encoder = MACCSEncoder()
        fingerprints = encoder.encode_batch(sample_smiles)
        
        assert isinstance(fingerprints, np.ndarray)
        assert fingerprints.shape == (len(sample_smiles), 166)
        assert fingerprints.dtype == np.uint8
        assert np.all(fingerprints.sum(axis=1) > 0)  # All should have keys set
    
    @pytest.mark.rdkit
    def test_maccs_encode_invalid_smiles(self, invalid_smiles):
        """Test encoding invalid SMILES strings."""
        from molenc.encoders.descriptors.fingerprints.maccs import MACCSEncoder
        
        encoder = MACCSEncoder()
        
        # Test with error_handling='raise'
        with pytest.raises(InvalidSMILESError):
            encoder.encode(invalid_smiles[0])
        
        # Test with error_handling='skip'
        encoder_skip = MACCSEncoder(error_handling='skip')
        result = encoder_skip.encode_batch(invalid_smiles)
        assert result.shape[0] == 0  # Should return empty array
    
    @pytest.mark.rdkit
    def test_maccs_get_output_dim(self):
        """Test getting output dimension."""
        from molenc.encoders.descriptors.fingerprints.maccs import MACCSEncoder
        
        encoder = MACCSEncoder()
        assert encoder.get_output_dim() == 166
    
    @pytest.mark.rdkit
    def test_maccs_get_feature_names(self):
        """Test getting feature names."""
        from molenc.encoders.descriptors.fingerprints.maccs import MACCSEncoder
        
        encoder = MACCSEncoder()
        feature_names = encoder.get_feature_names()
        
        assert len(feature_names) == 166
        assert all(name.startswith('maccs_key_') for name in feature_names)
        assert feature_names[0] == 'maccs_key_0'
        assert feature_names[-1] == 'maccs_key_165'
    
    @pytest.mark.rdkit
    def test_maccs_repr(self):
        """Test string representation."""
        from molenc.encoders.descriptors.fingerprints.maccs import MACCSEncoder
        
        encoder = MACCSEncoder()
        repr_str = repr(encoder)
        
        assert 'MACCSEncoder' in repr_str
        assert '166' in repr_str


class TestFingerprintComparison:
    """Tests comparing different fingerprint encoders."""
    
    @pytest.mark.rdkit
    def test_fingerprint_consistency(self, sample_smiles):
        """Test that fingerprints are consistent across multiple calls."""
        from molenc.encoders.descriptors.fingerprints.morgan import MorganEncoder
        from molenc.encoders.descriptors.fingerprints.maccs import MACCSEncoder
        
        morgan_encoder = MorganEncoder()
        maccs_encoder = MACCSEncoder()
        
        smiles = sample_smiles[0]
        
        # Test Morgan consistency
        morgan_fp1 = morgan_encoder.encode(smiles)
        morgan_fp2 = morgan_encoder.encode(smiles)
        np.testing.assert_array_equal(morgan_fp1, morgan_fp2)
        
        # Test MACCS consistency
        maccs_fp1 = maccs_encoder.encode(smiles)
        maccs_fp2 = maccs_encoder.encode(smiles)
        np.testing.assert_array_equal(maccs_fp1, maccs_fp2)
    
    @pytest.mark.rdkit
    def test_different_molecules_different_fingerprints(self, sample_smiles):
        """Test that different molecules produce different fingerprints."""
        from molenc.encoders.descriptors.fingerprints.morgan import MorganEncoder
        
        if len(sample_smiles) < 2:
            pytest.skip("Need at least 2 different SMILES for comparison")
        
        encoder = MorganEncoder()
        fp1 = encoder.encode(sample_smiles[0])
        fp2 = encoder.encode(sample_smiles[1])
        
        # Fingerprints should be different (unless molecules are identical)
        if sample_smiles[0] != sample_smiles[1]:
            assert not np.array_equal(fp1, fp2)
    
    @pytest.mark.rdkit
    def test_fingerprint_dimensions(self):
        """Test that fingerprint dimensions match expected values."""
        from molenc.encoders.descriptors.fingerprints.morgan import MorganEncoder
        from molenc.encoders.descriptors.fingerprints.maccs import MACCSEncoder
        
        # Test different Morgan dimensions
        for n_bits in [512, 1024, 2048, 4096]:
            encoder = MorganEncoder(n_bits=n_bits)
            assert encoder.get_output_dim() == n_bits
        
        # MACCS always has 166 bits
        encoder = MACCSEncoder()
        assert encoder.get_output_dim() == 166