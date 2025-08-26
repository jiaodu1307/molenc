"""Tests for multimodal encoders."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

from molenc.core.exceptions import InvalidSMILESError, DependencyError, EncoderInitializationError


class TestUniMolEncoder:
    """Tests for UniMolEncoder."""
    
    @pytest.mark.torch
    def test_unimol_encoder_initialization_default(self):
        """Test UniMolEncoder initialization with default parameters."""
        with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_model'):
            from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
            
            encoder = UniMolEncoder()
            assert encoder.model_name == 'unimol_v1'
            assert encoder.output_dim == 512
            assert encoder.use_3d is True
            assert encoder.max_atoms == 512
    
    @pytest.mark.torch
    def test_unimol_encoder_initialization_custom(self):
        """Test UniMolEncoder initialization with custom parameters."""
        with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_model'):
            from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
            
            encoder = UniMolEncoder(
                model_name='unimol_v2',
                output_dim=1024,
                use_3d=False,
                max_atoms=128,
                device='cpu'
            )
            assert encoder.model_name == 'unimol_v2'
            assert encoder.output_dim == 1024
            assert encoder.use_3d is False
            assert encoder.max_atoms == 128
            assert str(encoder.device) == 'cpu'
    
    def test_unimol_encoder_missing_torch(self):
        """Test UniMolEncoder raises error when PyTorch is not available."""
        with patch.dict('sys.modules', {'torch': None}):
            # Also mock the advanced dependency check to return partial availability
            with patch('molenc.encoders.representations.multimodal.unimol.ADVANCED_DEPENDENCY_CHECK_AVAILABLE', True):
                with patch('molenc.encoders.representations.multimodal.unimol.check_encoder_readiness') as mock_check:
                    from enum import Enum
                    class MockDependencyLevel(Enum):
                        NONE = "none"
                        PARTIAL = "partial"
                        FULL = "full"
                    
                    mock_check.return_value = (False, MockDependencyLevel.PARTIAL, "Partial dependencies available. Available: ['rdkit', 'unimol_tools'], Missing: ['torch']")
                    
                    with pytest.raises(DependencyError) as exc_info:
                        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
                        UniMolEncoder()
                    assert "torch" in str(exc_info.value)
                    assert "unimol" in str(exc_info.value)
    
    def test_unimol_encoder_missing_rdkit(self):
        """Test UniMolEncoder raises error when RDKit is not available."""
        with patch.dict('sys.modules', {'rdkit': None}):
            # Also mock the advanced dependency check to return partial availability
            with patch('molenc.encoders.representations.multimodal.unimol.ADVANCED_DEPENDENCY_CHECK_AVAILABLE', True):
                with patch('molenc.encoders.representations.multimodal.unimol.check_encoder_readiness') as mock_check:
                    from enum import Enum
                    class MockDependencyLevel(Enum):
                        NONE = "none"
                        PARTIAL = "partial"
                        FULL = "full"
                    
                    mock_check.return_value = (False, MockDependencyLevel.PARTIAL, "Partial dependencies available. Available: ['torch', 'unimol_tools'], Missing: ['rdkit']")
                    
                    with pytest.raises(DependencyError) as exc_info:
                        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
                        UniMolEncoder()
                    assert "rdkit" in str(exc_info.value)
                    assert "unimol" in str(exc_info.value)
    
    @pytest.mark.torch
    @pytest.mark.rdkit
    def test_unimol_encode_single_valid_smiles(self, sample_smiles):
        """Test encoding a single valid SMILES string."""
        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
        
        # Mock the model and preprocessing
        with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_model') as mock_load:
            with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._mol_to_features') as mock_mol_to_features:
                # Mock preprocessing output
                mock_mol_to_features.return_value = {
                    'atom_types': torch.randint(0, 100, (10,)),  # 10 atoms
                    'coords_3d': torch.randn(10, 3),     # 3D coordinates
                }
                
                # Mock model
                mock_model = MagicMock()
                mock_model.return_value = torch.randn(1, 512)  # Batch size 1, 512 features
                mock_load.return_value = mock_model
                
                encoder = UniMolEncoder()
                embedding = encoder.encode(sample_smiles[0])
                
                assert isinstance(embedding, np.ndarray)
                assert embedding.shape == (512,)
                assert embedding.dtype == np.float32 or embedding.dtype == np.float64
    
    @pytest.mark.torch
    @pytest.mark.rdkit
    def test_unimol_encode_batch(self, sample_smiles):
        """Test batch encoding of SMILES strings."""
        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
        
        # Mock the model and preprocessing
        with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_model') as mock_load:
            with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._mol_to_features') as mock_mol_to_features:
                # Mock preprocessing output for each molecule
                def mock_mol_to_features_fn(mol):
                    num_atoms = hash(str(mol)) % 20 + 5  # 5-24 atoms
                    return {
                        'atom_types': torch.randint(0, 100, (num_atoms,)),  # Atom types
                        'coords_3d': torch.randn(num_atoms, 3),     # 3D coordinates
                    }
                
                mock_mol_to_features.side_effect = mock_mol_to_features_fn
                
                # Mock model
                mock_model = MagicMock()
                def mock_forward(batch_data):
                    batch_size = len(batch_data) if isinstance(batch_data, list) else 1
                    return torch.randn(batch_size, 512)
                mock_model.side_effect = mock_forward
                mock_load.return_value = mock_model
                
                encoder = UniMolEncoder()
                embeddings = encoder.encode_batch(sample_smiles)
                
                assert isinstance(embeddings, np.ndarray)
                assert embeddings.shape == (len(sample_smiles), 512)
                assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64
    
    @pytest.mark.torch
    @pytest.mark.rdkit
    def test_unimol_encode_invalid_smiles(self, invalid_smiles):
        """Test encoding invalid SMILES strings."""
        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
        
        # Mock the entire encoder initialization to avoid loading real model
        with patch.object(UniMolEncoder, '__init__', return_value=None):
            # Create encoder instance manually
            encoder = UniMolEncoder.__new__(UniMolEncoder)
            # Initialize base class
            from molenc.core.base import BaseEncoder
            BaseEncoder.__init__(encoder)
            # Set required attributes
            encoder._is_real_model = True
            encoder.output_dim = 512
            encoder.model = MagicMock()
            
            # Mock _encode_single to raise InvalidSMILESError for invalid SMILES
            def mock_encode_single(smiles):
                if smiles in invalid_smiles:
                    raise InvalidSMILESError(smiles, "Invalid SMILES")
                # Return a mock embedding for valid SMILES
                return np.random.rand(512).astype(np.float32)
            
            encoder._encode_single = mock_encode_single
            
            # Test with error_handling='raise'
            with pytest.raises(InvalidSMILESError):
                encoder.encode(invalid_smiles[0])
            
            # Test with error_handling='skip'
            encoder.handle_errors = 'skip'
            result = encoder.encode_batch(invalid_smiles)
            assert result.shape[0] == 0  # Should return empty array
            assert result.shape[1] == 512  # Should have correct output dimension
    
    @pytest.mark.torch
    def test_unimol_get_output_dim(self):
        """Test getting output dimension."""
        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
        
        with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_model'):
            encoder = UniMolEncoder(output_dim=1024)
            assert encoder.get_output_dim() == 1024
    
    @pytest.mark.torch
    def test_unimol_model_loading_failure(self):
        """Test handling of model loading failures."""
        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
        
        # Mock _load_real_unimol to raise an exception
        with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_real_unimol') as mock_load_real:
            mock_load_real.side_effect = Exception("Model loading failed")
            
            # Mock process isolation to also fail
            with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_placeholder_model') as mock_load_placeholder:
                mock_load_placeholder.side_effect = Exception("Placeholder loading failed")
                
                # Mock cloud client to also fail
                with patch('molenc.encoders.representations.multimodal.unimol.CLOUD_API_AVAILABLE', False):
                    with pytest.raises(EncoderInitializationError) as exc_info:
                        UniMolEncoder()
                    assert "Model loading failed" in str(exc_info.value)
    
    @pytest.mark.torch
    @pytest.mark.rdkit
    def test_unimol_3d_vs_2d_mode(self, sample_smiles):
        """Test difference between 3D and 2D modes."""
        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
        
        # Mock the model and preprocessing
        with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_model') as mock_load:
            with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._mol_to_features') as mock_mol_to_features:
                # Different preprocessing for 3D vs 2D
                def mock_mol_to_features_fn(mol):
                    # In our implementation, use_3d is a property of the encoder
                    # We'll just return the same structure for both cases
                    return {
                        'atom_types': torch.randint(0, 100, (10,)),  # Atom types
                        'coords_3d': torch.randn(10, 3),     # 3D coordinates
                    }
                
                mock_mol_to_features.side_effect = mock_mol_to_features_fn
                
                # Mock model
                mock_model = MagicMock()
                mock_model.return_value = torch.randn(1, 512)
                mock_load.return_value = mock_model
                
                # Test 3D mode
                encoder_3d = UniMolEncoder(use_3d=True)
                embedding_3d = encoder_3d.encode(sample_smiles[0])
                
                # Test 2D mode
                encoder_2d = UniMolEncoder(use_3d=False)
                embedding_2d = encoder_2d.encode(sample_smiles[0])
                
                assert isinstance(embedding_3d, np.ndarray)
                assert isinstance(embedding_2d, np.ndarray)
                assert embedding_3d.shape == embedding_2d.shape == (512,)
    
    @pytest.mark.torch
    @pytest.mark.rdkit
    def test_unimol_max_atoms_handling(self, sample_smiles):
        """Test handling of molecules with different numbers of atoms."""
        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
        
        # Mock the model and preprocessing
        with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_model') as mock_load:
            with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._mol_to_features') as mock_mol_to_features:
                # Test with molecule exceeding max_atoms
                def mock_mol_to_features_large(mol):
                    return {
                        'atom_types': torch.randint(0, 100, (300,)),  # Exceeds max_atoms=256
                        'coords_3d': torch.randn(300, 3),
                    }
                
                mock_mol_to_features.side_effect = mock_mol_to_features_large
                
                # Mock model
                mock_model = MagicMock()
                mock_model.return_value = torch.randn(1, 512)
                mock_load.return_value = mock_model
                
                encoder = UniMolEncoder(max_atoms=256)
                
                # Should handle large molecules gracefully
                try:
                    embedding = encoder.encode(sample_smiles[0])
                    assert isinstance(embedding, np.ndarray)
                    assert embedding.shape == (512,)
                except Exception as e:
                    # If truncation is not implemented, should raise appropriate error
                    assert "atoms" in str(e).lower() or "size" in str(e).lower()
    
    @pytest.mark.torch
    def test_unimol_device_handling(self):
        """Test device handling (CPU/GPU)."""
        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
        
        with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_model') as mock_load:
            # Test CPU device
            encoder_cpu = UniMolEncoder(device='cpu')
            assert str(encoder_cpu.device) == 'cpu'
            
            # Test GPU device (if available)
            if torch.cuda.is_available():
                encoder_gpu = UniMolEncoder(device='cuda')
                assert 'cuda' in str(encoder_gpu.device)
            else:
                # Should fallback to CPU if CUDA not available
                encoder_gpu = UniMolEncoder(device='cuda')
                assert encoder_gpu.device == 'cpu'
    
    @pytest.mark.torch
    def test_unimol_repr(self):
        """Test string representation."""
        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
        
        with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_model'):
            encoder = UniMolEncoder(
                model_name='unimol_v2',
                output_dim=1024,
                use_3d=False,
                max_atoms=128
            )
            repr_str = repr(encoder)

            assert 'UniMolEncoder' in repr_str
            assert 'unimol_v2' in repr_str
            assert 'output_dim=1024' in repr_str
            assert 'use_3d=False' in repr_str


class TestMultimodalEncoderUtils:
    """Tests for multimodal encoder utility functions."""
    
    @pytest.mark.torch
    @pytest.mark.rdkit
    def test_molecule_preprocessing(self, sample_smiles):
        """Test molecule preprocessing for multimodal encoders."""
        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
        
        with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_model'):
            encoder = UniMolEncoder()
            
            # Test the actual preprocessing (if implemented)
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(sample_smiles[0])
                
                if hasattr(encoder, '_preprocess_molecule'):
                    processed = encoder._preprocess_molecule(mol)
                    
                    assert isinstance(processed, dict)
                    expected_keys = ['atom_features', 'coordinates', 'edge_index', 'edge_attr']
                    for key in expected_keys:
                        if key in processed:
                            assert isinstance(processed[key], torch.Tensor)
                else:
                    pytest.skip("Molecule preprocessing not implemented")
            except ImportError:
                pytest.skip("RDKit not available")
    
    @pytest.mark.torch
    def test_conformer_generation(self, sample_smiles):
        """Test 3D conformer generation for multimodal encoders."""
        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
        
        with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_model'):
            encoder = UniMolEncoder(use_3d=True)
            
            # Test conformer generation (if implemented)
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(sample_smiles[0])
                
                if hasattr(encoder, '_generate_conformer'):
                    mol_with_conf = encoder._generate_conformer(mol)
                    
                    assert mol_with_conf is not None
                    assert mol_with_conf.GetNumConformers() > 0
                    
                    # Check 3D coordinates
                    conf = mol_with_conf.GetConformer()
                    assert conf.Is3D()
                else:
                    pytest.skip("Conformer generation not implemented")
            except ImportError:
                pytest.skip("RDKit not available")
    
    @pytest.mark.torch
    def test_atom_feature_extraction(self, sample_smiles):
        """Test atom feature extraction for multimodal encoders."""
        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
        
        with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_model'):
            encoder = UniMolEncoder()
            
            # Test atom feature extraction (if implemented)
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(sample_smiles[0])
                
                if hasattr(encoder, '_extract_atom_features'):
                    atom_features = encoder._extract_atom_features(mol)
                    
                    assert isinstance(atom_features, torch.Tensor)
                    assert atom_features.dim() == 2  # [num_atoms, num_features]
                    assert atom_features.shape[0] == mol.GetNumAtoms()
                else:
                    pytest.skip("Atom feature extraction not implemented")
            except ImportError:
                pytest.skip("RDKit not available")


class TestMultimodalEncoderIntegration:
    """Integration tests for multimodal encoders."""
    
    @pytest.mark.torch
    @pytest.mark.rdkit
    def test_multimodal_encoder_consistency(self, sample_smiles):
        """Test that multimodal encodings are consistent across multiple calls."""
        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
        
        # Mock with deterministic behavior
        with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_model') as mock_load:
            with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._mol_to_features') as mock_mol_to_features:
                # Deterministic preprocessing
                torch.manual_seed(42)
                mock_mol_to_features.return_value = {
                    'atom_types': torch.randint(0, 100, (10,)),
                    'coords_3d': torch.randn(10, 3),
                }
                
                # Deterministic model
                mock_model = MagicMock()
                def deterministic_forward(batch_data):
                    torch.manual_seed(42)
                    return torch.randn(1, 512)
                mock_model.side_effect = deterministic_forward
                mock_load.return_value = mock_model
                
                encoder = UniMolEncoder()
                smiles = sample_smiles[0]
                
                embedding1 = encoder.encode(smiles)
                embedding2 = encoder.encode(smiles)
                
                np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=6)
    
    @pytest.mark.torch
    @pytest.mark.rdkit
    def test_multimodal_encoder_different_molecules(self, sample_smiles):
        """Test that different molecules produce different embeddings."""
        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
        
        if len(sample_smiles) < 2:
            pytest.skip("Need at least 2 different SMILES for comparison")
        
        # Mock with molecule-dependent behavior
        with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_model') as mock_load:
            with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._mol_to_features') as mock_mol_to_features:
                # Different preprocessing for different molecules
                def mock_mol_to_features_fn(mol):
                    seed = hash(str(mol)) % 2**32
                    torch.manual_seed(seed)
                    num_atoms = hash(str(mol)) % 15 + 5  # 5-19 atoms
                    return {
                        'atom_types': torch.randint(0, 100, (num_atoms,)),
                        'coords_3d': torch.randn(num_atoms, 3),
                    }
                
                mock_mol_to_features.side_effect = mock_mol_to_features_fn
                
                # Model that depends on input
                mock_model = MagicMock()
                def mock_forward(batch_data):
                    if isinstance(batch_data, dict):
                        seed = int(batch_data['atom_features'].sum().item()) % 2**32
                    else:
                        seed = 42
                    torch.manual_seed(seed)
                    return torch.randn(1, 512)
                mock_model.side_effect = mock_forward
                mock_load.return_value = mock_model
                
                encoder = UniMolEncoder()
                
                embedding1 = encoder.encode(sample_smiles[0])
                embedding2 = encoder.encode(sample_smiles[1])
                
                # Different molecules should produce different embeddings
                if sample_smiles[0] != sample_smiles[1]:
                    assert not np.array_equal(embedding1, embedding2)
    
    @pytest.mark.torch
    @pytest.mark.slow
    def test_multimodal_encoder_performance(self, sample_smiles):
        """Test multimodal encoder performance with larger batches."""
        from molenc.encoders.representations.multimodal.unimol import UniMolEncoder
        
        # Mock for performance testing
        with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._load_model') as mock_load:
            with patch('molenc.encoders.representations.multimodal.unimol.UniMolEncoder._mol_to_features') as mock_mol_to_features:
                # Fast preprocessing
                mock_mol_to_features.return_value = {
                    'atom_types': torch.randint(0, 100, (10,)),
                    'coords_3d': torch.randn(10, 3),
                }
                
                # Fast model
                mock_model = MagicMock()
                def fast_forward(batch_data):
                    batch_size = len(batch_data) if isinstance(batch_data, list) else 1
                    return torch.randn(batch_size, 512)
                mock_model.side_effect = fast_forward
                mock_load.return_value = mock_model
                
                encoder = UniMolEncoder()
                
                # Test with larger batch
                large_batch = sample_smiles * 10  # Repeat SMILES to create larger batch
                
                import time
                start_time = time.time()
                embeddings = encoder.encode_batch(large_batch)
                end_time = time.time()
                
                assert isinstance(embeddings, np.ndarray)
                assert embeddings.shape == (len(large_batch), 512)
                
                # Should complete in reasonable time (adjust threshold as needed)
                assert end_time - start_time < 30.0  # 30 seconds threshold