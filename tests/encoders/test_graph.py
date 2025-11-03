"""Fixed tests for graph-based encoders."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

from molenc.core.exceptions import InvalidSMILESError, DependencyError, EncoderInitializationError


class TestGCNEncoder:
    """Tests for GCNEncoder."""
    
    @pytest.mark.torch
    def test_gcn_encoder_initialization_default(self):
        """Test GCNEncoder initialization with default parameters."""
        from molenc.encoders.representations.graph.gcn import GCNEncoder
        
        encoder = GCNEncoder()
        assert encoder.hidden_dims == [64, 64]
        assert encoder.output_dim == 128  # Default output dim
        assert encoder.dropout == 0.1
        assert str(encoder.device) == "cpu" or "cuda" in str(encoder.device)
    
    @pytest.mark.torch
    def test_gcn_encoder_initialization_custom(self):
        """Test GCNEncoder initialization with custom parameters."""
        from molenc.encoders.representations.graph.gcn import GCNEncoder
        
        encoder = GCNEncoder(
            hidden_dims=[128, 256, 128],
            output_dim=256,
            dropout=0.2,
            device='cpu'
        )
        assert encoder.hidden_dims == [128, 256, 128]
        assert encoder.output_dim == 256
        assert encoder.dropout == 0.2
        assert str(encoder.device) == "cpu"
    
    def test_gcn_encoder_missing_torch(self):
        """Test GCNEncoder raises error when PyTorch is not available."""
        with patch.dict('sys.modules', {'torch': None}):
            with pytest.raises(DependencyError) as exc_info:
                from molenc.encoders.representations.graph.gcn import GCNEncoder
                GCNEncoder()
            assert "torch" in str(exc_info.value)
            assert "gcn" in str(exc_info.value)
    
    def test_gcn_encoder_missing_rdkit(self):
        """Test GCNEncoder raises error when RDKit is not available."""
        with patch.dict('sys.modules', {'rdkit': None}):
            with pytest.raises(DependencyError) as exc_info:
                from molenc.encoders.representations.graph.gcn import GCNEncoder
                GCNEncoder()
            assert "rdkit" in str(exc_info.value)
            assert "gcn" in str(exc_info.value)
    
    @pytest.mark.torch
    @pytest.mark.rdkit
    def test_gcn_encode_single_valid_smiles(self, sample_smiles):
        """Test encoding a single valid SMILES string."""
        from molenc.encoders.representations.graph.gcn import GCNEncoder
        
        # Mock the model and graph conversion
        with patch.object(GCNEncoder, '_init_model') as mock_init_model:
            with patch.object(GCNEncoder, '_mol_to_graph') as mock_graph:
                # Mock graph data
                mock_graph.return_value = (
                    torch.randn(10, 9),  # node_features: 10 atoms, 9 features each
                    torch.randint(0, 2, (10, 10)).float()  # adjacency_matrix: 10x10
                )
                
                # Create a real model for testing
                encoder = GCNEncoder()
                # Manually set the model attribute since _init_model is mocked
                mock_model = Mock()
                mock_model.eval = Mock()
                mock_model.to = Mock()
                encoder.model = mock_model
                
                # Mock the model's forward method
                mock_model.return_value = torch.randn(10, 128)  # 10 nodes, 128 features each
                
                embedding = encoder.encode(sample_smiles[0])
                
                assert isinstance(embedding, np.ndarray)
                assert embedding.shape == (128,)  # Should be pooled to single vector
                assert embedding.dtype == np.float32 or embedding.dtype == np.float64
    
    @pytest.mark.torch
    @pytest.mark.rdkit
    def test_gcn_encode_batch(self, sample_smiles):
        """Test batch encoding of SMILES strings."""
        from molenc.encoders.representations.graph.gcn import GCNEncoder
        
        # Mock the model and graph conversion
        with patch.object(GCNEncoder, '_init_model') as mock_init_model:
            with patch.object(GCNEncoder, '_mol_to_graph') as mock_graph:
                # Mock graph data for each molecule
                def mock_graph_fn(mol):
                    num_atoms = mol.GetNumAtoms() if mol else 10
                    return (
                        torch.randn(num_atoms, 9),
                        torch.randint(0, 2, (num_atoms, num_atoms)).float()
                    )
                
                mock_graph.side_effect = mock_graph_fn
                
                # Create encoder and mock model forward pass
                encoder = GCNEncoder()
                # Manually set the model attribute since _init_model is mocked
                mock_model = Mock()
                mock_model.eval = Mock()
                mock_model.to = Mock()
                encoder.model = mock_model
                
                def mock_forward_fn(x, adj):
                    return torch.randn(x.shape[0], 128)
                mock_model.side_effect = mock_forward_fn
                
                embeddings = encoder.encode_batch(sample_smiles)
                
                assert isinstance(embeddings, np.ndarray)
                assert embeddings.shape == (len(sample_smiles), 128)
                assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64
    
    @pytest.mark.torch
    @pytest.mark.rdkit
    def test_gcn_encode_invalid_smiles(self, invalid_smiles):
        """Test encoding invalid SMILES strings."""
        from molenc.encoders.representations.graph.gcn import GCNEncoder
        
        with patch.object(GCNEncoder, '_init_model') as mock_init_model:
            encoder = GCNEncoder()
            # Manually set the model attribute since _init_model is mocked
            mock_model = Mock()
            mock_model.eval = Mock()
            mock_model.to = Mock()
            encoder.model = mock_model
            
            # Test with error_handling='raise'
            with pytest.raises(InvalidSMILESError):
                encoder.encode(invalid_smiles[0])
            
            # Test with error_handling='skip'
            encoder_skip = GCNEncoder(error_handling='skip')
            # Manually set the model attribute since _init_model is mocked
            encoder_skip.model = mock_model
            result = encoder_skip.encode_batch(invalid_smiles)
            assert result.shape[0] == 0  # Should return empty array
    
    @pytest.mark.torch
    def test_gcn_get_output_dim(self):
        """Test getting output dimension."""
        from molenc.encoders.representations.graph.gcn import GCNEncoder
        
        with patch.object(GCNEncoder, '_init_model') as mock_init_model:
            encoder = GCNEncoder(output_dim=128)
            assert encoder.get_output_dim() == 128
    
    @pytest.mark.torch
    def test_gcn_model_building(self):
        """Test GCN model building."""
        from molenc.encoders.representations.graph.gcn import GCNModel
        
        # Test GCNModel directly
        model = GCNModel(
            input_dim=9,
            hidden_dims=[64, 128],
            output_dim=256,
            dropout=0.1
        )
        
        assert isinstance(model, nn.Module)
        assert len(model.layers) == 2  # Two hidden layers
        
        # Test forward pass
        x = torch.randn(10, 9)  # 10 nodes, 9 features
        adj = torch.randint(0, 2, (10, 10)).float()  # adjacency matrix
        
        output = model(x, adj)
        assert output.shape == (10, 256)  # 10 nodes, 256 output features
    
    @pytest.mark.torch
    def test_gcn_graph_pooling(self):
        """Test graph pooling strategies."""
        from molenc.encoders.representations.graph.gcn import GCNEncoder
        
        pooling_strategies = ['mean', 'max', 'sum']
        
        for strategy in pooling_strategies:
            with patch.object(GCNEncoder, '_init_model') as mock_init_model:
                # Manually set the model attribute since _init_model is mocked
                mock_model = Mock()
                mock_model.eval = Mock()
                mock_model.to = Mock()
                
                encoder = GCNEncoder(pooling=strategy)
                encoder.model = mock_model
                
                # Test the pooling method directly
                node_embeddings = torch.randn(10, 128)
                pooled = encoder._pool_node_embeddings(node_embeddings)
                
                assert isinstance(pooled, torch.Tensor)
                assert pooled.shape == (128,)
    
    @pytest.mark.torch
    def test_gcn_repr(self):
        """Test string representation."""
        from molenc.encoders.representations.graph.gcn import GCNEncoder
        
        with patch.object(GCNEncoder, '_init_model') as mock_init_model:
            encoder = GCNEncoder(
                hidden_dims=[128, 256],
                output_dim=512,
                dropout=0.2,
                pooling='mean'  # Add pooling explicitly to match __repr__
            )
            repr_str = repr(encoder)
            
            assert 'GCNEncoder' in repr_str
            assert '[128, 256]' in repr_str
            assert 'output_dim=512' in repr_str
            assert "pooling='mean'" in repr_str


class TestGraphEncoderUtils:
    """Tests for graph encoder utility functions."""
    
    @pytest.mark.torch
    @pytest.mark.rdkit
    def test_smiles_to_graph_conversion(self, sample_smiles):
        """Test SMILES to graph conversion."""
        from molenc.encoders.representations.graph.gcn import GCNEncoder
        
        with patch.object(GCNEncoder, '_init_model') as mock_init_model:
            encoder = GCNEncoder()
            # Manually set the model attribute since _init_model is mocked
            mock_model = Mock()
            mock_model.eval = Mock()
            mock_model.to = Mock()
            encoder.model = mock_model
            
            # Test the actual graph conversion (if implemented)
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(sample_smiles[0])
                node_features, adj_matrix = encoder._mol_to_graph(mol)
                
                assert isinstance(node_features, torch.Tensor)
                assert isinstance(adj_matrix, torch.Tensor)
                assert node_features.dim() == 2  # [num_nodes, num_features]
                assert adj_matrix.dim() == 2  # [num_nodes, num_nodes]
                assert node_features.shape[0] == adj_matrix.shape[0]  # Same number of nodes
                assert adj_matrix.shape[0] == adj_matrix.shape[1]  # Square matrix
            except NotImplementedError:
                pytest.skip("Graph conversion not implemented")
    
    @pytest.mark.torch
    def test_graph_pooling_functions(self):
        """Test different graph pooling functions."""
        from molenc.encoders.representations.graph.gcn import GCNEncoder
        
        with patch.object(GCNEncoder, '_init_model') as mock_init_model:
            # Manually set the model attribute since _init_model is mocked
            mock_model = Mock()
            mock_model.eval = Mock()
            mock_model.to = Mock()
            
            encoder = GCNEncoder()
            encoder.model = mock_model
            
            # Test pooling with mock node embeddings
            node_embeddings = torch.randn(10, 128)  # 10 nodes, 128 features
            
            # Test mean pooling
            pooled_mean = encoder._pool_node_embeddings(node_embeddings)
            assert pooled_mean.shape == (128,)


class TestGraphEncoderIntegration:
    """Integration tests for graph-based encoders."""
    
    @pytest.mark.torch
    @pytest.mark.rdkit
    def test_graph_encoder_consistency(self, sample_smiles):
        """Test that graph encodings are consistent across multiple calls."""
        from molenc.encoders.representations.graph.gcn import GCNEncoder
        
        # Mock with deterministic behavior
        with patch.object(GCNEncoder, '_init_model') as mock_init_model:
            with patch.object(GCNEncoder, '_mol_to_graph') as mock_graph:
                # Deterministic graph data
                torch.manual_seed(42)
                mock_graph.return_value = (
                    torch.randn(10, 9),
                    torch.randint(0, 2, (10, 10)).float()
                )
                
                # Deterministic model
                encoder = GCNEncoder()
                # Manually set the model attribute since _init_model is mocked
                mock_model = Mock()
                mock_model.eval = Mock()
                mock_model.to = Mock()
                encoder.model = mock_model
                
                def deterministic_forward(x, adj):
                    torch.manual_seed(42)
                    return torch.randn(x.shape[0], 128)
                mock_model.side_effect = deterministic_forward
                
                smiles = sample_smiles[0]
                
                embedding1 = encoder.encode(smiles)
                embedding2 = encoder.encode(smiles)
                
                np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=6)
    
    @pytest.mark.torch
    @pytest.mark.rdkit
    def test_graph_encoder_different_molecules(self, sample_smiles):
        """Test that different molecules produce different embeddings."""
        from molenc.encoders.representations.graph.gcn import GCNEncoder
        
        if len(sample_smiles) < 2:
            pytest.skip("Need at least 2 different SMILES for comparison")
        
        # Mock with molecule-dependent behavior
        with patch.object(GCNEncoder, '_init_model') as mock_init_model:
            with patch.object(GCNEncoder, '_mol_to_graph') as mock_graph:
                # Different graph data for different molecules
                def mock_graph_fn(mol):
                    seed = hash(mol.GetNumAtoms()) % 2**32
                    torch.manual_seed(seed)
                    num_atoms = mol.GetNumAtoms()
                    return (
                        torch.randn(num_atoms, 9),
                        torch.randint(0, 2, (num_atoms, num_atoms)).float()
                    )
                
                mock_graph.side_effect = mock_graph_fn
                
                # Model that depends on input
                encoder = GCNEncoder()
                # Manually set the model attribute since _init_model is mocked
                mock_model = Mock()
                mock_model.eval = Mock()
                mock_model.to = Mock()
                encoder.model = mock_model
                
                def mock_forward(x, adj):
                    seed = int(x.sum().item()) % 2**32
                    torch.manual_seed(seed)
                    return torch.randn(x.shape[0], 128)
                mock_model.side_effect = mock_forward
                
                embedding1 = encoder.encode(sample_smiles[0])
                embedding2 = encoder.encode(sample_smiles[1])
                
                # Different molecules should produce different embeddings
                if sample_smiles[0] != sample_smiles[1]:
                    assert not np.array_equal(embedding1, embedding2)
    
    @pytest.mark.torch
    def test_graph_encoder_edge_cases(self):
        """Test graph encoder edge cases."""
        from molenc.encoders.representations.graph.gcn import GCNEncoder
        
        with patch.object(GCNEncoder, '_init_model') as mock_init_model:
            with patch.object(GCNEncoder, '_mol_to_graph') as mock_graph:
                # Test single atom molecule
                mock_graph.return_value = (
                    torch.randn(1, 9),  # Single atom
                    torch.ones(1, 1)   # Self-loop
                )
                
                encoder = GCNEncoder()
                # Manually set the model attribute since _init_model is mocked
                mock_model = Mock()
                mock_model.eval = Mock()
                mock_model.to = Mock()
                encoder.model = mock_model
                mock_model.return_value = torch.randn(1, 128)
                
                from rdkit import Chem
                mol = Chem.MolFromSmiles("C")  # Methane
                embedding = encoder.encode("C")
                
                assert isinstance(embedding, np.ndarray)
                assert embedding.shape == (128,)