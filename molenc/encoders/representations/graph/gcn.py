"""Graph Convolutional Network (GCN) encoder implementation."""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import Mol

from molenc.core.base import BaseEncoder
from molenc.core.registry import register_encoder
from molenc.core.exceptions import InvalidSMILESError


class GCNLayer(nn.Module):
    """Graph Convolutional Layer."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1) -> None:
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GCN layer.

        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]

        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Apply linear transformation
        x = self.linear(x)

        # Graph convolution: A * X * W
        x = torch.matmul(adj, x)

        # Apply activation and dropout
        x = F.relu(x)
        x = self.dropout(x)

        return x


class GCNModel(nn.Module):
    """Graph Convolutional Network model."""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 dropout: float = 0.1) -> None:
        super(GCNModel, self).__init__()

        self.layers = nn.ModuleList()

        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(GCNLayer(prev_dim, hidden_dim, dropout))
            prev_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GCN model.

        Args:
            x: Node features [num_nodes, input_dim]
            adj: Adjacency matrix [num_nodes, num_nodes]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Apply GCN layers
        for layer in self.layers:
            x = layer(x, adj)

        # Apply output layer
        x = self.output_layer(x)

        return x


@register_encoder('gcn')
class GCNEncoder(BaseEncoder):
    """Graph Convolutional Network encoder for molecules.

    This encoder converts molecules to graphs and uses GCN to learn
    molecular representations.
    """

    def __init__(self,
                 hidden_dims: List[int] = [64, 64],
                 output_dim: int = 128,
                 dropout: float = 0.1,
                 pooling: str = "mean",
                 device: Optional[str] = None,
                 **kwargs) -> None:
        """
        Initialize GCN encoder.

        Args:
            hidden_dims: List of hidden layer dimensions (default: [64, 64])
            output_dim: Output embedding dimension (default: 128)
            dropout: Dropout rate (default: 0.1)
            pooling: Graph pooling method ("mean", "max", "sum") (default: "mean")
            device: Device to run the model on ("cpu", "cuda", or None for auto)
            **kwargs: Additional parameters passed to BaseEncoder
        """
        super().__init__(**kwargs)

        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.pooling = pooling

        # Check dependencies
        try:
            import torch
            from rdkit import Chem
        except ImportError as e:
            from molenc.core.exceptions import DependencyError
            missing_lib = str(e).split("'")[1] if "'" in str(e) else "torch/rdkit"
            raise DependencyError(missing_lib, "gcn")

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        self._init_model()

    def _init_model(self) -> None:
        """
        Initialize GCN model.
        """
        # Define atom feature dimension
        # This is a simplified version - in practice, you would use more features
        self.atom_feature_dim = 9  # atomic number, degree, formal charge, etc.

        self.model = GCNModel(
            input_dim=self.atom_feature_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            dropout=self.dropout
        )

        self.model.to(self.device)
        self.model.eval()

    def _mol_to_graph(self, mol: Mol) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert RDKit molecule to graph representation.

        Args:
            mol: RDKit molecule object

        Returns:
            Tuple of (node_features, adjacency_matrix)
        """
        # Get atom features
        atom_features: List[List[float]] = []
        for atom in mol.GetAtoms():
            features = [
                float(atom.GetAtomicNum()),
                float(atom.GetDegree()),
                float(atom.GetFormalCharge()),
                float(int(atom.GetHybridization())),
                float(int(atom.GetIsAromatic())),
                float(atom.GetMass()),
                float(atom.GetTotalValence()),
                float(int(atom.IsInRing())),
                float(atom.GetTotalNumHs())
            ]
            atom_features.append(features)

        # Convert to tensor
        node_features = torch.tensor(atom_features, dtype=torch.float32)

        # Get adjacency matrix
        num_atoms = mol.GetNumAtoms()
        adj_matrix = torch.zeros((num_atoms, num_atoms), dtype=torch.float32)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adj_matrix[i, j] = 1.0
            adj_matrix[j, i] = 1.0

        # Add self-loops
        adj_matrix += torch.eye(num_atoms)

        # Normalize adjacency matrix (simple normalization)
        degree = adj_matrix.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # Avoid division by zero
        adj_matrix = adj_matrix / degree

        return node_features, adj_matrix

    def _pool_node_embeddings(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Pool node embeddings to get graph-level representation.

        Args:
            node_embeddings: Node embeddings [num_nodes, output_dim]

        Returns:
            Graph embedding [output_dim]
        """
        if self.pooling == "mean":
            return torch.mean(node_embeddings, dim=0)
        elif self.pooling == "max":
            return torch.max(node_embeddings, dim=0)[0]
        elif self.pooling == "sum":
            return torch.sum(node_embeddings, dim=0)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def _encode_single(self, smiles: str) -> np.ndarray:
        """
        Encode a single SMILES string using GCN.

        Args:
            smiles: SMILES string to encode

        Returns:
            GCN embedding as numpy array

        Raises:
            InvalidSMILESError: If SMILES is invalid
        """
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise InvalidSMILESError(smiles, "Could not parse SMILES")

            # Convert molecule to graph
            node_features, adj_matrix = self._mol_to_graph(mol)

            # Move to device
            node_features = node_features.to(self.device)
            adj_matrix = adj_matrix.to(self.device)

            # Get node embeddings
            with torch.no_grad():
                node_embeddings = self.model(node_features, adj_matrix)

            # Pool to get graph embedding
            graph_embedding = self._pool_node_embeddings(node_embeddings)

            # Convert to numpy
            embedding = graph_embedding.cpu().numpy()

            return embedding.astype(np.float32)

        except Exception as e:
            if isinstance(e, InvalidSMILESError):
                raise e
            raise InvalidSMILESError(smiles, f"GCN encoding failed: {str(e)}")

    def get_output_dim(self) -> int:
        """
        Get the output dimension of GCN embeddings.

        Returns:
            Output dimension
        """
        return self.output_dim

    def get_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration.

        Returns:
            Configuration dictionary
        """
        config: Dict[str, Any] = super().get_config()
        config.update({
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'pooling': self.pooling,
            'device': str(self.device)
        })
        return config

    def get_feature_names(self) -> List[str]:
        """
        Get feature names for the embedding dimensions.

        Returns:
            List of feature names
        """
        return [f"gcn_dim_{i}" for i in range(self.output_dim)]

    def __repr__(self) -> str:
        return (
            f"GCNEncoder(hidden_dims={self.hidden_dims}, "
            f"output_dim={self.output_dim}, pooling='{self.pooling}')"
        )
