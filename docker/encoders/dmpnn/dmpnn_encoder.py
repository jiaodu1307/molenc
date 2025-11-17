"""
D-MPNN (Directed Message Passing Neural Network) Encoder Implementation

This module implements a simplified D-MPNN encoder for molecular representation learning.
D-MPNN is particularly effective for molecular property prediction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from rdkit import Chem
from rdkit.Chem import rdchem
import networkx as nx
from collections import defaultdict


class DMPNNEncoder(nn.Module):
    """
    D-MPNN (Directed Message Passing Neural Network) Encoder
    
    This implementation is based on the paper:
    "Analyzing Learned Molecular Representations for Property Prediction"
    by Yang et al. (2019)
    """
    
    def __init__(
        self,
        node_dim: Optional[int] = None,
        edge_dim: Optional[int] = None,
        hidden_size: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        activation: str = 'relu',
        aggregation: str = 'mean'
    ):
        """
        Initialize D-MPNN encoder
        
        Args:
            node_dim: Dimension of atom features (optional, for API compatibility)
            edge_dim: Dimension of bond features (optional, for API compatibility)
            hidden_size: Hidden layer size
            depth: Number of message passing steps
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', 'leaky_relu')
            aggregation: Pooling method over atom representations ('mean', 'sum', 'max')
        """
        super(DMPNNEncoder, self).__init__()
        
        # Internal feature sizes derived from feature extractors
        # Actual feature dimensions based on extractors
        # Atom features: 100 (Z one-hot) + 1 (degree) + 1 (charge) + 1 (Hs)
        #               + 6 (hybridization) + 1 (aromatic) + 7 (ring flags) = 117
        # Bond features: 4 (bond type) + 1 (conjugated) + 1 (ring) + 4 (stereo) = 10
        self.node_features = 117
        self.edge_features = 10
        
        # Accept API-compatible params but do not override internal feature sizes
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.aggregation = aggregation
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Input layers
        self.W_i = nn.Linear(self.node_features + self.edge_features, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(self.node_features + hidden_size, hidden_size)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer
        self.W_out = nn.Linear(hidden_size, hidden_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _mol_to_graph(self, mol: Chem.Mol) -> Dict:
        """
        Convert RDKit molecule to graph representation
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary containing graph information
        """
        if mol is None:
            raise ValueError("Invalid molecule")
        
        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = self._get_atom_features(atom)
            atom_features.append(features)
        
        atom_features = torch.tensor(atom_features, dtype=torch.float32)
        
        # Get bond features and edge information
        edge_list = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions for undirected graph
            edge_list.extend([[i, j], [j, i]])
            
            # Get bond features
            bond_feat = self._get_bond_features(bond)
            edge_features.extend([bond_feat, bond_feat])  # Same features for both directions
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        
        return {
            'atom_features': atom_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds()
        }
    
    def _get_atom_features(self, atom: rdchem.Atom) -> List[float]:
        """Extract atom features"""
        features = []
        
        # Atomic number (one-hot)
        atomic_num = atom.GetAtomicNum()
        features.extend(self._one_hot(atomic_num, list(range(1, 101))))
        
        # Degree
        features.append(atom.GetDegree())
        
        # Formal charge
        features.append(atom.GetFormalCharge())
        
        # Number of hydrogens
        features.append(atom.GetTotalNumHs())
        
        # Hybridization
        hybridization = atom.GetHybridization()
        features.extend(self._one_hot(hybridization, [1, 2, 3, 4, 5, 6]))
        
        # Aromaticity
        features.append(1 if atom.GetIsAromatic() else 0)
        
        # Ring information
        features.append(1 if atom.IsInRing() else 0)
        features.append(1 if atom.IsInRingSize(3) else 0)
        features.append(1 if atom.IsInRingSize(4) else 0)
        features.append(1 if atom.IsInRingSize(5) else 0)
        features.append(1 if atom.IsInRingSize(6) else 0)
        features.append(1 if atom.IsInRingSize(7) else 0)
        features.append(1 if atom.IsInRingSize(8) else 0)
        
        return features
    
    def _get_bond_features(self, bond: rdchem.Bond) -> List[float]:
        """Extract bond features"""
        features = []
        
        # Bond type
        bond_type = bond.GetBondType()
        features.extend(self._one_hot(bond_type, [1, 2, 3, 12]))
        
        # Conjugation
        features.append(1 if bond.GetIsConjugated() else 0)
        
        # Ring information
        features.append(1 if bond.IsInRing() else 0)
        
        # Stereo
        stereo = bond.GetStereo()
        features.extend(self._one_hot(stereo, [0, 1, 2, 3]))
        
        return features
    
    def _one_hot(self, value, choices):
        """Create one-hot encoding"""
        encoding = [0] * len(choices)
        if value in choices:
            encoding[choices.index(value)] = 1
        return encoding
    
    def forward(self, graph: Dict) -> torch.Tensor:
        """
        Forward pass of D-MPNN
        
        Args:
            graph: Graph dictionary from _mol_to_graph
            
        Returns:
            Molecular representation tensor
        """
        atom_features = graph['atom_features']
        edge_index = graph['edge_index']
        edge_features = graph['edge_features']
        num_atoms = graph['num_atoms']
        
        # Initialize messages
        messages = torch.zeros(edge_index.size(1), self.hidden_size)
        
        # Initial message computation
        for step in range(self.depth):
            # Message passing
            new_messages = torch.zeros_like(messages)
            
            for edge_idx in range(edge_index.size(1)):
                source_atom = edge_index[0, edge_idx]
                target_atom = edge_index[1, edge_idx]
                
                # Get neighboring edges (excluding reverse edge)
                neighbor_edges = []
                for neighbor_idx in range(edge_index.size(1)):
                    if neighbor_idx != edge_idx:  # Exclude self-loop
                        if edge_index[0, neighbor_idx] == target_atom:
                            neighbor_edges.append(neighbor_idx)
                
                # Aggregate messages from neighbors
                if neighbor_edges:
                    neighbor_messages = messages[neighbor_edges]
                    aggregated = torch.sum(neighbor_messages, dim=0)
                    
                    # Combine with edge features
                    edge_feat = edge_features[edge_idx]
                    combined = torch.cat([atom_features[target_atom], edge_feat])
                    
                    # Update message
                    message_input = self.W_i(combined) + self.W_h(aggregated)
                    new_messages[edge_idx] = self.activation(message_input)
                else:
                    # Initial message for leaf edges
                    edge_feat = edge_features[edge_idx]
                    combined = torch.cat([atom_features[target_atom], edge_feat])
                    new_messages[edge_idx] = self.activation(self.W_i(combined))
            
            messages = self.dropout_layer(new_messages)
        
        # Readout phase
        # Global pooling (mean)
        atom_representations = torch.zeros(num_atoms, self.hidden_size)
        
        for atom_idx in range(num_atoms):
            # Aggregate incoming messages
            incoming_edges = []
            for edge_idx in range(edge_index.size(1)):
                if edge_index[1, edge_idx] == atom_idx:
                    incoming_edges.append(edge_idx)
            
            if incoming_edges:
                incoming_messages = messages[incoming_edges]
                message_agg = torch.sum(incoming_messages, dim=0)
                
                # Combine with atom features
                combined = torch.cat([atom_features[atom_idx], message_agg])
                atom_representations[atom_idx] = self.activation(self.W_o(combined))
        
        # Global pooling according to aggregation method
        if self.aggregation == 'sum':
            mol_representation = torch.sum(atom_representations, dim=0)
        elif self.aggregation == 'max':
            mol_representation, _ = torch.max(atom_representations, dim=0)
        else:
            mol_representation = torch.mean(atom_representations, dim=0)
        
        # Final output layer
        output = self.activation(self.W_out(mol_representation))
        
        return output
    
    def encode_smiles(self, smiles: str) -> np.ndarray:
        """
        Encode a SMILES string to molecular representation
        
        Args:
            smiles: SMILES string
            
        Returns:
            Molecular embedding as numpy array
        """
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Convert to graph
        graph = self._mol_to_graph(mol)
        
        # Get molecular representation
        with torch.no_grad():
            representation = self.forward(graph)
            return representation.numpy()
    
    def encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Encode a batch of SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Batch of molecular embeddings as numpy array
        """
        representations = []
        
        for smiles in smiles_list:
            try:
                rep = self.encode_smiles(smiles)
                representations.append(rep)
            except Exception as e:
                print(f"Warning: Failed to encode {smiles}: {e}")
                # Use zero vector for failed molecules
                representations.append(np.zeros(self.hidden_size))
        
        return np.array(representations)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the molecular embeddings"""
        return self.hidden_size


# Example usage and testing
if __name__ == "__main__":
    # Create encoder
    encoder = DMPNNEncoder(
        node_features=133,
        edge_features=14,
        hidden_size=300,
        depth=3,
        dropout=0.0
    )
    
    # Test with some molecules
    test_smiles = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
    ]
    
    print("D-MPNN Encoder Test")
    print("=" * 40)
    
    # Single encoding
    for smiles in test_smiles:
        try:
            embedding = encoder.encode_smiles(smiles)
            print(f"SMILES: {smiles}")
            print(f"Embedding shape: {embedding.shape}")
            print(f"First 5 values: {embedding[:5]}")
            print()
        except Exception as e:
            print(f"Error encoding {smiles}: {e}")
    
    # Batch encoding
    print("Batch encoding test:")
    batch_embeddings = encoder.encode_batch(test_smiles)
    print(f"Batch shape: {batch_embeddings.shape}")
    print(f"Batch encoding successful!")