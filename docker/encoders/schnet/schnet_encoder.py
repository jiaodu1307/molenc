"""SchNet encoder implementation for 3D molecular representation learning."""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import List, Optional, Union, Dict, Any
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings

# Disable warnings for cleaner output
warnings.filterwarnings('ignore')


class SchNetEncoder:
    """
    SchNet encoder for 3D molecular representation learning.
    
    This is a simplified implementation that captures the core concepts of SchNet:
    - Continuous filter convolutions
    - Interaction blocks
    - Atom-wise representations
    
    Note: This is a demonstration implementation. For production use,
    consider using the full SchNetPack library.
    """
    
    def __init__(self, 
                 hidden_channels: int = 128,
                 num_filters: int = 128,
                 num_interactions: int = 6,
                 num_gaussians: int = 50,
                 cutoff: float = 10.0,
                 max_num_neighbors: int = 32,
                 device: str = 'cpu'):
        """
        Initialize SchNet encoder.
        
        Args:
            hidden_channels: Number of hidden channels
            num_filters: Number of filter channels
            num_interactions: Number of interaction blocks
            num_gaussians: Number of Gaussian basis functions
            cutoff: Cutoff distance for interactions
            max_num_neighbors: Maximum number of neighbors
            device: Computing device ('cpu' or 'cuda')
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model components
        self._initialize_model()
        
        logging.info(f"SchNet encoder initialized with {num_interactions} interaction blocks")
    
    def _initialize_model(self):
        """Initialize model components."""
        # Embedding layer for atomic numbers
        self.embedding = nn.Embedding(100, self.hidden_channels)  # Support up to element 99
        
        # Interaction blocks
        self.interactions = nn.ModuleList([
            InteractionBlock(self.hidden_channels, self.num_filters, self.num_gaussians)
            for _ in range(self.num_interactions)
        ])
        
        # Output layers
        self.lin1 = nn.Linear(self.hidden_channels, self.hidden_channels // 2)
        self.lin2 = nn.Linear(self.hidden_channels // 2, 1)  # Single output for representation
        
        # Gaussian basis for distance encoding
        self.distance_expansion = GaussianBasis(self.num_gaussians, self.cutoff)
    
    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """
        Forward pass through SchNet.
        
        Args:
            z: Atomic numbers [num_atoms]
            pos: Atomic positions [num_atoms, 3]
            batch: Batch indices [num_atoms]
            
        Returns:
            Molecular representations [num_molecules, hidden_channels]
        """
        # Embed atomic numbers
        h = self.embedding(z)
        
        # Compute distances and neighbors
        edge_index, edge_weight = self._compute_edges(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        
        # Apply interaction blocks
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        
        # Global pooling to get molecular representation
        if batch is None:
            # Single molecule case
            h_mol = h.mean(dim=0, keepdim=True)
        else:
            # Batch of molecules
            h_mol = self._global_mean_pool(h, batch)
        
        # Final linear layers for representation
        h_mol = torch.relu(self.lin1(h_mol))
        # Return hidden representation instead of property prediction
        return h_mol
    
    def _compute_edges(self, pos: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """Compute graph edges based on distances."""
        # Simplified neighbor computation based on cutoff distance
        num_atoms = pos.size(0)
        edge_index = []
        edge_weight = []
        
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i == j:
                    continue
                
                dist = torch.norm(pos[i] - pos[j])
                if dist < self.cutoff:
                    edge_index.append([i, j])
                    edge_weight.append(dist)
        
        if len(edge_index) == 0:
            # At least one self loop
            edge_index.append([0, 0])
            edge_weight.append(torch.tensor(0.0, device=self.device))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t()
        edge_weight = torch.stack(edge_weight)
        
        return edge_index, edge_weight
    
    def _global_mean_pool(self, h: torch.Tensor, batch: torch.Tensor):
        """Global mean pooling."""
        unique_batches = torch.unique(batch)
        h_mol = []
        
        for b in unique_batches:
            mask = batch == b
            h_mol.append(h[mask].mean(dim=0, keepdim=True))
        
        return torch.cat(h_mol, dim=0)
    
    def encode_smiles(self, smiles: str, max_conformers: int = 1) -> np.ndarray:
        """
        Encode a SMILES string to molecular representation.
        
        Args:
            smiles: SMILES string
            max_conformers: Maximum number of conformers to generate
            
        Returns:
            Molecular representation vector
        """
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D conformer
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            # If 3D embedding fails, try 2D with random coordinates
            AllChem.Compute2DCoords(mol)
            # Add random Z coordinates
            conf = mol.GetConformer()
            for i in range(conf.GetNumAtoms()):
                conf.SetAtomPosition(i, 
                    [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, np.random.random()])
        
        # Optimize geometry
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=50)
        except:
            pass  # Continue even if optimization fails
        
        # Extract atomic numbers and positions
        conf = mol.GetConformer()
        z = []
        pos = []
        
        for atom in mol.GetAtoms():
            z.append(atom.GetAtomicNum())
            pos_atom = conf.GetAtomPosition(atom.GetIdx())
            pos.append([pos_atom.x, pos_atom.y, pos_atom.z])
        
        z = torch.tensor(z, dtype=torch.long, device=self.device)
        pos = torch.tensor(pos, dtype=torch.float32, device=self.device)
        
        # Get molecular representation
        with torch.no_grad():
            representation = self.forward(z, pos)
        
        return representation.cpu().numpy().flatten()
    
    def encode_batch(self, smiles_list: List[str], max_conformers: int = 1) -> np.ndarray:
        """
        Encode a batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            max_conformers: Maximum number of conformers per molecule
            
        Returns:
            Array of molecular representations [num_molecules, hidden_channels]
        """
        representations = []
        
        for smiles in smiles_list:
            try:
                rep = self.encode_smiles(smiles, max_conformers)
                representations.append(rep)
            except Exception as e:
                logging.warning(f"Failed to encode {smiles}: {e}")
                # Return zero vector for failed molecules
                representations.append(np.zeros(self.get_output_dim()))
        
        return np.array(representations)
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.hidden_channels // 2


class InteractionBlock(nn.Module):
    """SchNet interaction block."""
    
    def __init__(self, hidden_channels: int, num_filters: int, num_gaussians: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, hidden_channels)
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
    
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: torch.Tensor, edge_attr: torch.Tensor):
        """Forward pass through interaction block."""
        # Continuous filter convolution
        W = self.mlp(edge_attr)
        
        # Message passing (simplified)
        h_new = h.clone()
        for i in range(h.size(0)):
            neighbors = edge_index[1][edge_index[0] == i]
            if len(neighbors) > 0:
                messages = W[edge_index[0] == i] * h[neighbors]
                h_new[i] = h_new[i] + messages.sum(dim=0)
        
        # Final linear transformation
        h_new = self.lin(h_new)
        return h_new


class GaussianBasis(nn.Module):
    """Gaussian basis expansion for distances."""
    
    def __init__(self, num_gaussians: int, cutoff: float):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.offsets = nn.Parameter(torch.linspace(0, cutoff, num_gaussians))
        self.widths = nn.Parameter(torch.ones(num_gaussians) * (cutoff / num_gaussians))
    
    def forward(self, distances: torch.Tensor):
        """Expand distances to Gaussian basis."""
        distances = distances.unsqueeze(-1)
        return torch.exp(-((distances - self.offsets) ** 2) / (2 * self.widths ** 2))


# Example usage and testing
if __name__ == "__main__":
    # Initialize encoder
    encoder = SchNetEncoder(hidden_channels=128, device='cpu')
    
    # Test with a simple molecule
    smiles = "CCO"  # Ethanol
    try:
        representation = encoder.encode_smiles(smiles)
        print(f"SchNet representation for {smiles}: shape={representation.shape}")
        print(f"First 10 dimensions: {representation[:10]}")
    except Exception as e:
        print(f"Error encoding {smiles}: {e}")
    
    # Test batch encoding
    smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
    try:
        batch_representations = encoder.encode_batch(smiles_list)
        print(f"Batch representations shape: {batch_representations.shape}")
    except Exception as e:
        print(f"Error in batch encoding: {e}")