"""
MACCS Keys fingerprint encoder for molecular representation.
"""
from typing import List, Union, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys


class MACCSEncoder:
    """
    MACCS (Molecular ACCess System) keys fingerprint encoder.
    
    MACCS keys are a set of 166 structural features that capture
    important molecular characteristics for similarity searching and
    machine learning applications.
    """
    
    def __init__(self):
        """Initialize MACCS encoder."""
        self.n_bits = 167  # MACCS keys are 167 bits (including bit 0)
        self.name = "maccs"
        
    def _validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    
    def _smiles_to_maccs(self, smiles: str) -> Optional[np.ndarray]:
        """Convert SMILES to MACCS fingerprint."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Generate MACCS keys fingerprint
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        
        # Convert to numpy array
        fp_array = np.zeros(self.n_bits, dtype=np.float32)
        for i in range(self.n_bits):
            fp_array[i] = maccs_fp.GetBit(i)
            
        return fp_array
    
    def encode(self, smiles: Union[str, List[str]]) -> np.ndarray:
        """
        Encode SMILES strings to MACCS fingerprints.
        
        Args:
            smiles: Single SMILES string or list of SMILES strings
            
        Returns:
            Numpy array of shape (n_molecules, 167) containing MACCS fingerprints
            
        Raises:
            ValueError: If invalid SMILES are provided
        """
        # Convert single SMILES to list
        if isinstance(smiles, str):
            smiles_list = [smiles]
        else:
            smiles_list = smiles
            
        # Validate all SMILES
        valid_smiles = []
        invalid_indices = []
        
        for i, smi in enumerate(smiles_list):
            if self._validate_smiles(smi):
                valid_smiles.append(smi)
            else:
                invalid_indices.append(i)
        
        if invalid_indices:
            raise ValueError(f"Invalid SMILES at indices: {invalid_indices}")
        
        # Encode all valid SMILES
        fingerprints = []
        for smi in valid_smiles:
            fp = self._smiles_to_maccs(smi)
            if fp is None:
                raise ValueError(f"Failed to generate MACCS fingerprint for: {smi}")
            fingerprints.append(fp)
        
        return np.array(fingerprints, dtype=np.float32)
    
    def encode_single(self, smiles: str) -> np.ndarray:
        """
        Encode a single SMILES string to MACCS fingerprint.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Numpy array of shape (167,) containing MACCS fingerprint
        """
        return self.encode(smiles)[0]
    
    def get_info(self) -> dict:
        """Get encoder information."""
        return {
            "name": self.name,
            "description": "MACCS keys molecular fingerprint encoder",
            "n_bits": self.n_bits,
            "features": [
                "Structural key-based fingerprint",
                "167-bit binary fingerprint",
                "Based on MACCS (Molecular ACCess System) keys",
                "Captures important molecular substructures"
            ]
        }