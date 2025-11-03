"""MACCS keys fingerprint encoder implementation."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from typing import Dict

from molenc.core.registry import register_encoder
from molenc.core.exceptions import InvalidSMILESError
from molenc.core.dependency_utils import require_dependencies
from .base import BaseFingerprintEncoder


@register_encoder('maccs')
@require_dependencies(['rdkit'], 'MACCS')
class MACCSEncoder(BaseFingerprintEncoder):
    """MACCS keys fingerprint encoder.

    MACCS (Molecular ACCess System) keys are a set of 166 predefined
    structural keys that represent the presence or absence of specific
    molecular substructures.
    """

    # MACCS keys always have 166 bits
    MACCS_SIZE = 166

    def __init__(self, **kwargs) -> None:
        """
        Initialize MACCS keys encoder.

        Args:
            **kwargs: Additional parameters passed to BaseEncoder
        """
        super().__init__(**kwargs)

    def _generate_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """
        Generate MACCS keys fingerprint from RDKit molecule object.

        Args:
            mol: RDKit molecule object

        Returns:
            MACCS keys fingerprint as numpy array
        """
        # Generate MACCS keys fingerprint
        fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)

        # Convert to numpy array
        arr = np.zeros((self.MACCS_SIZE,), dtype=np.uint8)
        for i in range(self.MACCS_SIZE):
            arr[i] = fp[i]

        return np.array(arr, dtype=np.uint8)

    def get_output_dim(self) -> int:
        """
        Get the output dimension of MACCS keys fingerprint.

        Returns:
            Number of MACCS keys (always 166)
        """
        return self.MACCS_SIZE

    def get_feature_names(self) -> list:
        """
        Get feature names for the MACCS keys.

        Returns:
            List of MACCS key names
        """
        # MACCS keys have predefined meanings
        # This is a simplified version - in practice, you might want to include
        # the actual chemical meanings of each key
        return [f"maccs_key_{i}" for i in range(self.MACCS_SIZE)]

    def get_key_descriptions(self) -> Dict[int, str]:
        """
        Get descriptions of MACCS keys.

        Returns:
            Dictionary mapping key indices to their chemical meanings
        """
        # This is a subset of MACCS key descriptions
        # In a full implementation, you would include all 166 descriptions
        descriptions: Dict[int, str] = {
            0: "Isotope",
            1: "103 < atomic number < 256",
            2: "Group IVa,Va,VIa,VIIa",
            3: "Actinide",
            4: "Group IIIB,IVB,VB,VIB,VIIB,VIII",
            5: "Lanthanide",
            6: "Group IA,IIA",
            7: "4M ring",
            8: "5M ring",
            9: "6M ring",
            10: "7M ring",
            # ... (would continue for all 166 keys)
        }

        # Fill in remaining keys with generic descriptions
        for i in range(len(descriptions), self.MACCS_SIZE):
            descriptions[i] = f"MACCS key {i}"

        return descriptions

    def __repr__(self) -> str:
        return f"MACCSEncoder(output_dim={self.MACCS_SIZE})"
