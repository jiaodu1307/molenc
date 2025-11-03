"""Morgan fingerprint encoder implementation."""

import numpy as np
from rdkit import Chem
from typing import Any

from molenc.core.registry import register_encoder
from molenc.core.exceptions import InvalidSMILESError
from molenc.core.dependency_utils import require_dependencies
from molenc.core.encoder_utils import EncoderUtils
from .base import BaseFingerprintEncoder


@register_encoder('morgan')
@require_dependencies(['rdkit'], 'Morgan')
class MorganEncoder(BaseFingerprintEncoder):
    """Morgan fingerprint encoder.

    Morgan fingerprints (also known as circular fingerprints) are based on
    the Morgan algorithm and are similar to ECFP (Extended Connectivity Fingerprints).
    """

    def __init__(self,
                 radius: int = 2,
                 n_bits: int = 2048,
                 use_features: bool = False,
                 use_chirality: bool = False,
                 use_bond_types: bool = True,
                 **kwargs) -> None:
        """
        Initialize Morgan fingerprint encoder.

        Args:
            radius: Radius of the fingerprint (default: 2)
            n_bits: Number of bits in the fingerprint (default: 2048)
            use_features: Whether to use feature-based fingerprints (default: False)
            use_chirality: Whether to include chirality information (default: False)
            use_bond_types: Whether to include bond type information (default: True)
            **kwargs: Additional parameters passed to BaseEncoder
        """
        # Validate parameters using mixin
        validated_params = self.validate_init_parameters(
            radius=radius,
            n_bits=n_bits
        )
        
        super().__init__(**kwargs)

        self.radius = validated_params['radius']
        self.n_bits = validated_params['n_bits']
        self.use_features = use_features
        self.use_chirality = use_chirality
        self.use_bond_types = use_bond_types

        # Create fingerprint generator using the new API
        try:
            from rdkit.Chem import rdFingerprintGenerator
            if self.use_features:
                self.fp_generator = rdFingerprintGenerator.GetMorganGenerator(
                    radius=self.radius,
                    fpSize=self.n_bits,
                    includeChirality=self.use_chirality,
                    useBondTypes=self.use_bond_types,
                    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
                )
            else:
                self.fp_generator = rdFingerprintGenerator.GetMorganGenerator(
                    radius=self.radius,
                    fpSize=self.n_bits,
                    includeChirality=self.use_chirality,
                    useBondTypes=self.use_bond_types
                )
        except ImportError:
            # Fallback for older RDKit versions (shouldn't happen with 2025.03.5)
            self.fp_generator = None  # type: ignore

    def _generate_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """
        Generate Morgan fingerprint from RDKit molecule object.

        Args:
            mol: RDKit molecule object

        Returns:
            Morgan fingerprint as numpy array
        """
        # Use the new fingerprint generator if available
        if self.fp_generator is not None:
            # Generate fingerprint using new API
            fp = self.fp_generator.GetFingerprint(mol)
            
            # Convert to numpy array
            arr = np.zeros((self.n_bits,), dtype=np.uint8)
            for bit_id in fp.GetOnBits():
                if bit_id < self.n_bits:  # Safety check
                    arr[bit_id] = 1
        else:
            # Fallback to old API (shouldn't be needed with RDKit 2025)
            from rdkit.Chem import rdMolDescriptors
            if self.use_features:
                # Feature-based Morgan fingerprint
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=self.radius,
                    nBits=self.n_bits,
                    useFeatures=True,
                    useChirality=self.use_chirality,
                    useBondTypes=self.use_bond_types
                )
            else:
                # Standard Morgan fingerprint
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=self.radius,
                    nBits=self.n_bits,
                    useChirality=self.use_chirality,
                    useBondTypes=self.use_bond_types
                )

            # Convert to numpy array
            arr = np.zeros((self.n_bits,), dtype=np.uint8)
            for i in range(self.n_bits):
                arr[i] = fp[i]

        return np.array(arr, dtype=np.uint8)

    def get_output_dim(self) -> int:
        """
        Get the output dimension of Morgan fingerprint.

        Returns:
            Number of bits in the fingerprint
        """
        return self.n_bits

    def get_config(self) -> dict:
        """
        Get encoder configuration.

        Returns:
            Configuration dictionary
        """
        config: dict = super().get_config()
        config.update({
            'radius': self.radius,
            'n_bits': self.n_bits,
            'use_features': self.use_features,
            'use_chirality': self.use_chirality,
            'use_bond_types': self.use_bond_types
        })
        return config

    def get_feature_names(self) -> list:
        """
        Get feature names for the fingerprint bits.

        Returns:
            List of feature names
        """
        return [f"morgan_bit_{i}" for i in range(self.n_bits)]

    def __repr__(self) -> str:
        return (
            f"MorganEncoder(radius={self.radius}, n_bits={self.n_bits}, "
            f"use_features={self.use_features}, use_chirality={self.use_chirality})"
        )
