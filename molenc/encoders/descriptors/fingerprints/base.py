"""Base fingerprint encoder class."""

import numpy as np
from abc import abstractmethod
from typing import Optional
from rdkit import Chem

from molenc.core.base import BaseEncoder
from molenc.core.exceptions import InvalidSMILESError
from molenc.core.encoder_mixins import SMILESValidationMixin, ParameterValidationMixin


class BaseFingerprintEncoder(BaseEncoder, SMILESValidationMixin, ParameterValidationMixin):
    """Base class for fingerprint encoders.
    
    This class provides common functionality for fingerprint-based encoders,
    including SMILES validation and error handling.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize base fingerprint encoder."""
        super().__init__(**kwargs)

    def _encode_single(self, smiles: str) -> np.ndarray:
        """
        Encode a single SMILES string to fingerprint.

        Args:
            smiles: SMILES string to encode

        Returns:
            Fingerprint as numpy array

        Raises:
            InvalidSMILESError: If SMILES is invalid
        """
        try:
            mol = self.validate_and_parse_smiles(smiles)
            return self._generate_fingerprint(mol)
        except Exception as e:
            if isinstance(e, InvalidSMILESError):
                raise e
            raise InvalidSMILESError(smiles, f"Fingerprint generation failed: {str(e)}")

    @abstractmethod
    def _generate_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """
        Generate fingerprint from RDKit molecule object.

        Args:
            mol: RDKit molecule object

        Returns:
            Fingerprint as numpy array
        """
        pass