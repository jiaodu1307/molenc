"""Base encoder class for all molecular encoders."""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Any, Dict
import numpy as np
from .exceptions import InvalidSMILESError

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    Chem = None  # type: ignore


class BaseEncoder(ABC):
    """Abstract base class for all molecular encoders.

    All encoder implementations should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, handle_errors: str = 'raise', error_handling: Optional[str] = None, **kwargs):
        """
        Initialize the base encoder.

        Args:
            handle_errors: How to handle invalid SMILES. Options: 'raise', 'skip', 'warn'
            error_handling: Alias for handle_errors (for backward compatibility)
            **kwargs: Additional parameters specific to the encoder
        """
        # Handle error_handling as alias for handle_errors
        if error_handling is not None:
            handle_errors = error_handling

        self.handle_errors = handle_errors
        self.config = kwargs

    @abstractmethod
    def _encode_single(self, smiles: str) -> np.ndarray:
        """
        Encode a single SMILES string to a vector.

        Args:
            smiles: A valid SMILES string

        Returns:
            numpy array representing the molecular encoding

        Raises:
            InvalidSMILESError: If the SMILES string is invalid
        """
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """
        Get the output dimension of the encoder.

        Returns:
            The dimension of the output vector
        """
        pass

    def validate_smiles(self, smiles: str) -> bool:
        """
        Validate a SMILES string using RDKit.

        Args:
            smiles: SMILES string to validate

        Returns:
            True if valid, False otherwise
        """
        if not HAS_RDKIT:
            # Basic validation without RDKit
            return isinstance(smiles, str) and len(smiles.strip()) > 0

        # Handle empty or whitespace-only strings
        if not smiles or not isinstance(smiles, str):
            return False

        try:
            mol = Chem.MolFromSmiles(smiles)
            # Additional check for empty molecules
            return mol is not None and mol.GetNumAtoms() > 0
        except Exception:
            return False

    def encode(self, smiles: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode SMILES string(s) to vector(s).

        Args:
            smiles: Single SMILES string or list of SMILES strings

        Returns:
            Encoded vector(s) as numpy array(s)
        """
        if isinstance(smiles, str):
            return self._encode_with_error_handling(smiles)
        elif isinstance(smiles, list):
            return self.encode_batch(smiles)
        else:
            raise TypeError("Input must be a string or list of strings")

    def encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Encode a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            2D numpy array where each row is an encoded vector
        """
        results: List[np.ndarray] = []
        for smiles in smiles_list:
            try:
                vector = self._encode_with_error_handling(smiles)
                if vector is not None:
                    results.append(vector)
            except Exception as e:
                if self.handle_errors == 'raise':
                    raise e
                elif self.handle_errors == 'warn':
                    print(f"Warning: Failed to encode {smiles}: {e}")
                # 'skip' case: just continue to next molecule

        if not results:
            # Return empty array with correct shape
            return np.empty((0, self.get_output_dim()), dtype=np.float32)

        # Stack results into 2D array
        return np.vstack(results)

    def _encode_with_error_handling(self, smiles: str) -> Optional[np.ndarray]:
        """
        Encode a single SMILES with error handling.

        Args:
            smiles: SMILES string to encode

        Returns:
            Encoded vector or None if encoding failed and errors are skipped
        """
        if not self.validate_smiles(smiles):
            error_msg = f"Invalid SMILES: {smiles}"
            if self.handle_errors == 'raise':
                raise InvalidSMILESError(error_msg)
            elif self.handle_errors == 'warn':
                print(f"Warning: {error_msg}")
                return None
            elif self.handle_errors == 'skip':
                return None

        return self._encode_single(smiles)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the encoder.

        Returns:
            Dictionary containing encoder configuration
        """
        return {
            'encoder_type': self.__class__.__name__,
            'handle_errors': self.handle_errors,
            'output_dim': self.get_output_dim(),
            **self.config
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output_dim={self.get_output_dim()})"
