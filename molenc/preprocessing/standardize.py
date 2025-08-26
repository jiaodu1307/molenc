"""SMILES standardization utilities."""

import logging
from typing import List, Tuple, Optional
from molenc.core.exceptions import InvalidSMILESError, DependencyError

try:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    Chem = None  # type: ignore
    rdMolStandardize = None  # type: ignore


class SMILESStandardizer:
    """SMILES standardization utility.

    This class provides methods to standardize SMILES strings using RDKit,
    including normalization, canonicalization, and cleanup operations.
    """

    def __init__(self,
                 remove_stereochemistry: bool = False,
                 neutralize: bool = True,
                 remove_salts: bool = True,
                 canonical: bool = True,
                 canonicalize: Optional[bool] = None) -> None:
        """
        Initialize SMILES standardizer.

        Args:
            remove_stereochemistry: Whether to remove stereochemistry information
            neutralize: Whether to neutralize charges
            remove_salts: Whether to remove salts and keep largest fragment
            canonical: Whether to canonicalize SMILES
            canonicalize: Alias for canonical (for backward compatibility)
        """
        # Check RDKit availability
        if not HAS_RDKIT:
            raise DependencyError(
                "RDKit is required for SMILES standardization. "
                "Install it with: conda install -c conda-forge rdkit"
            )

        # Handle canonicalize alias
        if canonicalize is not None:
            canonical = canonicalize

        self.remove_stereochemistry = remove_stereochemistry
        self.neutralize = neutralize
        self.remove_salts = remove_salts
        self.canonical = canonical

        # Initialize RDKit standardization tools
        self._init_standardization_tools()

    def _init_standardization_tools(self) -> None:
        """
        Initialize RDKit standardization tools.
        """
        # Normalizer for functional group standardization
        self.normalizer = rdMolStandardize.Normalizer()

        # Uncharger for neutralization
        if self.neutralize:
            self.uncharger = rdMolStandardize.Uncharger()

        # Fragment remover for salt removal
        if self.remove_salts:
            self.fragment_remover = rdMolStandardize.FragmentRemover()

    def standardize(self, smiles: str) -> str:
        """
        Standardize a single SMILES string.

        Args:
            smiles: Input SMILES string

        Returns:
            Standardized SMILES string

        Raises:
            InvalidSMILESError: If SMILES cannot be parsed or standardized
        """
        if not smiles or not isinstance(smiles, str):
            raise InvalidSMILESError(smiles, "Invalid SMILES format")

        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise InvalidSMILESError(smiles, "Could not parse SMILES")

            # Apply standardization steps
            mol = self._apply_standardization(mol)

            # Convert back to SMILES
            if self.canonical:
                standardized_smiles = Chem.MolToSmiles(mol, canonical=True)
            else:
                standardized_smiles = Chem.MolToSmiles(mol)

            return standardized_smiles

        except Exception as e:
            if isinstance(e, InvalidSMILESError):
                raise e
            raise InvalidSMILESError(smiles, f"Standardization failed: {str(e)}")

    def _apply_standardization(self, mol):
        """
        Apply standardization steps to a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            Standardized molecule
        """
        # Normalize functional groups
        mol = self.normalizer.normalize(mol)

        # Remove salts (keep largest fragment)
        if self.remove_salts:
            mol = self.fragment_remover.remove(mol)

        # Neutralize charges
        if self.neutralize:
            mol = self.uncharger.uncharge(mol)

        # Remove stereochemistry
        if self.remove_stereochemistry:
            Chem.RemoveStereochemistry(mol)

        # Sanitize molecule
        Chem.SanitizeMol(mol)

        return mol

    def standardize_batch(self, smiles_list: List[str],
                          skip_invalid: bool = True) -> Tuple[List[str], List[int]]:
        """
        Standardize a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings
            skip_invalid: Whether to skip invalid SMILES (True) or raise error (False)

        Returns:
            Tuple of (standardized_smiles_list, invalid_indices)
        """
        standardized: List[str] = []
        invalid_indices: List[int] = []

        for i, smiles in enumerate(smiles_list):
            try:
                std_smiles = self.standardize(smiles)
                standardized.append(std_smiles)
            except InvalidSMILESError as e:
                if skip_invalid:
                    invalid_indices.append(i)
                    logging.warning(
                        f"Skipping invalid SMILES at index {i}: {smiles} - {str(e)}")
                else:
                    raise e

        return standardized, invalid_indices

    def get_config(self) -> dict:
        """
        Get standardizer configuration.

        Returns:
            Configuration dictionary
        """
        return {
            'remove_stereochemistry': self.remove_stereochemistry,
            'neutralize': self.neutralize,
            'remove_salts': self.remove_salts,
            'canonical': self.canonical
        }

    def __repr__(self) -> str:
        return (
            f"SMILESStandardizer(remove_stereochemistry={self.remove_stereochemistry}, "
            f"neutralize={self.neutralize}, remove_salts={self.remove_salts}, "
            f"canonical={self.canonical})"
        )


def standardize_smiles(smiles: str, **kwargs) -> str:
    """
    Convenience function to standardize a single SMILES string.

    Args:
        smiles: SMILES string to standardize
        **kwargs: Arguments passed to SMILESStandardizer

    Returns:
        Standardized SMILES string
    """
    standardizer = SMILESStandardizer(**kwargs)
    return standardizer.standardize(smiles)


def standardize_smiles_list(smiles_list: List[str], **kwargs) -> Tuple[List[str], List[int]]:
    """
    Convenience function to standardize a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        **kwargs: Arguments passed to SMILESStandardizer

    Returns:
        Tuple of (standardized_smiles_list, invalid_indices)
    """
    standardizer = SMILESStandardizer(**kwargs)
    return standardizer.standardize_batch(smiles_list)
