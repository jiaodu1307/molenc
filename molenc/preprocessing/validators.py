"""SMILES validation utilities."""

import re
import logging
from typing import List, Tuple, Dict, Any, Optional
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from molenc.core.exceptions import InvalidSMILESError, DependencyError


class SMILESValidator:
    """SMILES validation utility.

    This class provides comprehensive validation for SMILES strings,
    including syntax checking, chemical validity, and structural analysis.
    """

    def __init__(self,
                 strict_mode: bool = False,
                 check_aromaticity: bool = True,
                 check_valence: bool = True,
                 max_atoms: Optional[int] = None,
                 allowed_elements: Optional[List[str]] = None):
        """
        Initialize SMILES validator.

        Args:
            strict_mode: Whether to apply strict validation rules
            check_aromaticity: Whether to validate aromaticity
            check_valence: Whether to validate valence
            max_atoms: Maximum number of atoms allowed (None for no limit)
            allowed_elements: List of allowed elements (None for all)
        """
        # Check RDKit availability
        try:
            import rdkit
        except ImportError:
            raise DependencyError("rdkit", "SMILESValidator")

        self.strict_mode = strict_mode
        self.check_aromaticity = check_aromaticity
        self.check_valence = check_valence
        self.max_atoms = max_atoms
        self.allowed_elements = set(allowed_elements) if allowed_elements else None

        # Common SMILES patterns
        self.valid_chars = set('()[]{}=#+\\/-@0123456789BCNOPSFIclnops')
        self.bracket_pairs = {'(': ')', '[': ']', '{': '}'}

        # Initialize validation statistics
        self.reset_stats()

    def reset_stats(self):
        """Reset validation statistics."""
        self.stats = {
            'total_validated': 0,
            'valid_count': 0,
            'invalid_count': 0,
            'error_types': {
                'syntax_error': 0,
                'rdkit_parse_error': 0,
                'aromaticity_error': 0,
                'valence_error': 0,
                'atom_count_error': 0,
                'element_error': 0,
                'other_error': 0
            }
        }

    def check_syntax(self, smiles: str) -> Tuple[bool, str]:
        """
        Check basic SMILES syntax.

        Args:
            smiles: SMILES string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not smiles or not isinstance(smiles, str):
            return False, "Empty or non-string SMILES"

        # Check for invalid characters
        if self.strict_mode:
            invalid_chars = set(smiles) - self.valid_chars
            if invalid_chars:
                return False, f"Invalid characters: {invalid_chars}"

        # Check bracket matching
        stack = []
        for char in smiles:
            if char in self.bracket_pairs:
                stack.append(char)
            elif char in self.bracket_pairs.values():
                if not stack:
                    return False, f"Unmatched closing bracket: {char}"
                opening = stack.pop()
                if self.bracket_pairs[opening] != char:
                    return False, f"Mismatched brackets: {opening} and {char}"

        if stack:
            return False, f"Unmatched opening brackets: {stack}"

        # Check for consecutive operators
        operators = set('=#+\\/-')
        prev_was_operator = False
        for char in smiles:
            if char in operators:
                if prev_was_operator:
                    return False, f"Consecutive operators found"
                prev_was_operator = True
            else:
                prev_was_operator = False

        return True, ""

    def check_rdkit_validity(self, smiles: str) -> Tuple[bool, str, Optional[object]]:
        """
        Check if SMILES can be parsed by RDKit.

        Args:
            smiles: SMILES string to validate

        Returns:
            Tuple of (is_valid, error_message, mol_object)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, "RDKit could not parse SMILES", None

            # Try to sanitize
            Chem.SanitizeMol(mol)

            return True, "", mol

        except Exception as e:
            return False, f"RDKit error: {str(e)}", None

    def check_structural_validity(self, mol) -> Tuple[bool, str]:
        """
        Check structural validity of the molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check atom count
            if self.max_atoms and mol.GetNumAtoms() > self.max_atoms:
                return False, f"Too many atoms: {mol.GetNumAtoms()} > {self.max_atoms}"

            # Check allowed elements
            if self.allowed_elements:
                mol_elements = set(atom.GetSymbol() for atom in mol.GetAtoms())
                forbidden = mol_elements - self.allowed_elements
                if forbidden:
                    return False, f"Forbidden elements: {forbidden}"

            # Check aromaticity
            if self.check_aromaticity:
                try:
                    Chem.rdmolops.SetAromaticity(mol)
                except Exception as e:
                    return False, f"Aromaticity error: {str(e)}"

            # Check valence
            if self.check_valence:
                for atom in mol.GetAtoms():
                    try:
                        valence = atom.GetTotalValence()
                        if valence < 0:
                            return False, f"Invalid valence for atom {atom.GetIdx()}"
                    except Exception as e:
                        return False, f"Valence error for atom {atom.GetIdx()}: {str(e)}"

            return True, ""

        except Exception as e:
            return False, f"Structural validation error: {str(e)}"

    def validate(self, smiles: str) -> Tuple[bool, str]:
        """
        Comprehensive validation of a SMILES string.

        Args:
            smiles: SMILES string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Update total validated count
        self.stats['total_validated'] += 1

        try:
            # Step 1: Syntax check
            syntax_valid, syntax_error = self.check_syntax(smiles)
            if not syntax_valid:
                self.stats['invalid_count'] += 1
                self.stats['error_types']['syntax_error'] += 1
                return False, f"Syntax error: {syntax_error}"

            # Step 2: RDKit parsing
            rdkit_valid, rdkit_error, mol = self.check_rdkit_validity(smiles)
            if not rdkit_valid:
                self.stats['invalid_count'] += 1
                self.stats['error_types']['rdkit_parse_error'] += 1
                return False, f"RDKit error: {rdkit_error}"

            # Step 3: Structural validation
            struct_valid, struct_error = self.check_structural_validity(mol)
            if not struct_valid:
                self.stats['invalid_count'] += 1
                # Categorize structural errors
                if 'aromaticity' in struct_error.lower():
                    self.stats['error_types']['aromaticity_error'] += 1
                elif 'valence' in struct_error.lower():
                    self.stats['error_types']['valence_error'] += 1
                elif 'atoms' in struct_error.lower():
                    self.stats['error_types']['atom_count_error'] += 1
                elif 'element' in struct_error.lower():
                    self.stats['error_types']['element_error'] += 1
                else:
                    self.stats['error_types']['other_error'] += 1
                return False, f"Structural error: {struct_error}"

            # All checks passed
            self.stats['valid_count'] += 1
            return True, ""

        except Exception as e:
            self.stats['invalid_count'] += 1
            self.stats['error_types']['other_error'] += 1
            return False, f"Unexpected error: {str(e)}"

    def validate_batch(self, smiles_list: List[str],
                       return_details: bool = False) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """
        Validate a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings
            return_details: Whether to return detailed validation results

        Returns:
            Tuple of (valid_smiles, invalid_indices, stats)
        """
        valid_smiles = []
        invalid_indices = []
        validation_details = [] if return_details else None

        # Reset stats for this batch
        self.reset_stats()

        for i, smiles in enumerate(smiles_list):
            self.stats['total_validated'] += 1
            is_valid, error_msg = self.validate(smiles)

            if is_valid:
                valid_smiles.append(smiles)
            else:
                invalid_indices.append(i)
                logging.debug(f"Invalid SMILES at index {i}: {smiles} - {error_msg}")

            if return_details:
                validation_details.append({
                    'smiles': smiles,
                    'is_valid': is_valid,
                    'error': error_msg if not is_valid else None
                })

        # Create stats summary
        stats = {
            'total': len(smiles_list),
            'valid': len(valid_smiles),
            'invalid': len(invalid_indices),
            'success_rate': len(valid_smiles) / len(smiles_list) if smiles_list else 0.0
        }

        return valid_smiles, invalid_indices, stats

    def get_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics.

        Returns:
            Dictionary with validation statistics
        """
        stats = self.stats.copy()
        if stats['total_validated'] > 0:
            stats['valid_rate'] = stats['valid_count'] / stats['total_validated']
            stats['invalid_rate'] = stats['invalid_count'] / stats['total_validated']
        else:
            stats['valid_rate'] = 0.0
            stats['invalid_rate'] = 0.0

        return stats

    def get_config(self) -> dict:
        """
        Get validator configuration.

        Returns:
            Configuration dictionary
        """
        return {
            'strict_mode': self.strict_mode,
            'check_aromaticity': self.check_aromaticity,
            'check_valence': self.check_valence,
            'max_atoms': self.max_atoms,
            'allowed_elements': list(self.allowed_elements) if self.allowed_elements else None
        }

    def __repr__(self) -> str:
        parts = [
            f"strict_mode={self.strict_mode}",
            f"check_aromaticity={self.check_aromaticity}",
            f"check_valence={self.check_valence}"
        ]

        if self.max_atoms is not None:
            parts.append(f"max_atoms={self.max_atoms}")

        if self.allowed_elements is not None:
            elements_list = sorted(list(self.allowed_elements))
            parts.append(f"allowed_elements={elements_list}")

        return f"SMILESValidator({', '.join(parts)})"


def validate_smiles(smiles: str, **kwargs) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to validate a single SMILES string.

    Args:
        smiles: SMILES string to validate
        **kwargs: Arguments passed to SMILESValidator

    Returns:
        Tuple of (is_valid, validation_results)
    """
    validator = SMILESValidator(**kwargs)
    return validator.validate(smiles)


def validate_smiles_list(smiles_list: List[str], **kwargs) -> Tuple[List[str], List[int], List[Dict]]:
    """
    Convenience function to validate a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        **kwargs: Arguments passed to SMILESValidator

    Returns:
        Tuple of (valid_smiles, invalid_indices, validation_details)
    """
    validator = SMILESValidator(**kwargs)
    return validator.validate_batch(smiles_list, return_details=True)
