"""Molecular filtering utilities."""

import logging
from typing import List, Tuple, Optional, Dict, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, FilterCatalog

from molenc.core.exceptions import DependencyError


class MolecularFilters:
    """Molecular filtering utility.

    This class provides methods to filter molecules based on various
    physicochemical properties and drug-likeness criteria.
    """

    def __init__(self,
                 mw_range: Tuple[float, float] = (0, 1000),
                 logp_range: Tuple[float, float] = (-10, 10),
                 hbd_range: Tuple[int, int] = (0, 20),
                 hba_range: Tuple[int, int] = (0, 20),
                 rotatable_bonds_range: Tuple[int, int] = (0, 50),
                 tpsa_range: Tuple[float, float] = (0, 300),
                 apply_lipinski: bool = False,
                 apply_pains: bool = False,
                 custom_filters: Optional[Dict[str, Any]] = None,
                 lipinski_rule: Optional[bool] = None,
                 pains_filter: Optional[bool] = None) -> None:
        """
        Initialize molecular filters.

        Args:
            mw_range: Molecular weight range (min, max)
            logp_range: LogP range (min, max)
            hbd_range: Hydrogen bond donor count range (min, max)
            hba_range: Hydrogen bond acceptor count range (min, max)
            rotatable_bonds_range: Rotatable bonds count range (min, max)
            tpsa_range: Topological polar surface area range (min, max)
            apply_lipinski: Whether to apply Lipinski's Rule of Five
            apply_pains: Whether to filter out PAINS (Pan Assay Interference Compounds)
            custom_filters: Custom filter functions
        """
        # Check RDKit availability
        try:
            import rdkit
        except ImportError:
            raise DependencyError("rdkit", "MolecularFilters")

        self.mw_range = mw_range
        self.logp_range = logp_range
        self.hbd_range = hbd_range
        self.hba_range = hba_range
        self.rotatable_bonds_range = rotatable_bonds_range
        self.tpsa_range = tpsa_range

        # Handle lipinski_rule as alias for apply_lipinski
        if lipinski_rule is not None:
            self.apply_lipinski = lipinski_rule
        else:
            self.apply_lipinski = apply_lipinski

        # Handle pains_filter as alias for apply_pains
        if pains_filter is not None:
            self.apply_pains = pains_filter
        else:
            self.apply_pains = apply_pains
        self.custom_filters = custom_filters or {}

        # Create aliases for backward compatibility
        self.lipinski_rule = self.apply_lipinski
        self.pains_filter = self.apply_pains

        # Initialize PAINS filters if needed
        if self.apply_pains:
            self._init_pains_filters()
        else:
            self.pains_patterns = None  # type: ignore

    def _init_pains_filters(self) -> None:
        """
        Initialize PAINS (Pan Assay Interference Compounds) filters.
        """
        try:
            # Create PAINS filter catalog
            params = FilterCatalog.FilterCatalogParams()
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
            self.pains_catalog = FilterCatalog.FilterCatalog(params)

            # Create patterns list for compatibility with tests
            # Each entry represents a PAINS pattern
            self.pains_patterns = []
            for i in range(self.pains_catalog.GetNumEntries()):
                entry = self.pains_catalog.GetEntry(i)
                # Use the entry itself as a pattern placeholder
                self.pains_patterns.append(entry)

        except ImportError:
            logging.warning("PAINS filters not available in this RDKit version")
            self.apply_pains = False
            self.pains_catalog = None
            self.pains_patterns = []

    def calculate_properties(self, smiles_or_mol) -> Optional[Dict[str, float]]:
        """
        Calculate molecular properties.

        Args:
            smiles_or_mol: SMILES string or RDKit molecule object

        Returns:
            Dictionary of molecular properties, or None if invalid SMILES
        """
        # Handle both SMILES strings and RDKit molecule objects
        if isinstance(smiles_or_mol, str):
            # Handle empty string as invalid
            if not smiles_or_mol.strip():
                return None
            mol = Chem.MolFromSmiles(smiles_or_mol)
            if mol is None:
                return None
            # Additional check for empty molecules
            if mol.GetNumAtoms() == 0:
                return None
        else:
            mol = smiles_or_mol

        try:
            properties: Dict[str, float] = {
                'mw': float(Descriptors.MolWt(mol)),
                'logp': float(Crippen.MolLogP(mol)),
                'hbd': float(Descriptors.NumHDonors(mol)),
                'hba': float(Descriptors.NumHAcceptors(mol)),
                'rotatable_bonds': float(Descriptors.NumRotatableBonds(mol)),
                'tpsa': float(Descriptors.TPSA(mol)),
                'num_atoms': float(mol.GetNumAtoms()),
                'num_rings': float(Descriptors.RingCount(mol)),
                'aromatic_rings': float(Descriptors.NumAromaticRings(mol))
            }

            return properties
        except Exception:
            return None

    def check_lipinski(self, properties: Dict[str, float]) -> bool:
        """
        Check Lipinski's Rule of Five.

        Args:
            properties: Dictionary of molecular properties

        Returns:
            True if molecule passes Lipinski's rule
        """
        violations = 0

        if properties['mw'] > 500:
            violations += 1
        if properties['logp'] > 5:
            violations += 1
        if properties['hbd'] > 5:
            violations += 1
        if properties['hba'] > 10:
            violations += 1

        # Allow up to 1 violation
        return violations <= 1

    def check_pains(self, mol) -> bool:
        """
        Check for PAINS (Pan Assay Interference Compounds).

        Args:
            mol: RDKit molecule object

        Returns:
            True if molecule does not contain PAINS
        """
        if not self.apply_pains or self.pains_catalog is None:
            return True

        return not self.pains_catalog.HasMatch(mol)

    def check_property_ranges(self, properties: Dict[str, float]) -> bool:
        """
        Check if molecular properties are within specified ranges.

        Args:
            properties: Dictionary of molecular properties

        Returns:
            True if all properties are within ranges
        """
        checks = [
            self.mw_range[0] <= properties['mw'] <= self.mw_range[1],
            self.logp_range[0] <= properties['logp'] <= self.logp_range[1],
            self.hbd_range[0] <= properties['hbd'] <= self.hbd_range[1],
            self.hba_range[0] <= properties['hba'] <= self.hba_range[1],
            self.rotatable_bonds_range[0] <= properties['rotatable_bonds'] <= self.rotatable_bonds_range[1],
            self.tpsa_range[0] <= properties['tpsa'] <= self.tpsa_range[1]
        ]

        return all(checks)

    def apply_custom_filters(self, mol, properties: Dict[str, float]) -> bool:
        """
        Apply custom filter functions.

        Args:
            mol: RDKit molecule object
            properties: Dictionary of molecular properties

        Returns:
            True if molecule passes all custom filters
        """
        for filter_name, filter_func in self.custom_filters.items():
            try:
                if not filter_func(mol, properties):
                    return False
            except Exception as e:
                logging.warning(f"Custom filter '{filter_name}' failed: {str(e)}")
                return False

        return True

    def passes_filters(self, smiles: str) -> bool:
        """
        Check if a molecule passes all filters.

        Args:
            smiles: SMILES string

        Returns:
            True if molecule passes all filters
        """
        passes, _ = self.filter_molecule(smiles)
        return passes

    def get_filter_reasons(self, smiles: str) -> List[str]:
        """
        Get reasons why a molecule failed filters.

        Args:
            smiles: SMILES string

        Returns:
            List of failure reasons (empty if molecule passes all filters)
        """
        passes, results = self.filter_molecule(smiles)

        if passes:
            return []

        reasons: List[str] = []

        # Check property range failures
        if not results.get('property_ranges', True):
            props = results.get('properties', {})

            # Check each property range
            if 'mw' in props:
                mw = props['mw']
                if mw < self.mw_range[0] or mw > self.mw_range[1]:
                    reasons.append(f"mw {mw:.2f} outside range {self.mw_range}")

            if 'logp' in props:
                logp = props['logp']
                if logp < self.logp_range[0] or logp > self.logp_range[1]:
                    reasons.append(f"logp {logp:.2f} outside range {self.logp_range}")

            if 'hbd' in props:
                hbd = props['hbd']
                if hbd < self.hbd_range[0] or hbd > self.hbd_range[1]:
                    reasons.append(f"hbd {hbd} outside range {self.hbd_range}")

            if 'hba' in props:
                hba = props['hba']
                if hba < self.hba_range[0] or hba > self.hba_range[1]:
                    reasons.append(f"hba {hba} outside range {self.hba_range}")

            if 'rotatable_bonds' in props:
                rotatable = props['rotatable_bonds']
                if rotatable < self.rotatable_bonds_range[0] or rotatable > self.rotatable_bonds_range[1]:
                    reasons.append(
                        f"rotatable {rotatable} outside range {self.rotatable_bonds_range}")

            if 'tpsa' in props:
                tpsa = props['tpsa']
                if tpsa < self.tpsa_range[0] or tpsa > self.tpsa_range[1]:
                    reasons.append(f"tpsa {tpsa:.2f} outside range {self.tpsa_range}")

        # Check Lipinski rule failure
        if self.apply_lipinski and not results.get('lipinski', True):
            reasons.append("Failed Lipinski rule of five")

        # Check PAINS filter failure
        if self.apply_pains and not results.get('pains', True):
            reasons.append(
                "Contains PAINS (Pan Assay Interference Compounds) substructure")

        # Check custom filter failure
        if not results.get('custom', True):
            reasons.append("Failed custom filters")

        return reasons

    def filter_molecule(self, smiles: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Filter a single molecule.

        Args:
            smiles: SMILES string

        Returns:
            Tuple of (passes_filter, filter_results)
        """
        try:
            # Handle empty string as invalid
            if not smiles.strip():
                return False, {'error': 'Invalid SMILES'}

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, {'error': 'Invalid SMILES'}

            # Additional check for empty molecules
            if mol.GetNumAtoms() == 0:
                return False, {'error': 'Invalid SMILES'}

            # Calculate properties
            properties = self.calculate_properties(mol)
            if properties is None:
                return False, {'error': 'Failed to calculate properties'}

            # Apply filters
            results: Dict[str, Any] = {
                'properties': properties,
                'property_ranges': self.check_property_ranges(properties),
                'lipinski': self.check_lipinski(properties) if self.apply_lipinski else True,
                'pains': self.check_pains(mol) if self.apply_pains else True,
                'custom': self.apply_custom_filters(mol, properties)
            }

            # Overall pass/fail
            passes = all([
                results['property_ranges'],
                results['lipinski'],
                results['pains'],
                results['custom']
            ])

            return passes, results

        except Exception as e:
            return False, {'error': str(e)}

    def filter_batch(self, smiles_list: List[str]) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """
        Filter a batch of molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Tuple of (passed_smiles, failed_indices, stats)
        """
        passed_smiles: List[str] = []
        failed_indices: List[int] = []
        failed_reasons: Dict[str, int] = {
            'invalid_smiles': 0,
            'property_ranges': 0,
            'lipinski': 0,
            'pains': 0,
            'custom': 0
        }

        for i, smiles in enumerate(smiles_list):
            passes, results = self.filter_molecule(smiles)

            if passes:
                passed_smiles.append(smiles)
            else:
                failed_indices.append(i)
                logging.debug(f"Rejected SMILES at index {i}: {smiles} - {results}")

                # Count failure reasons
                if 'error' in results:
                    failed_reasons['invalid_smiles'] += 1
                else:
                    if not results.get('property_ranges', True):
                        failed_reasons['property_ranges'] += 1
                    if not results.get('lipinski', True):
                        failed_reasons['lipinski'] += 1
                    if not results.get('pains', True):
                        failed_reasons['pains'] += 1
                    if not results.get('custom', True):
                        failed_reasons['custom'] += 1

        stats: Dict[str, Any] = {
            'total': len(smiles_list),
            'passed': len(passed_smiles),
            'failed': len(failed_indices),
            'filter_breakdown': failed_reasons
        }

        return passed_smiles, failed_indices, stats

    def get_filter_stats(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Get filtering statistics for a list of molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dictionary with filtering statistics
        """
        total = len(smiles_list)
        passed = 0
        failed_reasons: Dict[str, int] = {
            'invalid_smiles': 0,
            'property_ranges': 0,
            'lipinski': 0,
            'pains': 0,
            'custom': 0
        }

        for smiles in smiles_list:
            passes, results = self.filter_molecule(smiles)

            if passes:
                passed += 1
            else:
                if 'error' in results:
                    failed_reasons['invalid_smiles'] += 1
                else:
                    if not results.get('property_ranges', True):
                        failed_reasons['property_ranges'] += 1
                    if not results.get('lipinski', True):
                        failed_reasons['lipinski'] += 1
                    if not results.get('pains', True):
                        failed_reasons['pains'] += 1
                    if not results.get('custom', True):
                        failed_reasons['custom'] += 1

        return {
            'total': total,
            'passed': passed,
            'rejected': total - passed,
            'pass_rate': passed / total if total > 0 else 0,
            'failure_reasons': failed_reasons
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Get filter configuration.

        Returns:
            Configuration dictionary
        """
        return {
            'mw_range': self.mw_range,
            'logp_range': self.logp_range,
            'hbd_range': self.hbd_range,
            'hba_range': self.hba_range,
            'rotatable_bonds_range': self.rotatable_bonds_range,
            'tpsa_range': self.tpsa_range,
            'apply_lipinski': self.apply_lipinski,
            'apply_pains': self.apply_pains,
            'custom_filters': list(self.custom_filters.keys())
        }

    def __repr__(self) -> str:
        return (
            f"MolecularFilters(mw_range={self.mw_range}, logp_range={self.logp_range}, "
            f"lipinski_rule={self.apply_lipinski}, pains_filter={self.apply_pains})"
        )
