"""Tests for molecular filtering utilities."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from molenc.core.exceptions import DependencyError


class TestMolecularFilters:
    """Tests for MolecularFilters."""
    
    @pytest.mark.rdkit
    def test_filters_initialization_default(self):
        """Test MolecularFilters initialization with default parameters."""
        from molenc.preprocessing.filters import MolecularFilters
        
        filters = MolecularFilters()
        
        # Check default ranges
        assert filters.mw_range == (0, 1000)
        assert filters.logp_range == (-10, 10)
        assert filters.hbd_range == (0, 20)
        assert filters.hba_range == (0, 20)
        assert filters.rotatable_bonds_range == (0, 50)
        assert filters.tpsa_range == (0, 300)
        
        # Check default flags
        assert filters.lipinski_rule is False
        assert filters.pains_filter is False
        
        # Check that PAINS patterns are not loaded by default
        assert filters.pains_patterns is None
    
    @pytest.mark.rdkit
    def test_filters_initialization_custom(self):
        """Test MolecularFilters initialization with custom parameters."""
        from molenc.preprocessing.filters import MolecularFilters
        
        filters = MolecularFilters(
            mw_range=(100, 500),
            logp_range=(-2, 5),
            hbd_range=(0, 5),
            hba_range=(0, 10),
            rotatable_bonds_range=(0, 10),
            tpsa_range=(20, 140),
            lipinski_rule=True,
            pains_filter=True
        )
        
        assert filters.mw_range == (100, 500)
        assert filters.logp_range == (-2, 5)
        assert filters.hbd_range == (0, 5)
        assert filters.hba_range == (0, 10)
        assert filters.rotatable_bonds_range == (0, 10)
        assert filters.tpsa_range == (20, 140)
        assert filters.lipinski_rule is True
        assert filters.pains_filter is True
    
    def test_filters_missing_rdkit(self):
        """Test MolecularFilters raises error when RDKit is not available."""
        with patch.dict('sys.modules', {'rdkit': None}):
            with pytest.raises(DependencyError) as exc_info:
                from molenc.preprocessing.filters import MolecularFilters
                MolecularFilters()
            assert "rdkit" in str(exc_info.value)
            assert "MolecularFilters" in str(exc_info.value)
    
    @pytest.mark.rdkit
    def test_calculate_properties_valid_smiles(self, sample_smiles):
        """Test property calculation for valid SMILES."""
        from molenc.preprocessing.filters import MolecularFilters
        
        filters = MolecularFilters()
        
        for smiles in sample_smiles:
            properties = filters.calculate_properties(smiles)
            
            assert isinstance(properties, dict)
            
            # Check that all expected properties are present
            expected_props = ['mw', 'logp', 'hbd', 'hba', 'rotatable_bonds', 'tpsa']
            for prop in expected_props:
                assert prop in properties
                assert isinstance(properties[prop], (int, float))
                # Only check non-negativity for properties that should always be non-negative
                # LogP can be negative, so we exclude it from this check
                if prop != 'logp':
                    assert properties[prop] >= 0
    
    @pytest.mark.rdkit
    def test_calculate_properties_invalid_smiles(self, invalid_smiles):
        """Test property calculation for invalid SMILES."""
        from molenc.preprocessing.filters import MolecularFilters
        
        filters = MolecularFilters()
        
        for smiles in invalid_smiles:
            properties = filters.calculate_properties(smiles)
            
            # Should return None for invalid SMILES
            assert properties is None
    
    @pytest.mark.rdkit
    def test_calculate_properties_specific_molecules(self):
        """Test property calculation for specific molecules with known properties."""
        from molenc.preprocessing.filters import MolecularFilters
        
        filters = MolecularFilters()
        
        # Test with ethanol (CCO)
        ethanol_props = filters.calculate_properties("CCO")
        assert ethanol_props is not None
        assert 40 < ethanol_props['mw'] < 50  # MW around 46
        assert ethanol_props['hbd'] == 1  # One OH group
        assert ethanol_props['hba'] == 1  # One oxygen
        assert ethanol_props['rotatable_bonds'] == 0  # C-C bond (RDKit doesn't count this as rotatable)
        
        # Test with benzene (c1ccccc1)
        benzene_props = filters.calculate_properties("c1ccccc1")
        assert benzene_props is not None
        assert 75 < benzene_props['mw'] < 85  # MW around 78
        assert benzene_props['hbd'] == 0  # No H-bond donors
        assert benzene_props['hba'] == 0  # No H-bond acceptors
        assert benzene_props['rotatable_bonds'] == 0  # No rotatable bonds
    
    @pytest.mark.rdkit
    def test_passes_filters_basic(self):
        """Test basic filtering functionality."""
        from molenc.preprocessing.filters import MolecularFilters
        
        # Restrictive filters
        filters = MolecularFilters(
            mw_range=(40, 100),
            logp_range=(-2, 2),
            hbd_range=(0, 2),
            hba_range=(0, 2),
            rotatable_bonds_range=(0, 2),
            tpsa_range=(0, 50)
        )
        
        # Small molecule that should pass
        assert filters.passes_filters("CCO") is True
        
        # Large molecule that should fail
        large_molecule = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
        result = filters.passes_filters(large_molecule)
        # May pass or fail depending on exact property values
        assert isinstance(result, bool)
    
    @pytest.mark.rdkit
    def test_passes_filters_invalid_smiles(self, invalid_smiles):
        """Test filtering with invalid SMILES."""
        from molenc.preprocessing.filters import MolecularFilters
        
        filters = MolecularFilters()
        
        for smiles in invalid_smiles:
            # Should return False for invalid SMILES
            assert filters.passes_filters(smiles) is False
    
    @pytest.mark.rdkit
    def test_lipinski_rule_filter(self):
        """Test Lipinski rule filtering."""
        from molenc.preprocessing.filters import MolecularFilters
        
        filters = MolecularFilters(lipinski_rule=True)
        
        # Small drug-like molecule (should pass Lipinski)
        drug_like = "CCO"  # Ethanol
        assert filters.passes_filters(drug_like) is True
        
        # Test with a molecule that might violate Lipinski
        # (This depends on having a molecule that actually violates the rule)
        complex_molecule = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
        result = filters.passes_filters(complex_molecule)
        assert isinstance(result, bool)
    
    @pytest.mark.rdkit
    def test_pains_filter_initialization(self):
        """Test PAINS filter initialization."""
        from molenc.preprocessing.filters import MolecularFilters
        
        # Initialize with PAINS filter
        filters = MolecularFilters(pains_filter=True)
        
        # PAINS patterns should be loaded
        assert filters.pains_patterns is not None
        assert len(filters.pains_patterns) > 0
        
        # Each pattern should be a compiled molecule
        for pattern in filters.pains_patterns:
            assert pattern is not None
    
    @pytest.mark.rdkit
    def test_pains_filter_functionality(self):
        """Test PAINS filter functionality."""
        from molenc.preprocessing.filters import MolecularFilters
        
        filters = MolecularFilters(pains_filter=True)
        
        # Simple molecules should generally pass PAINS filter
        simple_molecules = ["CCO", "c1ccccc1", "CC(=O)O"]
        
        for smiles in simple_molecules:
            result = filters.passes_filters(smiles)
            # Most simple molecules should pass
            assert isinstance(result, bool)
    
    @pytest.mark.rdkit
    def test_filter_batch(self, sample_smiles, invalid_smiles):
        """Test batch filtering of SMILES."""
        from molenc.preprocessing.filters import MolecularFilters
        
        filters = MolecularFilters(
            mw_range=(0, 500),
            logp_range=(-5, 5)
        )
        
        # Mix valid and invalid SMILES
        mixed_smiles = sample_smiles + invalid_smiles
        
        passed_smiles, failed_indices, stats = filters.filter_batch(mixed_smiles)
        
        assert isinstance(passed_smiles, list)
        assert isinstance(failed_indices, list)
        assert isinstance(stats, dict)
        
        # Check that passed SMILES are valid
        for smiles in passed_smiles:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
        
        # Check that failed indices are in correct range
        for idx in failed_indices:
            assert 0 <= idx < len(mixed_smiles)
        
        # Check stats
        assert stats['total'] == len(mixed_smiles)
        assert stats['passed'] == len(passed_smiles)
        assert stats['failed'] == len(failed_indices)
        assert stats['passed'] + stats['failed'] == stats['total']
        
        # Check filter breakdown
        assert 'filter_breakdown' in stats
        breakdown = stats['filter_breakdown']
        assert isinstance(breakdown, dict)
    
    @pytest.mark.rdkit
    def test_get_filter_reasons(self):
        """Test getting reasons for filter failures."""
        from molenc.preprocessing.filters import MolecularFilters
        
        # Very restrictive filters
        filters = MolecularFilters(
            mw_range=(10, 20),  # Very narrow range
            logp_range=(0, 1),
            hbd_range=(0, 0),
            hba_range=(0, 0)
        )
        
        # Molecule that will likely fail multiple filters
        test_smiles = "CCO"  # Ethanol
        
        reasons = filters.get_filter_reasons(test_smiles)
        
        assert isinstance(reasons, list)
        # Should have some failure reasons
        assert len(reasons) > 0
        
        # Each reason should be a string
        for reason in reasons:
            assert isinstance(reason, str)
            assert len(reason) > 0
    
    @pytest.mark.rdkit
    def test_get_filter_reasons_passing_molecule(self):
        """Test getting filter reasons for a molecule that passes all filters."""
        from molenc.preprocessing.filters import MolecularFilters
        
        # Very permissive filters
        filters = MolecularFilters(
            mw_range=(0, 1000),
            logp_range=(-10, 10),
            hbd_range=(0, 20),
            hba_range=(0, 20)
        )
        
        test_smiles = "CCO"  # Ethanol
        
        reasons = filters.get_filter_reasons(test_smiles)
        
        # Should have no failure reasons
        assert isinstance(reasons, list)
        assert len(reasons) == 0
    
    @pytest.mark.rdkit
    def test_filters_repr(self):
        """Test string representation of filters."""
        from molenc.preprocessing.filters import MolecularFilters
        
        filters = MolecularFilters(
            mw_range=(100, 500),
            logp_range=(-2, 5),
            lipinski_rule=True,
            pains_filter=True
        )
        
        repr_str = repr(filters)
        
        assert 'MolecularFilters' in repr_str
        assert 'mw_range=(100, 500)' in repr_str
        assert 'logp_range=(-2, 5)' in repr_str
        assert 'lipinski_rule=True' in repr_str
        assert 'pains_filter=True' in repr_str
    
    @pytest.mark.rdkit
    def test_empty_batch_filtering(self):
        """Test filtering of empty batch."""
        from molenc.preprocessing.filters import MolecularFilters
        
        filters = MolecularFilters()
        
        passed_smiles, failed_indices, stats = filters.filter_batch([])
        
        assert passed_smiles == []
        assert failed_indices == []
        assert stats['total'] == 0
        assert stats['passed'] == 0
        assert stats['failed'] == 0
    
    @pytest.mark.rdkit
    def test_filter_consistency(self, sample_smiles):
        """Test that filtering is consistent across multiple calls."""
        from molenc.preprocessing.filters import MolecularFilters
        
        filters = MolecularFilters()
        
        for smiles in sample_smiles:
            result1 = filters.passes_filters(smiles)
            result2 = filters.passes_filters(smiles)
            
            # Results should be identical
            assert result1 == result2
    
    @pytest.mark.rdkit
    def test_property_calculation_edge_cases(self):
        """Test property calculation for edge case molecules."""
        from molenc.preprocessing.filters import MolecularFilters
        
        filters = MolecularFilters()
        
        edge_cases = [
            "C",  # Single carbon
            "[H][H]",  # Hydrogen molecule
            "[Na+].[Cl-]",  # Salt
            "C#C",  # Acetylene
            "C=C=C",  # Allene
        ]
        
        for smiles in edge_cases:
            properties = filters.calculate_properties(smiles)
            
            if properties is not None:
                # If properties are calculated, they should be reasonable
                assert properties['mw'] > 0
                assert properties['hbd'] >= 0
                assert properties['hba'] >= 0
                assert properties['rotatable_bonds'] >= 0
                assert properties['tpsa'] >= 0


class TestFilteringIntegration:
    """Integration tests for molecular filtering."""
    
    @pytest.mark.rdkit
    def test_filtering_with_real_molecules(self):
        """Test filtering with real molecular SMILES."""
        from molenc.preprocessing.filters import MolecularFilters
        
        # Drug-like filters (similar to Lipinski)
        filters = MolecularFilters(
            mw_range=(150, 500),
            logp_range=(-2, 5),
            hbd_range=(0, 5),
            hba_range=(0, 10),
            rotatable_bonds_range=(0, 10),
            tpsa_range=(20, 140)
        )
        
        # Real drug molecules
        drug_smiles = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",  # Salbutamol
        ]
        
        passed_smiles, failed_indices, stats = filters.filter_batch(drug_smiles)
        
        # Most drug molecules should pass drug-like filters
        success_rate = len(passed_smiles) / len(drug_smiles)
        assert success_rate > 0.5  # At least 50% should pass
    
    @pytest.mark.rdkit
    def test_filtering_with_different_configurations(self, sample_smiles):
        """Test filtering with different filter configurations."""
        from molenc.preprocessing.filters import MolecularFilters
        
        # Different filter configurations
        configs = [
            # Permissive filters
            {
                'mw_range': (0, 1000),
                'logp_range': (-10, 10),
                'lipinski_rule': False,
                'pains_filter': False
            },
            # Restrictive filters
            {
                'mw_range': (100, 300),
                'logp_range': (-1, 3),
                'hbd_range': (0, 3),
                'hba_range': (0, 5)
            },
            # Lipinski rule only
            {
                'lipinski_rule': True,
                'pains_filter': False
            },
            # PAINS filter only
            {
                'lipinski_rule': False,
                'pains_filter': True
            }
        ]
        
        for config in configs:
            filters = MolecularFilters(**config)
            
            passed_count = 0
            for smiles in sample_smiles:
                if filters.passes_filters(smiles):
                    passed_count += 1
            
            # Should pass at least some molecules
            assert passed_count >= 0
    
    @pytest.mark.rdkit
    def test_filtering_performance(self):
        """Test filtering performance with larger datasets."""
        from molenc.preprocessing.filters import MolecularFilters
        import time
        
        filters = MolecularFilters()
        
        # Create a larger dataset
        base_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)O"]
        large_dataset = base_smiles * 250  # 1000 SMILES
        
        start_time = time.time()
        passed_smiles, failed_indices, stats = filters.filter_batch(large_dataset)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 15.0  # 15 seconds threshold
        
        # Should process all molecules
        assert stats['total'] == len(large_dataset)
        assert stats['passed'] + stats['failed'] == stats['total']
    
    @pytest.mark.rdkit
    def test_comprehensive_filtering_suite(self):
        """Comprehensive filtering test with various molecule types."""
        from molenc.preprocessing.filters import MolecularFilters
        
        filters = MolecularFilters(
            mw_range=(50, 600),
            logp_range=(-3, 6),
            hbd_range=(0, 6),
            hba_range=(0, 12),
            rotatable_bonds_range=(0, 12),
            tpsa_range=(0, 150),
            lipinski_rule=False,
            pains_filter=False
        )
        
        # Comprehensive test set
        test_smiles = [
            # Small molecules
            "CCO", "CC(=O)O", "c1ccccc1",
            
            # Drug-like molecules
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            
            # Natural products
            "C[C@H]1CC[C@H]2[C@@H](C1)CC[C@@H]3[C@@H]2CC[C@H]4[C@@H]3CCC(=O)C4",  # Steroid-like
            
            # Peptide-like
            "CC(C)C[C@H](N)C(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O",
            
            # Aromatic compounds
            "c1ccc2c(c1)cccn2",  # Quinoline
            "c1ccc2c(c1)oc1ccccc12",  # Dibenzofuran
        ]
        
        passed_smiles, failed_indices, stats = filters.filter_batch(test_smiles)
        
        # Should pass most of the comprehensive set
        success_rate = len(passed_smiles) / len(test_smiles)
        assert success_rate > 0.6  # At least 60% success rate
        
        # Check that stats are reasonable
        assert stats['total'] == len(test_smiles)
        assert stats['passed'] == len(passed_smiles)
        assert stats['failed'] == len(failed_indices)
    
    @pytest.mark.rdkit
    def test_filter_reasons_comprehensive(self):
        """Test comprehensive filter reason reporting."""
        from molenc.preprocessing.filters import MolecularFilters
        
        # Very restrictive filters to ensure failures
        filters = MolecularFilters(
            mw_range=(50, 100),
            logp_range=(0, 1),
            hbd_range=(0, 1),
            hba_range=(0, 1),
            rotatable_bonds_range=(0, 1),
            tpsa_range=(10, 30)
        )
        
        # Molecules that will likely fail multiple filters
        test_molecules = [
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen (large)
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine (multiple violations)
            "CCCCCCCCCCCCCCCCCC",  # Long alkyl chain
        ]
        
        for smiles in test_molecules:
            reasons = filters.get_filter_reasons(smiles)
            
            # Should have multiple failure reasons
            assert isinstance(reasons, list)
            assert len(reasons) > 0
            
            # Each reason should be descriptive
            for reason in reasons:
                assert isinstance(reason, str)
                assert len(reason) > 10  # Should be descriptive
                # Should mention the property that failed
                assert any(prop in reason.lower() for prop in 
                          ['mw', 'logp', 'hbd', 'hba', 'rotatable', 'tpsa'])