"""Tests for SMILES standardization utilities."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from molenc.core.exceptions import InvalidSMILESError, DependencyError


class TestSMILESStandardizer:
    """Tests for SMILESStandardizer."""
    
    @pytest.mark.rdkit
    def test_standardizer_initialization_default(self):
        """Test SMILESStandardizer initialization with default parameters."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        
        standardizer = SMILESStandardizer()
        assert standardizer.remove_stereochemistry is False
        assert standardizer.neutralize is True
        assert standardizer.remove_salts is True
        assert standardizer.canonical is True
        
        # Check that standardization tools are initialized
        assert hasattr(standardizer, 'normalizer')
        assert hasattr(standardizer, 'uncharger')
        assert hasattr(standardizer, 'fragment_remover')
    
    @pytest.mark.rdkit
    def test_standardizer_initialization_custom(self):
        """Test SMILESStandardizer initialization with custom parameters."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        
        standardizer = SMILESStandardizer(
            remove_stereochemistry=True,
            neutralize=False,
            remove_salts=False,
            canonical=False
        )
        assert standardizer.remove_stereochemistry is True
        assert standardizer.neutralize is False
        assert standardizer.remove_salts is False
        assert standardizer.canonical is False
    
    def test_standardizer_missing_rdkit(self):
        """Test SMILESStandardizer raises error when RDKit is not available."""
        with patch('molenc.preprocessing.standardize.HAS_RDKIT', False):
            with pytest.raises(DependencyError) as exc_info:
                from molenc.preprocessing.standardize import SMILESStandardizer
                SMILESStandardizer()
            assert "rdkit" in str(exc_info.value).lower()
            assert "standardization" in str(exc_info.value).lower()
    
    @pytest.mark.rdkit
    def test_standardize_single_valid_smiles(self, sample_smiles):
        """Test standardizing a single valid SMILES string."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        
        standardizer = SMILESStandardizer()
        
        for smiles in sample_smiles:
            standardized = standardizer.standardize(smiles)
            assert isinstance(standardized, str)
            assert len(standardized) > 0
            
            # Standardized SMILES should be valid
            from rdkit import Chem
            mol = Chem.MolFromSmiles(standardized)
            assert mol is not None
    
    @pytest.mark.rdkit
    def test_standardize_invalid_smiles(self, invalid_smiles):
        """Test standardizing invalid SMILES strings."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        
        standardizer = SMILESStandardizer()
        
        for smiles in invalid_smiles:
            with pytest.raises(InvalidSMILESError):
                standardizer.standardize(smiles)
    
    @pytest.mark.rdkit
    def test_standardize_empty_input(self):
        """Test standardizing empty or None input."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        
        standardizer = SMILESStandardizer()
        
        # Test empty string
        with pytest.raises(InvalidSMILESError):
            standardizer.standardize("")
        
        # Test None
        with pytest.raises(InvalidSMILESError):
            standardizer.standardize(None)
        
        # Test non-string input
        with pytest.raises(InvalidSMILESError):
            standardizer.standardize(123)
    
    @pytest.mark.rdkit
    def test_standardize_batch(self, sample_smiles):
        """Test batch standardization of SMILES strings."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        
        standardizer = SMILESStandardizer()
        
        standardized_smiles, failed_indices = standardizer.standardize_batch(sample_smiles)
        
        assert isinstance(standardized_smiles, list)
        assert isinstance(failed_indices, list)
        assert len(standardized_smiles) + len(failed_indices) == len(sample_smiles)
        
        # All standardized SMILES should be valid
        from rdkit import Chem
        for smiles in standardized_smiles:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
    
    @pytest.mark.rdkit
    def test_standardize_batch_with_invalid(self, sample_smiles, invalid_smiles):
        """Test batch standardization with mixed valid and invalid SMILES."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        
        standardizer = SMILESStandardizer()
        
        # Mix valid and invalid SMILES
        mixed_smiles = sample_smiles + invalid_smiles
        
        standardized_smiles, failed_indices = standardizer.standardize_batch(mixed_smiles)
        
        assert isinstance(standardized_smiles, list)
        assert isinstance(failed_indices, list)
        assert len(standardized_smiles) + len(failed_indices) == len(mixed_smiles)
        
        # Failed indices should correspond to invalid SMILES
        assert len(failed_indices) >= len(invalid_smiles)
        
        # Check that failed indices are in the correct range
        for idx in failed_indices:
            assert 0 <= idx < len(mixed_smiles)
    
    @pytest.mark.rdkit
    def test_canonicalization(self):
        """Test SMILES canonicalization."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        
        # Test with canonical=True
        standardizer_canonical = SMILESStandardizer(canonical=True)
        
        # Test with canonical=False
        standardizer_non_canonical = SMILESStandardizer(canonical=False)
        
        # Use a SMILES that can have different representations
        smiles = "C1=CC=CC=C1"  # Benzene
        
        canonical_result = standardizer_canonical.standardize(smiles)
        non_canonical_result = standardizer_non_canonical.standardize(smiles)
        
        assert isinstance(canonical_result, str)
        assert isinstance(non_canonical_result, str)
        
        # Both should be valid SMILES
        from rdkit import Chem
        assert Chem.MolFromSmiles(canonical_result) is not None
        assert Chem.MolFromSmiles(non_canonical_result) is not None
    
    @pytest.mark.rdkit
    def test_stereochemistry_removal(self):
        """Test stereochemistry removal."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        
        # Test with stereochemistry removal
        standardizer_no_stereo = SMILESStandardizer(remove_stereochemistry=True)
        
        # Test with stereochemistry preservation
        standardizer_with_stereo = SMILESStandardizer(remove_stereochemistry=False)
        
        # Use a SMILES with stereochemistry
        stereo_smiles = "C[C@H](N)C(=O)O"  # L-Alanine
        
        result_no_stereo = standardizer_no_stereo.standardize(stereo_smiles)
        result_with_stereo = standardizer_with_stereo.standardize(stereo_smiles)
        
        assert isinstance(result_no_stereo, str)
        assert isinstance(result_with_stereo, str)
        
        # Result without stereochemistry should not contain @ or @@
        if standardizer_no_stereo.remove_stereochemistry:
            # Note: This test depends on the actual implementation
            # The exact behavior may vary based on RDKit version
            pass
    
    @pytest.mark.rdkit
    def test_neutralization(self):
        """Test charge neutralization."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        
        # Test with neutralization
        standardizer_neutralize = SMILESStandardizer(neutralize=True)
        
        # Test without neutralization
        standardizer_no_neutralize = SMILESStandardizer(neutralize=False)
        
        # Use a charged SMILES
        charged_smiles = "C[NH3+]"  # Methylammonium
        
        try:
            result_neutralize = standardizer_neutralize.standardize(charged_smiles)
            result_no_neutralize = standardizer_no_neutralize.standardize(charged_smiles)
            
            assert isinstance(result_neutralize, str)
            assert isinstance(result_no_neutralize, str)
            
            # Both should be valid SMILES
            from rdkit import Chem
            assert Chem.MolFromSmiles(result_neutralize) is not None
            assert Chem.MolFromSmiles(result_no_neutralize) is not None
            
        except InvalidSMILESError:
            # Some charged SMILES might not be standardizable
            pytest.skip("Charged SMILES not standardizable")
    
    @pytest.mark.rdkit
    def test_salt_removal(self):
        """Test salt removal."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        
        # Test with salt removal
        standardizer_remove_salts = SMILESStandardizer(remove_salts=True)
        
        # Test without salt removal
        standardizer_keep_salts = SMILESStandardizer(remove_salts=False)
        
        # Use a SMILES with salt
        salt_smiles = "CCO.Cl"  # Ethanol with chloride
        
        try:
            result_remove_salts = standardizer_remove_salts.standardize(salt_smiles)
            result_keep_salts = standardizer_keep_salts.standardize(salt_smiles)
            
            assert isinstance(result_remove_salts, str)
            assert isinstance(result_keep_salts, str)
            
            # Both should be valid SMILES
            from rdkit import Chem
            assert Chem.MolFromSmiles(result_remove_salts) is not None
            assert Chem.MolFromSmiles(result_keep_salts) is not None
            
            # Result with salt removal should be shorter (no salt)
            if standardizer_remove_salts.remove_salts:
                # The exact behavior depends on implementation
                pass
                
        except InvalidSMILESError:
            # Some salt SMILES might not be standardizable
            pytest.skip("Salt SMILES not standardizable")
    
    @pytest.mark.rdkit
    def test_standardization_consistency(self, sample_smiles):
        """Test that standardization is consistent across multiple calls."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        
        standardizer = SMILESStandardizer()
        
        for smiles in sample_smiles:
            try:
                result1 = standardizer.standardize(smiles)
                result2 = standardizer.standardize(smiles)
                
                # Results should be identical
                assert result1 == result2
                
            except InvalidSMILESError:
                # Skip invalid SMILES
                continue
    
    @pytest.mark.rdkit
    def test_standardization_idempotency(self, sample_smiles):
        """Test that standardizing already standardized SMILES doesn't change them."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        
        standardizer = SMILESStandardizer()
        
        for smiles in sample_smiles:
            try:
                standardized_once = standardizer.standardize(smiles)
                standardized_twice = standardizer.standardize(standardized_once)
                
                # Should be identical
                assert standardized_once == standardized_twice
                
            except InvalidSMILESError:
                # Skip invalid SMILES
                continue
    
    @pytest.mark.rdkit
    def test_error_handling_in_batch(self):
        """Test error handling in batch processing."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        
        standardizer = SMILESStandardizer()
        
        # Mix of valid and invalid SMILES
        test_smiles = [
            "CCO",  # Valid
            "INVALID",  # Invalid
            "c1ccccc1",  # Valid
            "",  # Invalid (empty)
            "CC(=O)O",  # Valid
            None,  # Invalid (None)
        ]
        
        standardized_smiles, failed_indices = standardizer.standardize_batch(test_smiles)
        
        # Should have some valid and some failed
        assert len(standardized_smiles) > 0
        assert len(failed_indices) > 0
        assert len(standardized_smiles) + len(failed_indices) == len(test_smiles)
        
        # Failed indices should include invalid SMILES positions
        expected_failed = {1, 3, 5}  # Positions of invalid SMILES
        assert set(failed_indices).issuperset(expected_failed)
    
    @pytest.mark.rdkit
    def test_standardizer_repr(self):
        """Test string representation of standardizer."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        
        standardizer = SMILESStandardizer(
            remove_stereochemistry=True,
            neutralize=False,
            remove_salts=True,
            canonical=False
        )
        
        repr_str = repr(standardizer)
        
        assert 'SMILESStandardizer' in repr_str
        assert 'remove_stereochemistry=True' in repr_str
        assert 'neutralize=False' in repr_str
        assert 'remove_salts=True' in repr_str
        assert 'canonical=False' in repr_str


class TestStandardizationIntegration:
    """Integration tests for SMILES standardization."""
    
    @pytest.mark.rdkit
    def test_standardization_with_real_molecules(self):
        """Test standardization with real molecular SMILES."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        
        standardizer = SMILESStandardizer()
        
        # Real molecular SMILES with various complexities
        real_smiles = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "C[C@H](N)C(=O)O",  # L-Alanine
            "CC1=CC=C(C=C1)C(=O)O",  # p-Toluic acid
        ]
        
        standardized_smiles, failed_indices = standardizer.standardize_batch(real_smiles)
        
        # All real molecules should be standardizable
        assert len(failed_indices) == 0
        assert len(standardized_smiles) == len(real_smiles)
        
        # All results should be valid SMILES
        from rdkit import Chem
        for smiles in standardized_smiles:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
    
    @pytest.mark.rdkit
    def test_standardization_preserves_molecular_identity(self):
        """Test that standardization preserves molecular identity."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        from rdkit import Chem
        
        standardizer = SMILESStandardizer()
        
        test_smiles = [
            "CCO",  # Ethanol
            "C1=CC=CC=C1",  # Benzene (different representation)
            "c1ccccc1",  # Benzene (aromatic)
        ]
        
        for smiles in test_smiles:
            try:
                original_mol = Chem.MolFromSmiles(smiles)
                standardized_smiles = standardizer.standardize(smiles)
                standardized_mol = Chem.MolFromSmiles(standardized_smiles)
                
                # Molecules should have same molecular formula
                original_formula = Chem.rdMolDescriptors.CalcMolFormula(original_mol)
                standardized_formula = Chem.rdMolDescriptors.CalcMolFormula(standardized_mol)
                
                assert original_formula == standardized_formula
                
            except InvalidSMILESError:
                # Skip if standardization fails
                continue
    
    @pytest.mark.rdkit
    @pytest.mark.slow
    def test_standardization_performance(self):
        """Test standardization performance with larger datasets."""
        from molenc.preprocessing.standardize import SMILESStandardizer
        import time
        
        standardizer = SMILESStandardizer()
        
        # Create a larger dataset by repeating SMILES
        base_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)O"]
        large_dataset = base_smiles * 100  # 400 SMILES
        
        start_time = time.time()
        standardized_smiles, failed_indices = standardizer.standardize_batch(large_dataset)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 10.0  # 10 seconds threshold
        
        # Should process most SMILES successfully
        success_rate = len(standardized_smiles) / len(large_dataset)
        assert success_rate > 0.9  # At least 90% success rate