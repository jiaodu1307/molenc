"""Tests for SMILES validation utilities."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from molenc.core.exceptions import InvalidSMILESError, DependencyError


class TestSMILESValidator:
    """Tests for SMILESValidator."""
    
    @pytest.mark.rdkit
    def test_validator_initialization_default(self):
        """Test SMILESValidator initialization with default parameters."""
        from molenc.preprocessing.validators import SMILESValidator
        
        validator = SMILESValidator()
        assert validator.strict_mode is False
        assert validator.check_aromaticity is True
        assert validator.check_valence is True
        assert validator.max_atoms is None
        assert validator.allowed_elements is None
        
        # Check that stats are initialized
        assert hasattr(validator, 'stats')
        assert validator.stats['total_validated'] == 0
    
    @pytest.mark.rdkit
    def test_validator_initialization_custom(self):
        """Test SMILESValidator initialization with custom parameters."""
        from molenc.preprocessing.validators import SMILESValidator
        
        validator = SMILESValidator(
            strict_mode=True,
            check_aromaticity=False,
            check_valence=False,
            max_atoms=50,
            allowed_elements=['C', 'N', 'O', 'H']
        )
        assert validator.strict_mode is True
        assert validator.check_aromaticity is False
        assert validator.check_valence is False
        assert validator.max_atoms == 50
        assert validator.allowed_elements == {'C', 'N', 'O', 'H'}
    
    def test_validator_missing_rdkit(self):
        """Test SMILESValidator raises error when RDKit is not available."""
        with patch.dict('sys.modules', {'rdkit': None}):
            with pytest.raises(DependencyError) as exc_info:
                from molenc.preprocessing.validators import SMILESValidator
                SMILESValidator()
            assert "rdkit" in str(exc_info.value)
            assert "SMILESValidator" in str(exc_info.value)
    
    @pytest.mark.rdkit
    def test_check_syntax_valid(self):
        """Test syntax checking with valid SMILES."""
        from molenc.preprocessing.validators import SMILESValidator
        
        validator = SMILESValidator()
        
        valid_smiles = [
            "CCO",
            "c1ccccc1",
            "CC(=O)O",
            "C[C@H](N)C(=O)O",
            "CC1=CC=C(C=C1)C(=O)O"
        ]
        
        for smiles in valid_smiles:
            is_valid, error_msg = validator.check_syntax(smiles)
            assert is_valid is True
            assert error_msg == ""
    
    @pytest.mark.rdkit
    def test_check_syntax_invalid(self):
        """Test syntax checking with invalid SMILES."""
        from molenc.preprocessing.validators import SMILESValidator
        
        validator = SMILESValidator(strict_mode=True)
        
        invalid_smiles = [
            "",  # Empty string
            "C(C",  # Unmatched parenthesis
            "C)C",  # Unmatched parenthesis
            "C[C",  # Unmatched bracket
            "C]C",  # Unmatched bracket
            "C{C",  # Unmatched brace
            "C}C",  # Unmatched brace
        ]
        
        for smiles in invalid_smiles:
            is_valid, error_msg = validator.check_syntax(smiles)
            assert is_valid is False
            assert len(error_msg) > 0
    
    @pytest.mark.rdkit
    def test_check_syntax_non_string_input(self):
        """Test syntax checking with non-string input."""
        from molenc.preprocessing.validators import SMILESValidator
        
        validator = SMILESValidator()
        
        invalid_inputs = [None, 123, [], {}]
        
        for invalid_input in invalid_inputs:
            is_valid, error_msg = validator.check_syntax(invalid_input)
            assert is_valid is False
            assert "non-string" in error_msg.lower() or "empty" in error_msg.lower()
    
    @pytest.mark.rdkit
    def test_validate_single_valid_smiles(self, sample_smiles):
        """Test validating single valid SMILES strings."""
        from molenc.preprocessing.validators import SMILESValidator
        
        validator = SMILESValidator()
        
        for smiles in sample_smiles:
            is_valid, error_msg = validator.validate(smiles)
            assert is_valid is True
            assert error_msg == ""
    
    @pytest.mark.rdkit
    def test_validate_single_invalid_smiles(self, invalid_smiles):
        """Test validating single invalid SMILES strings."""
        from molenc.preprocessing.validators import SMILESValidator
        
        validator = SMILESValidator()
        
        for smiles in invalid_smiles:
            is_valid, error_msg = validator.validate(smiles)
            assert is_valid is False
            assert len(error_msg) > 0
    
    @pytest.mark.rdkit
    def test_validate_batch(self, sample_smiles, invalid_smiles):
        """Test batch validation of SMILES strings."""
        from molenc.preprocessing.validators import SMILESValidator
        
        validator = SMILESValidator()
        
        # Mix valid and invalid SMILES
        mixed_smiles = sample_smiles + invalid_smiles
        
        valid_smiles, invalid_indices, stats = validator.validate_batch(mixed_smiles)
        
        assert isinstance(valid_smiles, list)
        assert isinstance(invalid_indices, list)
        assert isinstance(stats, dict)
        
        # Check that all valid SMILES are actually valid
        for smiles in valid_smiles:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
        
        # Check that invalid indices are in correct range
        for idx in invalid_indices:
            assert 0 <= idx < len(mixed_smiles)
        
        # Check stats
        assert stats['total'] == len(mixed_smiles)
        assert stats['valid'] == len(valid_smiles)
        assert stats['invalid'] == len(invalid_indices)
        assert stats['valid'] + stats['invalid'] == stats['total']
    
    @pytest.mark.rdkit
    def test_max_atoms_constraint(self):
        """Test maximum atoms constraint."""
        from molenc.preprocessing.validators import SMILESValidator
        
        # Validator with max_atoms constraint
        validator = SMILESValidator(max_atoms=5)
        
        # Small molecule (should pass)
        small_smiles = "CCO"  # 3 atoms
        is_valid, error_msg = validator.validate(small_smiles)
        assert is_valid is True
        
        # Large molecule (should fail)
        large_smiles = "CCCCCCCCCCCCCCCCCCCC"  # 20 atoms
        is_valid, error_msg = validator.validate(large_smiles)
        assert is_valid is False
        assert "atoms" in error_msg.lower()
    
    @pytest.mark.rdkit
    def test_allowed_elements_constraint(self):
        """Test allowed elements constraint."""
        from molenc.preprocessing.validators import SMILESValidator
        
        # Validator with allowed elements constraint
        validator = SMILESValidator(allowed_elements=['C', 'H', 'O'])
        
        # Molecule with allowed elements (should pass)
        allowed_smiles = "CCO"  # C, H, O only
        is_valid, error_msg = validator.validate(allowed_smiles)
        assert is_valid is True
        
        # Molecule with disallowed elements (should fail)
        disallowed_smiles = "CCN"  # Contains N
        is_valid, error_msg = validator.validate(disallowed_smiles)
        assert is_valid is False
        assert "element" in error_msg.lower()
    
    @pytest.mark.rdkit
    def test_aromaticity_checking(self):
        """Test aromaticity validation."""
        from molenc.preprocessing.validators import SMILESValidator
        
        # Validator with aromaticity checking
        validator_check = SMILESValidator(check_aromaticity=True)
        
        # Validator without aromaticity checking
        validator_no_check = SMILESValidator(check_aromaticity=False)
        
        # Valid aromatic SMILES
        aromatic_smiles = "c1ccccc1"  # Benzene
        
        is_valid_check, _ = validator_check.validate(aromatic_smiles)
        is_valid_no_check, _ = validator_no_check.validate(aromatic_smiles)
        
        assert is_valid_check is True
        assert is_valid_no_check is True
        
        # Test with potentially problematic aromatic SMILES
        # (This depends on the actual implementation)
    
    @pytest.mark.rdkit
    def test_valence_checking(self):
        """Test valence validation."""
        from molenc.preprocessing.validators import SMILESValidator
        
        # Validator with valence checking
        validator_check = SMILESValidator(check_valence=True)
        
        # Validator without valence checking
        validator_no_check = SMILESValidator(check_valence=False)
        
        # Valid SMILES
        valid_smiles = "CCO"
        
        is_valid_check, _ = validator_check.validate(valid_smiles)
        is_valid_no_check, _ = validator_no_check.validate(valid_smiles)
        
        assert is_valid_check is True
        assert is_valid_no_check is True
        
        # Test with potentially problematic valence SMILES
        # (This would require specific invalid valence examples)
    
    @pytest.mark.rdkit
    def test_strict_mode(self):
        """Test strict mode validation."""
        from molenc.preprocessing.validators import SMILESValidator
        
        # Strict validator
        validator_strict = SMILESValidator(strict_mode=True)
        
        # Non-strict validator
        validator_lenient = SMILESValidator(strict_mode=False)
        
        # SMILES with potentially problematic characters
        test_smiles = "CCO"  # Simple, should pass both
        
        is_valid_strict, _ = validator_strict.validate(test_smiles)
        is_valid_lenient, _ = validator_lenient.validate(test_smiles)
        
        assert is_valid_strict is True
        assert is_valid_lenient is True
        
        # Test with SMILES containing unusual characters
        # (This depends on what characters are considered invalid in strict mode)
    
    @pytest.mark.rdkit
    def test_validation_statistics(self, sample_smiles, invalid_smiles):
        """Test validation statistics tracking."""
        from molenc.preprocessing.validators import SMILESValidator
        
        validator = SMILESValidator()
        
        # Reset stats
        validator.reset_stats()
        initial_stats = validator.get_stats()
        assert initial_stats['total_validated'] == 0
        
        # Validate some SMILES
        mixed_smiles = sample_smiles + invalid_smiles
        
        for smiles in mixed_smiles:
            validator.validate(smiles)
        
        # Check updated stats
        final_stats = validator.get_stats()
        assert final_stats['total_validated'] == len(mixed_smiles)
        assert final_stats['valid_count'] >= len(sample_smiles)
        assert final_stats['invalid_count'] >= len(invalid_smiles)
        assert final_stats['valid_count'] + final_stats['invalid_count'] == final_stats['total_validated']
        
        # Check error type breakdown
        assert 'error_types' in final_stats
        error_types = final_stats['error_types']
        assert isinstance(error_types, dict)
        
        # Should have some error types recorded
        total_errors = sum(error_types.values())
        assert total_errors == final_stats['invalid_count']
    
    @pytest.mark.rdkit
    def test_reset_stats(self):
        """Test resetting validation statistics."""
        from molenc.preprocessing.validators import SMILESValidator
        
        validator = SMILESValidator()
        
        # Validate some SMILES to generate stats
        validator.validate("CCO")
        validator.validate("INVALID")
        
        # Check that stats are not zero
        stats_before = validator.get_stats()
        assert stats_before['total_validated'] > 0
        
        # Reset stats
        validator.reset_stats()
        
        # Check that stats are reset
        stats_after = validator.get_stats()
        assert stats_after['total_validated'] == 0
        assert stats_after['valid_count'] == 0
        assert stats_after['invalid_count'] == 0
        
        # Check that error types are reset
        for error_type, count in stats_after['error_types'].items():
            assert count == 0
    
    @pytest.mark.rdkit
    def test_validator_repr(self):
        """Test string representation of validator."""
        from molenc.preprocessing.validators import SMILESValidator
        
        validator = SMILESValidator(
            strict_mode=True,
            check_aromaticity=False,
            check_valence=False,
            max_atoms=100,
            allowed_elements=['C', 'N', 'O']
        )
        
        repr_str = repr(validator)
        
        assert 'SMILESValidator' in repr_str
        assert 'strict_mode=True' in repr_str
        assert 'check_aromaticity=False' in repr_str
        assert 'check_valence=False' in repr_str
        assert 'max_atoms=100' in repr_str
    
    @pytest.mark.rdkit
    def test_empty_batch_validation(self):
        """Test validation of empty batch."""
        from molenc.preprocessing.validators import SMILESValidator
        
        validator = SMILESValidator()
        
        valid_smiles, invalid_indices, stats = validator.validate_batch([])
        
        assert valid_smiles == []
        assert invalid_indices == []
        assert stats['total'] == 0
        assert stats['valid'] == 0
        assert stats['invalid'] == 0
    
    @pytest.mark.rdkit
    def test_validation_consistency(self, sample_smiles):
        """Test that validation is consistent across multiple calls."""
        from molenc.preprocessing.validators import SMILESValidator
        
        validator = SMILESValidator()
        
        for smiles in sample_smiles:
            result1, msg1 = validator.validate(smiles)
            result2, msg2 = validator.validate(smiles)
            
            # Results should be identical
            assert result1 == result2
            assert msg1 == msg2


class TestValidationIntegration:
    """Integration tests for SMILES validation."""
    
    @pytest.mark.rdkit
    def test_validation_with_real_molecules(self):
        """Test validation with real molecular SMILES."""
        from molenc.preprocessing.validators import SMILESValidator
        
        validator = SMILESValidator()
        
        # Real molecular SMILES
        real_smiles = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "C[C@H](N)C(=O)O",  # L-Alanine
            "CC1=CC=C(C=C1)C(=O)O",  # p-Toluic acid
            "c1ccc2c(c1)cccn2",  # Quinoline
        ]
        
        valid_smiles, invalid_indices, stats = validator.validate_batch(real_smiles)
        
        # All real molecules should be valid
        assert len(invalid_indices) == 0
        assert len(valid_smiles) == len(real_smiles)
        assert stats['valid'] == len(real_smiles)
        assert stats['invalid'] == 0
    
    @pytest.mark.rdkit
    def test_validation_with_edge_cases(self):
        """Test validation with edge case SMILES."""
        from molenc.preprocessing.validators import SMILESValidator
        
        validator = SMILESValidator()
        
        edge_cases = [
            "C",  # Single carbon
            "[H][H]",  # Hydrogen molecule
            "[Na+].[Cl-]",  # Salt
            "C#C",  # Acetylene
            "C=C=C",  # Allene
            "c1ccc2c(c1)oc1ccccc12",  # Dibenzofuran (fused rings)
        ]
        
        for smiles in edge_cases:
            is_valid, error_msg = validator.validate(smiles)
            
            # Most edge cases should be valid (depends on implementation)
            if not is_valid:
                # If invalid, should have a meaningful error message
                assert len(error_msg) > 0
    
    @pytest.mark.rdkit
    def test_validation_performance(self):
        """Test validation performance with larger datasets."""
        from molenc.preprocessing.validators import SMILESValidator
        import time
        
        validator = SMILESValidator()
        
        # Create a larger dataset
        base_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)O"]
        large_dataset = base_smiles * 250  # 1000 SMILES
        
        start_time = time.time()
        valid_smiles, invalid_indices, stats = validator.validate_batch(large_dataset)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 10.0  # 10 seconds threshold
        
        # Should validate most SMILES successfully
        success_rate = len(valid_smiles) / len(large_dataset)
        assert success_rate > 0.9  # At least 90% success rate
    
    @pytest.mark.rdkit
    def test_validation_with_different_configurations(self, sample_smiles):
        """Test validation with different validator configurations."""
        from molenc.preprocessing.validators import SMILESValidator
        
        # Different validator configurations
        configs = [
            {'strict_mode': True, 'check_aromaticity': True, 'check_valence': True},
            {'strict_mode': False, 'check_aromaticity': False, 'check_valence': False},
            {'max_atoms': 20, 'allowed_elements': ['C', 'H', 'O', 'N']},
            {'max_atoms': 100, 'allowed_elements': None},
        ]
        
        for config in configs:
            validator = SMILESValidator(**config)
            
            valid_count = 0
            for smiles in sample_smiles:
                is_valid, _ = validator.validate(smiles)
                if is_valid:
                    valid_count += 1
            
            # Should validate at least some SMILES
            assert valid_count > 0
    
    @pytest.mark.rdkit
    @pytest.mark.slow
    def test_comprehensive_validation_suite(self):
        """Comprehensive validation test with various SMILES types."""
        from molenc.preprocessing.validators import SMILESValidator
        
        validator = SMILESValidator()
        
        # Comprehensive test set
        test_smiles = [
            # Simple molecules
            "C", "CC", "CCO", "CCN",
            
            # Aromatic molecules
            "c1ccccc1", "c1ccc2ccccc2c1", "c1ccncc1",
            
            # Stereochemistry
            "C[C@H](N)C(=O)O", "C[C@@H](O)C(=O)O",
            
            # Multiple bonds
            "C=C", "C#C", "C=C=C",
            
            # Rings
            "C1CCCCC1", "C1CCC2CCCCC2C1",
            
            # Charged species
            "[NH4+]", "[OH-]", "C[NH3+]",
            
            # Salts
            "CCO.Cl", "[Na+].[Cl-]",
            
            # Complex molecules
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CC1=CC=C(C=C1)S(=O)(=O)NC(=O)NN2CCCC2",  # Complex drug-like
        ]
        
        valid_smiles, invalid_indices, stats = validator.validate_batch(test_smiles)
        
        # Should validate most of the comprehensive set
        success_rate = len(valid_smiles) / len(test_smiles)
        assert success_rate > 0.8  # At least 80% success rate
        
        # Check that stats are reasonable
        assert stats['total'] == len(test_smiles)
        assert stats['valid'] == len(valid_smiles)
        assert stats['invalid'] == len(invalid_indices)