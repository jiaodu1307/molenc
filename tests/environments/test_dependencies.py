"""Tests for dependency management functionality."""

import pytest
import sys
import warnings
from unittest.mock import patch, MagicMock
from packaging import version

from molenc.environments.dependencies import (
    DependencyChecker,
    check_dependencies,
    require_dependencies,
    warn_missing_optional
)


class TestDependencyChecker:
    """Test cases for DependencyChecker class."""
    
    def test_init(self):
        """Test DependencyChecker initialization."""
        checker = DependencyChecker()
        assert hasattr(checker, 'CORE_DEPENDENCIES')
        assert hasattr(checker, 'OPTIONAL_DEPENDENCIES')
        assert isinstance(checker.CORE_DEPENDENCIES, dict)
        assert isinstance(checker.OPTIONAL_DEPENDENCIES, dict)
        
        # Check core dependencies structure
        assert 'numpy' in checker.CORE_DEPENDENCIES
        assert 'pandas' in checker.CORE_DEPENDENCIES
        assert 'scikit-learn' in checker.CORE_DEPENDENCIES
        
        # Check optional dependencies structure
        assert 'rdkit' in checker.OPTIONAL_DEPENDENCIES
        assert 'torch' in checker.OPTIONAL_DEPENDENCIES
        
        # Check optional dependency structure
        rdkit_info = checker.OPTIONAL_DEPENDENCIES['rdkit']
        assert 'package' in rdkit_info
        assert 'version' in rdkit_info
        assert 'features' in rdkit_info
        assert 'install_hint' in rdkit_info
    
    def test_check_package_available(self):
        """Test checking available packages."""
        checker = DependencyChecker()
        
        # Test with a package that should be available (sys is built-in)
        with patch('importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_module.__version__ = '1.0.0'
            mock_import.return_value = mock_module
            
            available, installed_version, error = checker.check_package('sys', '>=0.1.0')
            assert available is True
            assert installed_version == '1.0.0'
            assert error is None
    
    def test_check_package_not_available(self):
        """Test checking unavailable packages."""
        checker = DependencyChecker()
        
        with patch('importlib.import_module', side_effect=ImportError("No module named 'nonexistent'")):
            available, installed_version, error = checker.check_package('nonexistent')
            assert available is False
            assert installed_version is None
            assert 'No module named' in error
    
    def test_check_package_version_mismatch(self):
        """Test checking packages with version mismatches."""
        checker = DependencyChecker()
        
        with patch('importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_module.__version__ = '0.5.0'
            mock_import.return_value = mock_module
            
            available, installed_version, error = checker.check_package('test_package', '>=1.0.0')
            assert available is False
            assert installed_version == '0.5.0'
            assert 'version mismatch' in error.lower()
    
    def test_check_package_no_version_attribute(self):
        """Test checking packages without version attribute."""
        checker = DependencyChecker()
        
        with patch('importlib.import_module') as mock_import:
            # Create a real object without version attributes
            class MockModule:
                pass
            
            mock_import.return_value = MockModule()
            
            available, installed_version, error = checker.check_package('test_package')
            assert available is True
            assert installed_version == 'unknown'
            assert error is None
            
            available, installed_version, error = checker.check_package('test_package')
            assert available is True
            assert installed_version == 'unknown'
            assert error is None
    
    def test_check_core_dependencies(self):
        """Test checking core dependencies."""
        checker = DependencyChecker()
        
        with patch.object(checker, 'check_package') as mock_check:
            mock_check.return_value = (True, '1.0.0', None)
            
            results = checker.check_core_dependencies()
            
            assert isinstance(results, dict)
            assert len(results) == len(checker.CORE_DEPENDENCIES)
            
            for dep_name in checker.CORE_DEPENDENCIES:
                assert dep_name in results
                assert results[dep_name]['available'] is True
                assert results[dep_name]['installed_version'] == '1.0.0'
                assert results[dep_name]['required_version'] == checker.CORE_DEPENDENCIES[dep_name]
    
    def test_check_optional_dependencies_all(self):
        """Test checking all optional dependencies."""
        checker = DependencyChecker()
        
        with patch.object(checker, 'check_package') as mock_check:
            mock_check.return_value = (True, '1.0.0', None)
            
            results = checker.check_optional_dependencies()
            
            assert isinstance(results, dict)
            assert len(results) == len(checker.OPTIONAL_DEPENDENCIES)
            
            for dep_name in checker.OPTIONAL_DEPENDENCIES:
                assert dep_name in results
                assert results[dep_name]['available'] is True
    
    def test_check_optional_dependencies_specific_features(self):
        """Test checking optional dependencies for specific features."""
        checker = DependencyChecker()
        
        with patch.object(checker, 'check_package') as mock_check:
            mock_check.return_value = (True, '1.0.0', None)
            
            results = checker.check_optional_dependencies(['molecular_descriptors'])
            
            # Should only include dependencies that support molecular_descriptors
            rdkit_included = any(
                'molecular_descriptors' in dep_info.get('features', [])
                for dep_info in checker.OPTIONAL_DEPENDENCIES.values()
            )
            
            if rdkit_included:
                assert 'rdkit' in results
    
    def test_check_all_dependencies(self):
        """Test checking all dependencies."""
        checker = DependencyChecker()
        
        with patch.object(checker, 'check_package') as mock_check:
            mock_check.return_value = (True, '1.0.0', None)
            
            results = checker.check_all_dependencies()
            
            assert 'core' in results
            assert 'optional' in results
            assert 'summary' in results
            
            assert isinstance(results['core'], dict)
            assert isinstance(results['optional'], dict)
            assert isinstance(results['summary'], dict)
            
            # Check summary structure
            summary = results['summary']
            assert 'total_core' in summary
            assert 'available_core' in summary
            assert 'total_optional' in summary
            assert 'available_optional' in summary
    
    def test_get_missing_dependencies(self):
        """Test getting missing dependencies."""
        checker = DependencyChecker()
        
        def mock_check_package(package, version_req=None):
            # Simulate numpy as missing, others as available
            if package == 'numpy' or (hasattr(checker.CORE_DEPENDENCIES, 'get') and 
                                     checker.CORE_DEPENDENCIES.get(package) and package == 'numpy'):
                return (False, None, 'Package not found')
            return (True, '1.0.0', None)
        
        with patch.object(checker, 'check_package', side_effect=mock_check_package):
            missing = checker.get_missing_dependencies()
            
            assert 'core' in missing
            assert 'optional' in missing
            
            # numpy should be in missing core dependencies
            assert 'numpy' in missing['core']
    
    def test_generate_install_commands(self):
        """Test generating installation commands."""
        checker = DependencyChecker()
        
        missing = {
            'core': ['numpy', 'pandas'],
            'optional': ['rdkit', 'torch']
        }
        
        commands = checker.generate_install_commands(missing)
        
        assert isinstance(commands, list)
        assert len(commands) > 0
        
        # Should contain pip install commands for core dependencies
        core_command = next((cmd for cmd in commands if 'numpy' in cmd and 'pandas' in cmd), None)
        assert core_command is not None
        assert 'pip install' in core_command
    
    def test_print_dependency_report(self, capsys):
        """Test printing dependency report."""
        checker = DependencyChecker()
        
        with patch.object(checker, 'check_package') as mock_check:
            mock_check.return_value = (True, '1.0.0', None)
            
            checker.print_dependency_report()
            
            captured = capsys.readouterr()
            assert 'Dependency Report' in captured.out
            assert 'Core Dependencies' in captured.out
    
    def test_str_representation(self):
        """Test string representation."""
        checker = DependencyChecker()
        str_repr = str(checker)
        assert 'DependencyChecker' in str_repr


class TestDependencyFunctions:
    """Test cases for dependency utility functions."""
    
    def test_check_dependencies_function(self):
        """Test check_dependencies convenience function."""
        with patch('molenc.environments.dependencies.DependencyChecker') as mock_checker_class:
            mock_checker = MagicMock()
            mock_checker.check_all_dependencies.return_value = {'test': 'result'}
            mock_checker_class.return_value = mock_checker
            
            result = check_dependencies(['test_feature'], print_report=False)
            
            assert result == {'test': 'result'}
            mock_checker.check_all_dependencies.assert_called_once_with(['test_feature'])
            mock_checker.print_dependency_report.assert_not_called()
    
    def test_check_dependencies_with_report(self):
        """Test check_dependencies with report printing."""
        with patch('molenc.environments.dependencies.DependencyChecker') as mock_checker_class:
            mock_checker = MagicMock()
            mock_checker.check_all_dependencies.return_value = {'test': 'result'}
            mock_checker_class.return_value = mock_checker
            
            result = check_dependencies(['test_feature'], print_report=True)
            
            assert result == {'test': 'result'}
            mock_checker.print_dependency_report.assert_called_once_with(['test_feature'])
    
    def test_require_dependencies_decorator_success(self):
        """Test require_dependencies decorator with available dependencies."""
        @require_dependencies(['numpy'], 'test_feature')
        def test_function():
            return 'success'
        
        with patch('molenc.environments.dependencies.DependencyChecker') as mock_checker_class:
            mock_checker = MagicMock()
            mock_checker.CORE_DEPENDENCIES = {'numpy': '>=1.0.0'}
            mock_checker.OPTIONAL_DEPENDENCIES = {}
            mock_checker.check_package.return_value = (True, '1.0.0', None)
            mock_checker_class.return_value = mock_checker
            
            result = test_function()
            assert result == 'success'
    
    def test_require_dependencies_decorator_failure(self):
        """Test require_dependencies decorator with missing dependencies."""
        @require_dependencies(['nonexistent'], 'test_feature')
        def test_function():
            return 'success'
        
        with patch('molenc.environments.dependencies.DependencyChecker') as mock_checker_class:
            mock_checker = MagicMock()
            mock_checker.CORE_DEPENDENCIES = {}
            mock_checker.OPTIONAL_DEPENDENCIES = {}
            mock_checker.check_package.return_value = (False, None, 'Not found')
            mock_checker_class.return_value = mock_checker
            
            with pytest.raises(ImportError, match='Missing required dependencies for test_feature'):
                test_function()
    
    def test_warn_missing_optional(self):
        """Test warning for missing optional dependencies."""
        with patch('molenc.environments.dependencies.DependencyChecker') as mock_checker_class:
            mock_checker = MagicMock()
            mock_checker.OPTIONAL_DEPENDENCIES = {
                'rdkit': {'install_hint': 'conda install -c conda-forge rdkit'}
            }
            mock_checker_class.return_value = mock_checker
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                warn_missing_optional('rdkit', 'molecular_descriptors')
                
                assert len(w) == 1
                assert 'rdkit' in str(w[0].message)
                assert 'molecular_descriptors' in str(w[0].message)
                assert 'conda install' in str(w[0].message)


class TestDependencyIntegration:
    """Integration tests for dependency management."""
    
    def test_real_dependency_check(self):
        """Test checking real dependencies (those that should be available)."""
        checker = DependencyChecker()
        
        # Check sys module (should always be available)
        available, version_str, error = checker.check_package('sys')
        assert available is True
        assert error is None
    
    def test_dependency_consistency(self):
        """Test consistency of dependency definitions."""
        checker = DependencyChecker()
        
        # All core dependencies should have version requirements
        for dep_name, version_req in checker.CORE_DEPENDENCIES.items():
            assert isinstance(dep_name, str)
            assert isinstance(version_req, str)
            assert version_req.startswith(('>=', '>', '==', '<=', '<', '~='))
        
        # All optional dependencies should have required structure
        for dep_name, dep_info in checker.OPTIONAL_DEPENDENCIES.items():
            assert isinstance(dep_name, str)
            assert isinstance(dep_info, dict)
            assert 'package' in dep_info
            assert 'version' in dep_info
            assert 'features' in dep_info
            assert 'install_hint' in dep_info
            assert isinstance(dep_info['features'], list)
    
    def test_feature_dependency_mapping(self):
        """Test that features are properly mapped to dependencies."""
        checker = DependencyChecker()
        
        # Collect all features mentioned in optional dependencies
        all_features = set()
        for dep_info in checker.OPTIONAL_DEPENDENCIES.values():
            all_features.update(dep_info['features'])
        
        # Test checking dependencies for each feature
        for feature in all_features:
            results = checker.check_optional_dependencies([feature])
            # Should return at least one dependency for each feature
            assert len(results) > 0
    
    def test_end_to_end_dependency_workflow(self):
        """Test complete dependency checking workflow."""
        # Test the complete workflow
        results = check_dependencies(print_report=False)
        
        assert 'core' in results
        assert 'optional' in results
        assert 'summary' in results
        
        # Test with specific features
        feature_results = check_dependencies(['molecular_descriptors'], print_report=False)
        assert isinstance(feature_results, dict)