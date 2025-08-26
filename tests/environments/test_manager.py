"""Tests for environment manager functionality."""

import pytest
import platform
import sys
import os
from unittest.mock import patch, MagicMock

from molenc.environments.manager import EnvironmentManager
from molenc.environments.config import EnvironmentConfig
from molenc.environments.dependencies import DependencyChecker


class TestEnvironmentManager:
    """Test cases for EnvironmentManager class."""
    
    def test_init_default(self):
        """Test EnvironmentManager initialization with default config."""
        manager = EnvironmentManager()
        
        assert isinstance(manager.config, EnvironmentConfig)
        assert isinstance(manager.dependency_checker, DependencyChecker)
        assert manager._system_info is None
    
    def test_init_custom_config(self):
        """Test EnvironmentManager initialization with custom config."""
        custom_config = EnvironmentConfig(
            max_workers=8,
            gpu_enabled=True,
            features=['deep_learning']
        )
        
        manager = EnvironmentManager(config=custom_config)
        
        assert manager.config is custom_config
        assert manager.config.max_workers == 8
        assert manager.config.gpu_enabled is True
        assert manager.config.features == ['deep_learning']
    
    def test_get_system_info(self):
        """Test getting system information."""
        manager = EnvironmentManager()
        
        system_info = manager.get_system_info()
        
        assert isinstance(system_info, dict)
        
        # Check required keys
        required_keys = [
            'platform', 'system', 'release', 'version',
            'machine', 'processor', 'python_version',
            'python_executable', 'environment_variables'
        ]
        
        for key in required_keys:
            assert key in system_info
        
        # Check data types
        assert isinstance(system_info['platform'], str)
        assert isinstance(system_info['system'], str)
        assert isinstance(system_info['python_version'], str)
        assert isinstance(system_info['python_executable'], str)
        assert isinstance(system_info['environment_variables'], dict)
        
        # Check that values are reasonable
        assert len(system_info['platform']) > 0
        assert system_info['python_executable'] == sys.executable
    
    def test_get_system_info_caching(self):
        """Test that system info is cached."""
        manager = EnvironmentManager()
        
        # First call
        system_info1 = manager.get_system_info()
        
        # Second call should return the same object (cached)
        system_info2 = manager.get_system_info()
        
        assert system_info1 is system_info2
    
    def test_check_environment(self):
        """Test checking the environment."""
        manager = EnvironmentManager()
        
        with patch.object(manager.dependency_checker, 'check_all_dependencies') as mock_check:
            mock_check.return_value = {
                'core': {'numpy': {'available': True}},
                'optional': {},
                'summary': {'total_core': 1, 'available_core': 1}
            }
            
            result = manager.check_environment()
            
            assert isinstance(result, dict)
            assert 'system_info' in result
            assert 'dependencies' in result
            assert 'config' in result
            
            # Check system_info structure
            assert isinstance(result['system_info'], dict)
            assert 'platform' in result['system_info']
            
            # Check dependencies structure
            assert isinstance(result['dependencies'], dict)
            assert 'core' in result['dependencies']
            
            # Check config structure
            assert isinstance(result['config'], dict)
            
            mock_check.assert_called_once_with(None)
    
    def test_check_environment_with_features(self):
        """Test checking environment with specific features."""
        manager = EnvironmentManager()
        features = ['molecular_descriptors', 'deep_learning']
        
        with patch.object(manager.dependency_checker, 'check_all_dependencies') as mock_check:
            mock_check.return_value = {'test': 'result'}
            
            result = manager.check_environment(features)
            
            mock_check.assert_called_once_with(features)
            assert result['dependencies'] == {'test': 'result'}
    
    def test_validate_environment_success(self):
        """Test environment validation with all dependencies available."""
        manager = EnvironmentManager()
        
        mock_results = {
            'system_info': {},
            'dependencies': {
                'core': {
                    'numpy': {'available': True},
                    'pandas': {'available': True}
                },
                'optional': {
                    'rdkit': {'available': True, 'features': ['molecular_descriptors']}
                }
            },
            'config': {}
        }
        
        with patch.object(manager, 'check_environment', return_value=mock_results):
            result = manager.validate_environment()
            assert result is True
    
    def test_validate_environment_missing_core(self):
        """Test environment validation with missing core dependencies."""
        manager = EnvironmentManager()
        
        mock_results = {
            'system_info': {},
            'dependencies': {
                'core': {
                    'numpy': {'available': True},
                    'pandas': {'available': False}  # Missing core dependency
                },
                'optional': {}
            },
            'config': {}
        }
        
        with patch.object(manager, 'check_environment', return_value=mock_results):
            result = manager.validate_environment()
            assert result is False
    
    def test_validate_environment_missing_feature_dependency(self):
        """Test environment validation with missing feature dependencies."""
        manager = EnvironmentManager()
        
        mock_results = {
            'system_info': {},
            'dependencies': {
                'core': {
                    'numpy': {'available': True},
                    'pandas': {'available': True}
                },
                'optional': {
                    'rdkit': {'available': False, 'features': ['molecular_descriptors']}
                }
            },
            'config': {}
        }
        
        with patch.object(manager, 'check_environment', return_value=mock_results):
            # Should pass without specific features
            result = manager.validate_environment()
            assert result is True
            
            # Should fail with molecular_descriptors feature
            result = manager.validate_environment(['molecular_descriptors'])
            assert result is False
    
    def test_validate_environment_with_available_features(self):
        """Test environment validation with available feature dependencies."""
        manager = EnvironmentManager()
        
        mock_results = {
            'system_info': {},
            'dependencies': {
                'core': {
                    'numpy': {'available': True}
                },
                'optional': {
                    'rdkit': {'available': True, 'features': ['molecular_descriptors']},
                    'torch': {'available': True, 'features': ['deep_learning']}
                }
            },
            'config': {}
        }
        
        with patch.object(manager, 'check_environment', return_value=mock_results):
            result = manager.validate_environment(['molecular_descriptors'])
            assert result is True
    
    def test_get_installation_instructions(self):
        """Test getting installation instructions."""
        manager = EnvironmentManager()
        
        mock_missing = {
            'core': ['numpy'],
            'optional': ['rdkit']
        }
        
        mock_commands = [
            'pip install numpy>=1.19.0',
            'conda install -c conda-forge rdkit'
        ]
        
        with patch.object(manager.dependency_checker, 'get_missing_dependencies', return_value=mock_missing):
            with patch.object(manager.dependency_checker, 'generate_install_commands', return_value=mock_commands):
                
                result = manager.get_installation_instructions()
                
                assert result == mock_commands
                manager.dependency_checker.get_missing_dependencies.assert_called_once_with(None)
                manager.dependency_checker.generate_install_commands.assert_called_once_with(mock_missing)
    
    def test_get_installation_instructions_with_features(self):
        """Test getting installation instructions for specific features."""
        manager = EnvironmentManager()
        features = ['molecular_descriptors']
        
        with patch.object(manager.dependency_checker, 'get_missing_dependencies') as mock_missing:
            with patch.object(manager.dependency_checker, 'generate_install_commands') as mock_commands:
                
                manager.get_installation_instructions(features)
                
                mock_missing.assert_called_once_with(features)
    
    def test_str_representation(self):
        """Test string representation."""
        config = EnvironmentConfig(device='cuda:0')
        manager = EnvironmentManager(config=config)
        
        str_repr = str(manager)
        assert 'EnvironmentManager' in str_repr
        assert 'config=' in str_repr
    
    def test_repr_representation(self):
        """Test repr representation."""
        manager = EnvironmentManager()
        assert repr(manager) == str(manager)


class TestEnvironmentManagerIntegration:
    """Integration tests for EnvironmentManager."""
    
    def test_real_system_info(self):
        """Test getting real system information."""
        manager = EnvironmentManager()
        system_info = manager.get_system_info()
        
        # Verify that we get real system information
        assert system_info['platform'] == platform.platform()
        assert system_info['system'] == platform.system()
        assert system_info['python_version'] == sys.version
        assert system_info['python_executable'] == sys.executable
        
        # Check environment variables
        assert 'PATH' in system_info['environment_variables']
        assert system_info['environment_variables']['PATH'] == os.environ['PATH']
    
    def test_environment_check_integration(self):
        """Test complete environment check integration."""
        manager = EnvironmentManager()
        
        # Perform real environment check
        result = manager.check_environment()
        
        # Verify structure
        assert 'system_info' in result
        assert 'dependencies' in result
        assert 'config' in result
        
        # Verify system info is populated
        system_info = result['system_info']
        assert len(system_info['platform']) > 0
        assert len(system_info['system']) > 0
        
        # Verify dependencies are checked
        dependencies = result['dependencies']
        assert 'core' in dependencies
        assert 'optional' in dependencies
        assert 'summary' in dependencies
        
        # Verify config is included
        config = result['config']
        assert isinstance(config, dict)
        assert 'max_workers' in config
    
    def test_validation_with_real_dependencies(self):
        """Test validation with real dependency checking."""
        manager = EnvironmentManager()
        
        # This should work with core Python packages
        result = manager.validate_environment()
        
        # The result depends on what's actually installed,
        # but the method should not raise an exception
        assert isinstance(result, bool)
    
    def test_configuration_integration(self):
        """Test integration with different configurations."""
        # Test with minimal configuration
        minimal_config = EnvironmentConfig(max_workers=1)
        minimal_manager = EnvironmentManager(config=minimal_config)
        
        result = minimal_manager.check_environment()
        assert result['config']['max_workers'] == 1
        
        # Test with feature-rich configuration
        rich_config = EnvironmentConfig(
            max_workers=8,
            gpu_enabled=True,
            features=['molecular_descriptors', 'deep_learning']
        )
        rich_manager = EnvironmentManager(config=rich_config)
        
        result = rich_manager.check_environment(rich_config.features)
        assert result['config']['max_workers'] == 8
        assert result['config']['gpu_enabled'] is True
        assert result['config']['features'] == ['molecular_descriptors', 'deep_learning']
    
    def test_installation_instructions_integration(self):
        """Test getting installation instructions integration."""
        manager = EnvironmentManager()
        
        # Get installation instructions
        instructions = manager.get_installation_instructions()
        
        # Should return a list of strings
        assert isinstance(instructions, list)
        
        # Each instruction should be a string
        for instruction in instructions:
            assert isinstance(instruction, str)
    
    def test_feature_specific_validation(self):
        """Test validation for specific features."""
        manager = EnvironmentManager()
        
        # Test different feature combinations
        feature_sets = [
            ['molecular_descriptors'],
            ['deep_learning'],
            ['graph_neural_networks'],
            ['molecular_descriptors', 'deep_learning'],
            []
        ]
        
        for features in feature_sets:
            # Should not raise an exception
            result = manager.validate_environment(features)
            assert isinstance(result, bool)
            
            # Check environment should also work
            env_result = manager.check_environment(features)
            assert isinstance(env_result, dict)
    
    def test_manager_with_custom_dependency_checker(self):
        """Test manager with custom dependency checker behavior."""
        config = EnvironmentConfig()
        manager = EnvironmentManager(config=config)
        
        # Verify that the dependency checker is properly initialized
        assert hasattr(manager.dependency_checker, 'CORE_DEPENDENCIES')
        assert hasattr(manager.dependency_checker, 'OPTIONAL_DEPENDENCIES')
        
        # Test that dependency checker methods are accessible
        core_deps = manager.dependency_checker.check_core_dependencies()
        assert isinstance(core_deps, dict)
    
    def test_environment_state_consistency(self):
        """Test that environment state remains consistent across calls."""
        manager = EnvironmentManager()
        
        # Multiple calls should return consistent system info
        info1 = manager.get_system_info()
        info2 = manager.get_system_info()
        
        assert info1 is info2  # Should be the same cached object
        
        # Multiple environment checks should be consistent
        env1 = manager.check_environment()
        env2 = manager.check_environment()
        
        # System info should be the same
        assert env1['system_info'] is env2['system_info']
        
        # Config should be the same
        assert env1['config'] == env2['config']