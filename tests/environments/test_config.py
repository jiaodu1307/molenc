"""Tests for environment configuration functionality."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from molenc.environments.config import EnvironmentConfig


class TestEnvironmentConfig:
    """Test cases for EnvironmentConfig class."""
    
    def test_init_default(self):
        """Test EnvironmentConfig initialization with default values."""
        config = EnvironmentConfig()
        
        assert isinstance(config.cache_dir, Path)
        assert isinstance(config.temp_dir, Path)
        assert config.cache_dir.name == 'cache'
        assert config.temp_dir.name == 'temp'
        assert config.max_workers is not None
        assert config.memory_limit is None
        assert config.gpu_enabled is False
        assert config.device == 'cpu'
        assert config.log_level == 'INFO'
        assert config.features == []
        assert config.custom_settings == {}
    
    def test_init_custom(self):
        """Test EnvironmentConfig initialization with custom values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / 'custom_cache'
            temp_dir_path = Path(temp_dir) / 'custom_temp'
            
            config = EnvironmentConfig(
                cache_dir=cache_dir,
                temp_dir=temp_dir_path,
                max_workers=4,
                memory_limit='2GB',
                gpu_enabled=True,
                device='cuda:0',
                log_level='DEBUG',
                features=['molecular_descriptors', 'deep_learning'],
                custom_settings={'custom_key': 'custom_value'}
            )
            
            assert config.cache_dir == cache_dir
            assert config.temp_dir == temp_dir_path
            assert config.max_workers == 4
            assert config.memory_limit == '2GB'
            assert config.gpu_enabled is True
            assert config.device == 'cuda:0'
            assert config.log_level == 'DEBUG'
            assert config.features == ['molecular_descriptors', 'deep_learning']
            assert config.custom_settings == {'custom_key': 'custom_value'}
    
    def test_init_gpu_enabled_auto_device(self):
        """Test automatic device selection when GPU is enabled."""
        config = EnvironmentConfig(gpu_enabled=True)
        assert config.device == 'cuda'
    
    def test_init_creates_directories(self):
        """Test that initialization creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / 'test_cache'
            temp_dir_path = Path(temp_dir) / 'test_temp'
            
            # Directories should not exist initially
            assert not cache_dir.exists()
            assert not temp_dir_path.exists()
            
            config = EnvironmentConfig(
                cache_dir=cache_dir,
                temp_dir=temp_dir_path
            )
            
            # Directories should be created
            assert cache_dir.exists()
            assert temp_dir_path.exists()
    
    def test_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            'max_workers': 8,
            'memory_limit': '4GB',
            'gpu_enabled': True,
            'device': 'cuda:1',
            'log_level': 'WARNING',
            'features': ['graph_neural_networks'],
            'custom_settings': {'test': 'value'}
        }
        
        config = EnvironmentConfig.from_dict(config_dict)
        
        assert config.max_workers == 8
        assert config.memory_limit == '4GB'
        assert config.gpu_enabled is True
        assert config.device == 'cuda:1'
        assert config.log_level == 'WARNING'
        assert config.features == ['graph_neural_networks']
        assert config.custom_settings == {'test': 'value'}
    
    def test_from_file_json(self):
        """Test loading configuration from JSON file."""
        config_dict = {
            'max_workers': 6,
            'memory_limit': '8GB',
            'gpu_enabled': False,
            'log_level': 'ERROR',
            'features': ['fingerprints']
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            config_path = f.name
        
        try:
            config = EnvironmentConfig.from_file(config_path)
            
            assert config.max_workers == 6
            assert config.memory_limit == '8GB'
            assert config.gpu_enabled is False
            assert config.log_level == 'ERROR'
            assert config.features == ['fingerprints']
        finally:
            Path(config_path).unlink()
    
    def test_from_file_not_found(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(FileNotFoundError, match='Configuration file not found'):
            EnvironmentConfig.from_file('/nonexistent/config.json')
    
    def test_from_file_unsupported_format(self):
        """Test loading configuration from unsupported file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('test: value')
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match='Unsupported configuration file format'):
                EnvironmentConfig.from_file(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = EnvironmentConfig(
            max_workers=4,
            memory_limit='2GB',
            gpu_enabled=True,
            device='cuda:0',
            log_level='DEBUG',
            features=['molecular_descriptors'],
            custom_settings={'custom': 'value'}
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['max_workers'] == 4
        assert config_dict['memory_limit'] == '2GB'
        assert config_dict['gpu_enabled'] is True
        assert config_dict['device'] == 'cuda:0'
        assert config_dict['log_level'] == 'DEBUG'
        assert config_dict['features'] == ['molecular_descriptors']
        assert config_dict['custom_settings'] == {'custom': 'value'}
        assert isinstance(config_dict['cache_dir'], str)
        assert isinstance(config_dict['temp_dir'], str)
    
    def test_save_to_file_json(self):
        """Test saving configuration to JSON file."""
        config = EnvironmentConfig(
            max_workers=8,
            memory_limit='4GB',
            features=['deep_learning']
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            config.save_to_file(config_path)
            
            # Verify file was created and contains correct data
            assert Path(config_path).exists()
            
            with open(config_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data['max_workers'] == 8
            assert saved_data['memory_limit'] == '4GB'
            assert saved_data['features'] == ['deep_learning']
        finally:
            Path(config_path).unlink()
    
    def test_save_to_file_creates_directory(self):
        """Test that save_to_file creates parent directories."""
        config = EnvironmentConfig()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'subdir' / 'config.json'
            
            # Parent directory should not exist
            assert not config_path.parent.exists()
            
            config.save_to_file(config_path)
            
            # File and parent directory should be created
            assert config_path.exists()
            assert config_path.parent.exists()
    
    def test_save_to_file_unsupported_format(self):
        """Test saving configuration to unsupported file format."""
        config = EnvironmentConfig()
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match='Unsupported configuration file format'):
                config.save_to_file(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_update(self):
        """Test updating configuration parameters."""
        config = EnvironmentConfig()
        
        # Update existing attributes
        config.update(
            max_workers=16,
            gpu_enabled=True,
            log_level='WARNING'
        )
        
        assert config.max_workers == 16
        assert config.gpu_enabled is True
        assert config.log_level == 'WARNING'
        
        # Update with custom settings
        config.update(
            custom_param='custom_value',
            another_param=42
        )
        
        assert config.custom_settings['custom_param'] == 'custom_value'
        assert config.custom_settings['another_param'] == 42
    
    def test_get_setting(self):
        """Test getting configuration settings."""
        config = EnvironmentConfig(
            max_workers=8,
            custom_settings={'custom_key': 'custom_value'}
        )
        
        # Get existing attribute
        assert config.get_setting('max_workers') == 8
        assert config.get_setting('gpu_enabled') is False
        
        # Get custom setting
        assert config.get_setting('custom_key') == 'custom_value'
        
        # Get non-existent setting with default
        assert config.get_setting('nonexistent', 'default') == 'default'
        assert config.get_setting('nonexistent') is None
    
    def test_str_representation(self):
        """Test string representation."""
        config = EnvironmentConfig(
            device='cuda:0',
            features=['molecular_descriptors', 'deep_learning']
        )
        
        str_repr = str(config)
        assert 'EnvironmentConfig' in str_repr
        assert 'cuda:0' in str_repr
        assert 'molecular_descriptors' in str_repr
        assert 'deep_learning' in str_repr
    
    def test_repr_representation(self):
        """Test repr representation."""
        config = EnvironmentConfig()
        assert repr(config) == str(config)


class TestEnvironmentConfigIntegration:
    """Integration tests for EnvironmentConfig."""
    
    def test_round_trip_serialization(self):
        """Test round-trip serialization (save and load)."""
        original_config = EnvironmentConfig(
            max_workers=12,
            memory_limit='6GB',
            gpu_enabled=True,
            device='cuda:1',
            log_level='DEBUG',
            features=['graph_neural_networks', 'molecular_transformers'],
            custom_settings={'batch_size': 64, 'learning_rate': 0.001}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Save configuration
            original_config.save_to_file(config_path)
            
            # Load configuration
            loaded_config = EnvironmentConfig.from_file(config_path)
            
            # Compare configurations
            assert loaded_config.max_workers == original_config.max_workers
            assert loaded_config.memory_limit == original_config.memory_limit
            assert loaded_config.gpu_enabled == original_config.gpu_enabled
            assert loaded_config.device == original_config.device
            assert loaded_config.log_level == original_config.log_level
            assert loaded_config.features == original_config.features
            assert loaded_config.custom_settings == original_config.custom_settings
        finally:
            Path(config_path).unlink()
    
    def test_configuration_inheritance(self):
        """Test configuration inheritance and modification."""
        base_config = EnvironmentConfig(
            max_workers=4,
            features=['fingerprints']
        )
        
        # Create derived configuration
        derived_dict = base_config.to_dict()
        derived_dict.update({
            'max_workers': 8,
            'features': ['fingerprints', 'deep_learning'],
            'gpu_enabled': True
        })
        
        derived_config = EnvironmentConfig.from_dict(derived_dict)
        
        # Verify inheritance and modifications
        assert derived_config.max_workers == 8
        assert derived_config.features == ['fingerprints', 'deep_learning']
        assert derived_config.gpu_enabled is True
        assert derived_config.log_level == base_config.log_level  # Inherited
    
    def test_directory_management(self):
        """Test directory creation and management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            cache_dir = base_path / 'custom_cache'
            temp_dir_path = base_path / 'custom_temp'
            
            # Create configuration with custom directories
            config = EnvironmentConfig(
                cache_dir=cache_dir,
                temp_dir=temp_dir_path
            )
            
            # Verify directories were created
            assert cache_dir.exists()
            assert temp_dir_path.exists()
            assert cache_dir.is_dir()
            assert temp_dir_path.is_dir()
            
            # Test that directories are accessible
            test_file = cache_dir / 'test.txt'
            test_file.write_text('test content')
            assert test_file.exists()
            assert test_file.read_text() == 'test content'
    
    def test_configuration_validation(self):
        """Test configuration validation and edge cases."""
        # Test with string paths
        with tempfile.TemporaryDirectory() as temp_dir:
            config = EnvironmentConfig(
                cache_dir=str(Path(temp_dir) / 'cache'),
                temp_dir=str(Path(temp_dir) / 'temp')
            )
            
            assert isinstance(config.cache_dir, Path)
            assert isinstance(config.temp_dir, Path)
        
        # Test with None values
        config = EnvironmentConfig(
            max_workers=None,
            memory_limit=None,
            device=None
        )
        
        assert config.max_workers is not None  # Should use os.cpu_count()
        assert config.memory_limit is None
        assert config.device == 'cpu'  # Default when gpu_enabled=False
    
    def test_feature_configuration(self):
        """Test feature-specific configuration."""
        # Test different feature combinations
        features_configs = [
            ['molecular_descriptors'],
            ['deep_learning', 'graph_neural_networks'],
            ['fingerprints', 'molecular_transformers', 'visualization'],
            []
        ]
        
        for features in features_configs:
            config = EnvironmentConfig(features=features)
            assert config.features == features
            
            # Test serialization with features
            config_dict = config.to_dict()
            assert config_dict['features'] == features
            
            # Test round-trip
            restored_config = EnvironmentConfig.from_dict(config_dict)
            assert restored_config.features == features