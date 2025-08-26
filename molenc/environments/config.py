"""Environment configuration for MolEnc.

This module provides the EnvironmentConfig class for managing
environment-specific settings and configurations.
"""

import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


class EnvironmentConfig:
    """Configuration for computational environments."""
    
    def __init__(self,
                 cache_dir: Optional[Union[str, Path]] = None,
                 temp_dir: Optional[Union[str, Path]] = None,
                 max_workers: Optional[int] = None,
                 memory_limit: Optional[str] = None,
                 gpu_enabled: bool = False,
                 device: Optional[str] = None,
                 log_level: str = 'INFO',
                 features: Optional[List[str]] = None,
                 custom_settings: Optional[Dict[str, Any]] = None) -> None:
        """Initialize environment configuration.
        
        Args:
            cache_dir: Directory for caching data
            temp_dir: Directory for temporary files
            max_workers: Maximum number of worker processes
            memory_limit: Memory limit (e.g., '4GB', '512MB')
            gpu_enabled: Whether GPU acceleration is enabled
            device: Specific device to use (e.g., 'cuda:0', 'cpu')
            log_level: Logging level
            features: List of enabled features
            custom_settings: Additional custom settings
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.molenc' / 'cache'
        self.temp_dir = Path(temp_dir) if temp_dir else Path.home() / '.molenc' / 'temp'
        self.max_workers = max_workers or os.cpu_count()
        self.memory_limit = memory_limit
        self.gpu_enabled = gpu_enabled
        self.device = device or ('cuda' if gpu_enabled else 'cpu')
        self.log_level = log_level
        self.features = features or []
        self.custom_settings = custom_settings or {}
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnvironmentConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            EnvironmentConfig instance
        """
        # Check if config is wrapped in 'environment' key
        if 'environment' in config_dict:
            env_config = config_dict['environment']
        else:
            env_config = config_dict
        
        # Map configuration keys to constructor parameters
        supported_params = {
            'cache_dir': env_config.get('cache_dir'),
            'temp_dir': env_config.get('temp_dir'),
            'max_workers': env_config.get('max_workers'),
            'memory_limit': env_config.get('memory_limit'),
            'gpu_enabled': env_config.get('gpu_enabled', False),
            'device': env_config.get('device'),
            'log_level': env_config.get('log_level', 'INFO'),
            'features': env_config.get('features'),
            'custom_settings': env_config.get('custom_settings')
        }
        
        # Remove None values
        supported_params = {k: v for k, v in supported_params.items() if v is not None}
        
        return cls(**supported_params)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'EnvironmentConfig':
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            EnvironmentConfig instance
        """
        import json
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                # Try to import yaml, but handle if not available
                try:
                    import yaml
                    config_dict = yaml.safe_load(f)
                except ImportError:
                    raise ImportError(
                        "PyYAML is required to load YAML configuration files. "
                        "Install it with: pip install PyYAML"
                    )
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            'cache_dir': str(self.cache_dir),
            'temp_dir': str(self.temp_dir),
            'max_workers': self.max_workers,
            'memory_limit': self.memory_limit,
            'gpu_enabled': self.gpu_enabled,
            'device': self.device,
            'log_level': self.log_level,
            'features': self.features,
            'custom_settings': self.custom_settings
        }
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file.
        
        Args:
            config_path: Path to save configuration
        """
        import json
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() == '.json':
                json.dump(self.to_dict(), f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    def update(self, **kwargs) -> None:
        """Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom_settings[key] = value
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a configuration setting.
        
        Args:
            key: Setting key
            default: Default value if key not found
            
        Returns:
            Setting value
        """
        if hasattr(self, key):
            return getattr(self, key)
        return self.custom_settings.get(key, default)
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"EnvironmentConfig(device={self.device}, features={self.features})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()