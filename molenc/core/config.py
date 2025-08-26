"""Configuration management for MolEnc library."""

import yaml
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from .exceptions import ConfigurationError


class Config:
    """Configuration manager for molecular encoders."""
    
    # Preset configurations
    PRESETS = {
        'drug_discovery': {
            'encoder_name': 'morgan',
            'radius': 3,
            'n_bits': 2048,
            'handle_errors': 'skip'
        },
        'fast_screening': {
            'encoder_name': 'maccs',
            'handle_errors': 'skip'
        },
        'high_accuracy': {
            'encoder_name': 'unimol',
            'model_path': None,  # Will use default pretrained model
            'handle_errors': 'warn'
        },
        'graph_based': {
            'encoder_name': 'gcn',
            'hidden_dim': 256,
            'num_layers': 3,
            'handle_errors': 'skip'
        },
        'sequence_based': {
            'encoder_name': 'chemberta',
            'model_name': 'seyonec/ChemBERTa-zinc-base-v1',
            'handle_errors': 'skip'
        }
    }
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize configuration.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        self._config: Dict[str, Any] = config_dict or {}
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not isinstance(self._config, dict):
            raise ConfigurationError("Configuration must be a dictionary")
        
        # Check for required encoder_name
        if 'encoder_name' not in self._config:
            raise ConfigurationError("'encoder_name' is required in configuration")
        
        # Validate handle_errors option
        handle_errors = self._config.get('handle_errors', 'raise')
        if handle_errors not in ['raise', 'skip', 'warn']:
            raise ConfigurationError(
                "'handle_errors' must be one of: 'raise', 'skip', 'warn'",
                'handle_errors'
            )
    
    @classmethod
    def from_file(cls, file_path: str) -> 'Config':
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file (YAML or JSON)
            
        Returns:
            Config instance
            
        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        file_path_obj: Path = Path(file_path)
        
        if not file_path_obj.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                if file_path_obj.suffix.lower() in ['.yml', '.yaml']:
                    config_dict = yaml.safe_load(f)
                elif file_path_obj.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported file format: {file_path_obj.suffix}. "
                        "Supported formats: .yml, .yaml, .json"
                    )
        except Exception as e:
            raise ConfigurationError(f"Failed to parse configuration file: {e}")
        
        return cls(config_dict)
    
    @classmethod
    def from_preset(cls, preset_name: str) -> 'Config':
        """
        Load configuration from preset.
        
        Args:
            preset_name: Name of the preset configuration
            
        Returns:
            Config instance
            
        Raises:
            ConfigurationError: If preset is not found
        """
        if preset_name not in cls.PRESETS:
            available_presets = list(cls.PRESETS.keys())
            raise ConfigurationError(
                f"Preset '{preset_name}' not found. "
                f"Available presets: {', '.join(available_presets)}"
            )
        
        preset_config: Dict[str, Any] = cls.PRESETS[preset_name].copy()
        return cls(preset_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
        self._validate_config()
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            config_dict: Dictionary containing new configuration values
        """
        self._config.update(config_dict)
        self._validate_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        # Create a copy of the configuration dictionary
        config_copy: Dict[str, Any] = {}
        for key, value in self._config.items():
            config_copy[key] = value
        return config_copy
    
    def save(self, file_path: str, format: str = 'yaml') -> None:
        """
        Save configuration to file.
        
        Args:
            file_path: Path to save configuration
            format: File format ('yaml' or 'json')
            
        Raises:
            ConfigurationError: If format is unsupported or save fails
        """
        if format not in ['yaml', 'json']:
            raise ConfigurationError(f"Unsupported format: {format}")
        
        file_path_obj: Path = Path(file_path)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path_obj, 'w', encoding='utf-8') as f:
                if format == 'yaml':
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
                else:  # json
                    json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    @staticmethod
    def list_presets() -> List[str]:
        """
        List available preset configurations.
        
        Returns:
            List of preset names
        """
        return list(Config.PRESETS.keys())
    
    def __repr__(self) -> str:
        config_str: str = str(self._config)
        return f"Config({config_str})"
    
    def __str__(self) -> str:
        config_dump: str = yaml.dump(self._config, default_flow_style=False, indent=2)
        return config_dump
