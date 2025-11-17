"""Enhanced environment manager for molecular encoders."""

import os
import sys
import subprocess
import logging
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

from .process_wrapper import ProcessWrapper, ProcessWrapperError
from ..environments.advanced_dependency_manager import (
    get_dependency_manager, 
    DependencyLevel, 
    check_encoder_readiness,
    EnvironmentType as AdvEnvType
)

# Optional imports with fallbacks
try:
    from ..core.api_client import CloudAPIClient, APIConfig, CloudAPIError
except Exception:
    CloudAPIClient = None  # type: ignore
    APIConfig = None  # type: ignore
    CloudAPIError = Exception


class EnvironmentType(Enum):
    """Types of environments for encoders."""
    LOCAL = "local"
    VIRTUAL_ENV = "virtual_env"
    CONDA_ENV = "conda_env"
    DOCKER = "docker"
    CLOUD_API = "cloud_api"
    PROCESS_ISOLATED = "process_isolated"
    
    @classmethod
    def from_advanced_type(cls, adv_type: AdvEnvType) -> 'EnvironmentType':
        """Convert from advanced environment type."""
        mapping = {
            AdvEnvType.LOCAL: cls.LOCAL,
            AdvEnvType.VIRTUAL_ENV: cls.VIRTUAL_ENV,
            AdvEnvType.CONDA_ENV: cls.CONDA_ENV
        }
        return mapping.get(adv_type, cls.LOCAL)


class EnvironmentStatus(Enum):
    """Status of an environment."""
    NOT_CONFIGURED = "not_configured"
    CONFIGURING = "configuring"
    READY = "ready"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


@dataclass
class EncoderEnvironmentConfig:
    """Configuration for an encoder's environment."""
    encoder_type: str
    environment_type: EnvironmentType
    environment_path: Optional[Path] = None
    python_executable: Optional[str] = None
    dependencies: List[str] = None
    capability_level: str = "CORE"
    cloud_fallback_enabled: bool = True
    process_isolation_enabled: bool = True
    auto_install_enabled: bool = True
    timeout: int = 300
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


class SmartEnvironmentManager:
    """Smart environment manager that handles encoder environments intelligently."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path.home() / ".molenc" / "environments"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._environments: Dict[str, EncoderEnvironmentConfig] = {}
        self._environment_status: Dict[str, EnvironmentStatus] = {}
        self._process_wrappers: Dict[str, ProcessWrapper] = {}
        self._cloud_clients: Dict[str, Any] = {}
        
        # Load existing configurations
        self._load_configurations()
        
        # Initialize managers
        if get_dependency_manager:
            self.dependency_manager = get_dependency_manager()
        else:
            self.dependency_manager = None
        
    def _load_configurations(self):
        """Load environment configurations from disk."""
        config_file = self.base_dir / "environments.json"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                
                for encoder_type, config_data in data.items():
                    self._environments[encoder_type] = EncoderEnvironmentConfig(
                        encoder_type=config_data["encoder_type"],
                        environment_type=EnvironmentType(config_data["environment_type"]),
                        environment_path=Path(config_data["environment_path"]) if config_data["environment_path"] else None,
                        python_executable=config_data["python_executable"],
                        dependencies=config_data["dependencies"],
                        capability_level=config_data["capability_level"],
                        cloud_fallback_enabled=config_data["cloud_fallback_enabled"],
                        process_isolation_enabled=config_data["process_isolation_enabled"],
                        auto_install_enabled=config_data["auto_install_enabled"],
                        timeout=config_data["timeout"],
                        metadata=config_data.get("metadata", {})
                    )
                    
                    # Set initial status
                    self._environment_status[encoder_type] = EnvironmentStatus.NOT_CONFIGURED
                    
            except Exception as e:
                self.logger.warning(f"Failed to load environment configurations: {e}")
    
    def _save_configurations(self):
        """Save environment configurations to disk."""
        config_file = self.base_dir / "environments.json"
        
        try:
            data = {}
            for encoder_type, config in self._environments.items():
                config_dict = asdict(config)
                config_dict["environment_type"] = config.environment_type.value
                if config.environment_path:
                    config_dict["environment_path"] = str(config.environment_path)
                data[encoder_type] = config_dict
            
            with open(config_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save environment configurations: {e}")
    
    def configure_encoder_environment(self, 
                                     encoder_type: str,
                                     preferred_environment: EnvironmentType = EnvironmentType.LOCAL,
                                     auto_configure: bool = True) -> EncoderEnvironmentConfig:
        """
        Configure environment for an encoder.
        
        Args:
            encoder_type: Type of encoder
            preferred_environment: Preferred environment type
            auto_configure: Whether to automatically configure if not already configured
            
        Returns:
            Encoder environment configuration
        """
        # Return existing configuration if available
        if encoder_type in self._environments:
            return self._environments[encoder_type]
        
        # Create new configuration
        config = EncoderEnvironmentConfig(
            encoder_type=encoder_type,
            environment_type=preferred_environment
        )
        
        # Check encoder readiness if possible and auto_configure is enabled
        if auto_configure and check_encoder_readiness:
            try:
                is_ready, capability_level, status_msg = check_encoder_readiness(encoder_type)
                config.capability_level = capability_level.value if hasattr(capability_level, 'value') else str(capability_level)
                self.logger.info(f"Encoder {encoder_type} readiness: {status_msg}")
                
                # If not ready, try to set up a virtual environment
                if not is_ready and preferred_environment == EnvironmentType.LOCAL:
                    self.logger.info(f"Setting up virtual environment for {encoder_type}")
                    env_config = self._setup_virtual_environment(encoder_type)
                    if env_config:
                        config = env_config
            except Exception as e:
                self.logger.warning(f"Failed to check encoder readiness: {e}")
        
        # Store configuration
        self._environments[encoder_type] = config
        self._environment_status[encoder_type] = EnvironmentStatus.NOT_CONFIGURED
        self._save_configurations()
        
        return config
    
    def _setup_virtual_environment(self, encoder_type: str) -> Optional[EncoderEnvironmentConfig]:
        """
        Set up a virtual environment for an encoder.
        
        Args:
            encoder_type: Type of encoder
            
        Returns:
            Encoder environment configuration or None if failed
        """
        try:
            if not get_dependency_manager:
                return None
                
            manager = get_dependency_manager()
            env_info = manager.get_encoder_environment(encoder_type)
            
            if env_info.status == "ready":
                config = EncoderEnvironmentConfig(
                    encoder_type=encoder_type,
                    environment_type=EnvironmentType.from_advanced_type(env_info.env_type),
                    environment_path=env_info.path,
                    python_executable=env_info.python_executable,
                    capability_level="FULL",  # Assuming it's fully configured
                    cloud_fallback_enabled=True,
                    process_isolation_enabled=True
                )
                return config
            else:
                self.logger.warning(f"Failed to set up environment for {encoder_type}: {env_info.error_message}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error setting up virtual environment for {encoder_type}: {e}")
            return None
    
    def get_process_wrapper(self, encoder_type: str) -> Optional[ProcessWrapper]:
        """Get process wrapper for encoder."""
        if encoder_type in self._process_wrappers:
            return self._process_wrappers[encoder_type]
        
        # Create new process wrapper if needed
        config = self.get_encoder_environment_config(encoder_type)
        
        if config.environment_type == EnvironmentType.VIRTUAL_ENV and config.environment_path:
            wrapper = ProcessWrapper(env_path=config.environment_path)
            self._process_wrappers[encoder_type] = wrapper
            return wrapper
        elif config.environment_type == EnvironmentType.PROCESS_ISOLATED:
            wrapper = ProcessWrapper()
            self._process_wrappers[encoder_type] = wrapper
            return wrapper
        else:
            return None
    
    def get_cloud_client(self, encoder_type: str):
        """Get cloud client for encoder."""
        if encoder_type in self._cloud_clients:
            return self._cloud_clients[encoder_type]
        
        # Create new cloud client
        try:
            if get_cloud_client:
                client = get_cloud_client()
                if hasattr(client, 'health_check') and client.health_check():
                    self._cloud_clients[encoder_type] = client
                    return client
                else:
                    self.logger.warning(f"Cloud API not available for {encoder_type}")
                    return None
        except Exception as e:
            self.logger.warning(f"Failed to initialize cloud client for {encoder_type}: {e}")
            return None
    
    def execute_encoder(self, 
                       encoder_type: str,
                       smiles_data: Union[str, List[str]],
                       encoder_config: Optional[Dict[str, Any]] = None,
                       timeout: Optional[int] = None) -> Any:
        """
        Execute encoder with intelligent environment selection.
        
        Args:
            encoder_type: Type of encoder to use
            smiles_data: SMILES string or list of SMILES strings
            encoder_config: Configuration for the encoder
            timeout: Timeout in seconds
            
        Returns:
            Encoder results
        """
        # Get environment configuration
        config = self.get_encoder_environment_config(encoder_type)
        actual_timeout = timeout or config.timeout
        
        # Try different execution strategies based on environment type
        if config.environment_type == EnvironmentType.LOCAL:
            return self._execute_local(encoder_type, smiles_data, encoder_config)
            
        elif config.environment_type == EnvironmentType.PROCESS_ISOLATED:
            return self._execute_process_isolated(encoder_type, smiles_data, encoder_config, actual_timeout)
            
        elif config.environment_type == EnvironmentType.VIRTUAL_ENV:
            return self._execute_virtual_env(encoder_type, smiles_data, encoder_config, actual_timeout)
            
        elif config.environment_type == EnvironmentType.CLOUD_API:
            return self._execute_cloud_api(encoder_type, smiles_data, encoder_config)
            
        else:
            # Last resort: try local execution and let it fail naturally
            return self._execute_local(encoder_type, smiles_data, encoder_config)
    
    def _execute_local(self, 
                       encoder_type: str,
                       smiles_data: Union[str, List[str]],
                       encoder_config: Optional[Dict[str, Any]] = None) -> Any:
        """Execute encoder in local environment."""
        try:
            from .. import MolEncoder
            encoder = MolEncoder(encoder_type, **(encoder_config or {}))
            
            if isinstance(smiles_data, str):
                return encoder.encode(smiles_data)
            else:
                return encoder.encode_batch(smiles_data)
                
        except Exception as e:
            self.logger.error(f"Local execution failed for {encoder_type}: {e}")
            raise
    
    def _execute_process_isolated(self, 
                                 encoder_type: str,
                                 smiles_data: Union[str, List[str]],
                                 encoder_config: Optional[Dict[str, Any]] = None,
                                 timeout: int = 300) -> Any:
        """Execute encoder in process-isolated environment."""
        try:
            wrapper = self.get_process_wrapper(encoder_type)
            if wrapper:
                return wrapper.execute_encoder(
                    encoder_type, smiles_data, encoder_config, timeout
                )
            else:
                # Fallback to local execution
                return self._execute_local(encoder_type, smiles_data, encoder_config)
                
        except ProcessWrapperError as e:
            self.logger.error(f"Process isolation failed for {encoder_type}: {e}")
            # Try cloud fallback if enabled
            config = self.get_encoder_environment_config(encoder_type)
            if config.cloud_fallback_enabled:
                try:
                    return self._execute_cloud_api(encoder_type, smiles_data, encoder_config)
                except Exception:
                    pass
            raise
        except Exception as e:
            self.logger.error(f"Process isolation failed unexpectedly for {encoder_type}: {e}")
            raise
    
    def _execute_virtual_env(self, 
                             encoder_type: str,
                             smiles_data: Union[str, List[str]],
                             encoder_config: Optional[Dict[str, Any]] = None,
                             timeout: int = 300) -> Any:
        """Execute encoder in virtual environment."""
        # This is similar to process isolation but with a specific virtual environment
        return self._execute_process_isolated(encoder_type, smiles_data, encoder_config, timeout)
    
    def _execute_cloud_api(self, 
                          encoder_type: str,
                          smiles_data: Union[str, List[str]],
                          encoder_config: Optional[Dict[str, Any]] = None) -> Any:
        """Execute encoder via cloud API."""
        try:
            client = self._get_cloud_client(encoder_type, encoder_config)
            if not client:
                raise Exception("Cloud API client not available")
            
            if isinstance(smiles_data, str):
                if hasattr(client, 'encode_single'):
                    response = client.encode_single(smiles_data, encoder_type, options=encoder_config)
                else:
                    raise Exception("Cloud client does not support single encoding")
            else:
                if hasattr(client, 'encode_batch'):
                    # Map kwargs for remote API
                    options = encoder_config or {}
                    response = client.encode_batch(smiles_data, encoder_type=encoder_type, **options)
                else:
                    raise Exception("Cloud client does not support batch encoding")
            
            if hasattr(response, 'success') and not response.success:
                raise Exception(f"Cloud API request failed: {getattr(response, 'error_message', 'Unknown error')}")
            
            return response
                
        except Exception as e:
            self.logger.error(f"Cloud API execution failed for {encoder_type}: {e}")
            raise

    def _get_cloud_client(self, encoder_type: str, encoder_config: Optional[Dict[str, Any]] = None):
        """Build cloud client from environment or config."""
        if CloudAPIClient is None or APIConfig is None:
            return None
        import os
        base_url = None
        # Prefer explicit config
        if encoder_config and 'base_url' in encoder_config:
            base_url = encoder_config.get('base_url')
        # Fallback to env
        if not base_url:
            base_url = os.environ.get('MOLENC_REMOTE_URL')
        api_key = os.environ.get('MOLENC_REMOTE_KEY')
        timeout = int(os.environ.get('MOLENC_REMOTE_TIMEOUT', '30'))
        if not base_url:
            return None
        return CloudAPIClient(APIConfig(base_url=base_url, api_key=api_key, timeout=timeout))
    
    def get_encoder_environment_config(self, encoder_type: str) -> EncoderEnvironmentConfig:
        """Get environment configuration for encoder."""
        if encoder_type not in self._environments:
            # Auto-configure if not already configured
            self.configure_encoder_environment(encoder_type)
        
        return self._environments[encoder_type]
    
    def update_encoder_environment(self, config: EncoderEnvironmentConfig):
        """Update encoder environment configuration."""
        self._environments[config.encoder_type] = config
        self._environment_status[config.encoder_type] = EnvironmentStatus.NOT_CONFIGURED
        self._save_configurations()
    
    def get_environment_status(self, encoder_type: str) -> EnvironmentStatus:
        """Get current status of encoder environment."""
        return self._environment_status.get(encoder_type, EnvironmentStatus.NOT_CONFIGURED)
    
    def set_environment_status(self, encoder_type: str, status: EnvironmentStatus):
        """Set status of encoder environment."""
        self._environment_status[encoder_type] = status
    
    def get_status_report(self) -> str:
        """Generate status report for all encoder environments."""
        report = ["\n=== Encoder Environment Status Report ==="]
        
        for encoder_type, config in self._environments.items():
            status = self.get_environment_status(encoder_type)
            report.append(f"\n[{encoder_type.upper()}]")
            report.append(f"  Environment Type: {config.environment_type.value}")
            report.append(f"  Status: {status.value}")
            report.append(f"  Capability Level: {config.capability_level}")
            
            if config.environment_path:
                report.append(f"  Environment Path: {config.environment_path}")
            
            if config.metadata:
                report.append(f"  Metadata: {config.metadata}")
        
        return "\n".join(report)


# Global instance
_environment_manager = None


def get_environment_manager(base_dir: Optional[Path] = None) -> SmartEnvironmentManager:
    """Get the global environment manager instance."""
    global _environment_manager
    if _environment_manager is None:
        _environment_manager = SmartEnvironmentManager(base_dir)
    return _environment_manager