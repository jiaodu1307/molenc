"""Environment manager for MolEnc.

This module provides the EnvironmentManager class for managing
computational environments and system configurations.
"""

import os
import sys
import platform
from typing import Dict, List, Optional, Any
from .dependencies import DependencyChecker
from .config import EnvironmentConfig


class EnvironmentManager:
    """Manages computational environments and system configurations."""
    
    def __init__(self, config: Optional[EnvironmentConfig] = None) -> None:
        """Initialize the environment manager.
        
        Args:
            config: Environment configuration to use
        """
        self.config = config or EnvironmentConfig()
        self.dependency_checker = DependencyChecker()
        self._system_info: Optional[Dict[str, Any]] = None
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information.
        
        Returns:
            Dictionary containing system information
        """
        if self._system_info is not None:
            return self._system_info
            
        self._system_info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'python_executable': sys.executable,
            'environment_variables': dict(os.environ)
        }
        
        return self._system_info
    
    def check_environment(self, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Check the current environment.
        
        Args:
            features: List of features to check dependencies for
            
        Returns:
            Dictionary containing environment information
        """
        return {
            'system_info': self.get_system_info(),
            'dependencies': self.dependency_checker.check_all_dependencies(features),
            'config': self.config.to_dict()
        }
    
    def validate_environment(self, required_features: Optional[List[str]] = None) -> bool:
        """Validate the current environment.
        
        Args:
            required_features: List of required features
            
        Returns:
            True if environment is valid, False otherwise
        """
        env_info = self.check_environment(required_features)
        dependencies = env_info['dependencies']
        
        # Check core dependencies
        core_results = dependencies['core']
        core_available = sum(1 for r in core_results.values() if r['available'])
        core_total = len(core_results)
        core_satisfied = core_available == core_total
        
        # Check optional dependencies for required features
        feature_satisfied = True
        
        if required_features:
            optional_results = dependencies['optional']
            for feature in required_features:
                feature_available = any(
                    r['available'] and feature in r.get('features', []) 
                    for r in optional_results.values()
                )
                if not feature_available:
                    feature_satisfied = False
        
        return core_satisfied and feature_satisfied
    
    def get_installation_instructions(self, required_features: Optional[List[str]] = None) -> List[str]:
        """Get installation instructions for missing dependencies.
        
        Args:
            required_features: List of required features
            
        Returns:
            List of installation commands
        """
        env_info = self.check_environment(required_features)
        dependencies = env_info['dependencies']
        
        missing_deps = self.dependency_checker.get_missing_dependencies(required_features)
        return self.dependency_checker.generate_install_commands(missing_deps)
    
    def __str__(self) -> str:
        """String representation of the environment manager."""
        return f"EnvironmentManager(config={self.config})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"EnvironmentManager(config={self.config})"