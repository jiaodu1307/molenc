"""Advanced dependency management system for MolEnc.

This module provides intelligent dependency management with support for:
- Optional dependencies with graceful fallbacks
- Progressive installation strategies
- Dependency conflict resolution
- Version compatibility checking
"""

import sys
import importlib
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from ..core.exceptions import DependencyError
from .dependency_utils import (
    check_optional_dependency,
    is_package_available,
    check_dependencies,
    get_missing_dependencies,
    require_dependencies,
    warn_missing_optional,
    get_dependency_status as get_basic_status
)


@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    version_spec: Optional[str] = None
    optional: bool = False
    fallback_available: bool = False
    install_command: Optional[str] = None
    description: Optional[str] = None


class AdvancedDependencyManager:
    """Advanced dependency manager with intelligent resolution."""
    
    def __init__(self):
        self._dependencies: Dict[str, DependencyInfo] = {}
        self._import_cache: Dict[str, Any] = {}
        
    def register_dependency(self, 
                          name: str,
                          version_spec: Optional[str] = None,
                          optional: bool = False,
                          fallback_available: bool = False,
                          install_command: Optional[str] = None,
                          description: Optional[str] = None) -> None:
        """Register a dependency with the manager.
        
        Args:
            name: Package name
            version_spec: Version specification (e.g., '>=1.0.0')
            optional: Whether the dependency is optional
            fallback_available: Whether a fallback implementation exists
            install_command: Custom installation command
            description: Human-readable description
        """
        self._dependencies[name] = DependencyInfo(
            name=name,
            version_spec=version_spec,
            optional=optional,
            fallback_available=fallback_available,
            install_command=install_command,
            description=description
        )
        
    def check_dependency(self, name: str) -> bool:
        """Check if a dependency is available.
        
        Args:
            name: Package name
            
        Returns:
            True if dependency is available, False otherwise
        """
        return is_package_available(name)
            
    def import_dependency(self, name: str) -> Any:
        """Import a dependency with caching.
        
        Args:
            name: Package name
            
        Returns:
            Imported module
            
        Raises:
            DependencyError: If dependency is not available
        """
        if name in self._import_cache:
            return self._import_cache[name]
            
        if not self.check_dependency(name):
            dep_info = self._dependencies.get(name)
            if dep_info and dep_info.optional and dep_info.fallback_available:
                return None
            raise DependencyError(name, f"Required dependency '{name}' not available")
            
        try:
            module = importlib.import_module(name)
            self._import_cache[name] = module
            return module
        except ImportError as e:
            raise DependencyError(name, str(e))
            
    def check_all_dependencies(self) -> Tuple[List[str], List[str]]:
        """Check all registered dependencies.
        
        Returns:
            Tuple of (available_deps, missing_deps)
        """
        available = []
        missing = []
        
        for name, info in self._dependencies.items():
            if self.check_dependency(name):
                available.append(name)
            else:
                missing.append(name)
                
        return available, missing
        
    def get_missing_required_dependencies(self) -> List[str]:
        """Get list of missing required dependencies.
        
        Returns:
            List of missing required dependency names
        """
        missing = []
        for name, info in self._dependencies.items():
            if not info.optional and not self.check_dependency(name):
                missing.append(name)
        return missing
        
    def suggest_installation(self, name: str) -> str:
        """Suggest installation command for a dependency.
        
        Args:
            name: Package name
            
        Returns:
            Installation command suggestion
        """
        dep_info = self._dependencies.get(name)
        if dep_info and dep_info.install_command:
            return dep_info.install_command
            
        if dep_info and dep_info.version_spec:
            return f"pip install '{name}{dep_info.version_spec}'"
        else:
            return f"pip install {name}"
            
    def install_dependency(self, name: str, force: bool = False) -> bool:
        """Attempt to install a dependency.
        
        Args:
            name: Package name
            force: Force installation even if already available
            
        Returns:
            True if installation succeeded, False otherwise
        """
        if not force and self.check_dependency(name):
            return True
            
        install_cmd = self.suggest_installation(name)
        try:
            subprocess.run(
                install_cmd.split(),
                check=True,
                capture_output=True,
                text=True
            )
            # Clear cache to force recheck
            self._import_cache.pop(name, None)
            return self.check_dependency(name)
        except subprocess.CalledProcessError:
            return False
            
    def get_dependency_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all dependencies.
        
        Returns:
            Dictionary with dependency status information
        """
        status = {}
        for name, info in self._dependencies.items():
            available = self.check_dependency(name)
            status[name] = {
                'available': available,
                'optional': info.optional,
                'fallback_available': info.fallback_available,
                'version_spec': info.version_spec,
                'description': info.description,
                'install_command': self.suggest_installation(name) if not available else None
            }
        return status
        
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._import_cache.clear()


# Global dependency manager instance
dependency_manager = AdvancedDependencyManager()

# Register common dependencies
dependency_manager.register_dependency(
    'torch',
    version_spec='>=1.8.0',
    optional=False,
    description='PyTorch for deep learning models'
)

dependency_manager.register_dependency(
    'rdkit',
    optional=False,
    description='RDKit for molecular processing'
)

dependency_manager.register_dependency(
    'transformers',
    version_spec='>=4.0.0',
    optional=True,
    fallback_available=True,
    description='Hugging Face Transformers for advanced models'
)

dependency_manager.register_dependency(
    'requests',
    optional=True,
    fallback_available=True,
    description='HTTP library for cloud API access'
)

# Convenience functions using dependency_utils
def check_advanced_dependencies(dependencies: List[str]) -> Tuple[bool, List[str]]:
    """Check multiple dependencies using basic utilities."""
    return check_dependencies(dependencies)

def require_advanced_dependencies(dependencies: List[str]):
    """Require dependencies with advanced error handling."""
    return require_dependencies(dependencies)