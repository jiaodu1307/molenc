"""Simplified dependency management for encoders.

This module provides a streamlined approach to dependency management
that reduces complexity while maintaining functionality.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from .dependency_utils import (
    check_optional_dependency,
    is_package_available,
    check_dependencies,
    get_missing_dependencies,
    check_encoder_dependencies as _check_encoder_dependencies,
    require_dependencies,
    warn_missing_optional,
    get_dependency_status
)

logger = logging.getLogger(__name__)


class SimpleDependencyManager:
    """Simplified dependency manager for encoder requirements."""
    
    # Core dependency groups
    DEPENDENCY_GROUPS = {
        'rdkit': ['rdkit', 'rdkit-pypi'],
        'torch': ['torch', 'pytorch'],
        'transformers': ['transformers', 'huggingface-hub'],
        'unimol': ['unimol_tools', 'unimol'],
        'dgl': ['dgl', 'dgllife'],
        'sklearn': ['scikit-learn', 'sklearn']
    }
    
    # Encoder requirements mapping
    ENCODER_REQUIREMENTS = {
        'morgan': ['rdkit'],
        'maccs': ['rdkit'],
        'chemberta': ['torch', 'transformers', 'rdkit'],
        'gcn': ['torch', 'dgl', 'rdkit'],
        'unimol': ['torch', 'rdkit', 'unimol']
    }
    
    def __init__(self):
        self._cache: Dict[str, bool] = {}
        self._missing_cache: Dict[str, List[str]] = {}
    
    def is_available(self, package: str) -> bool:
        """
        Check if a package is available for import.
        
        Args:
            package: Package name to check
            
        Returns:
            True if package is available, False otherwise
        """
        return is_package_available(package)
    
    def check_encoder_dependencies(self, encoder_name: str) -> Tuple[bool, List[str]]:
        """
        Check if all dependencies for an encoder are available.
        
        Args:
            encoder_name: Name of the encoder
            
        Returns:
            Tuple of (all_available, missing_packages)
        """
        return _check_encoder_dependencies(encoder_name)
    
    def _check_dependency_group(self, group: str) -> bool:
        """
        Check if any package in a dependency group is available.
        
        Args:
            group: Dependency group name
            
        Returns:
            True if at least one package in the group is available
        """
        if group in self._cache:
            return self._cache[group]
        
        packages = self.DEPENDENCY_GROUPS.get(group, [group])
        
        for package in packages:
            if check_optional_dependency(package):
                self._cache[group] = True
                return True
        
        self._cache[group] = False
        return False
    
    def get_missing_dependencies(self, encoder_name: str) -> List[str]:
        """
        Get list of missing dependencies for an encoder.
        
        Args:
            encoder_name: Name of the encoder
            
        Returns:
            List of missing dependency group names
        """
        if encoder_name not in self.ENCODER_REQUIREMENTS:
            return []
        
        required_groups = self.ENCODER_REQUIREMENTS[encoder_name]
        missing = []
        
        for group in required_groups:
            if not self._check_dependency_group(group):
                missing.append(group)
        
        return missing
    
    def get_installation_command(self, missing_deps: List[str]) -> str:
        """
        Generate installation command for missing dependencies.
        
        Args:
            missing_deps: List of missing dependency group names
            
        Returns:
            Installation command string
        """
        packages_to_install = []
        
        for dep_group in missing_deps:
            # Get the first (preferred) package from each group
            packages = self.DEPENDENCY_GROUPS.get(dep_group, [dep_group])
            packages_to_install.append(packages[0])
        
        if packages_to_install:
            return f"pip install {' '.join(packages_to_install)}"
        return ""
    
    def list_available_encoders(self) -> List[str]:
        """
        List encoders that have all dependencies available.
        
        Returns:
            List of available encoder names
        """
        available = []
        for encoder_name in self.ENCODER_REQUIREMENTS:
            all_available, _ = self.check_encoder_dependencies(encoder_name)
            if all_available:
                available.append(encoder_name)
        return available
    
    def get_all_missing_dependencies(self) -> Dict[str, List[str]]:
        """
        Get all missing dependencies for all encoders.
        
        Returns:
            Dictionary mapping encoder names to missing dependencies
        """
        all_missing = {}
        for encoder_name in self.ENCODER_REQUIREMENTS:
            missing = self.get_missing_dependencies(encoder_name)
            if missing:
                all_missing[encoder_name] = missing
        return all_missing


# Global instance
dependency_manager = SimpleDependencyManager()

# Convenience functions
def check_encoder_deps(encoder_name: str) -> Tuple[bool, List[str]]:
    """Check encoder dependencies using global manager."""
    return dependency_manager.check_encoder_dependencies(encoder_name)

def get_available_encoders() -> List[str]:
    """Get list of available encoders."""
    return dependency_manager.list_available_encoders()

def require_simple_dependencies(dependencies: List[str]):
    """Decorator to require dependencies for a class."""
    def decorator(cls):
        # Check dependencies when class is defined
        missing = get_missing_dependencies(dependencies)
        if missing:
            logger.warning(f"Missing dependencies for {cls.__name__}: {missing}")
        return cls
    return decorator