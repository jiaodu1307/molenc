"""Advanced dependency manager for MolEnc.

This module provides advanced dependency management capabilities,
including automatic virtual environment creation and configuration.
"""

import os
import sys
import subprocess
import logging
import json
import venv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict

try:
    import conda.cli.python_api
    CONDA_AVAILABLE = True
except ImportError:
    CONDA_AVAILABLE = False

from .dependencies import DependencyChecker
from ..core.exceptions import MolEncError


class EnvironmentType(Enum):
    """Types of environments."""
    LOCAL = "local"
    VIRTUAL_ENV = "virtual_env"
    CONDA_ENV = "conda_env"


class DependencyLevel(Enum):
    """Levels of dependency satisfaction."""
    NONE = "none"  # No dependencies available
    CORE = "core"  # Only core dependencies available
    PARTIAL = "partial"  # Some optional dependencies available
    FULL = "full"   # All dependencies available


@dataclass
class EnvironmentInfo:
    """Information about an environment."""
    env_type: EnvironmentType
    path: Path
    python_executable: str
    dependencies: Dict[str, bool]  # dependency_name -> is_available
    status: str  # "ready", "needs_setup", "error"
    error_message: Optional[str] = None


class AdvancedDependencyManager:
    """Advanced dependency manager with virtual environment support."""
    
    def __init__(self, base_env_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.base_env_dir = base_env_dir or Path.home() / ".molenc" / "environments"
        self.base_env_dir.mkdir(parents=True, exist_ok=True)
        self.dependency_checker = DependencyChecker()
        
    def check_encoder_dependencies(self, encoder_type: str) -> Tuple[DependencyLevel, List[str], List[str]]:
        """
        Check dependencies for a specific encoder.
        
        Args:
            encoder_type: Type of encoder to check
            
        Returns:
            Tuple of (dependency_level, available_deps, missing_deps)
        """
        # Define dependencies for each encoder type
        # In a real implementation, this would be more comprehensive
        encoder_dependencies = {
            'morgan': ['rdkit'],
            'maccs': ['rdkit'],
            'chemberta': ['rdkit', 'transformers', 'torch'],
            'gcn': ['rdkit', 'torch', 'torch_geometric'],
            'unimol': ['rdkit', 'torch', 'unimol_tools'],  # Assuming unimol_tools is the package
        }
        
        required_deps = encoder_dependencies.get(encoder_type, [])
        if not required_deps:
            # If we don't have specific dependencies listed, 
            # assume it needs at least rdkit and check what's available
            required_deps = ['rdkit']
        
        available_deps = []
        missing_deps = []
        
        for dep in required_deps:
            # For this implementation, we'll just check if it's importable
            # A more robust implementation would check versions, etc.
            try:
                __import__(dep)
                available_deps.append(dep)
            except ImportError:
                missing_deps.append(dep)
        
        # Determine dependency level
        if len(available_deps) == len(required_deps):
            level = DependencyLevel.FULL
        elif len(available_deps) > 0:
            level = DependencyLevel.PARTIAL
        elif self._check_core_dependencies():
            level = DependencyLevel.CORE
        else:
            level = DependencyLevel.NONE
            
        return level, available_deps, missing_deps
    
    def _check_core_dependencies(self) -> bool:
        """Check if core dependencies are available."""
        core_results = self.dependency_checker.check_core_dependencies()
        return all(dep_info['available'] for dep_info in core_results.values())
    
    def get_encoder_environment(self, encoder_type: str) -> EnvironmentInfo:
        """
        Get or create environment for an encoder.
        
        Args:
            encoder_type: Type of encoder
            
        Returns:
            Environment information
        """
        # Check current environment first
        level, available_deps, missing_deps = self.check_encoder_dependencies(encoder_type)
        
        if level == DependencyLevel.FULL:
            # Current environment is sufficient
            return EnvironmentInfo(
                env_type=EnvironmentType.LOCAL,
                path=Path(sys.prefix),
                python_executable=sys.executable,
                dependencies={dep: True for dep in available_deps},
                status="ready"
            )
        
        # Need to create or use an isolated environment
        env_path = self.base_env_dir / encoder_type
        env_info = self._get_or_create_virtual_environment(encoder_type, env_path, missing_deps)
        
        return env_info
    
    def _get_or_create_virtual_environment(self, encoder_type: str, env_path: Path, 
                                         missing_deps: List[str]) -> EnvironmentInfo:
        """
        Get or create a virtual environment for an encoder.
        
        Args:
            encoder_type: Type of encoder
            env_path: Path to environment
            missing_deps: List of missing dependencies
            
        Returns:
            Environment information
        """
        # Check if environment already exists and is valid
        if env_path.exists():
            env_info = self._validate_environment(env_path)
            if env_info.status == "ready":
                return env_info
        
        # Create new virtual environment
        try:
            self.logger.info(f"Creating virtual environment for {encoder_type} at {env_path}")
            venv.create(env_path, with_pip=True)
            
            # Upgrade pip
            pip_exe = self._get_pip_executable(env_path)
            subprocess.run([pip_exe, "install", "--upgrade", "pip"], 
                          check=True, capture_output=True)
            
            # Install dependencies
            self._install_dependencies(env_path, missing_deps)
            
            # Validate environment
            return self._validate_environment(env_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create virtual environment for {encoder_type}: {e}")
            return EnvironmentInfo(
                env_type=EnvironmentType.VIRTUAL_ENV,
                path=env_path,
                python_executable="",
                dependencies={},
                status="error",
                error_message=str(e)
            )
    
    def _validate_environment(self, env_path: Path) -> EnvironmentInfo:
        """
        Validate a virtual environment.
        
        Args:
            env_path: Path to environment
            
        Returns:
            Environment information
        """
        try:
            python_exe = self._get_python_executable(env_path)
            pip_exe = self._get_pip_executable(env_path)
            
            # Check if Python executable works
            result = subprocess.run([python_exe, "--version"], 
                                  capture_output=True, text=True, check=True)
            if not result.stdout.startswith("Python"):
                raise Exception("Invalid Python executable")
            
            # Check if pip works
            result = subprocess.run([pip_exe, "--version"], 
                                  capture_output=True, text=True, check=True)
            
            # Basic validation successful
            return EnvironmentInfo(
                env_type=EnvironmentType.VIRTUAL_ENV,
                path=env_path,
                python_executable=python_exe,
                dependencies={},  # We're not checking specific deps in this simplified version
                status="ready"
            )
            
        except Exception as e:
            self.logger.error(f"Environment validation failed for {env_path}: {e}")
            return EnvironmentInfo(
                env_type=EnvironmentType.VIRTUAL_ENV,
                path=env_path,
                python_executable="",
                dependencies={},
                status="error",
                error_message=str(e)
            )
    
    def _install_dependencies(self, env_path: Path, dependencies: List[str]):
        """
        Install dependencies in environment.
        
        Args:
            env_path: Path to environment
            dependencies: List of dependencies to install
        """
        pip_exe = self._get_pip_executable(env_path)
        
        if not dependencies:
            return
            
        # Simple mapping of dependency names to pip package names
        # In a real implementation, this would be more sophisticated
        dep_to_pip = {
            'rdkit': 'rdkit',
            'torch': 'torch',
            'torch_geometric': 'torch-geometric',
            'transformers': 'transformers',
            'gensim': 'gensim',
            'unimol_tools': 'unimol-tools'  # Hypothetical package name
        }
        
        pip_packages = [dep_to_pip.get(dep, dep) for dep in dependencies]
        
        self.logger.info(f"Installing dependencies: {pip_packages}")
        subprocess.run([pip_exe, "install"] + pip_packages, 
                      check=True, capture_output=True)
    
    def _get_python_executable(self, env_path: Path) -> str:
        """Get Python executable for environment."""
        if sys.platform == "win32":
            return str(env_path / "Scripts" / "python.exe")
        else:
            return str(env_path / "bin" / "python")
    
    def _get_pip_executable(self, env_path: Path) -> str:
        """Get pip executable for environment."""
        if sys.platform == "win32":
            return str(env_path / "Scripts" / "pip.exe")
        else:
            return str(env_path / "bin" / "pip")


def get_dependency_manager(base_env_dir: Optional[Path] = None) -> AdvancedDependencyManager:
    """Get the global dependency manager instance."""
    return AdvancedDependencyManager(base_env_dir)


def check_encoder_readiness(encoder_type: str) -> Tuple[bool, DependencyLevel, str]:
    """
    Check if an encoder is ready to use.
    
    Args:
        encoder_type: Type of encoder to check
        
    Returns:
        Tuple of (is_ready, dependency_level, status_message)
    """
    manager = get_dependency_manager()
    level, available_deps, missing_deps = manager.check_encoder_dependencies(encoder_type)
    
    if level == DependencyLevel.FULL:
        return True, level, f"All dependencies available: {available_deps}"
    elif level == DependencyLevel.PARTIAL:
        return False, level, f"Partial dependencies available. Available: {available_deps}, Missing: {missing_deps}"
    elif level == DependencyLevel.CORE:
        return False, level, f"Only core dependencies available. Missing encoder dependencies: {missing_deps}"
    else:
        return False, level, f"Core dependencies missing. Missing: {missing_deps}"