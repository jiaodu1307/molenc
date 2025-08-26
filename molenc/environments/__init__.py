"""Environment management for MolEnc.

This module provides tools for managing different computational environments,
handling dependencies, and ensuring compatibility across different setups.
"""

from .manager import EnvironmentManager
from .dependencies import DependencyChecker, check_dependencies
from .config import EnvironmentConfig
from .advanced_dependency_manager import (
    AdvancedDependencyManager, 
    get_dependency_manager, 
    check_encoder_readiness,
    EnvironmentType as AdvancedEnvironmentType,
    DependencyLevel
)

__all__ = [
    'EnvironmentManager',
    'DependencyChecker', 
    'check_dependencies',
    'EnvironmentConfig',
    'AdvancedDependencyManager',
    'get_dependency_manager',
    'check_encoder_readiness',
    'AdvancedEnvironmentType',
    'DependencyLevel'
]