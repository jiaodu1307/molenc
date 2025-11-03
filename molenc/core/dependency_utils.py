"""
Unified dependency management utilities.

This module provides centralized functions for checking and managing dependencies
across the MolEnc library.
"""

import importlib
import logging
import subprocess
import sys
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple, Union


logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def is_package_available(package_name: str, min_version: Optional[str] = None) -> bool:
    """
    Check if a package is available and optionally meets minimum version requirement.
    
    Args:
        package_name: Name of the package to check
        min_version: Minimum version requirement (optional)
        
    Returns:
        True if package is available and meets version requirement
    """
    try:
        module = importlib.import_module(package_name)
        
        if min_version:
            # Try to get version from common attributes
            version = getattr(module, '__version__', None)
            if version is None:
                version = getattr(module, 'version', None)
            if version is None:
                version = getattr(module, 'VERSION', None)
            
            if version and _compare_versions(version, min_version) < 0:
                return False
        
        return True
    except ImportError:
        return False


def _compare_versions(version1: str, version2: str) -> int:
    """
    Compare two version strings.
    
    Returns:
        -1 if version1 < version2
         0 if version1 == version2
         1 if version1 > version2
    """
    def normalize_version(v):
        return [int(x) for x in v.split('.') if x.isdigit()]
    
    v1_parts = normalize_version(version1)
    v2_parts = normalize_version(version2)
    
    # Pad shorter version with zeros
    max_len = max(len(v1_parts), len(v2_parts))
    v1_parts.extend([0] * (max_len - len(v1_parts)))
    v2_parts.extend([0] * (max_len - len(v2_parts)))
    
    for v1, v2 in zip(v1_parts, v2_parts):
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
    
    return 0


def check_optional_dependency(package_name: str, 
                            min_version: Optional[str] = None,
                            extra_message: str = "") -> bool:
    """
    Check if an optional dependency is available.
    
    Args:
        package_name: Name of the package to check
        min_version: Minimum version requirement
        extra_message: Additional message for logging
        
    Returns:
        True if dependency is available
    """
    available = is_package_available(package_name, min_version)
    
    if not available:
        message = f"Optional dependency '{package_name}' not available"
        if min_version:
            message += f" (requires >= {min_version})"
        if extra_message:
            message += f". {extra_message}"
        logger.debug(message)
    
    return available


def check_dependencies(dependencies: List[str], 
                      min_versions: Optional[Dict[str, str]] = None) -> Tuple[bool, List[str]]:
    """
    Check multiple dependencies.
    
    Args:
        dependencies: List of package names to check
        min_versions: Optional dict of minimum versions
        
    Returns:
        Tuple of (all_available, missing_packages)
    """
    min_versions = min_versions or {}
    missing = []
    
    for dep in dependencies:
        min_ver = min_versions.get(dep)
        if not is_package_available(dep, min_ver):
            missing.append(dep)
    
    return len(missing) == 0, missing


def get_missing_dependencies(dependencies: List[str],
                           min_versions: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Get list of missing dependencies.
    
    Args:
        dependencies: List of package names to check
        min_versions: Optional dict of minimum versions
        
    Returns:
        List of missing package names
    """
    _, missing = check_dependencies(dependencies, min_versions)
    return missing


def require_dependencies(dependencies: Union[List[str], str],
                        feature_name: Optional[str] = None,
                        min_versions: Optional[Dict[str, str]] = None,
                        error_message: Optional[str] = None):
    """
    Require dependencies to be available, can be used as decorator or function.
    
    Args:
        dependencies: List of package names to check or single package name
        feature_name: Name of the feature (for decorator usage)
        min_versions: Optional dict of minimum versions
        error_message: Custom error message
        
    Raises:
        ImportError: If any dependency is missing
    """
    # Handle both decorator and function usage
    if isinstance(dependencies, str):
        # Single dependency as string
        deps_list = [dependencies]
    else:
        deps_list = dependencies
    
    def decorator(cls_or_func):
        # Check dependencies when decorator is applied
        all_available, missing = check_dependencies(deps_list, min_versions)
        
        if not all_available:
            if error_message:
                message = error_message
            else:
                feature = feature_name or getattr(cls_or_func, '__name__', 'feature')
                message = f"Missing required dependencies for {feature}: {', '.join(missing)}"
            raise ImportError(message)
        
        return cls_or_func
    
    # If called as function (not decorator)
    if feature_name is None and min_versions is None and error_message is None:
        # Direct function call
        all_available, missing = check_dependencies(deps_list, {})
        
        if not all_available:
            message = f"Missing required dependencies: {', '.join(missing)}"
            raise ImportError(message)
        return
    
    # Return decorator
    return decorator


def warn_missing_optional(dependencies: List[str],
                         min_versions: Optional[Dict[str, str]] = None,
                         feature_name: str = "feature") -> None:
    """
    Warn about missing optional dependencies.
    
    Args:
        dependencies: List of package names to check
        min_versions: Optional dict of minimum versions
        feature_name: Name of the feature that requires these dependencies
    """
    missing = get_missing_dependencies(dependencies, min_versions)
    
    if missing:
        logger.warning(
            f"Optional dependencies missing for {feature_name}: {', '.join(missing)}. "
            f"Some functionality may be limited."
        )


def get_dependency_status(dependencies: List[str],
                         min_versions: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get detailed status of dependencies.
    
    Args:
        dependencies: List of package names to check
        min_versions: Optional dict of minimum versions
        
    Returns:
        Dict with dependency status information
    """
    min_versions = min_versions or {}
    status = {}
    
    for dep in dependencies:
        min_ver = min_versions.get(dep)
        available = is_package_available(dep, min_ver)
        
        dep_status = {
            'available': available,
            'required_version': min_ver,
            'installed_version': None
        }
        
        if available:
            try:
                module = importlib.import_module(dep)
                version = getattr(module, '__version__', None)
                if version is None:
                    version = getattr(module, 'version', None)
                if version is None:
                    version = getattr(module, 'VERSION', None)
                dep_status['installed_version'] = version
            except Exception:
                pass
        
        status[dep] = dep_status
    
    return status