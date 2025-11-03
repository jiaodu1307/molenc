"""Dependency management and checking utilities."""

import importlib
import logging
import sys
from typing import Dict, List, Optional, Tuple, Any
from packaging import version
import warnings

# Import from our unified dependency utilities
from ..core.dependency_utils import (
    check_optional_dependency,
    is_package_available,
    check_dependencies as check_deps_basic,
    get_missing_dependencies,
    require_dependencies as require_deps_basic,
    warn_missing_optional,
    get_dependency_status
)


class DependencyChecker:
    """Check and manage package dependencies."""
    
    # Core dependencies that are always required
    CORE_DEPENDENCIES = {
        'numpy': '>=1.19.0',
        'pandas': '>=1.3.0',
        'scikit-learn': '>=1.0.0'
    }
    
    # Optional dependencies for specific features
    OPTIONAL_DEPENDENCIES = {
        'rdkit': {
            'package': 'rdkit',
            'version': '>=2022.03.1',
            'features': ['molecular_descriptors', 'fingerprints', 'standardization'],
            'install_hint': 'conda install -c conda-forge rdkit'
        },
        'torch': {
            'package': 'torch',
            'version': '>=1.10.0',
            'features': ['deep_learning', 'graph_neural_networks'],
            'install_hint': 'pip install torch'
        },
        'transformers': {
            'package': 'transformers',
            'version': '>=4.20.0',
            'features': ['molecular_transformers'],
            'install_hint': 'pip install transformers'
        },
        'torch_geometric': {
            'package': 'torch_geometric',
            'version': '>=2.0.0',
            'features': ['graph_neural_networks'],
            'install_hint': 'pip install torch-geometric'
        },
        'gensim': {
            'package': 'gensim',
            'version': '>=4.0.0',
            'features': ['word2vec'],
            'install_hint': 'pip install gensim'
        },
        'matplotlib': {
            'package': 'matplotlib',
            'version': '>=3.3.0',
            'features': ['visualization'],
            'install_hint': 'pip install matplotlib'
        },
        'seaborn': {
            'package': 'seaborn',
            'version': '>=0.11.0',
            'features': ['visualization'],
            'install_hint': 'pip install seaborn'
        },
        'plotly': {
            'package': 'plotly',
            'version': '>=5.0.0',
            'features': ['interactive_visualization'],
            'install_hint': 'pip install plotly'
        }
    }

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
    
    def check_package(self, package_name: str, 
                     min_version: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if a package is available and meets version requirements.
        
        Args:
            package_name: Name of the package to check
            min_version: Minimum required version (e.g., '>=1.0.0')
            
        Returns:
            Tuple of (is_available, installed_version, error_message)
        """
        # Use basic availability check first
        if not is_package_available(package_name):
            return (False, None, f"Package {package_name} not found")
        
        try:
            # Try to import the package
            module = importlib.import_module(package_name)
            
            # Get version if available
            installed_version: Optional[str] = 'unknown'
            for attr in ['__version__', 'version', 'VERSION']:
                if hasattr(module, attr):
                    try:
                        version_val = getattr(module, attr)
                        if version_val is not None:
                            if callable(version_val):
                                version_val = version_val()
                            installed_version = str(version_val)
                            break
                    except Exception:
                        continue
            
            # Check version requirement
            if min_version and installed_version and installed_version != 'unknown':
                try:
                    # Parse version requirement
                    if min_version.startswith('>='):
                        required_version = min_version[2:]
                        if version.parse(installed_version) < version.parse(required_version):
                            error_msg = f"Version mismatch: {installed_version} < required {required_version}"
                            return (False, installed_version, error_msg)
                        else:
                            return (True, installed_version, None)
                    else:
                        return (True, installed_version, None)
                except Exception as e:
                    self.logger.warning(f"Could not parse version for {package_name}: {e}")
                    return (True, installed_version, None)
            else:
                return (True, installed_version, None)
            
        except ImportError as e:
            return (False, None, f"Package not found: {str(e)}")
        except Exception as e:
            return (False, None, f"Error checking package: {str(e)}")
    
    def check_core_dependencies(self) -> Dict[str, Dict[str, Any]]:
        """
        Check all core dependencies.
        
        Returns:
            Dictionary with dependency check results
        """
        results: Dict[str, Dict[str, Any]] = {}
        
        for package, min_version in self.CORE_DEPENDENCIES.items():
            is_available, installed_version, error = self.check_package(package, min_version)
            results[package] = {
                'available': is_available,
                'installed_version': installed_version,
                'required_version': min_version,
                'error': error,
                'required': True
            }
        
        return results
    
    def check_optional_dependencies(self, features: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Check optional dependencies.
        
        Args:
            features: List of features to check dependencies for
            
        Returns:
            Dictionary with dependency check results
        """
        results: Dict[str, Dict[str, Any]] = {}
        
        for dep_name, dep_info in self.OPTIONAL_DEPENDENCIES.items():
            # Skip if specific features requested and this dependency doesn't support them
            if features:
                if not any(feature in dep_info['features'] for feature in features):
                    continue
            
            package_name = dep_info['package']
            min_version = dep_info['version']
            
            is_available, installed_version, error = self.check_package(package_name, min_version)
            
            results[dep_name] = {
                'available': is_available,
                'installed_version': installed_version,
                'required_version': min_version,
                'error': error,
                'required': False,
                'features': dep_info['features'],
                'install_hint': dep_info['install_hint']
            }
        
        return results
    
    def check_all_dependencies(self, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Check all dependencies (core and optional).
        
        Args:
            features: List of features to check dependencies for
            
        Returns:
            Complete dependency check results
        """
        core_results = self.check_core_dependencies()
        optional_results = self.check_optional_dependencies(features)
        
        # Count statistics
        core_available = sum(1 for r in core_results.values() if r['available'])
        core_total = len(core_results)
        
        optional_available = sum(1 for r in optional_results.values() if r['available'])
        optional_total = len(optional_results)
        
        # Check if core requirements are met
        core_satisfied = core_available == core_total
        
        return {
            'core': core_results,
            'optional': optional_results,
            'summary': {
                'core_satisfied': core_satisfied,
                'available_core': core_available,
                'total_core': core_total,
                'available_optional': optional_available,
                'total_optional': optional_total
            },
            'python_version': sys.version,
            'platform': sys.platform
        }
    
    def get_missing_dependencies(self, features: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Get lists of missing dependencies.
        
        Args:
            features: List of features to check dependencies for
            
        Returns:
            Dictionary with missing core and optional dependencies
        """
        results = self.check_all_dependencies(features)
        
        missing_core: List[str] = [
            name for name, info in results['core'].items()
            if not info['available']
        ]
        
        missing_optional: List[str] = [
            name for name, info in results['optional'].items()
            if not info['available']
        ]
        
        return {
            'core': missing_core,
            'optional': missing_optional
        }
    
    def generate_install_commands(self, missing_dependencies: Dict[str, List[str]]) -> List[str]:
        """
        Generate installation commands for missing dependencies.
        
        Args:
            missing_dependencies: Output from get_missing_dependencies()
            
        Returns:
            List of installation commands
        """
        commands: List[str] = []
        
        # Core dependencies
        if missing_dependencies['core']:
            core_packages: List[str] = []
            for dep in missing_dependencies['core']:
                if dep in self.CORE_DEPENDENCIES:
                    core_packages.append(dep)
            
            if core_packages:
                core_install_cmd = f"pip install {' '.join(core_packages)}"
                commands.append(core_install_cmd)
        
        # Optional dependencies
        for dep in missing_dependencies['optional']:
            if dep in self.OPTIONAL_DEPENDENCIES:
                install_hint = self.OPTIONAL_DEPENDENCIES[dep]['install_hint']
                commands.append(install_hint)
        
        return commands
    
    def print_dependency_report(self, features: Optional[List[str]] = None) -> None:
        """
        Print a comprehensive dependency report.
        
        Args:
            features: List of features to check dependencies for
        """
        results = self.check_all_dependencies(features)
        
        print("Dependency Report")
        print("=" * 50)
        
        # Core dependencies
        print("\nCore Dependencies:")
        print("-" * 20)
        for name, info in results['core'].items():
            status = "✓" if info['available'] else "✗"
            version_info = f" (v{info['installed_version']})" if info['installed_version'] else ""
            print(f"{status} {name}{version_info}")
            if not info['available'] and info['error']:
                print(f"    Error: {info['error']}")
        
        # Optional dependencies
        print("\nOptional Dependencies:")
        print("-" * 25)
        for name, info in results['optional'].items():
            status = "✓" if info['available'] else "✗"
            version_info = f" (v{info['installed_version']})" if info['installed_version'] else ""
            features_info = f" [{', '.join(info['features'])}]"
            print(f"{status} {name}{version_info}{features_info}")
            if not info['available']:
                print(f"    Install: {info['install_hint']}")
        
        # Summary
        print("\nSummary:")
        print("-" * 30)
        print(f"Core dependencies: {results['summary']['available_core']}/{results['summary']['total_core']} available")
        print(f"Optional dependencies: {results['summary']['available_optional']}/{results['summary']['total_optional']} available")
        
        if not results['summary']['core_satisfied']:
            print("\n⚠️  Warning: Some core dependencies are missing!")
            missing = self.get_missing_dependencies(features)
            if missing['core']:
                commands = self.generate_install_commands(missing)
                print("\nTo install missing dependencies, run:")
                for cmd in commands:
                    print(f"  {cmd}")
        else:
            print("\n✓ All core dependencies are satisfied!")


def check_dependencies(features: Optional[List[str]] = None, 
                      print_report: bool = True) -> Dict[str, Any]:
    """
    Convenience function to check dependencies.
    
    Args:
        features: List of features to check dependencies for
        print_report: Whether to print the dependency report
        
    Returns:
        Dependency check results
    """
    checker = DependencyChecker()
    results = checker.check_all_dependencies(features)
    
    if print_report:
        checker.print_dependency_report(features)
    
    return results


def require_dependencies(dependencies: List[str], 
                        feature_name: Optional[str] = None) -> Any:
    """
    Decorator to require specific dependencies for a function or class.
    
    Args:
        dependencies: List of required dependency names
        feature_name: Name of the feature (for error messages)
    """
    def decorator(func_or_class):
        # Use basic dependency checking
        missing = get_missing_dependencies(dependencies)
        if missing:
            feature_desc = f" for {feature_name}" if feature_name else ""
            warn_missing_optional(dependencies, f"Missing dependencies{feature_desc}")
        
        return func_or_class
    
    return decorator


def warn_missing_optional(dependencies: List[str], 
                         message: Optional[str] = None) -> None:
    """
    Warn about missing optional dependencies.
    
    Args:
        dependencies: List of dependency names
        message: Custom warning message
    """
    # Use the unified warning function
    warn_missing_optional(dependencies, message)