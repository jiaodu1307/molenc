"""Pre-compiled package manager for molecular encoders.

This module manages pre-compiled binary packages to solve
complex dependency issues and provide easy installation
of molecular encoders across different platforms.
"""

import os
import sys
import platform
import hashlib
import tarfile
import zipfile
import shutil
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from urllib.parse import urljoin


class Platform(Enum):
    """Supported platforms."""
    LINUX_X86_64 = "linux_x86_64"
    LINUX_ARM64 = "linux_arm64"
    MACOS_X86_64 = "macos_x86_64"
    MACOS_ARM64 = "macos_arm64"
    WINDOWS_X86_64 = "windows_x86_64"
    UNKNOWN = "unknown"


class PackageType(Enum):
    """Types of pre-compiled packages."""
    ENCODER_BINARY = "encoder_binary"
    PYTHON_WHEEL = "python_wheel"
    CONDA_PACKAGE = "conda_package"
    DOCKER_IMAGE = "docker_image"
    STATIC_LIBRARY = "static_library"


class CompressionFormat(Enum):
    """Supported compression formats."""
    TAR_GZ = "tar.gz"
    TAR_XZ = "tar.xz"
    ZIP = "zip"
    NONE = "none"


@dataclass
class PackageSpec:
    """Specification for a pre-compiled package."""
    name: str
    version: str
    encoder_type: str
    platform: Platform
    package_type: PackageType
    compression: CompressionFormat
    download_url: str
    checksum: str
    checksum_algorithm: str = "sha256"
    size_bytes: Optional[int] = None
    dependencies: Optional[List[str]] = None
    python_version: Optional[str] = None
    cuda_version: Optional[str] = None
    description: Optional[str] = None
    install_script: Optional[str] = None
    uninstall_script: Optional[str] = None


@dataclass
class PackageInfo:
    """Information about an installed package."""
    spec: PackageSpec
    install_path: Path
    install_time: float
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None


class PackageManagerError(Exception):
    """Exception raised for package manager errors."""
    pass


class PrecompiledPackageManager:
    """Manager for pre-compiled molecular encoder packages."""
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 registry_url: str = "https://packages.molenc.org"):
        self.cache_dir = cache_dir or Path.home() / ".molenc" / "packages"
        self.registry_url = registry_url
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Package registry and installed packages
        self._registry: Dict[str, List[PackageSpec]] = {}
        self._installed: Dict[str, PackageInfo] = {}
        
        # Load installed packages info
        self._load_installed_packages()
        
        # Initialize HTTP session
        self._session = None
        self._init_session()
    
    def _init_session(self):
        """Initialize HTTP session for downloads."""
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from requests.packages.urllib3.util.retry import Retry
            
            self._session = requests.Session()
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
            
            # Set headers
            self._session.headers.update({
                'User-Agent': 'molenc-package-manager/1.0'
            })
            
        except ImportError:
            self.logger.warning("requests library not available, downloads will be disabled")
    
    def _get_current_platform(self) -> Platform:
        """Detect the current platform."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == "linux":
            if machine in ["x86_64", "amd64"]:
                return Platform.LINUX_X86_64
            elif machine in ["arm64", "aarch64"]:
                return Platform.LINUX_ARM64
        elif system == "darwin":
            if machine in ["x86_64", "amd64"]:
                return Platform.MACOS_X86_64
            elif machine in ["arm64", "aarch64"]:
                return Platform.MACOS_ARM64
        elif system == "windows":
            if machine in ["x86_64", "amd64"]:
                return Platform.WINDOWS_X86_64
        
        return Platform.UNKNOWN
    
    def _load_installed_packages(self):
        """Load information about installed packages."""
        installed_file = self.cache_dir / "installed.json"
        
        if not installed_file.exists():
            return
        
        try:
            with open(installed_file, 'r') as f:
                data = json.load(f)
            
            for package_id, package_data in data.items():
                spec_data = package_data['spec']
                spec = PackageSpec(
                    name=spec_data['name'],
                    version=spec_data['version'],
                    encoder_type=spec_data['encoder_type'],
                    platform=Platform(spec_data['platform']),
                    package_type=PackageType(spec_data['package_type']),
                    compression=CompressionFormat(spec_data['compression']),
                    download_url=spec_data['download_url'],
                    checksum=spec_data['checksum'],
                    checksum_algorithm=spec_data.get('checksum_algorithm', 'sha256'),
                    size_bytes=spec_data.get('size_bytes'),
                    dependencies=spec_data.get('dependencies'),
                    python_version=spec_data.get('python_version'),
                    cuda_version=spec_data.get('cuda_version'),
                    description=spec_data.get('description'),
                    install_script=spec_data.get('install_script'),
                    uninstall_script=spec_data.get('uninstall_script')
                )
                
                info = PackageInfo(
                    spec=spec,
                    install_path=Path(package_data['install_path']),
                    install_time=package_data['install_time'],
                    is_active=package_data.get('is_active', True),
                    metadata=package_data.get('metadata')
                )
                
                self._installed[package_id] = info
                
        except Exception as e:
            self.logger.error(f"Failed to load installed packages: {e}")
    
    def _save_installed_packages(self):
        """Save information about installed packages."""
        installed_file = self.cache_dir / "installed.json"
        
        try:
            data = {}
            for package_id, info in self._installed.items():
                data[package_id] = {
                    'spec': asdict(info.spec),
                    'install_path': str(info.install_path),
                    'install_time': info.install_time,
                    'is_active': info.is_active,
                    'metadata': info.metadata
                }
                
                # Convert enums to strings
                data[package_id]['spec']['platform'] = info.spec.platform.value
                data[package_id]['spec']['package_type'] = info.spec.package_type.value
                data[package_id]['spec']['compression'] = info.spec.compression.value
            
            with open(installed_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save installed packages: {e}")
    
    def update_registry(self) -> bool:
        """Update the package registry from remote."""
        if not self._session:
            self.logger.error("HTTP session not available")
            return False
        
        try:
            url = urljoin(self.registry_url, "/registry.json")
            response = self._session.get(url, timeout=30)
            response.raise_for_status()
            
            registry_data = response.json()
            
            # Parse registry data
            self._registry.clear()
            for encoder_type, packages in registry_data.items():
                package_specs = []
                for package_data in packages:
                    spec = PackageSpec(
                        name=package_data['name'],
                        version=package_data['version'],
                        encoder_type=package_data['encoder_type'],
                        platform=Platform(package_data['platform']),
                        package_type=PackageType(package_data['package_type']),
                        compression=CompressionFormat(package_data['compression']),
                        download_url=package_data['download_url'],
                        checksum=package_data['checksum'],
                        checksum_algorithm=package_data.get('checksum_algorithm', 'sha256'),
                        size_bytes=package_data.get('size_bytes'),
                        dependencies=package_data.get('dependencies'),
                        python_version=package_data.get('python_version'),
                        cuda_version=package_data.get('cuda_version'),
                        description=package_data.get('description'),
                        install_script=package_data.get('install_script'),
                        uninstall_script=package_data.get('uninstall_script')
                    )
                    package_specs.append(spec)
                
                self._registry[encoder_type] = package_specs
            
            self.logger.info(f"Updated registry with {len(self._registry)} encoder types")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update registry: {e}")
            return False
    
    def find_packages(self, 
                     encoder_type: str,
                     platform: Optional[Platform] = None,
                     package_type: Optional[PackageType] = None,
                     python_version: Optional[str] = None) -> List[PackageSpec]:
        """Find available packages matching criteria."""
        if encoder_type not in self._registry:
            return []
        
        packages = self._registry[encoder_type]
        current_platform = platform or self._get_current_platform()
        
        # Filter packages
        filtered = []
        for package in packages:
            # Platform check
            if package.platform != current_platform:
                continue
            
            # Package type check
            if package_type and package.package_type != package_type:
                continue
            
            # Python version check (basic compatibility)
            if python_version and package.python_version:
                if not self._is_python_compatible(python_version, package.python_version):
                    continue
            
            filtered.append(package)
        
        # Sort by version (newest first)
        filtered.sort(key=lambda p: p.version, reverse=True)
        return filtered
    
    def _is_python_compatible(self, current_version: str, required_version: str) -> bool:
        """Check if Python versions are compatible."""
        try:
            current_parts = [int(x) for x in current_version.split('.')[:2]]
            required_parts = [int(x) for x in required_version.split('.')[:2]]
            
            # Same major version, current minor >= required minor
            return (current_parts[0] == required_parts[0] and 
                   current_parts[1] >= required_parts[1])
        except Exception:
            return True  # Assume compatible if parsing fails
    
    def download_package(self, spec: PackageSpec) -> Path:
        """Download a package to cache."""
        if not self._session:
            raise PackageManagerError("HTTP session not available")
        
        # Generate cache filename
        filename = f"{spec.name}-{spec.version}-{spec.platform.value}.{spec.compression.value}"
        cache_path = self.cache_dir / "downloads" / filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded and verified
        if cache_path.exists() and self._verify_checksum(cache_path, spec):
            self.logger.info(f"Package already cached: {filename}")
            return cache_path
        
        # Download package
        self.logger.info(f"Downloading package: {spec.download_url}")
        
        try:
            response = self._session.get(spec.download_url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Simple progress logging
                        if total_size > 0 and downloaded % (1024 * 1024) == 0:  # Every MB
                            progress = (downloaded / total_size) * 100
                            self.logger.info(f"Download progress: {progress:.1f}%")
            
            # Verify checksum
            if not self._verify_checksum(cache_path, spec):
                cache_path.unlink()
                raise PackageManagerError(f"Checksum verification failed for {filename}")
            
            self.logger.info(f"Package downloaded successfully: {filename}")
            return cache_path
            
        except Exception as e:
            if cache_path.exists():
                cache_path.unlink()
            raise PackageManagerError(f"Failed to download package: {e}")
    
    def _verify_checksum(self, file_path: Path, spec: PackageSpec) -> bool:
        """Verify file checksum."""
        try:
            hash_func = hashlib.new(spec.checksum_algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_func.update(chunk)
            
            calculated = hash_func.hexdigest()
            return calculated.lower() == spec.checksum.lower()
            
        except Exception as e:
            self.logger.error(f"Checksum verification error: {e}")
            return False
    
    def install_package(self, spec: PackageSpec) -> bool:
        """Install a pre-compiled package."""
        package_id = f"{spec.encoder_type}-{spec.name}-{spec.version}"
        
        # Check if already installed
        if package_id in self._installed:
            self.logger.info(f"Package already installed: {package_id}")
            return True
        
        try:
            # Download package
            package_path = self.download_package(spec)
            
            # Create installation directory
            install_dir = self.cache_dir / "installed" / package_id
            install_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract package
            self._extract_package(package_path, install_dir, spec.compression)
            
            # Run install script if provided
            if spec.install_script:
                self._run_install_script(install_dir, spec.install_script)
            
            # Record installation
            import time
            info = PackageInfo(
                spec=spec,
                install_path=install_dir,
                install_time=time.time(),
                is_active=True
            )
            
            self._installed[package_id] = info
            self._save_installed_packages()
            
            self.logger.info(f"Package installed successfully: {package_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to install package {package_id}: {e}")
            return False
    
    def _extract_package(self, package_path: Path, install_dir: Path, compression: CompressionFormat):
        """Extract package to installation directory."""
        if compression == CompressionFormat.TAR_GZ:
            with tarfile.open(package_path, 'r:gz') as tar:
                tar.extractall(install_dir)
        elif compression == CompressionFormat.TAR_XZ:
            with tarfile.open(package_path, 'r:xz') as tar:
                tar.extractall(install_dir)
        elif compression == CompressionFormat.ZIP:
            with zipfile.ZipFile(package_path, 'r') as zip_file:
                zip_file.extractall(install_dir)
        elif compression == CompressionFormat.NONE:
            # Copy file directly
            shutil.copy2(package_path, install_dir / package_path.name)
        else:
            raise PackageManagerError(f"Unsupported compression format: {compression}")
    
    def _run_install_script(self, install_dir: Path, script_content: str):
        """Run installation script."""
        script_path = install_dir / "install.sh"
        
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable
            script_path.chmod(0o755)
            
            # Run script
            import subprocess
            result = subprocess.run(
                [str(script_path)],
                cwd=install_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                raise PackageManagerError(f"Install script failed: {result.stderr}")
            
        except Exception as e:
            raise PackageManagerError(f"Failed to run install script: {e}")
        finally:
            if script_path.exists():
                script_path.unlink()
    
    def uninstall_package(self, package_id: str) -> bool:
        """Uninstall a package."""
        if package_id not in self._installed:
            self.logger.warning(f"Package not installed: {package_id}")
            return False
        
        try:
            info = self._installed[package_id]
            
            # Run uninstall script if provided
            if info.spec.uninstall_script:
                self._run_uninstall_script(info.install_path, info.spec.uninstall_script)
            
            # Remove installation directory
            if info.install_path.exists():
                shutil.rmtree(info.install_path)
            
            # Remove from installed packages
            del self._installed[package_id]
            self._save_installed_packages()
            
            self.logger.info(f"Package uninstalled successfully: {package_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to uninstall package {package_id}: {e}")
            return False
    
    def _run_uninstall_script(self, install_dir: Path, script_content: str):
        """Run uninstallation script."""
        script_path = install_dir / "uninstall.sh"
        
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            script_path.chmod(0o755)
            
            import subprocess
            result = subprocess.run(
                [str(script_path)],
                cwd=install_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Uninstall script failed: {result.stderr}")
            
        except Exception as e:
            self.logger.warning(f"Failed to run uninstall script: {e}")
        finally:
            if script_path.exists():
                script_path.unlink()
    
    def list_installed(self) -> Dict[str, PackageInfo]:
        """List all installed packages."""
        return self._installed.copy()
    
    def get_package_path(self, package_id: str) -> Optional[Path]:
        """Get installation path for a package."""
        if package_id in self._installed:
            return self._installed[package_id].install_path
        return None
    
    def is_package_available(self, encoder_type: str, platform: Optional[Platform] = None) -> bool:
        """Check if a package is available for the encoder type."""
        packages = self.find_packages(encoder_type, platform)
        return len(packages) > 0
    
    def get_best_package(self, encoder_type: str, **kwargs) -> Optional[PackageSpec]:
        """Get the best available package for an encoder type."""
        packages = self.find_packages(encoder_type, **kwargs)
        
        if not packages:
            return None
        
        # Prefer Python wheels, then binaries, then others
        type_priority = {
            PackageType.PYTHON_WHEEL: 0,
            PackageType.ENCODER_BINARY: 1,
            PackageType.CONDA_PACKAGE: 2,
            PackageType.STATIC_LIBRARY: 3,
            PackageType.DOCKER_IMAGE: 4
        }
        
        packages.sort(key=lambda p: (type_priority.get(p.package_type, 999), p.version), reverse=True)
        return packages[0]
    
    def cleanup_cache(self, keep_days: int = 30):
        """Clean up old cached downloads."""
        downloads_dir = self.cache_dir / "downloads"
        
        if not downloads_dir.exists():
            return
        
        import time
        cutoff_time = time.time() - (keep_days * 24 * 3600)
        
        removed_count = 0
        for file_path in downloads_dir.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    removed_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to remove cached file {file_path}: {e}")
        
        self.logger.info(f"Cleaned up {removed_count} cached files")


# Global package manager instance
_package_manager = None


def get_package_manager(cache_dir: Optional[Path] = None, registry_url: Optional[str] = None) -> PrecompiledPackageManager:
    """Get the global package manager instance."""
    global _package_manager
    if _package_manager is None:
        _package_manager = PrecompiledPackageManager(cache_dir, registry_url or "https://packages.molenc.org")
    return _package_manager


def install_encoder_package(encoder_type: str, **kwargs) -> bool:
    """Install the best available package for an encoder type."""
    manager = get_package_manager()
    
    # Update registry first
    manager.update_registry()
    
    # Find best package
    package = manager.get_best_package(encoder_type, **kwargs)
    
    if not package:
        return False
    
    return manager.install_package(package)


def is_encoder_available(encoder_type: str) -> bool:
    """Check if an encoder is available as a pre-compiled package."""
    manager = get_package_manager()
    return manager.is_package_available(encoder_type)