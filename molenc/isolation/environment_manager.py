"""Environment isolation manager for molecular encoders.

This module provides environment isolation mechanisms to avoid
dependency conflicts between different molecular encoders,
using virtual environments, containers, and process isolation.
"""

import os
import sys
import subprocess
import shutil
import logging
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import threading
import queue
import time


class IsolationMethod(Enum):
    """Available isolation methods."""
    VIRTUAL_ENV = "virtual_env"
    CONDA_ENV = "conda_env"
    DOCKER_CONTAINER = "docker_container"
    PROCESS_ISOLATION = "process_isolation"
    SUBPROCESS = "subprocess"
    NONE = "none"


class EnvironmentStatus(Enum):
    """Environment status."""
    CREATED = "created"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DESTROYED = "destroyed"


@dataclass
class EnvironmentSpec:
    """Specification for an isolated environment."""
    name: str
    encoder_type: str
    isolation_method: IsolationMethod
    python_version: Optional[str] = None
    requirements: Optional[List[str]] = None
    conda_channels: Optional[List[str]] = None
    docker_image: Optional[str] = None
    environment_variables: Optional[Dict[str, str]] = None
    working_directory: Optional[str] = None
    resource_limits: Optional[Dict[str, Any]] = None
    cleanup_on_exit: bool = True


@dataclass
class EnvironmentInfo:
    """Information about an isolated environment."""
    spec: EnvironmentSpec
    status: EnvironmentStatus
    path: Optional[Path] = None
    container_id: Optional[str] = None
    process_id: Optional[int] = None
    created_time: Optional[float] = None
    last_used: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class EnvironmentError(Exception):
    """Exception raised for environment management errors."""
    pass


class VirtualEnvironmentManager:
    """Manager for Python virtual environments."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def create_environment(self, spec: EnvironmentSpec) -> EnvironmentInfo:
        """Create a virtual environment."""
        env_path = self.base_dir / spec.name
        
        if env_path.exists():
            shutil.rmtree(env_path)
        
        try:
            # Create virtual environment
            python_exe = spec.python_version or sys.executable
            
            cmd = [python_exe, "-m", "venv", str(env_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise EnvironmentError(f"Failed to create virtual environment: {result.stderr}")
            
            # Install requirements
            if spec.requirements:
                self._install_requirements(env_path, spec.requirements)
            
            return EnvironmentInfo(
                spec=spec,
                status=EnvironmentStatus.CREATED,
                path=env_path,
                created_time=time.time()
            )
            
        except Exception as e:
            if env_path.exists():
                shutil.rmtree(env_path)
            raise EnvironmentError(f"Failed to create virtual environment: {e}")
    
    def _install_requirements(self, env_path: Path, requirements: List[str]):
        """Install requirements in virtual environment."""
        if sys.platform == "win32":
            pip_exe = env_path / "Scripts" / "pip.exe"
        else:
            pip_exe = env_path / "bin" / "pip"
        
        for requirement in requirements:
            cmd = [str(pip_exe), "install", requirement]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                self.logger.warning(f"Failed to install {requirement}: {result.stderr}")
    
    def activate_environment(self, env_path: Path) -> Dict[str, str]:
        """Get environment variables for activating the virtual environment."""
        if sys.platform == "win32":
            bin_dir = env_path / "Scripts"
            python_exe = bin_dir / "python.exe"
        else:
            bin_dir = env_path / "bin"
            python_exe = bin_dir / "python"
        
        # Prepare environment variables
        env_vars = os.environ.copy()
        env_vars["VIRTUAL_ENV"] = str(env_path)
        env_vars["PATH"] = str(bin_dir) + os.pathsep + env_vars.get("PATH", "")
        
        # Remove PYTHONHOME if set
        env_vars.pop("PYTHONHOME", None)
        
        return env_vars
    
    def destroy_environment(self, env_path: Path):
        """Destroy a virtual environment."""
        if env_path.exists():
            shutil.rmtree(env_path)


class CondaEnvironmentManager:
    """Manager for Conda environments."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._check_conda_available()
    
    def _check_conda_available(self):
        """Check if conda is available."""
        try:
            result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                raise EnvironmentError("Conda is not available")
        except FileNotFoundError:
            raise EnvironmentError("Conda is not installed")
    
    def create_environment(self, spec: EnvironmentSpec) -> EnvironmentInfo:
        """Create a conda environment."""
        try:
            # Build conda create command
            cmd = ["conda", "create", "-n", spec.name, "-y"]
            
            if spec.python_version:
                cmd.extend([f"python={spec.python_version}"])
            
            # Add channels
            if spec.conda_channels:
                for channel in spec.conda_channels:
                    cmd.extend(["-c", channel])
            
            # Create environment
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                raise EnvironmentError(f"Failed to create conda environment: {result.stderr}")
            
            # Install requirements
            if spec.requirements:
                self._install_requirements(spec.name, spec.requirements)
            
            return EnvironmentInfo(
                spec=spec,
                status=EnvironmentStatus.CREATED,
                created_time=time.time()
            )
            
        except Exception as e:
            # Try to clean up
            self._destroy_environment(spec.name)
            raise EnvironmentError(f"Failed to create conda environment: {e}")
    
    def _install_requirements(self, env_name: str, requirements: List[str]):
        """Install requirements in conda environment."""
        for requirement in requirements:
            # Try conda install first, then pip
            conda_cmd = ["conda", "install", "-n", env_name, "-y", requirement]
            result = subprocess.run(conda_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                # Fall back to pip
                pip_cmd = ["conda", "run", "-n", env_name, "pip", "install", requirement]
                result = subprocess.run(pip_cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    self.logger.warning(f"Failed to install {requirement}: {result.stderr}")
    
    def activate_environment(self, env_name: str) -> List[str]:
        """Get command prefix for running in conda environment."""
        return ["conda", "run", "-n", env_name]
    
    def _destroy_environment(self, env_name: str):
        """Destroy a conda environment."""
        cmd = ["conda", "env", "remove", "-n", env_name, "-y"]
        subprocess.run(cmd, capture_output=True, text=True)


class DockerEnvironmentManager:
    """Manager for Docker container environments."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._check_docker_available()
    
    def _check_docker_available(self):
        """Check if Docker is available."""
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                raise EnvironmentError("Docker is not available")
        except FileNotFoundError:
            raise EnvironmentError("Docker is not installed")
    
    def create_environment(self, spec: EnvironmentSpec) -> EnvironmentInfo:
        """Create a Docker container environment."""
        try:
            # Build docker run command
            cmd = ["docker", "run", "-d", "--name", spec.name]
            
            # Add environment variables
            if spec.environment_variables:
                for key, value in spec.environment_variables.items():
                    cmd.extend(["-e", f"{key}={value}"])
            
            # Add working directory
            if spec.working_directory:
                cmd.extend(["-w", spec.working_directory])
            
            # Add resource limits
            if spec.resource_limits:
                if "memory" in spec.resource_limits:
                    cmd.extend(["--memory", spec.resource_limits["memory"]])
                if "cpus" in spec.resource_limits:
                    cmd.extend(["--cpus", str(spec.resource_limits["cpus"])])
            
            # Add image and command
            image = spec.docker_image or "python:3.9-slim"
            cmd.extend([image, "sleep", "infinity"])  # Keep container running
            
            # Create container
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise EnvironmentError(f"Failed to create Docker container: {result.stderr}")
            
            container_id = result.stdout.strip()
            
            # Install requirements
            if spec.requirements:
                self._install_requirements(container_id, spec.requirements)
            
            return EnvironmentInfo(
                spec=spec,
                status=EnvironmentStatus.CREATED,
                container_id=container_id,
                created_time=time.time()
            )
            
        except Exception as e:
            # Try to clean up
            self._destroy_container(spec.name)
            raise EnvironmentError(f"Failed to create Docker environment: {e}")
    
    def _install_requirements(self, container_id: str, requirements: List[str]):
        """Install requirements in Docker container."""
        for requirement in requirements:
            cmd = ["docker", "exec", container_id, "pip", "install", requirement]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.warning(f"Failed to install {requirement}: {result.stderr}")
    
    def execute_in_container(self, container_id: str, command: List[str]) -> subprocess.CompletedProcess:
        """Execute command in Docker container."""
        cmd = ["docker", "exec", container_id] + command
        return subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    def _destroy_container(self, container_name: str):
        """Destroy a Docker container."""
        # Stop container
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        # Remove container
        subprocess.run(["docker", "rm", container_name], capture_output=True)


class ProcessIsolationManager:
    """Manager for process-based isolation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._processes: Dict[str, subprocess.Popen] = {}
        self._queues: Dict[str, queue.Queue] = {}
    
    def create_environment(self, spec: EnvironmentSpec) -> EnvironmentInfo:
        """Create a process-isolated environment."""
        # For process isolation, we don't create anything upfront
        return EnvironmentInfo(
            spec=spec,
            status=EnvironmentStatus.CREATED,
            created_time=time.time()
        )
    
    def execute_isolated(self, 
                        env_name: str, 
                        func: Callable, 
                        args: tuple = (), 
                        kwargs: dict = None,
                        timeout: Optional[float] = None) -> Any:
        """Execute function in isolated process."""
        import pickle
        import multiprocessing as mp
        
        kwargs = kwargs or {}
        
        # Create a queue for communication
        result_queue = mp.Queue()
        
        def worker():
            try:
                result = func(*args, **kwargs)
                result_queue.put(("success", result))
            except Exception as e:
                result_queue.put(("error", str(e)))
        
        # Start process
        process = mp.Process(target=worker)
        process.start()
        
        try:
            # Wait for result
            if timeout:
                process.join(timeout)
                if process.is_alive():
                    process.terminate()
                    process.join()
                    raise EnvironmentError("Process execution timeout")
            else:
                process.join()
            
            # Get result
            if not result_queue.empty():
                status, result = result_queue.get()
                if status == "success":
                    return result
                else:
                    raise EnvironmentError(f"Process execution failed: {result}")
            else:
                raise EnvironmentError("No result from process")
                
        finally:
            if process.is_alive():
                process.terminate()
                process.join()


class EnvironmentManager:
    """Main environment isolation manager."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path.home() / ".molenc" / "environments"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize sub-managers
        self.venv_manager = VirtualEnvironmentManager(self.base_dir / "venv")
        
        try:
            self.conda_manager = CondaEnvironmentManager()
        except EnvironmentError:
            self.conda_manager = None
            self.logger.warning("Conda not available, conda environments disabled")
        
        try:
            self.docker_manager = DockerEnvironmentManager()
        except EnvironmentError:
            self.docker_manager = None
            self.logger.warning("Docker not available, container environments disabled")
        
        self.process_manager = ProcessIsolationManager()
        
        # Environment registry
        self._environments: Dict[str, EnvironmentInfo] = {}
        self._load_environments()
    
    def _load_environments(self):
        """Load environment registry from disk."""
        registry_file = self.base_dir / "registry.json"
        
        if not registry_file.exists():
            return
        
        try:
            with open(registry_file, 'r') as f:
                data = json.load(f)
            
            for env_name, env_data in data.items():
                spec_data = env_data['spec']
                spec = EnvironmentSpec(
                    name=spec_data['name'],
                    encoder_type=spec_data['encoder_type'],
                    isolation_method=IsolationMethod(spec_data['isolation_method']),
                    python_version=spec_data.get('python_version'),
                    requirements=spec_data.get('requirements'),
                    conda_channels=spec_data.get('conda_channels'),
                    docker_image=spec_data.get('docker_image'),
                    environment_variables=spec_data.get('environment_variables'),
                    working_directory=spec_data.get('working_directory'),
                    resource_limits=spec_data.get('resource_limits'),
                    cleanup_on_exit=spec_data.get('cleanup_on_exit', True)
                )
                
                info = EnvironmentInfo(
                    spec=spec,
                    status=EnvironmentStatus(env_data['status']),
                    path=Path(env_data['path']) if env_data.get('path') else None,
                    container_id=env_data.get('container_id'),
                    process_id=env_data.get('process_id'),
                    created_time=env_data.get('created_time'),
                    last_used=env_data.get('last_used'),
                    metadata=env_data.get('metadata')
                )
                
                self._environments[env_name] = info
                
        except Exception as e:
            self.logger.error(f"Failed to load environment registry: {e}")
    
    def _save_environments(self):
        """Save environment registry to disk."""
        registry_file = self.base_dir / "registry.json"
        
        try:
            data = {}
            for env_name, info in self._environments.items():
                data[env_name] = {
                    'spec': asdict(info.spec),
                    'status': info.status.value,
                    'path': str(info.path) if info.path else None,
                    'container_id': info.container_id,
                    'process_id': info.process_id,
                    'created_time': info.created_time,
                    'last_used': info.last_used,
                    'metadata': info.metadata
                }
                
                # Convert enum to string
                data[env_name]['spec']['isolation_method'] = info.spec.isolation_method.value
            
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save environment registry: {e}")
    
    def create_environment(self, spec: EnvironmentSpec) -> EnvironmentInfo:
        """Create an isolated environment."""
        if spec.name in self._environments:
            raise EnvironmentError(f"Environment already exists: {spec.name}")
        
        # Choose appropriate manager
        if spec.isolation_method == IsolationMethod.VIRTUAL_ENV:
            info = self.venv_manager.create_environment(spec)
        elif spec.isolation_method == IsolationMethod.CONDA_ENV:
            if not self.conda_manager:
                raise EnvironmentError("Conda not available")
            info = self.conda_manager.create_environment(spec)
        elif spec.isolation_method == IsolationMethod.DOCKER_CONTAINER:
            if not self.docker_manager:
                raise EnvironmentError("Docker not available")
            info = self.docker_manager.create_environment(spec)
        elif spec.isolation_method == IsolationMethod.PROCESS_ISOLATION:
            info = self.process_manager.create_environment(spec)
        else:
            raise EnvironmentError(f"Unsupported isolation method: {spec.isolation_method}")
        
        # Register environment
        self._environments[spec.name] = info
        self._save_environments()
        
        self.logger.info(f"Created environment: {spec.name} ({spec.isolation_method.value})")
        return info
    
    @contextmanager
    def use_environment(self, env_name: str):
        """Context manager for using an isolated environment."""
        if env_name not in self._environments:
            raise EnvironmentError(f"Environment not found: {env_name}")
        
        info = self._environments[env_name]
        info.last_used = time.time()
        
        try:
            if info.spec.isolation_method == IsolationMethod.VIRTUAL_ENV:
                # Set up virtual environment
                old_env = os.environ.copy()
                new_env = self.venv_manager.activate_environment(info.path)
                os.environ.update(new_env)
                
                yield info
                
                # Restore environment
                os.environ.clear()
                os.environ.update(old_env)
                
            elif info.spec.isolation_method == IsolationMethod.CONDA_ENV:
                # Conda environments are handled via command prefix
                yield info
                
            elif info.spec.isolation_method == IsolationMethod.DOCKER_CONTAINER:
                # Docker containers are handled via docker exec
                yield info
                
            elif info.spec.isolation_method == IsolationMethod.PROCESS_ISOLATION:
                # Process isolation is handled per execution
                yield info
                
            else:
                yield info
                
        finally:
            self._save_environments()
    
    def execute_in_environment(self, 
                              env_name: str, 
                              command: Union[List[str], Callable],
                              *args, **kwargs) -> Any:
        """Execute command or function in isolated environment."""
        with self.use_environment(env_name) as info:
            if info.spec.isolation_method == IsolationMethod.VIRTUAL_ENV:
                if callable(command):
                    return command(*args, **kwargs)
                else:
                    env_vars = self.venv_manager.activate_environment(info.path)
                    return subprocess.run(command, env=env_vars, **kwargs)
                    
            elif info.spec.isolation_method == IsolationMethod.CONDA_ENV:
                if callable(command):
                    # For functions, we need to run in subprocess
                    raise EnvironmentError("Function execution not supported in conda environments")
                else:
                    cmd_prefix = self.conda_manager.activate_environment(info.spec.name)
                    full_command = cmd_prefix + command
                    return subprocess.run(full_command, **kwargs)
                    
            elif info.spec.isolation_method == IsolationMethod.DOCKER_CONTAINER:
                if callable(command):
                    raise EnvironmentError("Function execution not supported in Docker environments")
                else:
                    return self.docker_manager.execute_in_container(info.container_id, command)
                    
            elif info.spec.isolation_method == IsolationMethod.PROCESS_ISOLATION:
                if callable(command):
                    return self.process_manager.execute_isolated(
                        env_name, command, args, kwargs, kwargs.get('timeout')
                    )
                else:
                    return subprocess.run(command, **kwargs)
    
    def destroy_environment(self, env_name: str) -> bool:
        """Destroy an isolated environment."""
        if env_name not in self._environments:
            self.logger.warning(f"Environment not found: {env_name}")
            return False
        
        try:
            info = self._environments[env_name]
            
            if info.spec.isolation_method == IsolationMethod.VIRTUAL_ENV:
                self.venv_manager.destroy_environment(info.path)
            elif info.spec.isolation_method == IsolationMethod.CONDA_ENV:
                if self.conda_manager:
                    self.conda_manager._destroy_environment(info.spec.name)
            elif info.spec.isolation_method == IsolationMethod.DOCKER_CONTAINER:
                if self.docker_manager:
                    self.docker_manager._destroy_container(info.spec.name)
            
            # Remove from registry
            del self._environments[env_name]
            self._save_environments()
            
            self.logger.info(f"Destroyed environment: {env_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy environment {env_name}: {e}")
            return False
    
    def list_environments(self) -> Dict[str, EnvironmentInfo]:
        """List all environments."""
        return self._environments.copy()
    
    def get_environment(self, env_name: str) -> Optional[EnvironmentInfo]:
        """Get environment information."""
        return self._environments.get(env_name)
    
    def cleanup_unused_environments(self, max_age_days: int = 30):
        """Clean up unused environments."""
        current_time = time.time()
        cutoff_time = current_time - (max_age_days * 24 * 3600)
        
        to_remove = []
        for env_name, info in self._environments.items():
            if info.spec.cleanup_on_exit and info.last_used and info.last_used < cutoff_time:
                to_remove.append(env_name)
        
        for env_name in to_remove:
            self.destroy_environment(env_name)
        
        self.logger.info(f"Cleaned up {len(to_remove)} unused environments")


# Global environment manager instance
_environment_manager = None


def get_environment_manager(base_dir: Optional[Path] = None) -> EnvironmentManager:
    """Get the global environment manager instance."""
    global _environment_manager
    if _environment_manager is None:
        _environment_manager = EnvironmentManager(base_dir)
    return _environment_manager


def create_encoder_environment(
    encoder_type: str,
    isolation_method: IsolationMethod = IsolationMethod.VIRTUAL_ENV,
    **kwargs
) -> EnvironmentInfo:
    """Create an isolated environment for an encoder."""
    manager = get_environment_manager()
    
    env_name = f"{encoder_type}_{isolation_method.value}"
    
    spec = EnvironmentSpec(
        name=env_name,
        encoder_type=encoder_type,
        isolation_method=isolation_method,
        **kwargs
    )
    
    return manager.create_environment(spec)


@contextmanager
def isolated_encoder_execution(encoder_type: str, isolation_method: IsolationMethod = IsolationMethod.PROCESS_ISOLATION):
    """Context manager for isolated encoder execution."""
    manager = get_environment_manager()
    env_name = f"{encoder_type}_{isolation_method.value}"
    
    # Create environment if it doesn't exist
    if env_name not in manager._environments:
        create_encoder_environment(encoder_type, isolation_method)
    
    with manager.use_environment(env_name) as info:
        yield info