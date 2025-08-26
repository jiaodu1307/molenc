"""Environment isolation for MolEnc.

This module provides environment isolation mechanisms to avoid
dependency conflicts between different molecular encoders,
using virtual environments, containers, and process isolation.
"""

from .environment_manager import (
    EnvironmentManager,
    VirtualEnvironmentManager,
    CondaEnvironmentManager,
    DockerEnvironmentManager
)
from .process_wrapper import ProcessWrapper, create_process_wrapper
from .smart_environment_manager import (
    SmartEnvironmentManager,
    get_environment_manager,
    EnvironmentType as IsolationEnvironmentType,
    EnvironmentStatus
)

__all__ = [
    'EnvironmentManager',
    'VirtualEnvironmentManager',
    'CondaEnvironmentManager',
    'DockerEnvironmentManager',
    'ProcessWrapper',
    'create_process_wrapper',
    'SmartEnvironmentManager',
    'get_environment_manager',
    'IsolationEnvironmentType',
    'EnvironmentStatus'
]