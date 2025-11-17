from enum import Enum
from typing import Optional, Dict, Any


class ExecutionBackend(Enum):
    LOCAL = "local"
    VENV = "venv"
    CONDA = "conda"
    DOCKER = "docker"
    HTTP = "http"


def build_preferences(backend: Optional[str], extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    if not backend and not extra:
        return None
    prefs: Dict[str, Any] = {}
    if backend:
        prefs['backend'] = backend
    if extra:
        prefs.update(extra)
    return prefs