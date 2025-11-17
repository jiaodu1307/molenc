import numpy as np
from typing import List, Any, Dict, Optional

from .base import BaseEncoder
from ..isolation.smart_environment_manager import (
    get_environment_manager,
    EncoderEnvironmentConfig,
    EnvironmentType,
)


class EnvironmentManagedEncoder(BaseEncoder):
    def __init__(self, encoder_type: str, backend: str, encoder_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self.encoder_type = encoder_type
        self.backend = backend
        self.encoder_config = encoder_config or {}
        self._env_mgr = get_environment_manager()
        # Configure environment based on backend
        try:
            if self.backend == EnvironmentType.CLOUD_API.value or self.backend == 'http':
                base_url = self.encoder_config.get('base_url')
                config = EncoderEnvironmentConfig(
                    encoder_type=self.encoder_type,
                    environment_type=EnvironmentType.CLOUD_API,
                    metadata={'base_url': base_url} if base_url else {}
                )
                self._env_mgr.update_encoder_environment(config)
            elif self.backend == EnvironmentType.VIRTUAL_ENV.value or self.backend == 'venv':
                self._env_mgr.configure_encoder_environment(self.encoder_type, preferred_environment=EnvironmentType.VIRTUAL_ENV)
            elif self.backend == EnvironmentType.CONDA_ENV.value or self.backend == 'conda':
                self._env_mgr.configure_encoder_environment(self.encoder_type, preferred_environment=EnvironmentType.CONDA_ENV)
            elif self.backend == EnvironmentType.DOCKER.value or self.backend == 'docker':
                self._env_mgr.configure_encoder_environment(self.encoder_type, preferred_environment=EnvironmentType.DOCKER)
            else:
                self._env_mgr.configure_encoder_environment(self.encoder_type, preferred_environment=EnvironmentType.LOCAL)
        except Exception:
            pass

    def _encode_single(self, smiles: str) -> np.ndarray:
        result = self._env_mgr.execute(self.encoder_type, smiles, self.encoder_config)
        arr = np.array(result, dtype=np.float32)
        return arr

    def encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        result = self._env_mgr.execute(self.encoder_type, smiles_list, self.encoder_config)
        return np.array(result, dtype=np.float32)

    def get_output_dim(self) -> int:
        return int(self.encoder_config.get('output_dim', 0))

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({'backend': self.backend, 'environment_managed': True})
        return cfg