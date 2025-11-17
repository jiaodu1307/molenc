import os
import numpy as np
from typing import Optional, List, Dict, Any

from molenc.core.base import BaseEncoder
from molenc.core.dependency_utils import require_dependencies
from molenc.core.exceptions import EncoderInitializationError
from molenc.core.api_client import CloudAPIClient, APIConfig


@require_dependencies(['requests'], 'HttpGCN')
class HttpGCNEncoder(BaseEncoder):
    def __init__(self,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 timeout: int = 30,
                 output_dim: Optional[int] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_url = base_url or os.environ.get('MOLENC_REMOTE_URL')
        self.api_key = api_key or os.environ.get('MOLENC_REMOTE_KEY')
        self.timeout = timeout
        self._output_dim: Optional[int] = output_dim
        if not self.base_url:
            raise EncoderInitializationError('HttpGCN', 'base_url is required for HTTP remote encoder')
        self._client = CloudAPIClient(APIConfig(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout))

    def _encode_single(self, smiles: str) -> np.ndarray:
        vec = self._client.encode_single(smiles, encoder_type='gcn')
        arr = np.array(vec, dtype=np.float32)
        if self._output_dim is None:
            self._output_dim = int(arr.shape[0])
        return arr

    def encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        encodings = self._client.encode_batch(smiles_list, encoder_type='gcn')
        arr = np.array(encodings, dtype=np.float32)
        if self._output_dim is None and arr.size > 0:
            self._output_dim = int(arr.shape[1])
        return arr

    def get_output_dim(self) -> int:
        if self._output_dim is not None:
            return self._output_dim
        return 256

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({'base_url': self.base_url, 'timeout': self.timeout, 'remote': True})
        return cfg