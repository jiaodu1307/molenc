"""Cloud API client for MolEnc.

This module provides cloud API integration for:
- Remote molecular encoding
- Fallback when local dependencies are unavailable
- Scalable batch processing
- Model serving and inference
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import time
from ..core.exceptions import CloudAPIError

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None


@dataclass
class APIConfig:
    """Configuration for API client."""
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100


class CloudAPIClient:
    """Client for cloud-based molecular encoding API."""
    
    def __init__(self, config: APIConfig):
        if not REQUESTS_AVAILABLE:
            raise CloudAPIError(
                "initialization",
                "requests library is required for cloud API access"
            )
            
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._session = requests.Session()
        
        # Set up session headers
        if config.api_key:
            self._session.headers.update({
                'Authorization': f'Bearer {config.api_key}',
                'Content-Type': 'application/json'
            })
            
    def encode_single(self, 
                     smiles: str,
                     encoder_type: str = 'unimol',
                     model_name: Optional[str] = None,
                     **kwargs) -> List[float]:
        """Encode a single SMILES string using cloud API.
        
        Args:
            smiles: SMILES string to encode
            encoder_type: Type of encoder to use
            model_name: Specific model name
            **kwargs: Additional parameters
            
        Returns:
            Encoding vector as list of floats
            
        Raises:
            CloudAPIError: If API request fails
        """
        endpoint = f"{self.config.base_url}/encode/single"
        
        payload = {
            'smiles': smiles,
            'encoder_type': encoder_type,
            **kwargs
        }
        
        if model_name:
            payload['model_name'] = model_name
            
        try:
            response = self._make_request('POST', endpoint, json=payload)
            return response['encoding']
        except Exception as e:
            raise CloudAPIError(
                "encode_single",
                f"Failed to encode SMILES '{smiles}': {e}"
            )
            
    def encode_batch(self, 
                    smiles_list: List[str],
                    encoder_type: str = 'unimol',
                    model_name: Optional[str] = None,
                    **kwargs) -> List[List[float]]:
        """Encode a batch of SMILES strings using cloud API.
        
        Args:
            smiles_list: List of SMILES strings to encode
            encoder_type: Type of encoder to use
            model_name: Specific model name
            **kwargs: Additional parameters
            
        Returns:
            List of encoding vectors
            
        Raises:
            CloudAPIError: If API request fails
        """
        if len(smiles_list) <= self.config.batch_size:
            return self._encode_batch_single_request(
                smiles_list, encoder_type, model_name, **kwargs
            )
        else:
            return self._encode_batch_chunked(
                smiles_list, encoder_type, model_name, **kwargs
            )
            
    def _encode_batch_single_request(self, 
                                   smiles_list: List[str],
                                   encoder_type: str,
                                   model_name: Optional[str],
                                   **kwargs) -> List[List[float]]:
        """Encode batch in a single request."""
        endpoint = f"{self.config.base_url}/encode/batch"
        
        payload = {
            'smiles_list': smiles_list,
            'encoder_type': encoder_type,
            **kwargs
        }
        
        if model_name:
            payload['model_name'] = model_name
            
        try:
            response = self._make_request('POST', endpoint, json=payload)
            return response['encodings']
        except Exception as e:
            raise CloudAPIError(
                "encode_batch",
                f"Failed to encode batch of {len(smiles_list)} SMILES: {e}"
            )
            
    def _encode_batch_chunked(self, 
                            smiles_list: List[str],
                            encoder_type: str,
                            model_name: Optional[str],
                            **kwargs) -> List[List[float]]:
        """Encode large batch in chunks."""
        all_encodings = []
        
        for i in range(0, len(smiles_list), self.config.batch_size):
            chunk = smiles_list[i:i + self.config.batch_size]
            chunk_encodings = self._encode_batch_single_request(
                chunk, encoder_type, model_name, **kwargs
            )
            all_encodings.extend(chunk_encodings)
            
        return all_encodings
        
    def get_available_encoders(self) -> List[Dict[str, Any]]:
        """Get list of available encoders from the API.
        
        Returns:
            List of encoder information dictionaries
            
        Raises:
            CloudAPIError: If API request fails
        """
        endpoint = f"{self.config.base_url}/encoders"
        
        try:
            response = self._make_request('GET', endpoint)
            return response['encoders']
        except Exception as e:
            raise CloudAPIError(
                "get_available_encoders",
                f"Failed to get available encoders: {e}"
            )
            
    def get_encoder_info(self, encoder_type: str) -> Dict[str, Any]:
        """Get information about a specific encoder.
        
        Args:
            encoder_type: Type of encoder
            
        Returns:
            Encoder information dictionary
            
        Raises:
            CloudAPIError: If API request fails
        """
        endpoint = f"{self.config.base_url}/encoders/{encoder_type}"
        
        try:
            response = self._make_request('GET', endpoint)
            return response['encoder_info']
        except Exception as e:
            raise CloudAPIError(
                "get_encoder_info",
                f"Failed to get info for encoder '{encoder_type}': {e}"
            )
            
    def health_check(self) -> Dict[str, Any]:
        """Check API health status.
        
        Returns:
            Health status dictionary
            
        Raises:
            CloudAPIError: If health check fails
        """
        endpoint = f"{self.config.base_url}/health"
        
        try:
            response = self._make_request('GET', endpoint)
            return response
        except Exception as e:
            raise CloudAPIError(
                "health_check",
                f"Health check failed: {e}"
            )
            
    def _make_request(self, 
                     method: str,
                     url: str,
                     **kwargs) -> Dict[str, Any]:
        """Make HTTP request with retry logic.
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request arguments
            
        Returns:
            Response JSON data
            
        Raises:
            CloudAPIError: If request fails after retries
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._session.request(
                    method,
                    url,
                    timeout=self.config.timeout,
                    **kwargs
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    if attempt < self.config.max_retries:
                        delay = self.config.retry_delay * (2 ** attempt)
                        self.logger.warning(
                            f"Rate limited, retrying in {delay}s (attempt {attempt + 1})"
                        )
                        time.sleep(delay)
                        continue
                        
                # Handle other HTTP errors
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += f": {error_data['error']}"
                except:
                    error_msg += f": {response.text}"
                    
                raise CloudAPIError(
                    "http_request",
                    error_msg,
                    status_code=response.status_code
                )
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Request failed, retrying in {delay}s (attempt {attempt + 1}): {e}"
                    )
                    time.sleep(delay)
                else:
                    break
                    
        raise CloudAPIError(
            "http_request",
            f"Request failed after {self.config.max_retries + 1} attempts: {last_exception}"
        )
        
    def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            self._session.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class CloudEncoderWrapper:
    """Wrapper to make cloud API behave like a local encoder."""
    
    def __init__(self, 
                 client: CloudAPIClient,
                 encoder_type: str = 'unimol',
                 model_name: Optional[str] = None,
                 **kwargs):
        self.client = client
        self.encoder_type = encoder_type
        self.model_name = model_name
        self.config = kwargs
        
        # Get encoder info from API
        try:
            self.encoder_info = client.get_encoder_info(encoder_type)
            self.output_dim = self.encoder_info.get('output_dim', 512)
        except Exception:
            self.encoder_info = {}
            self.output_dim = 512
            
    def encode(self, smiles: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Encode SMILES string(s).
        
        Args:
            smiles: SMILES string or list of SMILES strings
            
        Returns:
            Encoding vector(s)
        """
        if isinstance(smiles, str):
            return self.client.encode_single(
                smiles,
                self.encoder_type,
                self.model_name,
                **self.config
            )
        else:
            return self.client.encode_batch(
                smiles,
                self.encoder_type,
                self.model_name,
                **self.config
            )
            
    def encode_batch(self, smiles_list: List[str]) -> List[List[float]]:
        """Encode batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of encoding vectors
        """
        return self.client.encode_batch(
            smiles_list,
            self.encoder_type,
            self.model_name,
            **self.config
        )
        
    def get_output_dim(self) -> int:
        """Get output dimension.
        
        Returns:
            Output dimension
        """
        return self.output_dim
        
    def get_config(self) -> Dict[str, Any]:
        """Get encoder configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'encoder_type': self.encoder_type,
            'model_name': self.model_name,
            'cloud_api': True,
            'output_dim': self.output_dim,
            **self.config
        }


# Default API configurations
DEFAULT_API_CONFIGS = {
    'molenc_cloud': APIConfig(
        base_url='https://api.molenc.ai/v1',
        timeout=30,
        max_retries=3
    ),
    'local_dev': APIConfig(
        base_url='http://localhost:8000/api/v1',
        timeout=10,
        max_retries=1
    )
}


def create_cloud_client(config_name: str = 'molenc_cloud',
                       api_key: Optional[str] = None,
                       **kwargs) -> CloudAPIClient:
    """Create a cloud API client with predefined configuration.
    
    Args:
        config_name: Name of predefined configuration
        api_key: API key for authentication
        **kwargs: Additional configuration overrides
        
    Returns:
        CloudAPIClient instance
    """
    if config_name not in DEFAULT_API_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}")
        
    config = DEFAULT_API_CONFIGS[config_name]
    
    # Override with provided values
    if api_key:
        config.api_key = api_key
        
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            
    return CloudAPIClient(config)