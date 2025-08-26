"""Cloud API client for molecular encoders.

This module provides cloud-based molecular encoding services
to solve complex dependency issues and provide access to
high-performance models without local installation.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from urllib.parse import urljoin

import numpy as np


class CloudProvider(Enum):
    """Supported cloud providers."""
    MOLENC_OFFICIAL = "molenc_official"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class EncodingFormat(Enum):
    """Supported encoding formats."""
    NUMPY = "numpy"
    JSON = "json"
    BASE64 = "base64"


@dataclass
class CloudConfig:
    """Configuration for cloud API client."""
    provider: CloudProvider = CloudProvider.MOLENC_OFFICIAL
    api_endpoint: str = "https://api.molenc.org"
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    encoding_format: EncodingFormat = EncodingFormat.NUMPY


@dataclass
class EncodingRequest:
    """Request for molecular encoding."""
    smiles: Union[str, List[str]]
    encoder_type: str
    encoder_version: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    format: EncodingFormat = EncodingFormat.NUMPY
    request_id: Optional[str] = None


@dataclass
class EncodingResponse:
    """Response from molecular encoding."""
    success: bool
    embeddings: Optional[np.ndarray] = None
    error_message: Optional[str] = None
    request_id: Optional[str] = None
    processing_time: Optional[float] = None
    encoder_info: Optional[Dict[str, Any]] = None
    cache_hit: bool = False


class CloudAPIError(Exception):
    """Exception raised for cloud API errors."""
    pass


class CloudAPIClient:
    """Client for cloud-based molecular encoding APIs."""
    
    def __init__(self, config: Optional[CloudConfig] = None):
        self.config = config or CloudConfig()
        self.logger = logging.getLogger(__name__)
        self._session = None
        self._cache: Dict[str, Any] = {}
        
        # Initialize HTTP session
        self._init_session()
    
    def _init_session(self):
        """Initialize HTTP session with proper configuration."""
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from requests.packages.urllib3.util.retry import Retry
            
            self._session = requests.Session()
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=self.config.retry_delay,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
            
            # Set default headers
            self._session.headers.update({
                'Content-Type': 'application/json',
                'User-Agent': 'molenc-cloud-client/1.0'
            })
            
            # Add API key if provided
            if self.config.api_key:
                self._session.headers.update({
                    'Authorization': f'Bearer {self.config.api_key}'
                })
            
        except ImportError:
            raise CloudAPIError("requests library is required for cloud API client")
    
    def _generate_cache_key(self, request: EncodingRequest) -> str:
        """Generate cache key for request."""
        # Create a deterministic hash of the request
        request_str = json.dumps({
            'smiles': request.smiles,
            'encoder_type': request.encoder_type,
            'encoder_version': request.encoder_version,
            'options': request.options
        }, sort_keys=True)
        
        return hashlib.md5(request_str.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[EncodingResponse]:
        """Check if response is cached and still valid."""
        if not self.config.cache_enabled or cache_key not in self._cache:
            return None
        
        cached_data = self._cache[cache_key]
        
        # Check if cache is still valid
        if time.time() - cached_data['timestamp'] > self.config.cache_ttl:
            del self._cache[cache_key]
            return None
        
        response = cached_data['response']
        response.cache_hit = True
        return response
    
    def _store_cache(self, cache_key: str, response: EncodingResponse):
        """Store response in cache."""
        if self.config.cache_enabled:
            self._cache[cache_key] = {
                'response': response,
                'timestamp': time.time()
            }
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to API."""
        if not self._session:
            raise CloudAPIError("HTTP session not initialized")
        
        url = urljoin(self.config.api_endpoint, endpoint)
        
        try:
            response = self._session.post(
                url,
                json=data,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            raise CloudAPIError(f"API request failed: {e}")
    
    def encode_single(self, 
                     smiles: str, 
                     encoder_type: str,
                     encoder_version: Optional[str] = None,
                     options: Optional[Dict[str, Any]] = None) -> EncodingResponse:
        """Encode a single SMILES string."""
        request = EncodingRequest(
            smiles=smiles,
            encoder_type=encoder_type,
            encoder_version=encoder_version,
            options=options,
            format=self.config.encoding_format
        )
        
        return self._process_encoding_request(request)
    
    def encode_batch(self, 
                    smiles_list: List[str], 
                    encoder_type: str,
                    encoder_version: Optional[str] = None,
                    options: Optional[Dict[str, Any]] = None) -> EncodingResponse:
        """Encode a batch of SMILES strings."""
        # Split large batches
        if len(smiles_list) > self.config.batch_size:
            return self._encode_large_batch(smiles_list, encoder_type, encoder_version, options)
        
        request = EncodingRequest(
            smiles=smiles_list,
            encoder_type=encoder_type,
            encoder_version=encoder_version,
            options=options,
            format=self.config.encoding_format
        )
        
        return self._process_encoding_request(request)
    
    def _encode_large_batch(self, 
                           smiles_list: List[str], 
                           encoder_type: str,
                           encoder_version: Optional[str] = None,
                           options: Optional[Dict[str, Any]] = None) -> EncodingResponse:
        """Handle large batch encoding by splitting into smaller chunks."""
        all_embeddings = []
        total_time = 0.0
        
        for i in range(0, len(smiles_list), self.config.batch_size):
            batch = smiles_list[i:i + self.config.batch_size]
            
            response = self.encode_batch(batch, encoder_type, encoder_version, options)
            
            if not response.success:
                return response  # Return error immediately
            
            all_embeddings.append(response.embeddings)
            if response.processing_time:
                total_time += response.processing_time
        
        # Combine all embeddings
        combined_embeddings = np.vstack(all_embeddings)
        
        return EncodingResponse(
            success=True,
            embeddings=combined_embeddings,
            processing_time=total_time
        )
    
    def _process_encoding_request(self, request: EncodingRequest) -> EncodingResponse:
        """Process an encoding request with caching."""
        # Check cache first
        cache_key = self._generate_cache_key(request)
        cached_response = self._check_cache(cache_key)
        if cached_response:
            self.logger.debug(f"Cache hit for request {cache_key[:8]}...")
            return cached_response
        
        # Make API request
        start_time = time.time()
        
        try:
            # Prepare request data
            request_data = {
                'smiles': request.smiles,
                'encoder_type': request.encoder_type,
                'encoder_version': request.encoder_version,
                'options': request.options or {},
                'format': request.format.value
            }
            
            # Choose endpoint based on request type
            if isinstance(request.smiles, str):
                endpoint = '/v1/encode'
            else:
                endpoint = '/v1/encode_batch'
            
            # Make request
            response_data = self._make_request(endpoint, request_data)
            
            # Parse response
            response = self._parse_response(response_data, start_time)
            
            # Cache successful response
            if response.success:
                self._store_cache(cache_key, response)
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            return EncodingResponse(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _parse_response(self, response_data: Dict[str, Any], start_time: float) -> EncodingResponse:
        """Parse API response data."""
        processing_time = time.time() - start_time
        
        if not response_data.get('success', False):
            return EncodingResponse(
                success=False,
                error_message=response_data.get('error', 'Unknown error'),
                processing_time=processing_time
            )
        
        # Parse embeddings based on format
        embeddings_data = response_data.get('embeddings')
        if embeddings_data is None:
            return EncodingResponse(
                success=False,
                error_message="No embeddings in response",
                processing_time=processing_time
            )
        
        try:
            if self.config.encoding_format == EncodingFormat.NUMPY:
                embeddings = np.array(embeddings_data, dtype=np.float32)
            elif self.config.encoding_format == EncodingFormat.BASE64:
                import base64
                decoded_data = base64.b64decode(embeddings_data)
                embeddings = np.frombuffer(decoded_data, dtype=np.float32)
            else:  # JSON format
                embeddings = np.array(embeddings_data, dtype=np.float32)
            
            return EncodingResponse(
                success=True,
                embeddings=embeddings,
                processing_time=processing_time,
                encoder_info=response_data.get('encoder_info'),
                request_id=response_data.get('request_id')
            )
            
        except Exception as e:
            return EncodingResponse(
                success=False,
                error_message=f"Failed to parse embeddings: {e}",
                processing_time=processing_time
            )
    
    def get_available_encoders(self) -> Dict[str, Any]:
        """Get list of available encoders from the API."""
        try:
            response_data = self._make_request('/v1/encoders', {})
            return response_data
        except Exception as e:
            self.logger.error(f"Failed to get available encoders: {e}")
            return {'error': str(e)}
    
    def get_encoder_info(self, encoder_type: str, encoder_version: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific encoder."""
        try:
            data = {'encoder_type': encoder_type}
            if encoder_version:
                data['encoder_version'] = encoder_version
            
            response_data = self._make_request('/v1/encoder_info', data)
            return response_data
        except Exception as e:
            self.logger.error(f"Failed to get encoder info: {e}")
            return {'error': str(e)}
    
    def health_check(self) -> bool:
        """Check if the API is healthy."""
        try:
            response_data = self._make_request('/v1/health', {})
            return response_data.get('status') == 'healthy'
        except Exception:
            return False
    
    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()
        self.logger.info("Cloud API cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._cache)
        total_size = sum(len(str(entry)) for entry in self._cache.values())
        
        # Count expired entries
        current_time = time.time()
        expired_entries = sum(
            1 for entry in self._cache.values()
            if current_time - entry['timestamp'] > self.config.cache_ttl
        )
        
        return {
            'total_entries': total_entries,
            'expired_entries': expired_entries,
            'cache_size_bytes': total_size,
            'cache_ttl': self.config.cache_ttl
        }


class CloudEncoderProxy:
    """Proxy class that makes cloud encoders look like local encoders."""
    
    def __init__(self, 
                 encoder_type: str,
                 encoder_version: Optional[str] = None,
                 cloud_config: Optional[CloudConfig] = None,
                 **kwargs):
        self.encoder_type = encoder_type
        self.encoder_version = encoder_version
        self.client = CloudAPIClient(cloud_config)
        self.logger = logging.getLogger(__name__)
        
        # Get encoder info from API
        self._encoder_info = self.client.get_encoder_info(encoder_type, encoder_version)
        
        if 'error' in self._encoder_info:
            raise CloudAPIError(f"Encoder not available: {self._encoder_info['error']}")
    
    def encode(self, smiles: str) -> np.ndarray:
        """Encode a single SMILES string."""
        response = self.client.encode_single(smiles, self.encoder_type, self.encoder_version)
        
        if not response.success:
            raise CloudAPIError(f"Encoding failed: {response.error_message}")
        
        return response.embeddings.squeeze()  # Remove batch dimension for single encoding
    
    def encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        """Encode a batch of SMILES strings."""
        response = self.client.encode_batch(smiles_list, self.encoder_type, self.encoder_version)
        
        if not response.success:
            raise CloudAPIError(f"Batch encoding failed: {response.error_message}")
        
        return response.embeddings
    
    def get_output_dim(self) -> int:
        """Get the output dimension of the encoder."""
        return self._encoder_info.get('output_dim', 512)
    
    def get_config(self) -> Dict[str, Any]:
        """Get encoder configuration."""
        return {
            'encoder_type': self.encoder_type,
            'encoder_version': self.encoder_version,
            'cloud_provider': self.client.config.provider.value,
            'api_endpoint': self.client.config.api_endpoint,
            'output_dim': self.get_output_dim(),
            'encoder_info': self._encoder_info
        }


# Global client instance
_cloud_client = None


def get_cloud_client(config: Optional[CloudConfig] = None) -> CloudAPIClient:
    """Get the global cloud API client instance."""
    global _cloud_client
    if _cloud_client is None or config is not None:
        _cloud_client = CloudAPIClient(config)
    return _cloud_client


def create_cloud_encoder(
    encoder_type: str,
    encoder_version: Optional[str] = None,
    cloud_config: Optional[CloudConfig] = None
) -> CloudEncoderProxy:
    """Create a cloud-based encoder proxy."""
    return CloudEncoderProxy(encoder_type, encoder_version, cloud_config)


def is_cloud_available(config: Optional[CloudConfig] = None) -> bool:
    """Check if cloud API is available."""
    try:
        client = get_cloud_client(config)
        return client.health_check()
    except Exception:
        return False