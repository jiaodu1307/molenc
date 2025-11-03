"""Simplified encoder factory for creating molecular encoders.

This module provides a streamlined approach to encoder creation
that reduces complexity while maintaining core functionality.
"""

import logging
from typing import Dict, Optional, Any, Type
from molenc.core.base import BaseEncoder
from molenc.core.registry import get_registry
from molenc.core.exceptions import EncoderNotFoundError, EncoderInitializationError
from molenc.core.simple_dependency_manager import get_simple_dependency_manager

logger = logging.getLogger(__name__)


class SimpleEncoderFactory:
    """Simplified factory for creating molecular encoders."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.registry = get_registry()
        self.dependency_manager = get_simple_dependency_manager()
        self._encoder_cache: Dict[str, BaseEncoder] = {}
        
        # Simple encoder aliases
        self._aliases = {
            'fp': 'morgan',
            'fingerprint': 'morgan',
            'bert': 'chemberta',
            'transformer': 'chemberta',
            'graph': 'gcn',
            'gnn': 'gcn'
        }
    
    def create_encoder(
        self, 
        encoder_type: str, 
        cache: bool = False,
        **kwargs
    ) -> BaseEncoder:
        """
        Create an encoder instance.
        
        Args:
            encoder_type: Type of encoder to create
            cache: Whether to cache the encoder instance
            **kwargs: Additional arguments passed to encoder constructor
            
        Returns:
            Configured encoder instance
            
        Raises:
            EncoderNotFoundError: If encoder type is not found
            EncoderInitializationError: If encoder initialization fails
        """
        # Normalize encoder type
        encoder_type = self._normalize_encoder_type(encoder_type)
        
        # Check cache if enabled
        if cache and encoder_type in self._encoder_cache:
            cached_encoder = self._encoder_cache[encoder_type]
            if self._is_encoder_valid(cached_encoder):
                self.logger.debug(f"Returning cached encoder for {encoder_type}")
                return cached_encoder
            else:
                del self._encoder_cache[encoder_type]
        
        # Check dependencies
        is_available, missing = self.dependency_manager.check_encoder_dependencies(encoder_type)
        if not is_available:
            raise EncoderInitializationError(
                encoder_type,
                f"Missing dependencies: {missing}. "
                f"Install with: {' && '.join(self.dependency_manager.get_installation_command(missing))}"
            )
        
        # Create encoder
        try:
            encoder = self.registry.get_encoder(encoder_type, **kwargs)
            
            # Cache if requested
            if cache:
                self._encoder_cache[encoder_type] = encoder
            
            self.logger.info(f"Created {encoder_type} encoder successfully")
            return encoder
            
        except Exception as e:
            raise EncoderInitializationError(encoder_type, str(e)) from e
    
    def _normalize_encoder_type(self, encoder_type: str) -> str:
        """Normalize encoder type using aliases."""
        encoder_type_lower = encoder_type.lower()
        return self._aliases.get(encoder_type_lower, encoder_type_lower)
    
    def _is_encoder_valid(self, encoder: BaseEncoder) -> bool:
        """Check if a cached encoder is still valid."""
        try:
            # Simple validation - check if encoder has required methods
            return hasattr(encoder, 'encode') and hasattr(encoder, 'get_output_dim')
        except Exception:
            return False
    
    def get_available_encoders(self) -> Dict[str, bool]:
        """
        Get list of available encoders and their dependency status.
        
        Returns:
            Dictionary mapping encoder names to availability status
        """
        available = {}
        
        for encoder_name in self.registry.list_encoders():
            is_available, _ = self.dependency_manager.check_encoder_dependencies(encoder_name)
            available[encoder_name] = is_available
        
        return available
    
    def get_encoder_info(self, encoder_type: str) -> Dict[str, Any]:
        """
        Get information about an encoder.
        
        Args:
            encoder_type: Type of encoder
            
        Returns:
            Dictionary with encoder information
        """
        encoder_type = self._normalize_encoder_type(encoder_type)
        is_available, missing = self.dependency_manager.check_encoder_dependencies(encoder_type)
        
        info = {
            'name': encoder_type,
            'available': is_available,
            'missing_dependencies': missing,
            'installation_commands': self.dependency_manager.get_installation_command(missing) if missing else []
        }
        
        # Try to get encoder class info
        try:
            encoder_class = self.registry._encoders.get(encoder_type)
            if encoder_class:
                info['class_name'] = encoder_class.__name__
                info['module'] = encoder_class.__module__
        except Exception:
            pass
        
        return info
    
    def clear_cache(self) -> None:
        """Clear the encoder cache."""
        self._encoder_cache.clear()
        self.logger.debug("Encoder cache cleared")
    
    def create_encoder_with_fallback(
        self, 
        encoder_types: list, 
        **kwargs
    ) -> BaseEncoder:
        """
        Try to create encoders in order until one succeeds.
        
        Args:
            encoder_types: List of encoder types to try in order
            **kwargs: Additional arguments passed to encoder constructor
            
        Returns:
            First successfully created encoder
            
        Raises:
            EncoderNotFoundError: If no encoder could be created
        """
        last_error = None
        
        for encoder_type in encoder_types:
            try:
                return self.create_encoder(encoder_type, **kwargs)
            except (EncoderNotFoundError, EncoderInitializationError) as e:
                last_error = e
                self.logger.debug(f"Failed to create {encoder_type}: {e}")
                continue
        
        raise EncoderNotFoundError(
            f"All encoder types failed: {encoder_types}. Last error: {last_error}"
        )


# Global instance
_simple_factory = SimpleEncoderFactory()


def get_simple_encoder_factory() -> SimpleEncoderFactory:
    """Get the global simple encoder factory instance."""
    return _simple_factory


def create_encoder(encoder_type: str, **kwargs) -> BaseEncoder:
    """
    Convenience function to create an encoder.
    
    Args:
        encoder_type: Type of encoder to create
        **kwargs: Additional arguments passed to encoder constructor
        
    Returns:
        Configured encoder instance
    """
    return _simple_factory.create_encoder(encoder_type, **kwargs)