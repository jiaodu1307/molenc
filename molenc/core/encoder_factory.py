"""Unified encoder factory with smart selection and dependency management.

This module provides a high-level interface for creating molecular encoders
with automatic dependency management and intelligent fallback strategies.
"""

import logging
from typing import Dict, List, Optional, Any, Type, Union
from dataclasses import dataclass
from enum import Enum

from .base import BaseEncoder
from .smart_encoder_selector import (
    get_encoder_selector, 
    SelectionStrategy, 
    EncoderSelector
)
from ..environments.advanced_dependency_manager import (
    get_dependency_manager,
    DependencyLevel,
    check_encoder_readiness
)
from .exceptions import EncoderNotAvailableError
from .execution_backend import ExecutionBackend
from .environment_managed_encoder import EnvironmentManagedEncoder


class EncoderMode(Enum):
    """Modes for encoder operation."""
    AUTO = "auto"  # Automatic selection based on available dependencies
    PERFORMANCE = "performance"  # Prioritize performance
    COMPATIBILITY = "compatibility"  # Prioritize compatibility
    MINIMAL = "minimal"  # Use minimal dependencies
    CLOUD = "cloud"  # Prefer cloud-based solutions


@dataclass
class EncoderConfig:
    """Configuration for encoder creation."""
    mode: EncoderMode = EncoderMode.AUTO
    allow_fallback: bool = True
    auto_install: bool = False
    max_dependency_level: DependencyLevel = DependencyLevel.FULL
    user_preferences: Optional[Dict[str, Any]] = None
    cache_enabled: bool = True
    performance_benchmark: bool = False


class EncoderFactory:
    """Unified factory for creating molecular encoders."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.selector = get_encoder_selector()
        self.dependency_manager = get_dependency_manager()
        self._encoder_cache: Dict[str, BaseEncoder] = {}
        
        # Map encoder types to their canonical names
        self._encoder_aliases = {
            'unimol': ['unimol', 'uni-mol', 'UniMol'],
            
            'gcn': ['gcn', 'graph', 'GraphConv'],
            'fingerprint': ['fingerprint', 'fp', 'morgan'],
            'transformer': ['transformer', 'bert', 'roberta']
        }
    
    def create_encoder(
        self, 
        encoder_type: str, 
        config: Optional[EncoderConfig] = None,
        **kwargs
    ) -> BaseEncoder:
        """Create an encoder with smart selection and dependency management.
        
        Args:
            encoder_type: Type of encoder to create
            config: Configuration for encoder creation
            **kwargs: Additional arguments passed to encoder constructor
            
        Returns:
            Configured encoder instance
            
        Raises:
            EncoderNotAvailableError: If no suitable encoder variant is available
        """
        if config is None:
            config = EncoderConfig()
        
        # Normalize encoder type
        encoder_type = self._normalize_encoder_type(encoder_type)
        
        # Check cache if enabled
        cache_key = f"{encoder_type}_{config.mode.value}"
        if config.cache_enabled and cache_key in self._encoder_cache:
            cached_encoder = self._encoder_cache[cache_key]
            if self._is_encoder_still_valid(cached_encoder):
                self.logger.debug(f"Returning cached encoder for {encoder_type}")
                return cached_encoder
            else:
                del self._encoder_cache[cache_key]
        
        # Check encoder readiness
        is_ready, capability_level, status_message = check_encoder_readiness(encoder_type)
        
        if not is_ready and not config.allow_fallback:
            raise EncoderNotAvailableError(
                f"Encoder '{encoder_type}' is not ready: {status_message}"
            )
        
        # Auto-install missing dependencies if requested
        if config.auto_install and not is_ready:
            self.logger.info(f"Auto-installing dependencies for {encoder_type}...")
            success = self.dependency_manager.auto_install_missing(
                encoder_type, 
                config.max_dependency_level
            )
            if success:
                # Recheck readiness after installation
                is_ready, capability_level, status_message = check_encoder_readiness(encoder_type)
        
        # Determine selection strategy based on mode
        strategy = self._mode_to_strategy(config.mode)

        # Create encoder using smart selector
        try:
            # If backend preference indicates non-local execution, wrap with EnvironmentManagedEncoder
            backend_pref = (config.user_preferences or {}).get('backend') if config.user_preferences else None
            if backend_pref in [ExecutionBackend.VENV.value, ExecutionBackend.CONDA.value, ExecutionBackend.DOCKER.value, ExecutionBackend.HTTP.value]:
                # For HTTP backend, perform a quick health check before using environment-managed execution
                if backend_pref == ExecutionBackend.HTTP.value:
                    is_ready = True
                    try:
                        import os
                        base_url = kwargs.get('base_url') or os.environ.get('MOLENC_REMOTE_URL')
                        if base_url:
                            import requests
                            resp = requests.get(base_url.rstrip('/') + '/health', timeout=3)
                            is_ready = resp.status_code == 200
                        else:
                            is_ready = False
                    except Exception:
                        is_ready = False
                    if not is_ready:
                        # Fall back to selector/local path if HTTP backend not ready
                        encoder = self.selector.create_encoder(
                            encoder_type=encoder_type,
                            strategy=strategy,
                            user_preferences=config.user_preferences,
                            **kwargs
                        )
                    else:
                        encoder = EnvironmentManagedEncoder(encoder_type, backend_pref, encoder_config=kwargs)
                else:
                    encoder = EnvironmentManagedEncoder(encoder_type, backend_pref, encoder_config=kwargs)
            else:
                encoder = self.selector.create_encoder(
                    encoder_type=encoder_type,
                    strategy=strategy,
                    user_preferences=config.user_preferences,
                    **kwargs
                )
            
            # Add factory metadata
            encoder._factory_config = config
            encoder._creation_timestamp = self._get_timestamp()
            
            # Benchmark performance if requested
            if config.performance_benchmark:
                self._benchmark_encoder(encoder)
            
            # Cache the encoder if enabled
            if config.cache_enabled:
                self._encoder_cache[cache_key] = encoder
            
            self.logger.info(
                f"Successfully created {encoder_type} encoder "
                f"(variant: {getattr(encoder, '_selected_variant', {}).name if hasattr(encoder, '_selected_variant') else 'unknown'}, "
                f"mode: {config.mode.value})"
            )
            
            return encoder

        except Exception as e:
            self.logger.error(f"Failed to create {encoder_type} encoder: {e}")
            
            if config.allow_fallback:
                return self._create_fallback_encoder(encoder_type, config, **kwargs)
            else:
                raise EncoderNotAvailableError(
                    f"Failed to create {encoder_type} encoder and fallback is disabled: {e}"
                )
    
    def _normalize_encoder_type(self, encoder_type: str) -> str:
        """Normalize encoder type name using aliases."""
        encoder_type_lower = encoder_type.lower()
        
        for canonical_name, aliases in self._encoder_aliases.items():
            if encoder_type_lower in [alias.lower() for alias in aliases]:
                return canonical_name
        
        return encoder_type_lower
    
    def _mode_to_strategy(self, mode: EncoderMode) -> SelectionStrategy:
        """Convert encoder mode to selection strategy."""
        mode_mapping = {
            EncoderMode.AUTO: SelectionStrategy.BALANCED,
            EncoderMode.PERFORMANCE: SelectionStrategy.PERFORMANCE_FIRST,
            EncoderMode.COMPATIBILITY: SelectionStrategy.COMPATIBILITY_FIRST,
            EncoderMode.MINIMAL: SelectionStrategy.COMPATIBILITY_FIRST,
            EncoderMode.CLOUD: SelectionStrategy.USER_PREFERENCE
        }
        return mode_mapping.get(mode, SelectionStrategy.BALANCED)
    
    def _is_encoder_still_valid(self, encoder: BaseEncoder) -> bool:
        """Check if a cached encoder is still valid."""
        try:
            # Try a simple operation to verify the encoder still works
            encoder.get_output_dim()
            return True
        except Exception:
            return False
    
    def _create_fallback_encoder(
        self, 
        encoder_type: str, 
        config: EncoderConfig, 
        **kwargs
    ) -> BaseEncoder:
        """Create a fallback encoder when the primary creation fails."""
        self.logger.warning(f"Creating fallback encoder for {encoder_type}")
        
        # Try with minimal dependencies
        fallback_config = EncoderConfig(
            mode=EncoderMode.MINIMAL,
            allow_fallback=False,
            max_dependency_level=DependencyLevel.CORE
        )
        
        try:
            return self.selector.create_encoder(
                encoder_type=encoder_type,
                strategy=SelectionStrategy.COMPATIBILITY_FIRST,
                **kwargs
            )
        except Exception as e:
            raise EncoderNotAvailableError(
                f"All fallback options exhausted for {encoder_type}: {e}"
            )
    
    def _benchmark_encoder(self, encoder: BaseEncoder):
        """Benchmark encoder performance."""
        try:
            test_smiles = ["CCO", "CC(=O)O", "c1ccccc1"]  # Simple test molecules
            
            import time
            start_time = time.time()
            
            # Test encoding
            for smiles in test_smiles:
                encoder.encode(smiles)
            
            end_time = time.time()
            benchmark_time = end_time - start_time
            
            encoder._benchmark_time = benchmark_time
            self.logger.info(f"Encoder benchmark: {benchmark_time:.3f}s for {len(test_smiles)} molecules")
            
        except Exception as e:
            self.logger.warning(f"Benchmarking failed: {e}")
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    def get_available_encoders(self) -> Dict[str, List[str]]:
        """Get list of available encoders and their variants."""
        available = {}
        
        for encoder_type in self._encoder_aliases.keys():
            variants = self.selector.get_available_variants(encoder_type)
            if variants:
                available[encoder_type] = [v.name for v in variants]
        
        return available
    
    def get_encoder_status(self, encoder_type: str = None) -> str:
        """Get detailed status report for encoders."""
        if encoder_type:
            encoder_type = self._normalize_encoder_type(encoder_type)
            return self.selector.get_selection_report(encoder_type)
        else:
            # Get status for all known encoders
            reports = []
            for enc_type in self._encoder_aliases.keys():
                try:
                    report = self.selector.get_selection_report(enc_type)
                    reports.append(report)
                except Exception as e:
                    reports.append(f"Error getting status for {enc_type}: {e}")
            
            # Append environment status
            try:
                from ..isolation.smart_environment_manager import get_environment_manager
                env_mgr = get_environment_manager()
                reports.append(env_mgr.get_status_report())
            except Exception:
                pass
            return "\n\n".join(reports)
    
    def install_dependencies(
        self, 
        encoder_type: str, 
        level: DependencyLevel = DependencyLevel.FULL
    ) -> bool:
        """Install dependencies for a specific encoder."""
        encoder_type = self._normalize_encoder_type(encoder_type)
        
        self.logger.info(f"Installing dependencies for {encoder_type} (level: {level.value})")
        
        success = self.dependency_manager.auto_install_missing(encoder_type, level)
        
        if success:
            # Clear caches to force recheck
            self.clear_cache()
            self.dependency_manager.clear_cache()
            self.selector.clear_cache()
        
        return success
    
    def suggest_installation(self, encoder_type: str) -> List[str]:
        """Get installation suggestions for an encoder."""
        encoder_type = self._normalize_encoder_type(encoder_type)
        return self.dependency_manager.suggest_installation(encoder_type)
    
    def clear_cache(self):
        """Clear encoder cache."""
        self._encoder_cache.clear()
        self.logger.debug("Encoder cache cleared")
    
    def create_encoder_with_fallback_chain(
        self, 
        encoder_types: List[str], 
        config: Optional[EncoderConfig] = None,
        **kwargs
    ) -> BaseEncoder:
        """Try to create encoders in order until one succeeds."""
        if config is None:
            config = EncoderConfig()
        
        last_error = None
        
        for encoder_type in encoder_types:
            try:
                return self.create_encoder(encoder_type, config, **kwargs)
            except EncoderNotAvailableError as e:
                last_error = e
                self.logger.debug(f"Failed to create {encoder_type}: {e}")
                continue
        
        raise EncoderNotAvailableError(
            f"All encoder types failed: {encoder_types}. Last error: {last_error}"
        )


# Global factory instance
_encoder_factory = None


def get_encoder_factory() -> EncoderFactory:
    """Get the global encoder factory instance."""
    global _encoder_factory
    if _encoder_factory is None:
        _encoder_factory = EncoderFactory()
    return _encoder_factory


# Convenience functions
def create_encoder(
    encoder_type: str, 
    mode: EncoderMode = EncoderMode.AUTO,
    **kwargs
) -> BaseEncoder:
    """Convenience function to create an encoder."""
    factory = get_encoder_factory()
    config = EncoderConfig(mode=mode)
    return factory.create_encoder(encoder_type, config, **kwargs)


def create_best_available_encoder(
    encoder_types: List[str],
    mode: EncoderMode = EncoderMode.AUTO,
    **kwargs
) -> BaseEncoder:
    """Create the best available encoder from a list of options."""
    factory = get_encoder_factory()
    config = EncoderConfig(mode=mode)
    return factory.create_encoder_with_fallback_chain(encoder_types, config, **kwargs)


def get_encoder_status(encoder_type: str = None) -> str:
    """Get encoder status report."""
    factory = get_encoder_factory()
    return factory.get_encoder_status(encoder_type)


def install_encoder_dependencies(
    encoder_type: str, 
    level: DependencyLevel = DependencyLevel.FULL
) -> bool:
    """Install dependencies for an encoder."""
    factory = get_encoder_factory()
    return factory.install_dependencies(encoder_type, level)