"""Smart encoder selection system for molenc.

This module provides intelligent encoder selection based on:
- Available dependencies
- Performance requirements
- Fallback strategies
- User preferences
"""

import logging
from typing import Dict, List, Optional, Type, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from abc import ABC, abstractmethod

from ..environments.advanced_dependency_manager import (
    get_dependency_manager, 
    DependencyLevel, 
    check_encoder_readiness
)
from .base import BaseEncoder
from .exceptions import EncoderNotAvailableError


class EncoderPriority(Enum):
    """Priority levels for encoder selection."""
    HIGHEST = "highest"  # Best performance, latest features
    HIGH = "high"       # Good performance, stable
    MEDIUM = "medium"   # Moderate performance, widely compatible
    LOW = "low"         # Basic functionality, maximum compatibility
    FALLBACK = "fallback"  # Last resort, always available


class SelectionStrategy(Enum):
    """Strategies for encoder selection."""
    PERFORMANCE_FIRST = "performance_first"  # Prioritize best performance
    COMPATIBILITY_FIRST = "compatibility_first"  # Prioritize compatibility
    BALANCED = "balanced"  # Balance performance and compatibility
    USER_PREFERENCE = "user_preference"  # Follow user-defined preferences


@dataclass
class EncoderVariant:
    """Represents a variant of an encoder with specific capabilities."""
    name: str
    encoder_class: Type[BaseEncoder]
    priority: EncoderPriority
    dependency_level: DependencyLevel
    required_packages: List[str]
    performance_score: float  # 0.0 to 1.0
    compatibility_score: float  # 0.0 to 1.0
    description: str
    fallback_for: Optional[str] = None  # Name of encoder this is a fallback for


class EncoderSelector:
    """Smart encoder selector with automatic fallback."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dependency_manager = get_dependency_manager()
        self._encoder_registry: Dict[str, List[EncoderVariant]] = {}
        self._selection_cache: Dict[str, EncoderVariant] = {}
        self._performance_cache: Dict[str, float] = {}
        
        # Register default encoder variants
        self._register_default_variants()
    
    def _register_default_variants(self):
        """Register default encoder variants."""
        # Import encoder classes (lazy import to avoid circular dependencies)
        try:
            from ..encoders.representations.multimodal.unimol import UniMolEncoder
            
            # UniMol variants
            self.register_encoder_variant(EncoderVariant(
                name="unimol_full",
                encoder_class=UniMolEncoder,
                priority=EncoderPriority.HIGHEST,
                dependency_level=DependencyLevel.ENHANCED,
                required_packages=["unimol_tools", "torch"],
                performance_score=0.95,
                compatibility_score=0.6,
                description="Full UniMol implementation with unimol_tools"
            ))
            
            self.register_encoder_variant(EncoderVariant(
                name="unimol_torch",
                encoder_class=UniMolEncoder,
                priority=EncoderPriority.HIGH,
                dependency_level=DependencyLevel.ENHANCED,
                required_packages=["torch", "transformers"],
                performance_score=0.85,
                compatibility_score=0.75,
                description="UniMol with PyTorch backend",
                fallback_for="unimol_full"
            ))
            
            self.register_encoder_variant(EncoderVariant(
                name="unimol_placeholder",
                encoder_class=UniMolEncoder,
                priority=EncoderPriority.FALLBACK,
                dependency_level=DependencyLevel.CORE,
                required_packages=["numpy"],
                performance_score=0.3,
                compatibility_score=0.95,
                description="UniMol placeholder implementation",
                fallback_for="unimol_torch"
            ))
            
        except ImportError:
            self.logger.warning("Could not import UniMolEncoder")
        
        # Fingerprint (Morgan/MACCS) - local RDKit variants
        try:
            from ..encoders.descriptors.fingerprints.morgan import MorganEncoder
            self.register_encoder_variant(EncoderVariant(
                name="fingerprint_morgan_local",
                encoder_class=MorganEncoder,
                priority=EncoderPriority.HIGH,
                dependency_level=DependencyLevel.FULL,
                required_packages=["rdkit"],
                performance_score=0.9,
                compatibility_score=0.8,
                description="Morgan fingerprint via RDKit"
            ))
        except ImportError:
            self.logger.debug("MorganEncoder not available")

        try:
            from ..encoders.descriptors.fingerprints.maccs import MACCSEncoder
            self.register_encoder_variant(EncoderVariant(
                name="fingerprint_maccs_local",
                encoder_class=MACCSEncoder,
                priority=EncoderPriority.MEDIUM,
                dependency_level=DependencyLevel.FULL,
                required_packages=["rdkit"],
                performance_score=0.85,
                compatibility_score=0.85,
                description="MACCS fingerprint via RDKit"
            ))
        except ImportError:
            self.logger.debug("MACCSEncoder not available")

        # Fingerprint Morgan - HTTP remote variant
        try:
            from ..encoders.remote.http_morgan import HttpMorganEncoder
            self.register_encoder_variant(EncoderVariant(
                name="fingerprint_morgan_http",
                encoder_class=HttpMorganEncoder,
                priority=EncoderPriority.MEDIUM,
                dependency_level=DependencyLevel.CORE,
                required_packages=["requests"],
                performance_score=0.6,
                compatibility_score=0.95,
                description="Morgan fingerprint via HTTP remote service"
            ))
        except ImportError:
            self.logger.debug("HttpMorganEncoder not available")

        # Sequence (ChemBERTa) - local torch/transformers variant
        try:
            from ..encoders.representations.sequence.chemberta import ChemBERTaEncoder
            self.register_encoder_variant(EncoderVariant(
                name="transformer_chemberta_local",
                encoder_class=ChemBERTaEncoder,
                priority=EncoderPriority.MEDIUM,
                dependency_level=DependencyLevel.PARTIAL,
                required_packages=["torch", "transformers"],
                performance_score=0.7,
                compatibility_score=0.6,
                description="ChemBERTa embeddings via transformers"
            ))
        except ImportError:
            self.logger.debug("ChemBERTaEncoder not available")

        # Graph (GCN) - local torch/DGL variant
        try:
            from ..encoders.representations.graph.gcn import GCNEncoder
            self.register_encoder_variant(EncoderVariant(
                name="gcn_local",
                encoder_class=GCNEncoder,
                priority=EncoderPriority.MEDIUM,
                dependency_level=DependencyLevel.PARTIAL,
                required_packages=["torch", "dgl"],
                performance_score=0.75,
                compatibility_score=0.6,
                description="Graph convolution encoder"
            ))
        except ImportError:
            self.logger.debug("GCNEncoder not available")

        # Transformer (ChemBERTa) - HTTP remote variant
        try:
            from ..encoders.remote.http_chemberta import HttpChemBERTaEncoder
            self.register_encoder_variant(EncoderVariant(
                name="transformer_chemberta_http",
                encoder_class=HttpChemBERTaEncoder,
                priority=EncoderPriority.MEDIUM,
                dependency_level=DependencyLevel.CORE,
                required_packages=["requests"],
                performance_score=0.6,
                compatibility_score=0.95,
                description="ChemBERTa embeddings via HTTP remote service"
            ))
        except ImportError:
            self.logger.debug("HttpChemBERTaEncoder not available")

        # Graph (GCN) - HTTP remote variant
        try:
            from ..encoders.remote.http_gcn import HttpGCNEncoder
            self.register_encoder_variant(EncoderVariant(
                name="gcn_http",
                encoder_class=HttpGCNEncoder,
                priority=EncoderPriority.MEDIUM,
                dependency_level=DependencyLevel.CORE,
                required_packages=["requests"],
                performance_score=0.6,
                compatibility_score=0.9,
                description="GCN graph embeddings via HTTP remote service"
            ))
        except ImportError:
            self.logger.debug("HttpGCNEncoder not available")
    
    def register_encoder_variant(self, variant: EncoderVariant):
        """Register a new encoder variant."""
        encoder_type = variant.name.split('_')[0]  # Extract base encoder name
        
        if encoder_type not in self._encoder_registry:
            self._encoder_registry[encoder_type] = []
        
        self._encoder_registry[encoder_type].append(variant)
        
        # Sort by priority (highest first)
        priority_order = {
            EncoderPriority.HIGHEST: 5,
            EncoderPriority.HIGH: 4,
            EncoderPriority.MEDIUM: 3,
            EncoderPriority.LOW: 2,
            EncoderPriority.FALLBACK: 1
        }
        
        self._encoder_registry[encoder_type].sort(
            key=lambda x: priority_order[x.priority], 
            reverse=True
        )
    
    def get_available_variants(self, encoder_type: str) -> List[EncoderVariant]:
        """Get all available variants for an encoder type."""
        if encoder_type not in self._encoder_registry:
            return []
        
        available = []
        for variant in self._encoder_registry[encoder_type]:
            if self._is_variant_available(variant):
                available.append(variant)
        
        return available
    
    def _is_variant_available(self, variant: EncoderVariant) -> bool:
        """Check if a variant is available based on dependencies."""
        try:
            # Check if required packages are available
            for package in variant.required_packages:
                __import__(package)
            return True
        except ImportError:
            return False
    
    def select_best_variant(
        self, 
        encoder_type: str, 
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Optional[EncoderVariant]:
        """Select the best available variant for an encoder type."""
        
        cache_key = f"{encoder_type}_{strategy.value}"
        if cache_key in self._selection_cache:
            variant = self._selection_cache[cache_key]
            if self._is_variant_available(variant):
                return variant
            else:
                # Remove from cache if no longer available
                del self._selection_cache[cache_key]
        
        available_variants = self.get_available_variants(encoder_type)
        if not available_variants:
            return None
        
        if strategy == SelectionStrategy.PERFORMANCE_FIRST:
            selected = max(available_variants, key=lambda x: x.performance_score)
        elif strategy == SelectionStrategy.COMPATIBILITY_FIRST:
            selected = max(available_variants, key=lambda x: x.compatibility_score)
        elif strategy == SelectionStrategy.BALANCED:
            selected = max(available_variants, 
                         key=lambda x: (x.performance_score + x.compatibility_score) / 2)
        elif strategy == SelectionStrategy.USER_PREFERENCE:
            selected = self._select_by_user_preference(available_variants, user_preferences)
        else:
            # Default to highest priority
            selected = available_variants[0]
        
        self._selection_cache[cache_key] = selected
        return selected
    
    def _select_by_user_preference(
        self, 
        variants: List[EncoderVariant], 
        preferences: Optional[Dict[str, Any]]
    ) -> EncoderVariant:
        """Select variant based on user preferences."""
        if not preferences:
            return variants[0]  # Default to highest priority
        
        # Apply user preferences
        scored_variants = []
        for variant in variants:
            score = 0.0
            
            # Preference for specific dependency level
            if 'dependency_level' in preferences:
                preferred_level = preferences['dependency_level']
                if variant.dependency_level == preferred_level:
                    score += 0.5
            
            # Preference for backend by naming convention
            if 'backend' in preferences and isinstance(preferences['backend'], str):
                backend = preferences['backend'].lower()
                name = variant.name.lower()
                backend_hit = (
                    (backend == 'http' and 'http' in name) or
                    (backend == 'docker' and 'docker' in name) or
                    (backend == 'conda' and 'conda' in name) or
                    (backend == 'venv' and 'venv' in name) or
                    (backend == 'local' and 'local' in name)
                )
                if backend_hit:
                    score += 1.0
            
            # Preference for performance vs compatibility
            performance_weight = preferences.get('performance_weight', 0.5)
            compatibility_weight = preferences.get('compatibility_weight', 0.5)
            
            score += (variant.performance_score * performance_weight + 
                     variant.compatibility_score * compatibility_weight)
            
            # Preference for specific packages
            if 'preferred_packages' in preferences:
                preferred_packages = set(preferences['preferred_packages'])
                variant_packages = set(variant.required_packages)
                if preferred_packages.intersection(variant_packages):
                    score += 0.3
            
            scored_variants.append((variant, score))
        
        # Return variant with highest score
        return max(scored_variants, key=lambda x: x[1])[0]
    
    def create_encoder(
        self, 
        encoder_type: str, 
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        user_preferences: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseEncoder:
        """Create an encoder instance using the best available variant."""
        
        variant = self.select_best_variant(encoder_type, strategy, user_preferences)
        
        if variant is None:
            raise EncoderNotAvailableError(
                f"No available variants for encoder type '{encoder_type}'. "
                f"Please install required dependencies."
            )
        
        self.logger.info(
            f"Creating {encoder_type} encoder using variant '{variant.name}' "
            f"(priority: {variant.priority.value}, "
            f"performance: {variant.performance_score:.2f}, "
            f"compatibility: {variant.compatibility_score:.2f})"
        )
        
        try:
            # Create encoder instance
            encoder = variant.encoder_class(**kwargs)
            
            # Add metadata about the selected variant
            encoder._selected_variant = variant
            encoder._selection_strategy = strategy
            
            return encoder
            
        except Exception as e:
            self.logger.error(f"Failed to create encoder with variant '{variant.name}': {e}")
            
            # Try fallback if available
            if variant.fallback_for:
                fallback_variants = [
                    v for v in self.get_available_variants(encoder_type)
                    if v.priority == EncoderPriority.FALLBACK
                ]
                
                if fallback_variants:
                    fallback_variant = fallback_variants[0]
                    self.logger.info(f"Falling back to variant '{fallback_variant.name}'")
                    
                    try:
                        encoder = fallback_variant.encoder_class(**kwargs)
                        encoder._selected_variant = fallback_variant
                        encoder._selection_strategy = strategy
                        return encoder
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback also failed: {fallback_error}")
            
            raise EncoderNotAvailableError(
                f"Failed to create {encoder_type} encoder: {e}"
            )
    
    def benchmark_variant(self, variant: EncoderVariant, test_smiles: List[str]) -> float:
        """Benchmark a variant's performance."""
        if not self._is_variant_available(variant):
            return 0.0
        
        cache_key = f"{variant.name}_benchmark"
        if cache_key in self._performance_cache:
            return self._performance_cache[cache_key]
        
        try:
            encoder = variant.encoder_class()
            
            # Measure encoding time
            start_time = time.time()
            
            # Test single encoding
            for smiles in test_smiles[:5]:  # Test with first 5 molecules
                encoder.encode(smiles)
            
            # Test batch encoding
            if len(test_smiles) > 1:
                encoder.encode_batch(test_smiles[:10])
            
            end_time = time.time()
            
            # Calculate performance score (lower time = higher score)
            elapsed_time = end_time - start_time
            performance_score = max(0.1, 1.0 / (1.0 + elapsed_time))
            
            self._performance_cache[cache_key] = performance_score
            return performance_score
            
        except Exception as e:
            self.logger.warning(f"Benchmark failed for variant '{variant.name}': {e}")
            return 0.0
    
    def get_selection_report(self, encoder_type: str) -> str:
        """Generate a report of available variants and selection logic."""
        report = [f"\n=== Encoder Selection Report: {encoder_type.upper()} ==="]
        
        if encoder_type not in self._encoder_registry:
            report.append(f"No variants registered for '{encoder_type}'")
            return "\n".join(report)
        
        all_variants = self._encoder_registry[encoder_type]
        available_variants = self.get_available_variants(encoder_type)
        
        report.append(f"Total variants: {len(all_variants)}")
        report.append(f"Available variants: {len(available_variants)}")
        
        report.append("\nVariant Details:")
        for variant in all_variants:
            available = variant in available_variants
            status = "✓ Available" if available else "✗ Unavailable"
            
            report.append(f"  {variant.name} ({variant.priority.value}) - {status}")
            report.append(f"    Performance: {variant.performance_score:.2f}, "
                         f"Compatibility: {variant.compatibility_score:.2f}")
            report.append(f"    Required: {', '.join(variant.required_packages)}")
            report.append(f"    Description: {variant.description}")
            
            if not available:
                missing_packages = []
                for package in variant.required_packages:
                    try:
                        __import__(package)
                    except ImportError:
                        missing_packages.append(package)
                
                if missing_packages:
                    report.append(f"    Missing: {', '.join(missing_packages)}")
        
        # Show selection for different strategies
        report.append("\nRecommended Selection by Strategy:")
        for strategy in SelectionStrategy:
            try:
                selected = self.select_best_variant(encoder_type, strategy)
                if selected:
                    report.append(f"  {strategy.value}: {selected.name}")
                else:
                    report.append(f"  {strategy.value}: None available")
            except Exception:
                report.append(f"  {strategy.value}: Selection failed")
        
        return "\n".join(report)
    
    def clear_cache(self):
        """Clear all caches."""
        self._selection_cache.clear()
        self._performance_cache.clear()


# Global instance
_encoder_selector = None


def get_encoder_selector() -> EncoderSelector:
    """Get the global encoder selector instance."""
    global _encoder_selector
    if _encoder_selector is None:
        _encoder_selector = EncoderSelector()
    return _encoder_selector


def create_smart_encoder(
    encoder_type: str,
    strategy: SelectionStrategy = SelectionStrategy.BALANCED,
    **kwargs
) -> BaseEncoder:
    """Convenience function to create an encoder with smart selection."""
    selector = get_encoder_selector()
    return selector.create_encoder(encoder_type, strategy, **kwargs)