"""Common mixins and utilities for encoder implementations.

This module provides reusable components that can be mixed into encoder
implementations to reduce code duplication and ensure consistency.
"""

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List
from rdkit import Chem

from .exceptions import InvalidSMILESError, EncoderInitializationError
from .encoder_utils import EncoderUtils


class SMILESValidationMixin:
    """Mixin providing common SMILES validation functionality."""
    
    def validate_and_parse_smiles(self, smiles: str) -> Chem.Mol:
        """
        Validate and parse SMILES string.

        Args:
            smiles: SMILES string to validate and parse

        Returns:
            RDKit molecule object

        Raises:
            InvalidSMILESError: If SMILES is invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise InvalidSMILESError(smiles, "Could not parse SMILES")
            return mol
        except Exception as e:
            if isinstance(e, InvalidSMILESError):
                raise e
            raise InvalidSMILESError(smiles, f"SMILES parsing failed: {str(e)}")

    def validate_smiles_batch(self, smiles_list: List[str]) -> List[Chem.Mol]:
        """
        Validate and parse a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of RDKit molecule objects

        Raises:
            InvalidSMILESError: If any SMILES is invalid
        """
        molecules = []
        for smiles in smiles_list:
            molecules.append(self.validate_and_parse_smiles(smiles))
        return molecules


class ParameterValidationMixin:
    """Mixin providing common parameter validation functionality."""
    
    def __init__(self, **kwargs):
        """Initialize parameter validation mixin."""
        self.logger = logging.getLogger(self.__class__.__name__)
        super().__init__(**kwargs)
    
    def validate_init_parameters(self, **params) -> Dict[str, Any]:
        """
        Validate initialization parameters.

        Args:
            **params: Parameters to validate

        Returns:
            Validated parameters dictionary

        Raises:
            EncoderInitializationError: If parameters are invalid
        """
        validated = {}
        
        for key, value in params.items():
            try:
                validated[key] = self._validate_single_parameter(key, value)
            except Exception as e:
                raise EncoderInitializationError(
                    self.__class__.__name__,
                    f"Invalid parameter '{key}': {str(e)}"
                )
        
        return validated
    
    def _validate_single_parameter(self, key: str, value: Any) -> Any:
        """
        Validate a single parameter.

        Args:
            key: Parameter name
            value: Parameter value

        Returns:
            Validated parameter value
        """
        # Common parameter validations
        if key in ['radius', 'n_bits', 'output_dim', 'max_atoms', 'max_length']:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        
        if key in ['n_bits'] and value & (value - 1) != 0:
            raise ValueError(f"{key} must be a power of 2")
        
        if key in ['dropout'] and (not isinstance(value, (int, float)) or value < 0 or value > 1):
            raise ValueError(f"{key} must be a float between 0 and 1")
        
        return value


class DeviceManagementMixin:
    """Mixin providing device management for PyTorch-based encoders."""
    
    def setup_device(self, device: Optional[str] = None) -> str:
        """
        Setup and validate device for PyTorch operations.

        Args:
            device: Device specification ("cpu", "cuda", or None for auto)

        Returns:
            Validated device string
        """
        return EncoderUtils.setup_device(device)
    
    def move_to_device(self, tensor_or_model, device: Optional[str] = None):
        """
        Move tensor or model to specified device.

        Args:
            tensor_or_model: PyTorch tensor or model
            device: Target device (uses self.device if None)

        Returns:
            Tensor or model on target device
        """
        target_device = device or getattr(self, 'device', 'cpu')
        
        try:
            return tensor_or_model.to(target_device)
        except Exception as e:
            self.logger.warning(f"Failed to move to device {target_device}: {e}")
            return tensor_or_model


class ConfigurationMixin:
    """Mixin providing configuration management functionality."""
    
    def __init__(self, **kwargs):
        """Initialize configuration mixin."""
        self.config = kwargs.copy()
        super().__init__(**kwargs)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration.

        Returns:
            Configuration dictionary
        """
        base_config = {
            'encoder_type': self.__class__.__name__,
            'output_dim': self.get_output_dim() if hasattr(self, 'get_output_dim') else None,
        }
        
        # Add handle_errors if available
        if hasattr(self, 'handle_errors'):
            base_config['handle_errors'] = self.handle_errors
        
        # Merge with stored config
        base_config.update(self.config)
        return base_config
    
    def update_config(self, **kwargs) -> None:
        """
        Update encoder configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        self.config.update(kwargs)


class ModelLoadingMixin:
    """Mixin providing common model loading patterns."""
    
    def __init__(self, **kwargs):
        """Initialize model loading mixin."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model_loaded = False
        super().__init__(**kwargs)
    
    def ensure_model_loaded(self) -> None:
        """
        Ensure model is loaded before use.

        Raises:
            EncoderInitializationError: If model loading fails
        """
        if not self._model_loaded:
            try:
                self._load_model()
                self._model_loaded = True
            except Exception as e:
                raise EncoderInitializationError(
                    self.__class__.__name__,
                    f"Failed to load model: {str(e)}"
                )
    
    @abstractmethod
    def _load_model(self) -> None:
        """
        Load the model. Must be implemented by subclasses.
        """
        pass


class BatchProcessingMixin:
    """Mixin providing efficient batch processing capabilities."""
    
    def optimize_batch_size(self, total_items: int, max_batch_size: int = 32) -> int:
        """
        Optimize batch size based on available resources and total items.

        Args:
            total_items: Total number of items to process
            max_batch_size: Maximum allowed batch size

        Returns:
            Optimized batch size
        """
        if total_items <= max_batch_size:
            return total_items
        
        # Find optimal batch size that divides evenly or leaves small remainder
        for batch_size in range(max_batch_size, 0, -1):
            if total_items % batch_size <= batch_size // 4:  # Allow small remainder
                return batch_size
        
        return max_batch_size
    
    def create_batches(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """
        Create batches from a list of items.

        Args:
            items: List of items to batch
            batch_size: Size of each batch

        Returns:
            List of batches
        """
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


class EncoderMixin(
    SMILESValidationMixin,
    ParameterValidationMixin,
    DeviceManagementMixin,
    ConfigurationMixin,
    ModelLoadingMixin,
    BatchProcessingMixin
):
    """
    Combined mixin providing all common encoder functionality.
    
    This mixin combines all the individual mixins to provide a comprehensive
    set of common functionality for encoder implementations.
    """
    
    def __init__(self, **kwargs):
        """Initialize all mixins."""
        super().__init__(**kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def __repr__(self) -> str:
        """String representation of the encoder."""
        output_dim = getattr(self, 'output_dim', 'unknown')
        if hasattr(self, 'get_output_dim'):
            try:
                output_dim = self.get_output_dim()
            except:
                pass
        
        return f"{self.__class__.__name__}(output_dim={output_dim})"