"""Encoder registry and factory for managing different molecular encoders."""

from typing import Dict, Type, Any, List, Optional, Union
import importlib
import numpy as np
from .base import BaseEncoder
from .exceptions import EncoderNotFoundError, EncoderInitializationError, DependencyError


class EncoderRegistry:
    """Registry for managing molecular encoders."""

    def __init__(self) -> None:
        self._encoders: Dict[str, Type[BaseEncoder]] = {}
        self._encoder_modules: Dict[str, str] = {}

    def register(self, name: str, encoder_class: Optional[Type[BaseEncoder]] = None, 
                 module_path: Optional[str] = None) -> None:
        """
        Register an encoder class.

        Args:
            name: Name to register the encoder under
            encoder_class: The encoder class to register
            module_path: Optional module path for lazy loading
        """
        if encoder_class is not None:
            self._encoders[name] = encoder_class
        if module_path:
            self._encoder_modules[name] = module_path

    def get_encoder(self, name: str, **kwargs) -> BaseEncoder:
        """
        Get an encoder instance by name.

        Args:
            name: Name of the encoder
            **kwargs: Parameters to pass to the encoder constructor

        Returns:
            Initialized encoder instance

        Raises:
            EncoderNotFoundError: If encoder is not found
            EncoderInitializationError: If encoder fails to initialize
        """
        if name not in self._encoders or self._encoders.get(name) is None:
            if name in self._encoder_modules:
                self._lazy_load_encoder(name)
            else:
                raise EncoderNotFoundError(name, list(self._encoders.keys()))

        encoder_class = self._encoders[name]

        try:
            return encoder_class(**kwargs)
        except ImportError as e:
            raise DependencyError(str(e), name)
        except Exception as e:
            raise EncoderInitializationError(name, str(e))

    def _lazy_load_encoder(self, name: str) -> None:
        """
        Lazy load an encoder from its module.

        Args:
            name: Name of the encoder to load
        """
        module_path = self._encoder_modules[name]
        try:
            importlib.import_module(module_path)
            # The module should register the encoder when imported
        except ImportError as e:
            raise DependencyError(f"Failed to import {module_path}: {e}", name)

    def list_encoders(self) -> List[str]:
        """
        List all registered encoder names.

        Returns:
            List of encoder names
        """
        return list(self._encoders.keys())

    def is_registered(self, name: str) -> bool:
        """
        Check if an encoder is registered.

        Args:
            name: Name of the encoder

        Returns:
            True if registered, False otherwise
        """
        return name in self._encoders or name in self._encoder_modules


# Global registry instance
_registry = EncoderRegistry()


def register_encoder(name: str, module_path: Optional[str] = None) -> Any:
    """
    Decorator to register an encoder class.

    Args:
        name: Name to register the encoder under
        module_path: Optional module path for lazy loading

    Returns:
        Decorator function
    """
    def decorator(encoder_class: Type[BaseEncoder]) -> Type[BaseEncoder]:
        _registry.register(name, encoder_class, module_path)
        return encoder_class
    return decorator


class MolEncoder:
    """
    Main interface for creating molecular encoders.

    This class provides a unified interface for creating and using different
    types of molecular encoders.
    """

    def __init__(self, encoder_name: str, **kwargs) -> None:
        """
        Initialize a molecular encoder.

        Args:
            encoder_name: Name of the encoder to use
            **kwargs: Parameters to pass to the encoder
        """
        self.encoder_name = encoder_name
        self.encoder = _registry.get_encoder(encoder_name, **kwargs)

    def encode(self, smiles: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode SMILES string(s) to vector(s).

        Args:
            smiles: Single SMILES string or list of SMILES strings

        Returns:
            Encoded vector(s)
        """
        return self.encoder.encode(smiles)

    def encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Encode a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of encoded vectors
        """
        return self.encoder.encode_batch(smiles_list)

    def get_output_dim(self) -> int:
        """
        Get the output dimension of the encoder.

        Returns:
            Output dimension
        """
        return self.encoder.get_output_dim()

    @classmethod
    def from_config(cls, config_path: str) -> 'MolEncoder':
        """
        Create encoder from configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            Initialized MolEncoder instance
        """
        from .config import Config
        config = Config.from_file(config_path)
        return cls(**config.to_dict())

    @classmethod
    def from_preset(cls, preset_name: str) -> 'MolEncoder':
        """
        Create encoder from preset configuration.

        Args:
            preset_name: Name of the preset

        Returns:
            Initialized MolEncoder instance
        """
        from .config import Config
        config = Config.from_preset(preset_name)
        return cls(**config.to_dict())

    @staticmethod
    def list_encoders() -> List[str]:
        """
        List all available encoders.

        Returns:
            List of encoder names
        """
        return _registry.list_encoders()

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the encoder.

        Returns:
            Dictionary containing encoder configuration
        """
        return self.encoder.get_config()

    def __repr__(self) -> str:
        return f"MolEncoder(encoder='{self.encoder_name}', output_dim={self.get_output_dim()})"


# Register built-in encoders with lazy loading
def _register_builtin_encoders() -> None:
    """Register built-in encoders with lazy loading."""
    # Descriptors
    _registry.register(
        'morgan', module_path='molenc.encoders.descriptors.fingerprints.morgan')
    _registry.register('maccs', module_path='molenc.encoders.descriptors.fingerprints.maccs')

    # Representations - Sequence-based
    _registry.register('chemberta', module_path='molenc.encoders.representations.sequence.chemberta')

    # Representations - Graph-based
    _registry.register('gcn', module_path='molenc.encoders.representations.graph.gcn')

    # Representations - Multimodal
    _registry.register(
        'unimol', module_path='molenc.encoders.representations.multimodal.unimol')


# Register built-in encoders on module import
_register_builtin_encoders()
