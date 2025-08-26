"""Custom exceptions for MolEnc library."""

from typing import Optional, List, Any


class MolEncError(Exception):
    """Base exception class for MolEnc library."""
    pass


class EncoderNotFoundError(MolEncError):
    """Raised when a requested encoder is not found in the registry."""

    def __init__(self, encoder_name: str, 
                 available_encoders: Optional[List[Any]] = None) -> None:
        self.encoder_name = encoder_name
        self.available_encoders = available_encoders or []

        if self.available_encoders:
            message = (
                f"Encoder '{encoder_name}' not found. "
                f"Available encoders: {', '.join(self.available_encoders)}"
            )
        else:
            message = f"Encoder '{encoder_name}' not found."

        super().__init__(message)


class InvalidSMILESError(MolEncError):
    """Raised when an invalid SMILES string is provided."""

    def __init__(self, smiles: str, reason: Optional[str] = None) -> None:
        self.smiles = smiles
        self.reason = reason

        if reason:
            message = f"Invalid SMILES '{smiles}': {reason}"
        else:
            message = f"Invalid SMILES: '{smiles}'"

        super().__init__(message)


class EncoderInitializationError(MolEncError):
    """Raised when an encoder fails to initialize properly."""

    def __init__(self, encoder_name: str, reason: Optional[str] = None) -> None:
        self.encoder_name = encoder_name
        self.reason = reason

        if reason:
            message = f"Failed to initialize encoder '{encoder_name}': {reason}"
        else:
            message = f"Failed to initialize encoder '{encoder_name}'"

        super().__init__(message)


class EncoderNotAvailableError(MolEncError):
    """Raised when a requested encoder is not available due to missing dependencies."""

    def __init__(self, encoder_name: str, 
                 missing_dependencies: Optional[List[Any]] = None) -> None:
        self.encoder_name = encoder_name
        self.missing_dependencies = missing_dependencies or []

        if self.missing_dependencies:
            message = (
                f"Encoder '{encoder_name}' not available. "
                f"Missing dependencies: {', '.join(self.missing_dependencies)}"
            )
        else:
            message = f"Encoder '{encoder_name}' not available."

        super().__init__(message)


class DependencyError(MolEncError):
    """Raised when there are dependency-related issues."""

    def __init__(self, dependency: str, encoder_name: Optional[str] = None) -> None:
        self.dependency = dependency
        self.encoder_name = encoder_name

        if encoder_name:
            message = f"Dependency '{dependency}' not available for encoder '{encoder_name}'"
        else:
            message = f"Dependency '{dependency}' not available"

        super().__init__(message)


class CloudAPIError(MolEncError):
    """Raised when cloud API operations fail."""

    def __init__(self, operation: str, reason: Optional[str] = None, 
                 status_code: Optional[int] = None) -> None:
        self.operation = operation
        self.reason = reason
        self.status_code = status_code

        message_parts = [f"Cloud API error during '{operation}'"]
        if status_code:
            message_parts.append(f"(status: {status_code})")
        if reason:
            message_parts.append(f": {reason}")

        super().__init__(' '.join(message_parts))


class IsolationError(MolEncError):
    """Raised when environment isolation operations fail."""

    def __init__(self, environment: str, operation: str, reason: Optional[str] = None) -> None:
        self.environment = environment
        self.operation = operation
        self.reason = reason

        if reason:
            message = f"Isolation error in '{environment}' during '{operation}': {reason}"
        else:
            message = f"Isolation error in '{environment}' during '{operation}'"

        super().__init__(message)


class ConfigurationError(MolEncError):
    """Raised when there's an error in encoder configuration."""

    def __init__(self, message: str, config_key: Optional[str] = None) -> None:
        self.config_key = config_key

        if config_key:
            full_message = f"Configuration error for '{config_key}': {message}"
        else:
            full_message = f"Configuration error: {message}"

        super().__init__(full_message)


class EncodingError(MolEncError):
    """Raised when encoding process fails."""

    def __init__(self, smiles: str, encoder_name: str, reason: Optional[str] = None) -> None:
        self.smiles = smiles
        self.encoder_name = encoder_name
        self.reason = reason

        if reason:
            message = f"Failed to encode '{smiles}' with {encoder_name}: {reason}"
        else:
            message = f"Failed to encode '{smiles}' with {encoder_name}"

        super().__init__(message)
