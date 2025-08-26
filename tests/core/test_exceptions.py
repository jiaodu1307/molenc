"""Tests for core.exceptions module."""

import pytest
from molenc.core.exceptions import (
    MolEncError,
    EncoderNotFoundError,
    InvalidSMILESError,
    EncoderInitializationError,
    DependencyError,
    ConfigurationError,
    EncodingError,
)


class TestMolEncError:
    """Test the base MolEncError exception."""
    
    def test_molenc_error_creation(self):
        """Test creating a MolEncError."""
        error = MolEncError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_molenc_error_inheritance(self):
        """Test that MolEncError inherits from Exception."""
        error = MolEncError("Test")
        assert isinstance(error, Exception)


class TestEncoderNotFoundError:
    """Test the EncoderNotFoundError exception."""
    
    def test_encoder_not_found_basic(self):
        """Test basic EncoderNotFoundError creation."""
        error = EncoderNotFoundError("test_encoder")
        assert error.encoder_name == "test_encoder"
        assert error.available_encoders == []
        assert "test_encoder" in str(error)
        assert "not found" in str(error)
    
    def test_encoder_not_found_with_available(self):
        """Test EncoderNotFoundError with available encoders list."""
        available = ["morgan", "maccs", "unimol"]
        error = EncoderNotFoundError("invalid_encoder", available)
        
        assert error.encoder_name == "invalid_encoder"
        assert error.available_encoders == available
        assert "invalid_encoder" in str(error)
        assert "Available encoders" in str(error)
        assert "morgan" in str(error)
        assert "maccs" in str(error)
        assert "unimol" in str(error)
    
    def test_encoder_not_found_inheritance(self):
        """Test that EncoderNotFoundError inherits from MolEncError."""
        error = EncoderNotFoundError("test")
        assert isinstance(error, MolEncError)
        assert isinstance(error, Exception)


class TestInvalidSMILESError:
    """Test the InvalidSMILESError exception."""
    
    def test_invalid_smiles_basic(self):
        """Test basic InvalidSMILESError creation."""
        smiles = "invalid_smiles"
        error = InvalidSMILESError(smiles)
        
        assert error.smiles == smiles
        assert error.reason is None
        assert smiles in str(error)
        assert "Invalid SMILES" in str(error)
    
    def test_invalid_smiles_with_reason(self):
        """Test InvalidSMILESError with reason."""
        smiles = "C[C@H](C)C[C@H](C)C"
        reason = "Stereochemistry parsing failed"
        error = InvalidSMILESError(smiles, reason)
        
        assert error.smiles == smiles
        assert error.reason == reason
        assert smiles in str(error)
        assert reason in str(error)
    
    def test_invalid_smiles_inheritance(self):
        """Test that InvalidSMILESError inherits from MolEncError."""
        error = InvalidSMILESError("test")
        assert isinstance(error, MolEncError)
        assert isinstance(error, Exception)


class TestEncoderInitializationError:
    """Test the EncoderInitializationError exception."""
    
    def test_encoder_init_error_basic(self):
        """Test basic EncoderInitializationError creation."""
        encoder_name = "test_encoder"
        error = EncoderInitializationError(encoder_name)
        
        assert error.encoder_name == encoder_name
        assert error.reason is None
        assert encoder_name in str(error)
        assert "Failed to initialize" in str(error)
    
    def test_encoder_init_error_with_reason(self):
        """Test EncoderInitializationError with reason."""
        encoder_name = "unimol"
        reason = "Model file not found"
        error = EncoderInitializationError(encoder_name, reason)
        
        assert error.encoder_name == encoder_name
        assert error.reason == reason
        assert encoder_name in str(error)
        assert reason in str(error)
    
    def test_encoder_init_error_inheritance(self):
        """Test that EncoderInitializationError inherits from MolEncError."""
        error = EncoderInitializationError("test")
        assert isinstance(error, MolEncError)
        assert isinstance(error, Exception)


class TestDependencyError:
    """Test the DependencyError exception."""
    
    def test_dependency_error_basic(self):
        """Test basic DependencyError creation."""
        dependency = "rdkit"
        error = DependencyError(dependency)
        
        assert error.dependency == dependency
        assert error.encoder_name is None
        assert dependency in str(error)
        assert "not available" in str(error)
    
    def test_dependency_error_with_encoder(self):
        """Test DependencyError with encoder name."""
        dependency = "torch"
        encoder_name = "gcn"
        error = DependencyError(dependency, encoder_name)
        
        assert error.dependency == dependency
        assert error.encoder_name == encoder_name
        assert dependency in str(error)
        assert encoder_name in str(error)
    
    def test_dependency_error_inheritance(self):
        """Test that DependencyError inherits from MolEncError."""
        error = DependencyError("test")
        assert isinstance(error, MolEncError)
        assert isinstance(error, Exception)


class TestConfigurationError:
    """Test the ConfigurationError exception."""
    
    def test_configuration_error_basic(self):
        """Test basic ConfigurationError creation."""
        message = "Invalid configuration"
        error = ConfigurationError(message)
        
        assert error.config_key is None
        assert message in str(error)
    
    def test_configuration_error_with_key(self):
        """Test ConfigurationError with config key."""
        message = "Invalid value"
        config_key = "radius"
        error = ConfigurationError(message, config_key)
        
        assert error.config_key == config_key
        assert message in str(error)
        assert config_key in str(error)
    
    def test_configuration_error_inheritance(self):
        """Test that ConfigurationError inherits from MolEncError."""
        error = ConfigurationError("test")
        assert isinstance(error, MolEncError)
        assert isinstance(error, Exception)


class TestEncodingError:
    """Test the EncodingError exception."""
    
    def test_encoding_error_basic(self):
        """Test basic EncodingError creation."""
        smiles = "CCO"
        encoder_name = "morgan"
        error = EncodingError(smiles, encoder_name)
        
        assert error.smiles == smiles
        assert error.encoder_name == encoder_name
        assert error.reason is None
        assert smiles in str(error)
        assert encoder_name in str(error)
    
    def test_encoding_error_with_reason(self):
        """Test EncodingError with reason."""
        smiles = "invalid"
        encoder_name = "morgan"
        reason = "RDKit parsing failed"
        error = EncodingError(smiles, encoder_name, reason)
        
        assert error.smiles == smiles
        assert error.encoder_name == encoder_name
        assert error.reason == reason
        assert smiles in str(error)
        assert encoder_name in str(error)
        assert reason in str(error)
    
    def test_encoding_error_inheritance(self):
        """Test that EncodingError inherits from MolEncError."""
        error = EncodingError("CCO", "morgan")
        assert isinstance(error, MolEncError)
        assert isinstance(error, Exception)


class TestExceptionChaining:
    """Test exception chaining and context."""
    
    def test_exception_chaining(self):
        """Test that exceptions can be chained properly."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise EncoderInitializationError("test_encoder", "Initialization failed") from e
        except EncoderInitializationError as error:
            assert error.__cause__ is not None
            assert isinstance(error.__cause__, ValueError)
            assert "Original error" in str(error.__cause__)
    
    def test_exception_context(self):
        """Test exception context handling."""
        try:
            try:
                raise InvalidSMILESError("invalid")
            except InvalidSMILESError:
                raise EncodingError("invalid", "morgan", "Failed due to invalid SMILES")
        except EncodingError as error:
            assert error.__context__ is not None
            assert isinstance(error.__context__, InvalidSMILESError)