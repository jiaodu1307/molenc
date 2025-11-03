"""Decorators for unified exception handling across encoders.

This module provides decorators that standardize exception handling
patterns across different encoder implementations.
"""

import functools
import logging
from typing import Callable, Any, Optional, Type, Union, List

from .exceptions import (
    InvalidSMILESError, 
    EncoderInitializationError,
    DependencyError,
    MolEncError
)


def handle_encoding_errors(
    default_return: Any = None,
    reraise_as: Optional[Type[Exception]] = None,
    log_errors: bool = True
):
    """
    Decorator to handle common encoding errors with consistent patterns.

    Args:
        default_return: Value to return on error (if not re-raising)
        reraise_as: Exception type to re-raise errors as
        log_errors: Whether to log errors

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (InvalidSMILESError, MolEncError):
                # Re-raise MolEnc-specific errors as-is
                raise
            except Exception as e:
                if log_errors:
                    logger = getattr(self, 'logger', logging.getLogger(self.__class__.__name__))
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                
                if reraise_as:
                    raise reraise_as(str(e)) from e
                
                if default_return is not None:
                    return default_return
                
                raise
        
        return wrapper
    return decorator


def handle_smiles_validation(
    reraise_as: Type[Exception] = InvalidSMILESError
):
    """
    Decorator specifically for SMILES validation errors.

    Args:
        reraise_as: Exception type to re-raise validation errors as

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, smiles: str, *args, **kwargs):
            try:
                return func(self, smiles, *args, **kwargs)
            except InvalidSMILESError:
                # Re-raise InvalidSMILESError as-is
                raise
            except Exception as e:
                # Convert other errors to InvalidSMILESError
                raise reraise_as(smiles, f"Validation failed: {str(e)}") from e
        
        return wrapper
    return decorator


def handle_initialization_errors(
    encoder_name: Optional[str] = None
):
    """
    Decorator for handling encoder initialization errors.

    Args:
        encoder_name: Name of the encoder (uses class name if None)

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except EncoderInitializationError:
                # Re-raise EncoderInitializationError as-is
                raise
            except Exception as e:
                name = encoder_name or self.__class__.__name__
                raise EncoderInitializationError(name, str(e)) from e
        
        return wrapper
    return decorator


def handle_batch_processing_errors(
    skip_invalid: bool = True,
    log_skipped: bool = True
):
    """
    Decorator for handling errors in batch processing.

    Args:
        skip_invalid: Whether to skip invalid items or raise error
        log_skipped: Whether to log skipped items

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, items: List[Any], *args, **kwargs):
            if not skip_invalid:
                return func(self, items, *args, **kwargs)
            
            results = []
            skipped_count = 0
            logger = getattr(self, 'logger', logging.getLogger(self.__class__.__name__))
            
            for i, item in enumerate(items):
                try:
                    result = func(self, [item], *args, **kwargs)
                    if result is not None and len(result) > 0:
                        results.extend(result)
                except Exception as e:
                    skipped_count += 1
                    if log_skipped:
                        logger.warning(f"Skipped item {i}: {str(e)}")
            
            if log_skipped and skipped_count > 0:
                logger.info(f"Skipped {skipped_count}/{len(items)} items in batch processing")
            
            return results
        
        return wrapper
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], tuple] = Exception
):
    """
    Decorator to retry function calls on failure.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff_factor: Factor to multiply delay by after each retry
        exceptions: Exception types to retry on

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
            raise last_exception
        
        return wrapper
    return decorator


class ErrorHandlingMixin:
    """Mixin providing standardized error handling methods."""
    
    def __init__(self, **kwargs):
        """Initialize error handling mixin."""
        self.logger = logging.getLogger(self.__class__.__name__)
        super().__init__(**kwargs)
    
    def handle_error(self, error: Exception, context: str = "") -> None:
        """
        Handle an error according to the encoder's error handling strategy.

        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
        """
        error_mode = getattr(self, 'handle_errors', 'raise')
        
        if error_mode == 'raise':
            raise error
        elif error_mode == 'warn':
            self.logger.warning(f"Warning in {context}: {str(error)}")
        elif error_mode == 'skip':
            self.logger.debug(f"Skipping error in {context}: {str(error)}")
        else:
            # Default to raising
            raise error
    
    def safe_execute(self, func: Callable, *args, default=None, context: str = "", **kwargs):
        """
        Safely execute a function with error handling.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            default: Default value to return on error
            context: Context description for error logging
            **kwargs: Keyword arguments for the function

        Returns:
            Function result or default value
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, context)
            return default