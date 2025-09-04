"""
Common exceptions used throughout the application.
Corresponds to section 11 of the detailed design document.
"""

class BaseAppException(Exception):
    """Base exception for all application-specific errors."""
    pass

class ConfigurationError(BaseAppException):
    """Error related to application configuration."""
    pass

class ValidationError(BaseAppException):
    """Error related to data validation (e.g., JSON schema)."""
    pass

class InferenceError(BaseAppException):
    """Error occurring during model inference."""
    pass

class RoutingError(BaseAppException):
    """Error occurring in the ConfidenceRouter."""
    pass

class ToolingError(BaseAppException):
    """Error related to function/tool handling."""
    pass

class SecurityError(BaseAppException):
    """Error related to a security policy violation."""
    pass
