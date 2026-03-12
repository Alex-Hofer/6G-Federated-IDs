"""Custom exceptions for the federated_ids package.

Provides domain-specific exception classes for validation failures
that must never be silently disabled (unlike assert statements).
"""


class DataValidationError(ValueError):
    """Raised when input data fails validation checks.

    All messages include actual vs expected values for debugging.
    Unlike assert statements, these checks cannot be disabled by
    running Python with the -O (optimize) flag.
    """
