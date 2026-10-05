class DMapError(Exception):
    """Base exception for user-actionable d3map failures."""


class ValidationError(DMapError):
    """Raised when scientific inputs do not satisfy the analysis contract."""
