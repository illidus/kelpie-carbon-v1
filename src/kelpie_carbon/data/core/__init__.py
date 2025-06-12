"""
Data integration modules for Kelpie Carbon v1.

This package provides integration with external datasets for validation and training.
"""

from .skema_integration import (
    SKEMADataIntegrator,
    SKEMAValidationPoint,
    get_skema_validation_data,
)

__all__ = ["SKEMAValidationPoint", "SKEMADataIntegrator", "get_skema_validation_data"]
