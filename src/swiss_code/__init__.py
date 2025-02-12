# src/swiss_code/__init__.py
"""
Swiss Code - A data visualization and geospatial analysis package.
"""

# Import submodules to expose them at the package level
from .data_viz import geospatial, plot_types, utilities

__all__ = ["geospatial", "plot_types", "utilities"]
