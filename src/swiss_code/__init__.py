# src/swiss_code/__init__.py
"""
Swiss Code - A data visualization and geospatial analysis package.
Automatically loads all submodules and their functions/classes.
"""

import importlib
import pkgutil
import sys
from pathlib import Path

package_name = __name__

# Dynamically import all submodules
for _, module_name, _ in pkgutil.iter_modules([Path(__file__).parent]):
    full_module_name = f"{package_name}.{module_name}"
    module = importlib.import_module(full_module_name)

    # Add functions and classes to the namespace
    for name, obj in vars(module).items():
        if callable(obj):  # Functions and classes
            setattr(sys.modules[package_name], name, obj)

# Cleanup
del importlib, pkgutil, sys, Path
