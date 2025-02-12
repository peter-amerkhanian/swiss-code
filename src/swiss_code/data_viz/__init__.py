# src/swiss_code/data_viz/__init__.py
"""
Swiss Code - Data visualization tools including geospatial plotting.
Automatically loads all functions and classes from submodules.
"""

import importlib
import pkgutil
import inspect
import sys
from pathlib import Path

# Get the current module (data_viz)
package_name = __name__

# Dynamically import all submodules
for _, module_name, _ in pkgutil.iter_modules([Path(__file__).parent]):
    full_module_name = f"{package_name}.{module_name}"
    module = importlib.import_module(full_module_name)

    # Add functions and classes to the namespace
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) or inspect.isclass(obj):
            setattr(sys.modules[package_name], name, obj)

# Cleanup
del importlib, pkgutil, inspect, sys, Path
