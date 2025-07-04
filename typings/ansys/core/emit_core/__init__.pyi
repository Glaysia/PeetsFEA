"""
This type stub file was generated by pyright.
"""

import os
import sys
import imp
from importlib import import_module
from ansys.aedt.core.aedt_logger import pyaedt_logger as logger
from ansys.aedt.core.emit_core.emit_constants import EmiCategoryFilter, InterfererType, ResultType, TxRxMode, UnitType

if sys.version_info < (3, 12):
    ...
else:
    ...
EMIT_API_PYTHON = ...
def emit_api_python(): # -> ModuleType:
    """
    Get the EMIT backend API.

    The backend API is available once a ansys.aedt.core.Emit() object has been created.
    An exception is raised if this method is called before a ``ansys.aedt.core.Emit()`` object has been created.
    """
    ...

