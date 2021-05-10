# Licensed under a GPLv3 license - see LICENSE.rst

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import sys

__minimum_python_version__ = "3.4"

class UnsupportedPythonError(Exception):
    pass

if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    raise UnsupportedPythonError("astrojoni_tools does not support Python < {}".format(__minimum_python_version__))
