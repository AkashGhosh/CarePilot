#!/usr/bin/env python3
"""
Path setup utility for Medical Visual Agent.
This ensures imports work correctly whether running as a script or as a module.
"""

import os
import sys

# Get the project root directory (parent of this file's directory)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add to sys.path if not already there
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Export project root for other modules
__all__ = ['_PROJECT_ROOT']

