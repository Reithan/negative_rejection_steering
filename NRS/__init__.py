"""NRS package init.

Re-exports `__version__` from nodes_NRS.py so `NRS.__version__` is
importable. This value must match the `version` field in pyproject.toml —
bump both together at release time.
"""

from .nodes_NRS import __version__

__all__ = ["__version__"]
