"""Compat shim — re-export native-dialog primitives from the vendored
:mod:`._filechooser`.
"""
from __future__ import annotations

from ._filechooser import (  # noqa: F401
    display_ok,
    is_available,
    open_path,
    save_path,
)
