"""Compat shim — re-export workfile-io config from the vendored
:mod:`._filechooser._config`.

Kept as a separate module so the rest of this package can keep doing
``from . import config`` without referencing the vendored layout.
"""
from __future__ import annotations

from ._filechooser._config import (  # noqa: F401
    ENV_VAR,
    IS_LINUX,
    IS_WINDOWS,
    NATIVE_DIALOG_PATH,
    NATIVE_DIALOG_TOOL,
    OS_NAME,
    RECENT_MAX,
    ROOTS,
    Root,
    recent_store_path,
    root_display_strings,
)

# Workfile-io's recent-files store stays in its own per-app subdir.
RECENT_STORE = recent_store_path(app="workfile-io")
