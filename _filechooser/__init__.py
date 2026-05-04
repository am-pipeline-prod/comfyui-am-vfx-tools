"""am_vfx_tools.filechooser — shared OS-native file dialog + sandbox.

Public surface:

* :func:`open_path`          — show a native "Open file" dialog.
* :func:`save_path`          — show a native "Save file" dialog.
* :func:`is_available`       — does the host have a usable native-dialog tool?
* :func:`display_ok`         — does the host have a usable display (Linux)?
* :func:`validate`           — sandbox-check a user-supplied path.
* :data:`ROOTS`              — configured sandbox roots (list of :class:`Root`).
* :data:`RECENT_MAX`         — capacity for the recent-files store (callers
  using a recent-files store should respect this).

Cross-OS layout: :mod:`.native_linux` (zenity/kdialog/yad) and
:mod:`.native_windows` (PowerShell + ``System.Windows.Forms``) each
implement the same dispatch keys; this ``__init__`` selects the right one
at import time based on :data:`._config.OS_NAME`.

Vendored from am-pipe-comfy / am_pipe.filechooser. Adapted defaults +
env-var name for the public ``comfyui-am-vfx-tools`` distribution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import _config
from . import _sandbox

ROOTS = _config.ROOTS
RECENT_MAX = _config.RECENT_MAX
ENV_VAR = _config.ENV_VAR

if _config.IS_WINDOWS:
    from . import native_windows as _native
else:
    from . import native_linux as _native


def is_available() -> bool:
    """True if a native-dialog tool was detected at import time."""
    return _config.NATIVE_DIALOG_TOOL is not None


def display_ok() -> bool:
    """Best-effort check whether a usable display is available (Linux only)."""
    return _native.display_ok()


def open_path(
    default_dir: Optional[Path] = None,
    title: str = "Open File",
    file_filters: Optional[list] = None,
) -> Optional[Path]:
    """Show an "Open file" dialog. Returns the chosen path or ``None`` on cancel.

    Raises :class:`RuntimeError` on tool/display failure — caller may fall
    back to an in-browser file browser.
    """
    return _native.open_path(default_dir=default_dir, title=title, file_filters=file_filters)


def save_path(
    default_dir: Optional[Path] = None,
    default_filename: Optional[str] = None,
    title: str = "Save File",
    file_filters: Optional[list] = None,
) -> Optional[Path]:
    """Show a "Save file" dialog. Returns the chosen path or ``None`` on cancel."""
    return _native.save_path(
        default_dir=default_dir,
        default_filename=default_filename,
        title=title,
        file_filters=file_filters,
    )


def validate(path: str, *, must_have_suffix: Optional[str] = None) -> Path:
    """Sandbox-check *path*; return a canonical :class:`Path` or raise.

    See :func:`am_vfx_tools.filechooser._sandbox.validate` for full semantics —
    rejects paths outside the configured roots, paths with NUL bytes, and
    (when *must_have_suffix* is set) paths whose extension doesn't match.
    """
    return _sandbox.validate(path, must_have_suffix=must_have_suffix)


def root_display_strings() -> list:
    """Canonical (as-given) root strings for user-facing output."""
    return _config.root_display_strings()


__all__ = [
    "open_path",
    "save_path",
    "validate",
    "is_available",
    "display_ok",
    "ROOTS",
    "RECENT_MAX",
    "ENV_VAR",
    "root_display_strings",
]
