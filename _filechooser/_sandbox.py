"""Path validation for the public pack — format-checks only, no path-roots gate.

The internal pipeline pack (`am_pipe.filechooser._sandbox`) enforces a
roots-allowlist: any path outside the configured roots is rejected. That
makes sense in a studio context where there's a real "outside the
pipeline" boundary. For the public ``comfyui-am-vfx-tools`` pack, paths
are unrestricted — same as ComfyUI core, which lets users read/write
anywhere.

What's still checked:
* Empty / NUL-byte paths (defensive)
* Empty final-component paths (e.g. trailing slash with no name)
* Optional ``must_have_suffix`` (used by the workfile-IO routes to enforce
  ``.json`` so a typo doesn't load random binary files)

The exception classes are kept for caller compatibility (``sandbox.py``
shim still translates them to aiohttp HTTPException). ``SandboxOutsideRoots``
is intentionally never raised by this implementation but kept in the
public surface so external callers don't break if they catch it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


log = logging.getLogger("am_vfx_tools.filechooser")


class SandboxError(ValueError):
    """Base for all path-rejection errors."""


class SandboxBadInput(SandboxError):
    """Path was malformed (empty, contains NUL, has empty final component, etc.)."""


class SandboxOutsideRoots(SandboxError):
    """Kept for caller compatibility; never raised by the public pack."""


class SandboxBadSuffix(SandboxError):
    """Path's extension didn't match the required suffix."""


def validate(path: str, *, must_have_suffix: Optional[str] = None) -> Path:
    """Resolve *path* and run format-checks. No roots-allowlist enforcement.

    Returns a resolved absolute :class:`Path` on success.
    Raises :class:`SandboxBadInput` for malformed paths or
    :class:`SandboxBadSuffix` if *must_have_suffix* is given and doesn't match.
    """
    if not isinstance(path, str) or not path:
        raise SandboxBadInput("path is empty")
    if "\x00" in path:
        raise SandboxBadInput("path contains NUL byte")

    try:
        resolved = Path(path).expanduser().resolve(strict=False)
    except (OSError, ValueError) as e:
        raise SandboxBadInput(f"invalid path: {e}")

    if not resolved.name and resolved != resolved.parent:
        raise SandboxBadInput("path has empty final component")

    if must_have_suffix is not None:
        wanted = must_have_suffix.lower().lstrip(".")
        got = resolved.suffix.lower().lstrip(".")
        if got != wanted:
            raise SandboxBadSuffix(
                f"path must have .{wanted} extension, got .{got or '(none)'}"
            )

    return resolved


__all__ = [
    "validate",
    "SandboxError",
    "SandboxBadInput",
    "SandboxOutsideRoots",
    "SandboxBadSuffix",
]
