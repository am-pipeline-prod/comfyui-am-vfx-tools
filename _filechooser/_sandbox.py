"""Sandbox check for user-supplied paths.

The single function :func:`validate` is the only place that decides whether
a path is permitted. It returns a canonical (as-given-prefixed) ``Path`` on
success or raises one of the module-local exception types so the caller can
translate to its own transport error (HTTP, GUI, etc.).

Originally lived in ``work-file-io/sandbox.py`` and raised aiohttp HTTP
exceptions directly; extracted in Phase 2 so non-HTTP callers (the AM Read
/ AM Write Comfy nodes that use the chooser inline) can reuse it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from . import _config

log = logging.getLogger("am_vfx_tools.filechooser")


class SandboxError(ValueError):
    """Base for all sandbox-rejection errors."""


class SandboxBadInput(SandboxError):
    """Path was malformed (empty, contains NUL, has empty final component, etc.)."""


class SandboxOutsideRoots(SandboxError):
    """Path resolved outside every configured root."""


class SandboxBadSuffix(SandboxError):
    """Path's extension didn't match the required suffix."""


def validate(path: str, *, must_have_suffix: Optional[str] = None) -> Path:
    """Resolve *path* and verify it sits under one of the configured roots.

    Returns a canonical :class:`Path` (as-given root prefix preserved) on
    success. Raises :class:`SandboxBadInput`, :class:`SandboxOutsideRoots`,
    or :class:`SandboxBadSuffix` on failure.

    *must_have_suffix* — if given, the resolved path's extension is
    compared case-insensitively against this string (with or without a
    leading ``.``).
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

    for root in _config.ROOTS:
        try:
            if resolved == root.resolved or _is_relative_to(resolved, root.resolved):
                if resolved == root.resolved:
                    return Path(root.as_given)
                rel = resolved.relative_to(root.resolved)
                return Path(root.as_given) / rel
        except ValueError:
            continue

    log.warning(
        "[am-vfx-tools/filechooser] sandbox reject: %s (roots=%s)",
        resolved, [r.as_given for r in _config.ROOTS],
    )
    raise SandboxOutsideRoots(
        f"path {path} is outside configured filechooser roots "
        f"({', '.join(r.as_given for r in _config.ROOTS) or 'none'})"
    )


def _is_relative_to(p: Path, root: Path) -> bool:
    """``Path.is_relative_to`` polyfill for Python < 3.9 (defensive)."""
    try:
        p.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = [
    "validate",
    "SandboxError",
    "SandboxBadInput",
    "SandboxOutsideRoots",
    "SandboxBadSuffix",
]
