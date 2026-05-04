"""Compat shim — sandbox check delegated to vendored
:mod:`._filechooser._sandbox` and translated to aiohttp HTTPException.

The shim preserves the return-or-HTTPException contract that
``routes_workfile.py`` and ``recent.py`` call into.
"""
from __future__ import annotations

import logging
from pathlib import Path

from aiohttp import web

from ._filechooser._sandbox import (
    SandboxBadInput,
    SandboxBadSuffix,
    SandboxError,
    SandboxOutsideRoots,
)
from ._filechooser._sandbox import validate as _validate_core

log = logging.getLogger("am_vfx_tools.workfile-io")


def validate(path: str, *, must_be_json: bool = False) -> Path:
    """Sandbox-validate *path*; raise :class:`aiohttp.web.HTTPException` on failure.

    *must_be_json* keeps the old keyword name; internally translates to the
    shared chooser's *must_have_suffix=".json"*.
    """
    try:
        return _validate_core(
            path,
            must_have_suffix="json" if must_be_json else None,
        )
    except SandboxBadInput as e:
        raise web.HTTPBadRequest(reason=str(e))
    except SandboxBadSuffix as e:
        raise web.HTTPBadRequest(reason=str(e))
    except SandboxOutsideRoots as e:
        raise web.HTTPForbidden(reason=str(e))
    except SandboxError as e:  # defensive: any future subclass
        raise web.HTTPBadRequest(reason=str(e))


__all__ = ["validate", "SandboxError"]
