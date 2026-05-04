"""Shared aiohttp route handlers for the in-browser file chooser.

The pack's own ``routes.py`` registers these handlers under a URL prefix
of its choice and brings its own decorators. The handlers themselves are
plain async functions over :class:`aiohttp.web.Request`.

Errors are raised as :mod:`._sandbox` exceptions; the package-level
``routes.py`` is expected to translate them to aiohttp HTTPException via
the small helper :func:`http_error_for` provided here.

Vendored from am-pipe-comfy / am_vfx_tools.filechooser.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from . import _config, _sandbox

# These are imported lazily inside register_filechooser_routes() so this
# module remains importable from environments without aiohttp installed
# (e.g. the Phase 1 unit-test runner on Rocky).
log = logging.getLogger("am_vfx_tools.filechooser")


def http_error_for(err: _sandbox.SandboxError):
    """Translate a sandbox exception into an aiohttp HTTPException class+text."""
    from aiohttp import web

    if isinstance(err, _sandbox.SandboxBadInput):
        return web.HTTPBadRequest(reason=str(err))
    if isinstance(err, _sandbox.SandboxBadSuffix):
        return web.HTTPBadRequest(reason=str(err))
    return web.HTTPForbidden(reason=str(err))


# ---------------------------------------------------------------------------
# Async dialog runner — keep aiohttp event loop responsive while the user
# sits in a native dialog.
# ---------------------------------------------------------------------------

async def _run_blocking(fn: Callable):
    return await asyncio.get_event_loop().run_in_executor(None, fn)


# ---------------------------------------------------------------------------
# Generic handler factories. Each takes a ``prefix`` (URL prefix without
# trailing slash) and a ``must_have_suffix`` for path-bearing routes; the
# ``register_filechooser_routes`` helper wires them up for you.
# ---------------------------------------------------------------------------

def make_get_roots():
    from aiohttp import web

    async def handler(_request):
        return web.json_response({
            "os": _config.OS_NAME,
            "roots": _config.root_display_strings(),
        })
    return handler


def make_native_available():
    from aiohttp import web

    from . import _config as cfg

    if _config.IS_WINDOWS:
        from . import native_windows as native
    else:
        from . import native_linux as native

    async def handler(_request):
        tool_present = cfg.NATIVE_DIALOG_TOOL is not None
        display_ok = native.display_ok() if tool_present else False
        return web.json_response({
            "available": tool_present and display_ok,
            "tool": cfg.NATIVE_DIALOG_TOOL,
            "tool_present": tool_present,
            "display_ok": display_ok,
        })
    return handler


def make_list_dir(must_have_suffix: Optional[str] = None):
    """List directory entries; only files matching *must_have_suffix* are emitted."""
    from aiohttp import web

    async def handler(request):
        raw = request.query.get("dir", "")
        try:
            p = _sandbox.validate(raw)
        except _sandbox.SandboxError as e:
            raise http_error_for(e)
        if not p.is_dir():
            raise web.HTTPBadRequest(text=f"not a directory: {p}")
        wanted = (
            None if must_have_suffix is None
            else "." + must_have_suffix.lower().lstrip(".")
        )
        entries = []
        try:
            for child in sorted(p.iterdir(), key=lambda c: (c.is_file(), c.name.lower())):
                try:
                    st = child.stat()
                except OSError:
                    continue
                if child.is_dir():
                    entries.append({"name": child.name, "type": "dir", "mtime": st.st_mtime, "size": None})
                elif child.is_file():
                    if wanted is None or child.suffix.lower() == wanted:
                        entries.append({
                            "name": child.name,
                            "type": "file",
                            "mtime": st.st_mtime,
                            "size": st.st_size,
                        })
        except OSError as e:
            raise web.HTTPInternalServerError(text=f"list {p}: {e}")
        return web.json_response({"dir": str(p), "entries": entries})
    return handler


def make_reveal_path():
    """POST {path}: open the OS file manager at the directory."""
    from aiohttp import web

    async def handler(request):
        try:
            payload = await request.json()
        except ValueError as e:
            raise web.HTTPBadRequest(reason=f"invalid JSON body: {e}")
        raw = payload.get("path")
        if not isinstance(raw, str):
            raise web.HTTPBadRequest(reason="missing or non-string 'path'")
        try:
            p = _sandbox.validate(raw)
        except _sandbox.SandboxError as e:
            raise http_error_for(e)
        target = p if p.is_dir() else p.parent
        if not target.is_dir():
            raise web.HTTPNotFound(reason=f"directory not found: {target}")
        cmd = (
            ["explorer.exe", str(target)]
            if _config.IS_WINDOWS
            else ["xdg-open", str(target)]
        )
        try:
            subprocess.Popen(
                cmd,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as e:
            raise web.HTTPInternalServerError(reason=f"could not launch file manager: {e}")
        log.info("[am-vfx-tools/filechooser] reveal %s via %s", target, cmd[0])
        return web.json_response({"opened": str(target)})
    return handler


def make_mkdir():
    from aiohttp import web

    async def handler(request):
        try:
            payload = await request.json()
        except ValueError as e:
            raise web.HTTPBadRequest(reason=f"invalid JSON body: {e}")
        raw = payload.get("path")
        if not isinstance(raw, str):
            raise web.HTTPBadRequest(reason="missing or non-string 'path'")
        try:
            p = _sandbox.validate(raw)
        except _sandbox.SandboxError as e:
            raise http_error_for(e)
        if p.exists():
            if p.is_dir():
                return web.json_response({"path": str(p), "existed": True})
            raise web.HTTPConflict(text=f"path exists and is not a directory: {p}")
        try:
            p.mkdir(parents=True, exist_ok=False)
        except OSError as e:
            raise web.HTTPInternalServerError(text=f"mkdir {p}: {e}")
        log.info("[am-vfx-tools/filechooser] mkdir %s", p)
        return web.json_response({"path": str(p), "existed": False})
    return handler


def make_native_open(must_have_suffix: Optional[str] = None, default_title: str = "Open File"):
    from aiohttp import web

    if _config.IS_WINDOWS:
        from . import native_windows as native
    else:
        from . import native_linux as native

    async def handler(request):
        payload = await request.json() if request.body_exists else {}
        title = payload.get("title") or default_title
        raw_dir = payload.get("default_dir")
        default_dir = None
        if raw_dir:
            # Expand ${ENV}/... and ~ before sandbox validation. JS callers
            # may pass tokenised paths sourced from saved-workflow widget
            # values — e.g. AM Pipe (post-2026-05-01) persists widget
            # `file_path` as ${PROJECT_ROOT}/<rel> for cross-OS portability,
            # so the dirname coming back from JS contains the literal token.
            # Sandbox + native dialog need an absolute path on this OS.
            expanded_dir = os.path.expandvars(os.path.expanduser(raw_dir))
            try:
                p = _sandbox.validate(expanded_dir)
                default_dir = p if p.is_dir() else None
            except _sandbox.SandboxError:
                default_dir = None

        if _config.NATIVE_DIALOG_TOOL is None:
            return web.json_response({"fallback": "unavailable"})
        if not native.display_ok():
            return web.json_response({"fallback": "headless"})

        try:
            result = await _run_blocking(
                functools.partial(native.open_path, default_dir=default_dir, title=title)
            )
        except RuntimeError as e:
            log.warning("[am-vfx-tools/filechooser] native-dialog/open failed: %s", e)
            return web.json_response({"fallback": "error", "message": str(e)})

        if result is None:
            return web.json_response({"cancelled": True})

        try:
            _sandbox.validate(str(result), must_have_suffix=must_have_suffix)
        except _sandbox.SandboxError as e:
            return web.json_response(
                {"error": "outside-sandbox", "path": str(result), "message": str(e)},
                status=403,
            )
        return web.json_response({"path": str(result)})
    return handler


def make_native_save(must_have_suffix: Optional[str] = None, default_title: str = "Save File"):
    from aiohttp import web

    if _config.IS_WINDOWS:
        from . import native_windows as native
    else:
        from . import native_linux as native

    async def handler(request):
        payload = await request.json() if request.body_exists else {}
        title = payload.get("title") or default_title
        default_filename = payload.get("default_filename") or None
        raw_dir = payload.get("default_dir")
        default_dir = None
        if raw_dir:
            # Expand ${ENV}/... and ~ — see make_native_open for the full
            # rationale (tokenised widget paths from saved workflows).
            expanded_dir = os.path.expandvars(os.path.expanduser(raw_dir))
            try:
                p = _sandbox.validate(expanded_dir)
                default_dir = p if p.is_dir() else None
            except _sandbox.SandboxError:
                default_dir = None

        if _config.NATIVE_DIALOG_TOOL is None:
            return web.json_response({"fallback": "unavailable"})
        if not native.display_ok():
            return web.json_response({"fallback": "headless"})

        try:
            result = await _run_blocking(
                functools.partial(
                    native.save_path,
                    default_dir=default_dir,
                    default_filename=default_filename,
                    title=title,
                )
            )
        except RuntimeError as e:
            log.warning("[am-vfx-tools/filechooser] native-dialog/save failed: %s", e)
            return web.json_response({"fallback": "error", "message": str(e)})

        if result is None:
            return web.json_response({"cancelled": True})

        try:
            _sandbox.validate(str(result), must_have_suffix=must_have_suffix)
        except _sandbox.SandboxError as e:
            return web.json_response(
                {"error": "outside-sandbox", "path": str(result), "message": str(e)},
                status=403,
            )
        return web.json_response({"path": str(result), "overwrite": True})
    return handler


# ---------------------------------------------------------------------------
# Convenience: register every standard chooser route at one prefix.
# ---------------------------------------------------------------------------

def register_filechooser_routes(
    routes,
    *,
    prefix: str,
    must_have_suffix: Optional[str] = None,
    open_title: str = "Open File",
    save_title: str = "Save File",
):
    """Register the standard set of chooser routes on a Comfy ``PromptServer.instance.routes``.

    Routes registered (under *prefix*):

    * ``GET  {prefix}/roots``
    * ``GET  {prefix}/native-dialog/available``
    * ``GET  {prefix}/list``
    * ``POST {prefix}/reveal``
    * ``POST {prefix}/mkdir``
    * ``POST {prefix}/native-dialog/open``
    * ``POST {prefix}/native-dialog/save``

    The package's own ``routes.py`` is responsible for any package-specific
    routes (load/save workflow JSON, recent files, active workflow, etc.).
    """
    p = prefix.rstrip("/")

    routes.get(f"{p}/roots")(make_get_roots())
    routes.get(f"{p}/native-dialog/available")(make_native_available())
    routes.get(f"{p}/list")(make_list_dir(must_have_suffix=must_have_suffix))
    routes.post(f"{p}/reveal")(make_reveal_path())
    routes.post(f"{p}/mkdir")(make_mkdir())
    routes.post(f"{p}/native-dialog/open")(
        make_native_open(must_have_suffix=must_have_suffix, default_title=open_title)
    )
    routes.post(f"{p}/native-dialog/save")(
        make_native_save(must_have_suffix=must_have_suffix, default_title=save_title)
    )

    log.info(
        "[am-vfx-tools/filechooser] registered chooser routes at %s "
        "(suffix=%s, native=%s, roots=%s)",
        p, must_have_suffix, _config.NATIVE_DIALOG_TOOL or "none",
        _config.root_display_strings(),
    )


__all__ = [
    "register_filechooser_routes",
    "http_error_for",
    "make_get_roots",
    "make_native_available",
    "make_list_dir",
    "make_reveal_path",
    "make_mkdir",
    "make_native_open",
    "make_native_save",
]
