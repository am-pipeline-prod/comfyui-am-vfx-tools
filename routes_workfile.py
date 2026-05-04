"""HTTP routes for AM Pipe Work File I/O.

All routes are prefixed ``/am-vfx-tools/workfile-io/``. Path-bearing routes run
through ``sandbox.validate`` before any disk access.
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import re
import time
from pathlib import Path

from aiohttp import web
from server import PromptServer

from . import config, native_dialog, recent, sandbox

log = logging.getLogger("am_vfx_tools.workfile-io")

routes = PromptServer.instance.routes

# Match a version token (case-insensitive) bordered by non-alphanumerics or
# string boundaries. Captures only the digits, so we can preserve the literal
# "v" / "V" the user used.
#
# Examples that match:                              digits captured
#   wf_v001                                          001
#   wf_v01_am                                        01
#   test_workflow-main_v001_am                       001
#   wf.v3.json (in stem if any non-letter precedes)  3
#   v002_only                                        002
# Examples that don't match (no detected version):
#   workflow.json                                    —
#   version_test                                     — (no digits after v)
#   xyz_2 / xyz2                                     — (no v marker)
_VERSION_RE = re.compile(r"(?<![A-Za-z0-9])([vV])(\d+)(?![A-Za-z0-9])")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _wrap_oserror(e: OSError, *, context: str) -> web.HTTPException:
    """Map an OSError to an appropriate HTTP exception with a useful body."""
    text = f"{context}: {e.strerror or e.__class__.__name__}: {e.filename or ''}".strip()
    if isinstance(e, PermissionError):
        return web.HTTPForbidden(text=text)
    if isinstance(e, FileNotFoundError):
        return web.HTTPNotFound(text=text)
    if isinstance(e, IsADirectoryError) or isinstance(e, NotADirectoryError):
        return web.HTTPBadRequest(text=text)
    if isinstance(e, FileExistsError):
        return web.HTTPConflict(text=text)
    return web.HTTPInternalServerError(text=text)


def _fs_safe(handler):
    """Decorator: catch any uncaught OSError raised by an aiohttp handler and
    convert it to the closest HTTP error. Lets handler bodies be simple
    ``p.exists() / p.read_bytes()`` without per-call try/except (Python 3.12's
    pathlib propagates PermissionError where 3.11 swallowed it).
    """
    @functools.wraps(handler)
    async def wrapped(request: web.Request) -> web.Response:
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except OSError as e:
            raise _wrap_oserror(e, context=request.path)
    return wrapped


@routes.get("/am-vfx-tools/workfile-io/roots")
async def get_roots(request: web.Request) -> web.Response:
    return web.json_response({
        "os": config.OS_NAME,
        "roots": config.root_display_strings(),
    })


@routes.get("/am-vfx-tools/workfile-io/native-dialog/available")
async def get_native_available(request: web.Request) -> web.Response:
    tool_present = native_dialog.is_available()
    display_ok = native_dialog.display_ok() if tool_present else False
    return web.json_response({
        "available": tool_present and display_ok,
        "tool": config.NATIVE_DIALOG_TOOL,
        "tool_present": tool_present,
        "display_ok": display_ok,
    })


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------

@routes.get("/am-vfx-tools/workfile-io/list")
@_fs_safe
async def list_dir(request: web.Request) -> web.Response:
    raw = request.query.get("dir", "")
    p = sandbox.validate(raw)
    if not p.is_dir():
        raise web.HTTPBadRequest(text=f"not a directory: {p}")
    entries: list[dict] = []
    try:
        for child in sorted(p.iterdir(), key=lambda c: (c.is_file(), c.name.lower())):
            try:
                st = child.stat()
            except OSError:
                continue
            if child.is_dir():
                entries.append({
                    "name": child.name,
                    "type": "dir",
                    "mtime": st.st_mtime,
                    "size": None,
                })
            elif child.is_file() and child.suffix.lower() == ".json":
                entries.append({
                    "name": child.name,
                    "type": "file",
                    "mtime": st.st_mtime,
                    "size": st.st_size,
                })
    except OSError as e:
        raise _wrap_oserror(e, context=f"list {p}")
    return web.json_response({"dir": str(p), "entries": entries})


@routes.post("/am-vfx-tools/workfile-io/reveal")
@_fs_safe
async def reveal_path(request: web.Request) -> web.Response:
    """Open the OS file manager at a directory (for "Open Current Folder").

    Accepts either a directory path or a file path; in the latter case we
    open the parent directory. Sandboxed via the same validate() as every
    other path-bearing route.
    """
    import subprocess

    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        raise web.HTTPBadRequest(reason=f"invalid JSON body: {e}")
    raw = payload.get("path")
    if not isinstance(raw, str):
        raise web.HTTPBadRequest(reason="missing or non-string 'path'")

    p = sandbox.validate(raw)
    target = p if p.is_dir() else p.parent
    if not target.is_dir():
        raise web.HTTPNotFound(reason=f"directory not found: {target}")

    cmd = ["explorer.exe", str(target)] if config.IS_WINDOWS else ["xdg-open", str(target)]
    try:
        subprocess.Popen(cmd, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError as e:
        raise web.HTTPInternalServerError(reason=f"could not launch file manager: {e}")

    log.info("[am-vfx-tools/workfile-io] reveal %s via %s", target, cmd[0])
    return web.json_response({"opened": str(target)})


@routes.post("/am-vfx-tools/workfile-io/mkdir")
@_fs_safe
async def make_directory(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        raise web.HTTPBadRequest(reason=f"invalid JSON body: {e}")
    raw = payload.get("path")
    if not isinstance(raw, str):
        raise web.HTTPBadRequest(reason="missing or non-string 'path'")
    p = sandbox.validate(raw)
    if p.exists():
        if p.is_dir():
            return web.json_response({"path": str(p), "existed": True})
        raise web.HTTPConflict(text=f"path exists and is not a directory: {p}")
    try:
        p.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        raise _wrap_oserror(e, context=f"mkdir {p}")
    log.info("[am-vfx-tools/workfile-io] mkdir %s", p)
    return web.json_response({"path": str(p), "existed": False})


# ---------------------------------------------------------------------------
# Workflow load/save
# ---------------------------------------------------------------------------

@routes.get("/am-vfx-tools/workfile-io/load")
@_fs_safe
async def load_workflow(request: web.Request) -> web.Response:
    raw = request.query.get("path", "")
    p = sandbox.validate(raw, must_be_json=True)
    if not p.is_file():
        raise web.HTTPNotFound(text=f"file not found: {p}")
    try:
        body = p.read_bytes()
    except OSError as e:
        raise _wrap_oserror(e, context=f"read {p}")
    recent.add(str(p), "load")
    return web.Response(body=body, content_type="application/json")


@routes.post("/am-vfx-tools/workfile-io/save")
@_fs_safe
async def save_workflow(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        raise web.HTTPBadRequest(reason=f"invalid JSON body: {e}")

    raw_path = payload.get("path")
    workflow = payload.get("workflow")
    overwrite = bool(payload.get("overwrite", False))
    if not isinstance(raw_path, str):
        raise web.HTTPBadRequest(reason="missing or non-string 'path'")
    if workflow is None:
        raise web.HTTPBadRequest(reason="missing 'workflow'")

    p = sandbox.validate(raw_path, must_be_json=True)

    if p.exists() and not overwrite:
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = None
        return web.json_response(
            {"error": "exists", "path": str(p), "existing_mtime": mtime},
            status=409,
        )

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise _wrap_oserror(e, context=f"mkdir {p.parent}")

    body = json.dumps(workflow, indent=2, ensure_ascii=False).encode("utf-8")
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}.{int(time.time() * 1000)}")
    try:
        tmp.write_bytes(body)
        os.replace(tmp, p)
    except OSError as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise _wrap_oserror(e, context=f"write {p}")

    recent.add(str(p), "save")
    log.info("[am-vfx-tools/workfile-io] saved %s (%d bytes)", p, len(body))
    return web.json_response({"path": str(p), "bytes_written": len(body)})


@routes.get("/am-vfx-tools/workfile-io/next-version")
@_fs_safe
async def next_version(request: web.Request) -> web.Response:
    raw = request.query.get("path", "")
    p = sandbox.validate(raw, must_be_json=True)
    parent = p.parent
    stem = p.stem
    suffix = p.suffix  # ".json"

    matches = list(_VERSION_RE.finditer(stem))
    if not matches:
        return web.json_response(
            {
                "error": "no-version",
                "name": p.name,
                "message": (
                    f"No version pattern (vNN) found in '{p.name}'. "
                    "Use Save As to give the file a versioned name first "
                    "(e.g. add '_v001' before .json)."
                ),
            },
            status=422,
        )

    # Use the rightmost match — version tokens conventionally sit near the
    # end of the name, even when there's a trailing artist tag like '_am'.
    last = matches[-1]
    v_letter = last.group(1)               # "v" or "V" — preserve user's case
    digits = last.group(2)
    width = len(digits)
    prefix = stem[: last.start()]          # everything before the "v"
    tail = stem[last.end():]               # everything after the digits

    # Look at siblings sharing the same prefix/v/tail/extension shape so we
    # bump past existing higher versions, not just the current one.
    sibling_pat = re.compile(
        rf"^{re.escape(prefix)}{v_letter}(\d+){re.escape(tail)}$",
        re.IGNORECASE,
    )
    max_num = int(digits)
    if parent.is_dir():
        for sib in parent.iterdir():
            if sib.suffix.lower() != suffix.lower():
                continue
            sm = sibling_pat.match(sib.stem)
            if sm:
                n = int(sm.group(1))
                if n > max_num:
                    max_num = n

    next_num = max_num + 1
    new_digits = str(next_num).zfill(width)  # preserve original zero-padding;
                                             # zfill(width) is a no-op if the
                                             # number outgrew the original width
    new_stem = f"{prefix}{v_letter}{new_digits}{tail}"
    return web.json_response({"suggestion": str(parent / f"{new_stem}{suffix}")})


# ---------------------------------------------------------------------------
# Native dialogs
# ---------------------------------------------------------------------------

async def _run_dialog(fn) -> Path | None:
    """Run a blocking dialog function in the default executor so we don't
    block aiohttp's event loop while the user is in the dialog."""
    return await asyncio.get_event_loop().run_in_executor(None, fn)


def _validate_default_dir(raw: str | None) -> Path | None:
    if not raw:
        return None
    p = sandbox.validate(raw)
    return p if p.is_dir() else None


@routes.post("/am-vfx-tools/workfile-io/native-dialog/open")
async def native_dialog_open_route(request: web.Request) -> web.Response:
    payload = await request.json() if request.body_exists else {}
    title = payload.get("title") or "Open Workflow"
    default_dir = _validate_default_dir(payload.get("default_dir"))

    if not native_dialog.is_available():
        return web.json_response({"fallback": "unavailable"})
    if not native_dialog.display_ok():
        return web.json_response({"fallback": "headless"})

    try:
        result = await _run_dialog(
            functools.partial(native_dialog.open_path, default_dir=default_dir, title=title)
        )
    except RuntimeError as e:
        log.warning("[am-vfx-tools/workfile-io] native-dialog/open failed: %s", e)
        return web.json_response({"fallback": "error", "message": str(e)})

    if result is None:
        return web.json_response({"cancelled": True})

    try:
        sandbox.validate(str(result), must_be_json=True)
    except web.HTTPException as e:
        return web.json_response(
            {"error": "outside-sandbox", "path": str(result), "message": e.reason},
            status=403,
        )
    return web.json_response({"path": str(result)})


@routes.post("/am-vfx-tools/workfile-io/native-dialog/save")
async def native_dialog_save_route(request: web.Request) -> web.Response:
    payload = await request.json() if request.body_exists else {}
    title = payload.get("title") or "Save Workflow"
    default_dir = _validate_default_dir(payload.get("default_dir"))
    default_filename = payload.get("default_filename") or None

    if not native_dialog.is_available():
        return web.json_response({"fallback": "unavailable"})
    if not native_dialog.display_ok():
        return web.json_response({"fallback": "headless"})

    try:
        result = await _run_dialog(
            functools.partial(
                native_dialog.save_path,
                default_dir=default_dir,
                default_filename=default_filename,
                title=title,
            )
        )
    except RuntimeError as e:
        log.warning("[am-vfx-tools/workfile-io] native-dialog/save failed: %s", e)
        return web.json_response({"fallback": "error", "message": str(e)})

    if result is None:
        return web.json_response({"cancelled": True})

    try:
        sandbox.validate(str(result), must_be_json=True)
    except web.HTTPException as e:
        return web.json_response(
            {"error": "outside-sandbox", "path": str(result), "message": e.reason},
            status=403,
        )
    # Native save dialog already prompted for overwrite; pass that signal through.
    return web.json_response({"path": str(result), "overwrite": True})


# ---------------------------------------------------------------------------
# Recent files
# ---------------------------------------------------------------------------

def _workflow_meta(p: Path) -> dict:
    """Compute the AM Pipe metadata schema for a given workflow path.

    Mirrors the JS-side stampGraphMetadata() in dialogs.js so callers (e.g.
    the /active route) can return the same shape custom nodes get from
    EXTRA_PNGINFO. Keep these two in sync — see runbook §7.5.
    """
    stem = p.stem
    m = _VERSION_RE.search(stem)
    return {
        "path": str(p),
        "filename": p.name,
        "stem": stem,
        "version_label": (m.group(1) + m.group(2)) if m else None,
        "version_num": int(m.group(2)) if m else None,
        "version_width": len(m.group(2)) if m else None,
        "extension": "comfyui-am-vfx-tools",
    }


@routes.get("/am-vfx-tools/workfile-io/active")
@_fs_safe
async def get_active(request: web.Request) -> web.Response:
    """Return metadata for the most-recently-loaded-or-saved workflow.

    Backed by the recent-files store (auto-updated by /save and /load), so no
    extra state machine to keep in sync. Single-process, single-user scope —
    "most recent" is per-server-process. See runbook §7.5 channel B.
    """
    entries = recent.read()
    if not entries:
        return web.json_response({"active": None})
    most_recent = entries[0]
    return web.json_response({
        "active": _workflow_meta(Path(most_recent["path"])),
        "ts": most_recent.get("ts"),
        "action": most_recent.get("action"),
    })


@routes.get("/am-vfx-tools/workfile-io/recent")
@_fs_safe
async def get_recent(request: web.Request) -> web.Response:
    return web.json_response({"entries": recent.read()})


@routes.post("/am-vfx-tools/workfile-io/recent/clear")
@_fs_safe
async def clear_recent(request: web.Request) -> web.Response:
    recent.clear()
    return web.Response(status=204)


log.info(
    "[am-vfx-tools/workfile-io] routes registered | os=%s roots=%s native=%s",
    config.OS_NAME,
    config.root_display_strings(),
    config.NATIVE_DIALOG_TOOL or "none",
)
