"""HTTP routes for AM VFX Tools — Read / Write media nodes.

Backs three pieces of UI on the AM Read Image / AM Read Video / AM Write
Image / AM Write Video nodes:

* ``POST /am-vfx-tools/detect-range`` — the 🔍 Detect Range button on the
  Read nodes. Scans a literal path the JS sends and returns the on-disk
  frame range (and width/height/fps for video). Two probe backends picked
  by file extension:

  * video extensions (``mov / mp4 / mkv / webm / avi / m4v / mpg / mpeg``)
    → PyAV header read, no decode. Returns
    ``{first, last, count, fps, width, height}``.
  * anything else → :func:`_core.sequence.detect_sequence_range`
    (``os.scandir``-based frame scan). Returns
    ``{first, last, count, pattern, padding}``.

* ``POST /am-vfx-tools/drop`` — the drag-drop handler for files dropped
  onto an AM Read node. Saves the file under ComfyUI's ``input/``
  directory (collisions get a numeric suffix), then probes its metadata
  for an embedded ComfyUI ``workflow`` / ``prompt`` so the JS can offer
  to load that workflow instead of just creating a Read node.

* ``POST /am-vfx-tools/open-in-explorer`` — the 📁 Open in Explorer
  button on Read / Write nodes. Reveals the resolved path in the
  OS-native file manager (Windows Explorer / macOS Finder / Linux
  ``xdg-open``). Walks up to the deepest existing ancestor when the
  literal path doesn't exist yet (e.g. fresh write target before any
  frames have rendered).

All routes (the three above + the filechooser routes registered at the
bottom of this module) accept arbitrary absolute paths — there is no
path-allowlist sandbox. The configured "roots" (default: user home +
`~/Documents`; override via `AM_VFX_TOOLS_FILECHOOSER_ROOTS`) are used
purely as the default starting directory for file dialogs and as the
roots dropdown in the in-browser browser. Same model as stock ComfyUI.
"""
from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
from typing import Any, Dict, Optional

from aiohttp import web

from server import PromptServer

from ._core import sequence
from ._filechooser.server import register_filechooser_routes

log = logging.getLogger("am_vfx_tools.routes")

routes = PromptServer.instance.routes


# ---------------------------------------------------------------------------
# Open-in-Explorer helpers — used by the "📁 Open in Explorer" button on
# AM Read / Write nodes.
# ---------------------------------------------------------------------------

def _walk_up_to_existing(path: str) -> Optional[str]:
    """Walk up from *path* until an existing directory is found.

    Used for write-mode targets where the rendered output dir frequently
    doesn't exist yet on a fresh shot — we want to open the deepest
    existing ancestor rather than fail. Returns ``None`` when nothing in
    the walk-up resolves.
    """
    if not path:
        return None
    if os.path.isdir(path):
        return path
    p = os.path.dirname(path) or path
    while p:
        if os.path.isdir(p):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent
    return None


def _spawn_file_manager(path: str) -> None:
    """Open the OS-native file manager at *path*.

    Selects the file when *path* is a file and the platform supports
    reveal-with-select; opens the directory plainly otherwise.

    Cross-platform spawn:

    * Windows — ``explorer.exe /select,<file>`` for files (the comma is
      glued to the flag, no space; passed as a single arg). ``explorer.exe
      <dir>`` for directories.
    * macOS — ``open -R <file>`` reveals + selects in Finder; ``open
      <dir>`` opens the directory.
    * Linux — ``xdg-open <dir>``. Always pass a directory (the parent
      when *path* is a file) because ``xdg-open`` on a file would launch
      the file's associated viewer rather than the file manager. File
      selection is desktop-specific (Nautilus ``--select-uri``, Dolphin
      ``--select``, others have nothing) — KISS and skip it.

    Spawn is fully detached: stdin/stdout/stderr → DEVNULL, no shell.
    The frontend gets a 200 immediately; the file-manager process
    outlives this request.
    """
    is_file = os.path.isfile(path)
    is_dir  = os.path.isdir(path)
    target_dir = path if is_dir else (os.path.dirname(path) or path)

    if sys.platform.startswith("win"):
        if is_file:
            args = ["explorer.exe", f"/select,{path}"]
        else:
            args = ["explorer.exe", target_dir]
    elif sys.platform == "darwin":
        if is_file:
            args = ["open", "-R", path]
        else:
            args = ["open", target_dir]
    else:
        # Linux + others — xdg-open with the directory.
        args = ["xdg-open", target_dir]

    subprocess.Popen(  # noqa: S603 — args is a fixed list, no shell
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )


# ---------------------------------------------------------------------------
# Drag-drop helpers — workflow extraction from dropped files.
# ---------------------------------------------------------------------------

# Image / video formats whose backends preserve the workflow metadata.
# Mirrors :data:`_core.image_backend._WORKFLOW_METADATA_FORMATS` plus the
# four containers in :mod:`_core.video_backend.CODECS`. Any other extension
# returns ``found=false`` from :func:`_extract_workflow_from_path`.
_WF_IMAGE_EXTS = frozenset({"png", "exr"})
_WF_VIDEO_EXTS = frozenset({"mov", "mp4", "mkv", "webm"})


def _strip_nonjson_floats(obj):
    """Recursively replace ``NaN`` / ``+Inf`` / ``-Inf`` floats with ``None``.

    Python's ``json`` module accepts these values both ways (parse + emit)
    via its non-standard ``allow_nan=True`` default. JavaScript's
    ``JSON.parse`` strictly rejects them per ECMA-404 — passing through a
    response containing ``NaN`` blows up at ``await response.json()`` and
    the JS handler gets nothing.

    ComfyUI embeds ``"is_changed": [NaN]`` in saved prompts as an
    "always re-execute" sentinel, so any drop carrying an embedded
    workflow triggers this path. Sanitize once before serializing the
    HTTP response and the browser sees clean JSON.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _strip_nonjson_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strip_nonjson_floats(v) for v in obj]
    return obj


def _safe_basename(name: str) -> str:
    """Strip directory separators + leading dots from an uploaded filename
    so a malicious client can't write outside the input dir.

    Mirrors the shape ComfyUI's stock /upload/image handler uses.
    """
    bare = os.path.basename(name or "")
    bare = bare.lstrip(".")
    return bare or "dropped-file.bin"


def _load_workflow_keys(found_pairs):
    """Pick ``workflow`` / ``prompt`` / ``source_path`` out of a list of
    (key, value) pairs sourced from a file's metadata. Case-insensitive,
    accepts both flat keys (``workflow``/``prompt``) and namespaced keys
    (``comfyui/workflow`` / ``comfyui/prompt`` / ``comfyui/source_path``).

    Returns ``{"workflow": dict|None, "prompt": dict|None,
              "source_path": str|None}``.
    """
    workflow_raw: Optional[str] = None
    prompt_raw: Optional[str] = None
    source_path: Optional[str] = None
    for raw_key, raw_value in found_pairs:
        if raw_value is None:
            continue
        key_lc = str(raw_key).lower()
        if key_lc == "comfyui/workflow" or key_lc == "workflow":
            workflow_raw = workflow_raw or str(raw_value)
        elif key_lc == "comfyui/prompt" or key_lc == "prompt":
            prompt_raw = prompt_raw or str(raw_value)
        elif key_lc == "comfyui/source_path":
            source_path = source_path or str(raw_value)

    import json as _json

    def _parse(s):
        if not s:
            return None
        try:
            return _json.loads(s)
        except Exception:
            return None

    return {
        "workflow": _parse(workflow_raw),
        "prompt":   _parse(prompt_raw),
        "source_path": source_path,
    }


def _find_source_under_workfile(
    workfile_path: Optional[str], basename: str,
) -> Optional[str]:
    """Walk the dcc folder (parent of ``work/``) and look for a file
    matching *basename*. Returns the absolute path iff exactly one match
    exists; ``None`` otherwise (no workfile, no work-dir convention,
    no matches, or multiple matches — ambiguous).

    Lets a dropped file snap back to its on-disk location without
    keeping the uploaded copy. Scope is bounded to the workflow's own
    dcc folder (``…/<dcc>/``) so the walk stays fast and won't
    accidentally match files from sibling shots.
    """
    if not workfile_path or not basename:
        return None
    work_dir = os.path.dirname(workfile_path)
    if os.path.basename(work_dir) != "work":
        return None  # workfile not under the conventional `…/<dcc>/work/`
    dcc_folder = os.path.dirname(work_dir)
    if not dcc_folder or not os.path.isdir(dcc_folder):
        return None
    matches: list = []
    for root, _dirs, files in os.walk(dcc_folder):
        if basename in files:
            matches.append(os.path.join(root, basename))
            if len(matches) > 1:
                return None  # ambiguous, bail
    return matches[0] if matches else None


def _extract_workflow_from_path(path: str) -> Dict[str, Any]:
    """Read *path* and pull `workflow` / `prompt` from its metadata.

    Returns ``{"found": bool, "workflow": dict|None, "prompt": dict|None,
    "format": str|None}``. ``found`` is True iff at least one of workflow
    or prompt was successfully parsed.
    """
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    pairs: list = []
    fmt: Optional[str] = None

    if ext in _WF_IMAGE_EXTS:
        try:
            import OpenImageIO as _oiio  # type: ignore[import-not-found]
        except ImportError:
            return {"found": False, "workflow": None, "prompt": None, "format": ext}
        inp = _oiio.ImageInput.open(path)
        if inp is None:
            return {"found": False, "workflow": None, "prompt": None, "format": ext}
        try:
            spec = inp.spec()
            for attr in spec.extra_attribs:
                try:
                    pairs.append((attr.name, spec.get_string_attribute(attr.name)))
                except Exception:
                    continue
        finally:
            inp.close()
        fmt = ext
    elif ext in _WF_VIDEO_EXTS:
        try:
            import av  # type: ignore[import-not-found]
        except ImportError:
            return {"found": False, "workflow": None, "prompt": None, "format": ext}
        try:
            container = av.open(path)
            try:
                for k, v in dict(container.metadata or {}).items():
                    pairs.append((k, v))
            finally:
                container.close()
            fmt = ext
        except Exception:
            return {"found": False, "workflow": None, "prompt": None, "format": ext}
    else:
        return {"found": False, "workflow": None, "prompt": None, "format": ext or None}

    parsed = _load_workflow_keys(pairs)
    parsed["found"] = bool(parsed["workflow"]) or bool(parsed["prompt"])
    parsed["format"] = fmt
    return parsed


# ---------------------------------------------------------------------------
# Detect-range — backs the 🔍 Detect Range button on AM Read Image /
# AM Read Video. Two probe backends picked by extension.
# ---------------------------------------------------------------------------

_VIDEO_EXTENSIONS = frozenset({
    ".mov", ".mp4", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg",
})


def _probe_video(path: str) -> Dict[str, Any]:
    """Read the first video stream's header via PyAV. No decode.

    Returns a dict mirroring the sequence-scan result shape:
    ``{pattern, padding, first, last, count, fps, width, height}``.
    """
    try:
        import av  # type: ignore
    except ImportError as e:
        raise web.HTTPInternalServerError(
            reason=f"PyAV not available for video probe: {e}"
        )

    container = av.open(path)
    try:
        if not container.streams.video:
            raise web.HTTPBadRequest(
                reason=f"no video stream in container: {path}"
            )
        stream = container.streams.video[0]

        # `stream.frames` is the most reliable source when set. Some
        # containers (raw-stream, fragmented MP4) report 0; fall back to
        # duration * average_rate. Last resort: 0 (unknown).
        total = int(stream.frames or 0)
        rate = stream.average_rate
        fps = float(rate) if rate is not None else 0.0
        if total <= 0 and stream.duration is not None and rate is not None and stream.time_base:
            try:
                total = int(round(
                    float(stream.duration * stream.time_base) * fps
                ))
            except Exception:
                total = 0

        ctx = stream.codec_context
        return {
            "pattern": path,
            "padding": 0,                       # video isn't a printf-form
            "first":   1 if total > 0 else None,
            "last":    total if total > 0 else None,
            "count":   total,
            "fps":     fps,
            "width":   int(getattr(ctx, "width", 0) or 0),
            "height":  int(getattr(ctx, "height", 0) or 0),
        }
    finally:
        container.close()


@routes.post("/am-vfx-tools/detect-range")
async def _detect_range(request: web.Request):
    """POST ``{path: "<abs path>"}`` -> sequence/video info.

    Body: ``{"path": "<absolute path or printf-form sequence pattern>"}``.

    The probe backend is picked by the file extension:

    * video extensions → PyAV header read; returns
      ``{pattern, padding, first, last, count, fps, width, height}``.
    * anything else → ``os.scandir``-based sequence scan; returns
      ``{pattern, padding, first, last, count}``.
    """
    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(reason="body must be JSON")

    body = body or {}
    raw = body.get("path", "")
    if not isinstance(raw, str) or not raw.strip():
        raise web.HTTPBadRequest(reason="missing 'path' string")
    path = os.path.expandvars(os.path.expanduser(raw))

    # Decide probe backend from extension. Video paths must literally
    # exist; sequence patterns may not (the parent directory is what
    # matters for the scandir walk).
    ext = os.path.splitext(path)[1].lower()
    is_video = ext in _VIDEO_EXTENSIONS

    if is_video:
        if not os.path.isfile(path):
            return web.json_response({
                "error": "file-not-found",
                "path": path,
                "message": f"video file not found: {path}",
            }, status=404)
        try:
            return web.json_response(_probe_video(path))
        except web.HTTPException:
            raise
        except Exception as e:
            return web.json_response({
                "error": "video-probe-failed",
                "path": path,
                "message": f"PyAV header read failed: {e}",
            }, status=500)

    # Sequence scan path.
    info = sequence.detect_sequence_range(path, scan_dir=True)
    return web.json_response({
        "pattern": info.pattern,
        "padding": info.padding,
        "first": info.first,
        "last": info.last,
        "count": len(info.present_set),
    })


# ---------------------------------------------------------------------------
# Drag-drop drop handler — saves the uploaded file into ComfyUI's input dir,
# extracts any embedded workflow metadata, and returns both so the JS-side
# drop interceptor can pick (per the user's drag-drop-mode setting) whether
# to load the workflow or create an AM Read node.
# ---------------------------------------------------------------------------

@routes.post("/am-vfx-tools/drop")
async def _drop(request: web.Request):
    """POST multipart/form-data with a single ``file`` field.

    Saves the file into ComfyUI's input directory under its basename
    (collisions get a numeric suffix), then probes its metadata for an
    embedded ComfyUI workflow / prompt.

    Returns::

        {
            "absolute_path": "/.../comfy-ui/input/<filename>",
            "input_dir":     "/.../comfy-ui/input",
            "filename":      "<filename>",
            "resolved_via":  "upload" | "source_path" | "search",
            "frame_padding": int,
            "workflow":      {...} | null,
            "prompt":        {...} | null,
            "found":         bool,
            "format":        "exr" | "png" | "mov" | ...
        }

    The frontend reads the active drag-drop-mode setting and:

    * mode=workflow + ``found`` -> ``app.loadGraphData(workflow)``
    * mode=workflow + not found, OR mode=media -> create AM Read node
      with file_path = ``absolute_path``.
    """
    try:
        reader = await request.multipart()
    except Exception as e:
        raise web.HTTPBadRequest(reason=f"expected multipart upload: {e}")

    field = await reader.next()
    if field is None or field.name != "file":
        raise web.HTTPBadRequest(reason="missing 'file' multipart field")

    filename = _safe_basename(field.filename or "dropped-file.bin")

    # Resolve ComfyUI's input directory the same way stock /upload/image does.
    try:
        import folder_paths  # type: ignore[import-not-found]
        input_dir = folder_paths.get_input_directory()
    except Exception as e:
        raise web.HTTPInternalServerError(
            reason=f"could not resolve ComfyUI input directory: {e}"
        )

    os.makedirs(input_dir, exist_ok=True)

    # Pick a non-clobbering destination — append `_2`, `_3`, ... if needed.
    base, ext = os.path.splitext(filename)
    dest = os.path.join(input_dir, filename)
    counter = 2
    while os.path.exists(dest):
        candidate = f"{base}_{counter}{ext}"
        dest = os.path.join(input_dir, candidate)
        counter += 1
        if counter > 9999:
            raise web.HTTPInternalServerError(
                reason="too many name collisions in input dir"
            )
    saved_filename = os.path.basename(dest)

    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await field.read_chunk(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        # Best-effort cleanup of a partial write.
        try:
            os.remove(dest)
        except Exception:
            pass
        raise web.HTTPInternalServerError(reason=f"upload write failed: {e}")

    # Best-effort metadata extraction. Failure here is non-fatal; the JS
    # falls back to the media path when ``found`` is False.
    try:
        meta = _extract_workflow_from_path(dest)
    except Exception as e:
        log.warning("[am_vfx_tools] workflow extraction failed for %s: %s",
                    dest, e)
        meta = {"found": False, "workflow": None, "prompt": None,
                "source_path": None,
                "format": (ext or "").lstrip(".").lower() or None}

    # Resolve the file's authoritative on-disk path. Three tiers:
    #   1. ``comfyui/source_path`` embedded by AM Image/Video Write at
    #      write time. Most reliable; covers any file written by the
    #      AM Write nodes (image or video).
    #   2. Filename search under the workflow's dcc folder
    #      (``parent-of-work/`` from any workfile path embedded in the
    #      workflow). Covers files that don't have ``source_path`` for
    #      any reason — provided the basename is unique within the dcc
    #      folder.
    #   3. Fall back to the uploaded copy in ComfyUI's input/.
    #
    # On tier 1/2 success, the upload is deleted to avoid clutter — the
    # dropped file is a duplicate of an authoritative on-disk copy.
    resolved_path: Optional[str] = None
    resolved_via: str = "upload"

    src_in_meta = meta.get("source_path")
    if isinstance(src_in_meta, str) and src_in_meta:
        # Embedded source paths are passed through verbatim — no
        # tokenisation or sandbox in the public pack.
        if os.path.isfile(src_in_meta):
            resolved_path = src_in_meta
            resolved_via = "source_path"

    if not resolved_path:
        wf = meta.get("workflow") or {}
        # `extra.am_vfx_tools.path` is a breadcrumb stamped into saved
        # workflows by this pack's workfile-io (see web/lib/dialogs.js
        # `stampGraphMetadata`) — surfaces the path of the workfile that
        # produced this image. Honor it when present so dropped files
        # snap back to their on-disk source.
        wf_extra = (wf.get("extra") if isinstance(wf, dict) else {}) or {}
        wf_meta = wf_extra.get("am_vfx_tools") or {}
        workfile_path = wf_meta.get("path") or wf_meta.get("workfile")
        if workfile_path:
            candidate = _find_source_under_workfile(workfile_path, filename)
            if candidate:
                resolved_path = candidate
                resolved_via = "search"

    if resolved_path:
        # The uploaded copy is redundant. Delete it.
        try:
            os.remove(dest)
        except Exception as e:
            log.warning(
                "[am_vfx_tools] failed to remove redundant upload %s: %s",
                dest, e,
            )
        absolute_path = resolved_path
        out_filename = os.path.basename(resolved_path)
        out_input_dir = os.path.dirname(resolved_path)
    else:
        absolute_path = dest
        out_filename = saved_filename
        out_input_dir = input_dir

    # Frame-sequence sniff — lets the JS-side AM Read Image override the
    # default `frame_mode=all` to `single` when the dropped file isn't
    # part of a recognisable sequence. ``parse_frame_pattern`` returns
    # padding=0 when no ``####`` / ``%0Nd`` / dot-separated trailing
    # digit token is present, so non-sequence drops like
    # ``output_b0001.exr`` / ``photo42.png`` aren't misread as a
    # sequence of one.
    _, _, frame_padding = sequence.parse_frame_pattern(absolute_path)

    log.info(
        "[am_vfx_tools] drop resolved via=%s path=%s padding=%d",
        resolved_via, absolute_path, frame_padding,
    )

    return web.json_response(_strip_nonjson_floats({
        "absolute_path": absolute_path,
        "input_dir":     out_input_dir,
        "filename":      out_filename,
        "resolved_via":  resolved_via,
        "frame_padding": frame_padding,
        "workflow":      meta.get("workflow"),
        "prompt":        meta.get("prompt"),
        "found":         bool(meta.get("found")),
        "format":        meta.get("format"),
    }))


# ---------------------------------------------------------------------------
# Open-in-Explorer route — backs the "📁 Open in Explorer" button on AM
# Read / AM Write nodes. Reveals the resolved path in the OS-native file
# manager (Windows Explorer / macOS Finder / Linux file manager via
# ``xdg-open``). Distinct from a Browse button — Browse opens a chooser
# dialog the user picks from; this opens a file-manager window the user
# navigates / interacts with normally.
# ---------------------------------------------------------------------------

@routes.post("/am-vfx-tools/open-in-explorer")
async def _open_in_explorer(request: web.Request):
    """POST -> ``{ok: true, opened: <abs path>}`` on success.

    Body: ``{"path": "<abs path>"}`` — the path the JS resolved (the
    ``file_path`` widget value on the node).

    If the literal path doesn't exist, walk up to the deepest existing
    ancestor and open that — write outputs frequently don't exist yet
    on a fresh shot.

    Errors:

    * ``400 missing-path`` — body has no ``path`` string.
    * ``404 no-existing-ancestor`` — neither the path nor any ancestor
      exists on disk.
    * ``500 no-file-manager`` — file-manager binary not on PATH (e.g.
      headless Linux without xdg-utils).
    * ``500 spawn-failed`` — anything else from the subprocess spawn.
    """
    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(reason="body must be JSON")

    raw = (body or {}).get("path", "")
    if not isinstance(raw, str) or not raw.strip():
        return web.json_response({
            "error": "missing-path",
            "message": "body must include a non-empty 'path' string",
        }, status=400)

    path = os.path.expandvars(os.path.expanduser(raw.strip()))

    # Walk up if the path doesn't exist (write outputs on a fresh shot).
    target = path if os.path.exists(path) else _walk_up_to_existing(path)
    if not target:
        return web.json_response({
            "error": "no-existing-ancestor",
            "path": path,
            "message": (
                f"neither {path!r} nor any of its ancestors exist on "
                "disk — cannot open a file-manager window."
            ),
        }, status=404)

    try:
        _spawn_file_manager(target)
    except FileNotFoundError as e:
        # The file-manager binary isn't on PATH (e.g. xdg-open missing on
        # a minimal headless Linux install). Surface a clean message
        # instead of HTTP 500 + a stack trace.
        return web.json_response({
            "error": "no-file-manager",
            "platform": sys.platform,
            "message": (
                f"could not launch the OS file manager: {e}. "
                "On headless Linux instances install xdg-utils "
                "(`dnf install xdg-utils`) or run ComfyUI from a "
                "desktop session."
            ),
        }, status=500)
    except Exception as e:
        return web.json_response({
            "error": "spawn-failed",
            "platform": sys.platform,
            "message": f"file-manager spawn failed: {e}",
        }, status=500)

    log.info("[am_vfx_tools] open-in-explorer -> %s", target)
    return web.json_response({"ok": True, "opened": target})


log.info(
    "[am_vfx_tools] media-IO routes registered at /am-vfx-tools/* "
    "(POST /detect-range, POST /drop, POST /open-in-explorer)"
)


# ---------------------------------------------------------------------------
# Filechooser routes — backs the 📂 Browse button on the AM Read / Write
# nodes and the workfile-io menu. Default starting dir comes from
# _filechooser/_config.py (user home + ~/Documents by default, override
# via AM_VFX_TOOLS_FILECHOOSER_ROOTS env). NOT sandboxed — paths are
# accepted verbatim, matching stock ComfyUI behaviour.
# ---------------------------------------------------------------------------

# AM Read accepts any image OIIO understands. We don't constrain
# must_have_suffix — the AM Read node validates the extension after the
# user picks one (so the dialog can show jpg + exr + tif together).
register_filechooser_routes(
    routes,
    prefix="/am-vfx-tools",
    must_have_suffix=None,
    open_title="AM Read — open",
    save_title="AM Write — choose folder",
)

log.info(
    "[am_vfx_tools] filechooser routes registered at /am-vfx-tools/* "
    "(GET /roots, /list, /native-dialog/available; POST /mkdir, /reveal, "
    "/native-dialog/{open,save})"
)
