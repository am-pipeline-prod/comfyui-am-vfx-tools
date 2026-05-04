"""Linux native file dialog wrappers.

Subprocesses zenity (preferred), kdialog, or yad. The dialog appears on the
same machine that runs the Python process; the process must therefore have
access to the user's display (X11 ``DISPLAY`` or ``WAYLAND_DISPLAY``).

Each public function returns a :class:`pathlib.Path` on success, ``None`` if
the user cancelled, or raises :class:`RuntimeError` on tool/display failure
(caller may fall back to an in-browser browser).
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from . import _config

log = logging.getLogger("am_vfx_tools.filechooser")

# Generous: user might leave the dialog open while thinking.
_TIMEOUT = 600

# Default file filter pair (label, glob). Each filter is a 2-tuple.
_DEFAULT_FILTERS = [("All files", "*")]


def display_ok() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _require_display() -> None:
    if not display_ok():
        raise RuntimeError(
            "no DISPLAY or WAYLAND_DISPLAY in process env — "
            "running headless, native dialogs cannot show"
        )


def _run(args: List[str], label: str) -> Tuple[int, str, str]:
    log.info("[am-vfx-tools/filechooser] %s: %s", label, args)
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=_TIMEOUT, check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{label}: dialog timed out after {_TIMEOUT}s")
    return result.returncode, (result.stdout or ""), (result.stderr or "")


def _take_first_line(s: str) -> Optional[str]:
    s = s.strip()
    if not s:
        return None
    # zenity multi-select uses '|' as separator; we never request multi but
    # be defensive.
    return s.splitlines()[0].split("|")[0].strip() or None


# ---------------------------------------------------------------------------
# Zenity (GTK)
# ---------------------------------------------------------------------------

def _zenity_open(default_dir, title, file_filters):
    _require_display()
    # NOTE: --ok-label / --cancel-label are NOT supported by zenity's
    # --file-selection mode (verified on Rocky 9 / zenity 3.32). Passing
    # them makes zenity exit 255 with an unhelpful error. Accept zenity's
    # default labels.
    args = [_config.NATIVE_DIALOG_PATH, "--file-selection", f"--title={title}"]
    for label, glob in (file_filters or _DEFAULT_FILTERS):
        args.append(f"--file-filter={label} | {glob}")
    if default_dir:
        args.append(f"--filename={str(default_dir).rstrip('/')}/")
    rc, out, err = _run(args, "zenity-open")
    if rc == 1 and not out.strip():
        return None
    if rc != 0:
        raise RuntimeError(f"zenity-open: exit {rc}, stderr={err.strip()[:200]}")
    first = _take_first_line(out)
    return Path(first) if first else None


def _zenity_save(default_dir, default_filename, title, file_filters):
    _require_display()
    args = [
        _config.NATIVE_DIALOG_PATH, "--file-selection", "--save",
        "--confirm-overwrite", f"--title={title}",
    ]
    for label, glob in (file_filters or _DEFAULT_FILTERS):
        args.append(f"--file-filter={label} | {glob}")
    initial = ""
    if default_dir:
        initial = str(default_dir).rstrip("/") + "/"
    if default_filename:
        initial += default_filename
    if initial:
        args.append(f"--filename={initial}")
    rc, out, err = _run(args, "zenity-save")
    if rc == 1 and not out.strip():
        return None
    if rc != 0:
        raise RuntimeError(f"zenity-save: exit {rc}, stderr={err.strip()[:200]}")
    first = _take_first_line(out)
    return Path(first) if first else None


# ---------------------------------------------------------------------------
# KDialog (KDE)
# ---------------------------------------------------------------------------

def _kdialog_open(default_dir, title, file_filters):
    _require_display()
    start = str(default_dir) if default_dir else str(Path.home())
    # kdialog filter format: "Label (*.ext1 *.ext2)"
    filt = " | ".join(
        f"{label} (*.{glob.lstrip('*.')})" for label, glob in (file_filters or _DEFAULT_FILTERS)
    ) or "All files (*)"
    args = [
        _config.NATIVE_DIALOG_PATH, "--title", title, "--getopenfilename", start, filt,
    ]
    rc, out, err = _run(args, "kdialog-open")
    if rc == 1 and not out.strip():
        return None
    if rc != 0:
        raise RuntimeError(f"kdialog-open: exit {rc}, stderr={err.strip()[:200]}")
    first = _take_first_line(out)
    return Path(first) if first else None


def _kdialog_save(default_dir, default_filename, title, file_filters):
    _require_display()
    start = str(default_dir) if default_dir else str(Path.home())
    if default_filename:
        start = start.rstrip("/") + "/" + default_filename
    filt = " | ".join(
        f"{label} (*.{glob.lstrip('*.')})" for label, glob in (file_filters or _DEFAULT_FILTERS)
    ) or "All files (*)"
    args = [
        _config.NATIVE_DIALOG_PATH, "--title", title, "--getsavefilename", start, filt,
    ]
    rc, out, err = _run(args, "kdialog-save")
    if rc == 1 and not out.strip():
        return None
    if rc != 0:
        raise RuntimeError(f"kdialog-save: exit {rc}, stderr={err.strip()[:200]}")
    first = _take_first_line(out)
    return Path(first) if first else None


# ---------------------------------------------------------------------------
# YAD
# ---------------------------------------------------------------------------

def _yad_open(default_dir, title, file_filters):
    _require_display()
    args = [_config.NATIVE_DIALOG_PATH, "--file", f"--title={title}"]
    for _label, glob in (file_filters or _DEFAULT_FILTERS):
        args.append(f"--file-filter={glob}")
    if default_dir:
        args.append(f"--filename={str(default_dir).rstrip('/')}/")
    rc, out, err = _run(args, "yad-open")
    if rc == 1 and not out.strip():
        return None
    if rc != 0:
        raise RuntimeError(f"yad-open: exit {rc}, stderr={err.strip()[:200]}")
    first = _take_first_line(out)
    return Path(first) if first else None


def _yad_save(default_dir, default_filename, title, file_filters):
    _require_display()
    args = [
        _config.NATIVE_DIALOG_PATH, "--file", "--save", "--confirm-overwrite",
        f"--title={title}",
    ]
    for _label, glob in (file_filters or _DEFAULT_FILTERS):
        args.append(f"--file-filter={glob}")
    initial = ""
    if default_dir:
        initial = str(default_dir).rstrip("/") + "/"
    if default_filename:
        initial += default_filename
    if initial:
        args.append(f"--filename={initial}")
    rc, out, err = _run(args, "yad-save")
    if rc == 1 and not out.strip():
        return None
    if rc != 0:
        raise RuntimeError(f"yad-save: exit {rc}, stderr={err.strip()[:200]}")
    first = _take_first_line(out)
    return Path(first) if first else None


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

_OPEN_DISPATCH = {
    "zenity": _zenity_open,
    "kdialog": _kdialog_open,
    "yad": _yad_open,
}
_SAVE_DISPATCH = {
    "zenity": _zenity_save,
    "kdialog": _kdialog_save,
    "yad": _yad_save,
}


def open_path(default_dir=None, title="Open File", file_filters=None):
    fn = _OPEN_DISPATCH.get(_config.NATIVE_DIALOG_TOOL)
    if fn is None:
        raise RuntimeError(
            f"no native dialog tool available "
            f"(detected: {_config.NATIVE_DIALOG_TOOL!r})"
        )
    return fn(default_dir, title, file_filters)


def save_path(default_dir=None, default_filename=None, title="Save File", file_filters=None):
    fn = _SAVE_DISPATCH.get(_config.NATIVE_DIALOG_TOOL)
    if fn is None:
        raise RuntimeError(
            f"no native dialog tool available "
            f"(detected: {_config.NATIVE_DIALOG_TOOL!r})"
        )
    return fn(default_dir, default_filename, title, file_filters)
