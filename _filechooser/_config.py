"""Default-directory roots + native-dialog tool detection.

ROOTS in the public pack are a UX hint, **not** a sandbox: they're used
as the starting directory the file dialog opens at, and as the entries
in the in-browser browser's "roots" dropdown. Path access is NOT
restricted to roots — the AM Read / Write / workfile-io routes accept
any absolute path the user types or picks. This matches stock ComfyUI's
behaviour (no path-allowlist).

Resolves once at import time. Inputs:

* Env var ``AM_VFX_TOOLS_FILECHOOSER_ROOTS`` — colon-separated on Linux,
  semicolon-separated on Windows. Override the default starting points
  by exporting it in your ComfyUI launch environment. Useful if your
  work lives somewhere other than ``~/Documents``.

* If unset, defaults to the user's home directory plus ``~/Documents``
  (when it exists). Open these so a fresh install lands somewhere
  reasonable — but the user can always navigate / type elsewhere.

Each root is tracked as a :class:`Root` ``(as_given, resolved)`` pair.

Vendored from am-pipe-comfy / am_pipe.filechooser. Adapted defaults +
env-var name + sandbox-removed for the public distribution.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


log = logging.getLogger("am_vfx_tools.filechooser")

RECENT_MAX = 20

IS_WINDOWS = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")
OS_NAME = "windows" if IS_WINDOWS else "linux"

ENV_VAR = "AM_VFX_TOOLS_FILECHOOSER_ROOTS"


@dataclass(frozen=True)
class Root:
    as_given: str
    resolved: Path

    def __str__(self) -> str:  # pragma: no cover — trivial
        return self.as_given


def _parse_roots_env(raw: str) -> List[str]:
    sep = ";" if IS_WINDOWS else ":"
    return [p for p in (s.strip() for s in raw.split(sep)) if p]


def _default_roots() -> List[str]:
    """Reasonable starting points for a fresh install with no env-var."""
    home = Path.home()
    candidates = [str(home)]
    docs = home / "Documents"
    if docs.is_dir():
        candidates.append(str(docs))
    return candidates


def _resolve_roots() -> List[Root]:
    raw = os.environ.get(ENV_VAR, "").strip()
    if raw:
        candidates = _parse_roots_env(raw)
        log.info("[am-vfx-tools/filechooser] using roots from env: %s", candidates)
    else:
        candidates = _default_roots()
        log.info(
            "[am-vfx-tools/filechooser] using default roots for %s: %s "
            "(override via env %s)",
            OS_NAME, candidates, ENV_VAR,
        )

    resolved: List[Root] = []
    for c in candidates:
        try:
            r = Path(c).expanduser().resolve(strict=False)
        except (OSError, ValueError) as e:
            log.warning("[am-vfx-tools/filechooser] could not resolve root %r: %s", c, e)
            continue
        if not r.exists():
            log.warning("[am-vfx-tools/filechooser] root does not exist, skipping: %s", r)
            continue
        if not r.is_dir():
            log.warning("[am-vfx-tools/filechooser] root is not a directory, skipping: %s", r)
            continue
        as_given = c
        if len(as_given) > 1 and as_given.endswith(("/", "\\")):
            as_given = as_given.rstrip("/\\")
        resolved.append(Root(as_given=as_given, resolved=r))

    if not resolved:
        log.warning(
            "[am-vfx-tools/filechooser] no resolvable default roots — file dialogs "
            "will open at the OS default location. Path I/O still works (no "
            "sandbox); this only affects where dialogs first land.",
        )
    return resolved


def _detect_native_dialog_tool() -> Tuple[Optional[str], Optional[str]]:
    """Returns ``(tool_name, tool_path)`` or ``(None, None)``."""
    if IS_WINDOWS:
        for cand in ("powershell.exe", "powershell", "pwsh.exe", "pwsh"):
            path = shutil.which(cand)
            if path:
                return ("powershell", path)
        return (None, None)
    for name in ("zenity", "kdialog", "yad"):
        path = shutil.which(name)
        if path:
            return (name, path)
    return (None, None)


ROOTS: List[Root] = _resolve_roots()
NATIVE_DIALOG_TOOL, NATIVE_DIALOG_PATH = _detect_native_dialog_tool()


def root_display_strings() -> List[str]:
    return [r.as_given for r in ROOTS]


def recent_store_path(app: str = "filechooser") -> Path:
    """Per-app recent-files JSON path (for callers that maintain one)."""
    if IS_WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData/Local"))
        return base / "comfyui-am-vfx-tools" / app / "recent.json"
    base_env = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(base_env) if base_env else Path.home() / ".local/state"
    return base / "comfyui-am-vfx-tools" / app / "recent.json"


if NATIVE_DIALOG_TOOL:
    log.info(
        "[am-vfx-tools/filechooser] native dialog tool: %s (%s)",
        NATIVE_DIALOG_TOOL, NATIVE_DIALOG_PATH,
    )
else:
    log.info(
        "[am-vfx-tools/filechooser] no native dialog tool detected — in-browser fallback only"
    )
