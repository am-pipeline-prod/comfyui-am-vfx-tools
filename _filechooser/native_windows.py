"""Windows native file dialog wrappers.

Subprocesses PowerShell with ``System.Windows.Forms`` for the OpenFileDialog
/ SaveFileDialog. This is the same pattern that worked in the pre-Phase-2
work-file-io extension and avoids the tkinter-on-Comfy-server complications.

Requires the calling process to run in the user's interactive session.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from . import _config

log = logging.getLogger("am_vfx_tools.filechooser")

_TIMEOUT = 600
_DEFAULT_FILTERS = [("All files", "*.*")]


def display_ok() -> bool:
    """On Windows the interactive-session check is non-trivial; assume yes."""
    return True


def _ps_quote(s: str) -> str:
    """Escape for embedding in a double-quoted PowerShell string."""
    return s.replace("`", "``").replace('"', '`"').replace("$", "`$")


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
    return s.splitlines()[0].strip() or None


def _format_filter(file_filters) -> str:
    """Build the PowerShell ``Filter`` string from ``[(label, glob), ...]``."""
    parts: List[str] = []
    for label, glob in (file_filters or _DEFAULT_FILTERS):
        parts.append(f"{label} ({glob})|{glob}")
    return "|".join(parts)


_PS_OPEN_TPL = """
[System.Threading.Thread]::CurrentThread.SetApartmentState([System.Threading.ApartmentState]::STA) | Out-Null
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$d = New-Object System.Windows.Forms.OpenFileDialog
$d.Filter = "{filter}"
$d.Title = "{title}"
{init_dir}
$result = $d.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {{ Write-Output $d.FileName }}
"""

_PS_SAVE_TPL = """
[System.Threading.Thread]::CurrentThread.SetApartmentState([System.Threading.ApartmentState]::STA) | Out-Null
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$d = New-Object System.Windows.Forms.SaveFileDialog
$d.Filter = "{filter}"
$d.OverwritePrompt = $true
$d.Title = "{title}"
{init_dir}
{init_filename}
$result = $d.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {{ Write-Output $d.FileName }}
"""


def open_path(default_dir=None, title="Open File", file_filters=None):
    if _config.NATIVE_DIALOG_TOOL is None:
        raise RuntimeError("no PowerShell on PATH; cannot open native dialog")
    init = (
        f'$d.InitialDirectory = "{_ps_quote(str(default_dir))}"' if default_dir else ""
    )
    script = _PS_OPEN_TPL.format(
        title=_ps_quote(title),
        init_dir=init,
        filter=_ps_quote(_format_filter(file_filters)),
    )
    args = [_config.NATIVE_DIALOG_PATH, "-NoProfile", "-STA", "-Command", script]
    rc, out, err = _run(args, "powershell-open")
    if rc != 0:
        raise RuntimeError(f"powershell-open: exit {rc}, stderr={err.strip()[:200]}")
    first = _take_first_line(out)
    return Path(first) if first else None


def save_path(default_dir=None, default_filename=None, title="Save File", file_filters=None):
    if _config.NATIVE_DIALOG_TOOL is None:
        raise RuntimeError("no PowerShell on PATH; cannot open native dialog")
    init = (
        f'$d.InitialDirectory = "{_ps_quote(str(default_dir))}"' if default_dir else ""
    )
    fn = (
        f'$d.FileName = "{_ps_quote(default_filename)}"' if default_filename else ""
    )
    script = _PS_SAVE_TPL.format(
        title=_ps_quote(title),
        init_dir=init,
        init_filename=fn,
        filter=_ps_quote(_format_filter(file_filters)),
    )
    args = [_config.NATIVE_DIALOG_PATH, "-NoProfile", "-STA", "-Command", script]
    rc, out, err = _run(args, "powershell-save")
    if rc != 0:
        raise RuntimeError(f"powershell-save: exit {rc}, stderr={err.strip()[:200]}")
    first = _take_first_line(out)
    return Path(first) if first else None
