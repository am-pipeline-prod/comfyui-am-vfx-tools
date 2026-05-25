"""am-vfx-tools-media-io._core.batch_suffix — runtime-discovered ``_bNNNN`` slot.

Mirrors stock ComfyUI's ``folder_paths.get_save_image_path`` counter: scan the
parent directory for files matching the rendered stem, parse the existing
``_bNNNN`` values, return ``max + 1``. The widget on the node is just the
``use_batch`` BOOLEAN — when on, this module owns the integer.

Two entry points by mode:

* :func:`resolve_for_template` — Auto mode. Caller renders the path with
  ``batch=0`` (sentinel), passes the result here. We derive the regex
  prefix/suffix from the sentinel, scan, and return ``(resolved_n,
  final_path)``.
* :func:`resolve_for_manual_path` — Manual mode. Caller passes the literal
  ``file_path`` (post user/var expansion). We inject ``_bNNNN`` before the
  frame token / extension via :func:`inject_into_manual`, scan, and return
  ``(resolved_n, final_path)``.

The slot format ``_b{N:04d}`` is fixed by the dcc-core spec
(``project-structure.md §5.5.6``) — lowercase ``_b`` followed by four digits,
no key/value separator (mirrors ``v001``: numeric-suffix-after-letter).
Padding stays at 4 digits — values past 9999 widen but stay unique. There is
no concurrency guard — the queue is serial; if a future parallel-execute
mode lands, we'll need a lock or a single-pass per queue iteration counter
shared across calls.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple


# Same shape used by am_image_write._has_frame_token. Defined locally so this
# module is self-contained.
_FRAME_TOKEN_RE = re.compile(r"#+|%0?\d*d")

# The sentinel value rendered when batch=0. The dcc-core template uses
# `<_b{batch:04d}>` so batch=0 produces exactly this substring.
_SENTINEL_MARKER = "_b0000"


def _frame_token_to_digits_re(s: str) -> str:
    """Escape *s* as a regex literal, replacing frame tokens with ``\\d+``.

    The on-disk filename has the frame slot expanded to literal digits
    (e.g. ``01001``), so the scan-pattern must match digits where the
    template had ``####`` or ``%05d``.
    """
    out = []
    last = 0
    for m in _FRAME_TOKEN_RE.finditer(s):
        out.append(re.escape(s[last:m.start()]))
        out.append(r"\d+")
        last = m.end()
    out.append(re.escape(s[last:]))
    return "".join(out)


def _scan_max_in_dir(parent_dir: str, regex: re.Pattern) -> int:
    """Return the max integer captured by *regex* over ``os.listdir(parent_dir)``,
    or 0 if no files match (or the dir is missing / unreadable).
    """
    if not os.path.isdir(parent_dir):
        return 0
    max_n = 0
    try:
        for name in os.listdir(parent_dir):
            m = regex.fullmatch(name)
            if m:
                try:
                    n = int(m.group(1))
                except (ValueError, IndexError):
                    continue
                if n > max_n:
                    max_n = n
    except OSError:
        return 0
    return max_n


def resolve_for_template(sentinel_path: str) -> Tuple[Optional[int], str]:
    """Resolve the next ``_bNNNN`` for an Auto-mode rendered path.

    *sentinel_path* is the result of rendering the template with
    ``batch=0`` — it must contain the literal substring ``_b0000``
    exactly where the dcc-core template's ``<_b{batch:04d}>`` segment
    sits. We split around that, build a regex matching files with any
    ``_b(\\d+)`` value (and any frame digits in the rest), find the
    max, and return ``(max+1, final_path_with_substitution)``.

    Returns ``(None, sentinel_path)`` when the sentinel marker is
    missing — the caller passed a template that doesn't have a
    ``<_b{batch}>`` slot, so there's nothing to resolve. The path is
    returned unchanged.
    """
    parent_dir = os.path.dirname(sentinel_path)
    name = os.path.basename(sentinel_path)
    idx = name.rfind(_SENTINEL_MARKER)
    if idx < 0:
        return None, sentinel_path
    pre = name[:idx]
    post = name[idx + len(_SENTINEL_MARKER):]
    pattern = re.escape(pre) + r"_b(\d+)" + _frame_token_to_digits_re(post)
    regex = re.compile(rf"^{pattern}$")
    n = _scan_max_in_dir(parent_dir, regex) + 1
    final_name = f"{pre}_b{n:04d}{post}"
    final_path = os.path.join(parent_dir, final_name) if parent_dir else final_name
    return n, final_path


def inject_into_manual(path: str, batch_n: int) -> str:
    """Insert ``_b{batch_n:04d}`` into a literal Manual-mode path.

    Placement (mirrors the dcc-core template's slot — params, then frame,
    then ext):

    * Path with a frame token (``####`` or ``%0Nd``) — insert before the
      separator that immediately precedes the token (``.`` or ``_``).
      Falls back to inserting just before the token if no separator
      anchors it.
    * Path without a frame token — insert before the final extension.
    """
    base, ext = os.path.splitext(path)
    m = _FRAME_TOKEN_RE.search(base)
    suffix = f"_b{batch_n:04d}"
    if m:
        start = m.start()
        sep_pos = max(base.rfind(".", 0, start), base.rfind("_", 0, start))
        if sep_pos > 0:
            return f"{base[:sep_pos]}{suffix}{base[sep_pos:]}{ext}"
        return f"{base[:start]}{suffix}.{base[start:]}{ext}"
    return f"{base}{suffix}{ext}"


_BATCH_SLOT_RE = re.compile(r"_b(\d+)")


def resolve_latest_existing(path_with_slot: str) -> Optional[str]:
    """Read-side complement to :func:`resolve_for_template` /
    :func:`resolve_for_manual_path`.

    Where those functions return ``max + 1`` (the NEXT free slot), this
    one returns the LATEST EXISTING slot — the one a fresh write would
    have just produced before its iteration count rolled forward.

    The path passed in contains a ``_bNNNN`` segment (e.g. ``_b0000``
    sentinel from Auto-mode render, or a real value from Manual-mode
    injection). Scans the parent directory for any file whose basename
    starts with ``<head>_bNNNN`` (any digits in the slot), picks the
    highest, and substring-replaces it back into the input path.
    Returns ``None`` when the directory is missing / no matching file
    is on disk.

    The pattern is anchored on the prefix only (head + ``_bNNNN``) — we
    don't constrain what comes AFTER the slot, so frame-token /
    extension differences don't confuse the match.
    """
    parent = os.path.dirname(path_with_slot) or "."
    if not os.path.isdir(parent):
        return None
    base = os.path.basename(path_with_slot)
    m = _BATCH_SLOT_RE.search(base)
    if m is None:
        return None
    slot_start, slot_end = m.span()
    slot_width = slot_end - slot_start - 2
    head = base[:slot_start]
    rx = re.compile(rf"^{re.escape(head)}_b(\d+)")
    best: Optional[int] = None
    try:
        with os.scandir(parent) as it:
            for entry in it:
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                em = rx.match(entry.name)
                if em is None:
                    continue
                try:
                    n = int(em.group(1))
                except ValueError:
                    continue
                if best is None or n > best:
                    best = n
    except OSError:
        return None
    if best is None:
        return None
    new_slot = f"_b{best:0{slot_width}d}"
    return path_with_slot[:slot_start] + new_slot + path_with_slot[slot_end:]


def resolve_for_manual_path(path: str) -> Tuple[int, str]:
    """Resolve the next ``_bNNNN`` for a Manual-mode literal path.

    Renders a sentinel via :func:`inject_into_manual` with ``batch_n=0``
    so the scan pattern is identical to the Auto-mode case, then defers
    to the same scanning logic. Always returns an integer (>=1) — there
    is no "no slot" case in Manual mode because we always inject.
    """
    sentinel = inject_into_manual(path, 0)
    n, final = resolve_for_template(sentinel)
    # Manual-mode injection always produces a marker, so n is never None.
    # Guard anyway in case inject_into_manual ever changes its placement.
    if n is None:
        return 1, inject_into_manual(path, 1)
    return n, final


__all__ = [
    "inject_into_manual",
    "resolve_for_manual_path",
    "resolve_for_template",
    "resolve_latest_existing",
]
