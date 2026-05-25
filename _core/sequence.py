"""am-vfx-tools-media-io._core.sequence — frame-pattern parsing + NFS-friendly scan.

Public surface:

* :func:`parse_frame_pattern` — recognize ``####`` / ``%0Nd`` / trailing
  literal frame number; return a normalized printf pattern.
* :func:`expand_frame_pattern` — substitute the pattern's frame token.
* :func:`detect_sequence_range` — single-syscall directory scan via
  ``os.scandir()`` returning :class:`SequenceInfo` (first / last /
  padding / present_set).

Ported from ``custom-nodes/nuke-nodes/io_nodes.py`` (the upstream
sumitchatterjee13/nuke-nodes-comfyui port). Substantive deltas:

* Directory enumeration uses :func:`os.scandir` rather than
  :func:`glob.glob`. ``glob`` issues a stat per entry plus opendir per
  directory level, which on NFS over OVH<->TrueNAS<->laptop trips a
  multi-second pause for long EXR sequences. ``scandir`` returns
  metadata in a single syscall per dir.
* :func:`detect_sequence_range` returns first / last / padding /
  ``present_set`` directly so callers can resolve missing-frame
  policy and gap detection without re-scanning.

See ``../NOTICE`` for upstream attribution.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple

log = logging.getLogger("am_vfx_tools.media-io.sequence")


_PRINTF_RE = re.compile(r"%0?(\d*)d")
_HASHES_RE = re.compile(r"#+")
# Trailing-literal: digits before the final extension, *separated from the
# rest of the basename by a dot* — e.g. ``image.0001.exr`` or
# ``shot01.0042.png``. The leading-dot requirement is what distinguishes
# a real frame sequence from filenames that merely happen to end with
# digits (AM Write's ``_b0001`` queue-iteration suffix, the ``_2`` collision
# suffix from drag-drop, random web images like ``photo42.png``). Without
# the dot the file is treated as a single literal (no printf form), so
# single-mode reads do not silently snap to a synthetic neighbour and
# range/all modes do not try to iterate a "sequence" of one.
#
# Match groups: (1) the frame digits, (2) the trailing extension incl. the
# dot before it. The lookbehind on the leading dot is zero-width, so
# ``m.start()`` lands on the first digit and the head slice naturally
# preserves the separator dot when rebuilding the printf form.
_TRAILING_DIGITS_RE = re.compile(r"(?<=\.)(\d+)(\.[^.\\/]+)$")


@dataclass(frozen=True)
class SequenceInfo:
    """Result of a directory scan for an image sequence.

    *pattern*       — printf-form path (``…/img.%05d.exr``); equals the
                      input filepath when no sequence pattern is detected.
    *padding*       — digit count; ``0`` for non-sequence single files.
    *first*/*last*  — smallest / largest frame number found on disk, or
                      ``None`` when no on-disk frames were found.
    *present_set*   — frozenset of all frame numbers found on disk.
    """
    pattern: str
    padding: int
    first: Optional[int]
    last: Optional[int]
    present_set: FrozenSet[int] = field(default_factory=frozenset)


def parse_frame_pattern(filepath: str) -> Tuple[str, Optional[str], int]:
    """Parse *filepath* for a frame token.

    Returns ``(printf_pattern, frame_spec, padding)``:

    * ``printf_pattern`` — *filepath* with the frame token normalized to
      ``%0Nd`` form (or returned unchanged when no token is found).
    * ``frame_spec`` — the original token text (``"####"`` / ``"%05d"`` /
      the literal frame digits) or ``None`` when no token is present.
    * ``padding`` — the digit count; ``0`` when *frame_spec* is ``None``.

    Detection rules:

    * Explicit tokens (``####``, ``%0Nd``) always win regardless of how
      they are positioned in the basename.
    * Trailing literal digits are only recognised as a frame token when
      they are preceded by a ``.`` separator (the studio convention,
      e.g. ``plate.0001.exr``). Filenames that merely end with digits
      (``output_b0001.exr``, ``photo_42.jpg``, ``image123.png``) are
      reported as non-sequences — ``frame_spec=None``, ``padding=0``.
      Callers in single-frame mode read these as literals; range/all
      modes treat them as a sequence of one.
    """
    fp = filepath.replace("\\", "/")

    m = _PRINTF_RE.search(fp)
    if m:
        digits = m.group(1)
        padding = int(digits) if digits else 4
        return fp, m.group(0), padding

    m = _HASHES_RE.search(fp)
    if m:
        hashes = m.group(0)
        padding = len(hashes)
        printf = fp.replace(hashes, f"%0{padding}d", 1)
        return printf, hashes, padding

    m = _TRAILING_DIGITS_RE.search(fp)
    if m:
        digits = m.group(1)
        padding = len(digits)
        ext = m.group(2)
        head = fp[: m.start()]
        printf = f"{head}%0{padding}d{ext}"
        return printf, digits, padding

    return fp, None, 0


def expand_frame_pattern(pattern: str, frame: int, padding: int = 0) -> str:
    """Substitute the frame token in *pattern*.

    Accepts ``%0Nd`` and ``####`` forms. *padding* is used only for
    bare ``%d`` (no inline width). Always returns a string.
    """
    f = int(frame)

    m = _PRINTF_RE.search(pattern)
    if m:
        digits = m.group(1)
        width = int(digits) if digits else (padding or 4)
        return pattern.replace(m.group(0), str(f).zfill(width), 1)

    m = _HASHES_RE.search(pattern)
    if m:
        hashes = m.group(0)
        return pattern.replace(hashes, str(f).zfill(len(hashes)), 1)

    return pattern


def _basename_regex(printf_pattern: str, padding: int) -> Optional[re.Pattern]:
    """Build a regex matching just the basename of *printf_pattern* against
    *padding*-digit frame numbers. Returns ``None`` when *printf_pattern*
    contains no printf token."""
    base = os.path.basename(printf_pattern)
    m = _PRINTF_RE.search(base)
    if not m:
        return None
    head = re.escape(base[: m.start()])
    tail = re.escape(base[m.end():])
    return re.compile(f"^{head}(\\d{{{padding}}}){tail}$")


def detect_sequence_range(filepath: str, *, scan_dir: bool = True) -> SequenceInfo:
    """Detect the frame range for the sequence containing *filepath*.

    When *filepath* contains no recognized frame token, returns a
    single-file :class:`SequenceInfo` with ``padding=0`` and an empty
    ``present_set`` regardless of *scan_dir*.

    When *scan_dir* is ``False``, returns the parsed pattern + padding
    without touching disk (used by the per-frame fast path so single-file
    reads never trigger a directory scan on NFS).

    When *scan_dir* is ``True``, lists the parent directory exactly once
    via :func:`os.scandir` and returns the matched frame numbers.
    """
    pattern, frame_spec, padding = parse_frame_pattern(filepath)
    if frame_spec is None or padding == 0:
        return SequenceInfo(
            pattern=pattern, padding=0, first=None, last=None,
            present_set=frozenset(),
        )

    if not scan_dir:
        return SequenceInfo(
            pattern=pattern, padding=padding, first=None, last=None,
            present_set=frozenset(),
        )

    rx = _basename_regex(pattern, padding)
    if rx is None:
        return SequenceInfo(
            pattern=pattern, padding=padding, first=None, last=None,
            present_set=frozenset(),
        )

    directory = os.path.dirname(pattern) or "."
    frames = set()
    try:
        with os.scandir(directory) as it:
            for entry in it:
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                m = rx.match(entry.name)
                if m is None:
                    continue
                try:
                    frames.add(int(m.group(1)))
                except ValueError:
                    continue
    except (FileNotFoundError, NotADirectoryError) as e:
        log.warning("[am_vfx_tools/sequence] directory missing for %s: %s", pattern, e)
        return SequenceInfo(
            pattern=pattern, padding=padding, first=None, last=None,
            present_set=frozenset(),
        )
    except OSError as e:
        log.warning("[am_vfx_tools/sequence] scandir failed for %s: %s", directory, e)
        return SequenceInfo(
            pattern=pattern, padding=padding, first=None, last=None,
            present_set=frozenset(),
        )

    if not frames:
        return SequenceInfo(
            pattern=pattern, padding=padding, first=None, last=None,
            present_set=frozenset(),
        )

    return SequenceInfo(
        pattern=pattern, padding=padding,
        first=min(frames), last=max(frames),
        present_set=frozenset(frames),
    )


__all__ = [
    "SequenceInfo",
    "parse_frame_pattern",
    "expand_frame_pattern",
    "detect_sequence_range",
]
