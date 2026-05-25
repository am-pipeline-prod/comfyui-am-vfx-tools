"""AM Image Write — ComfyUI node.

Writes an IMAGE batch to disk via OpenImageIO with OCIO 2.x color
management, optional per-frame reformat, and embedded ComfyUI workflow
metadata. Use the ``Browse`` button (or paste a path into ``file_path``)
to point at any directory + base filename; the ``ext`` widget picks the
output format (``exr`` / ``png`` / ``jpg`` / ...).

Frame-range knobs mirror AM Image Read symmetrically — same enum names,
same default sentinels — so artists move between Read and Write panels
without remapping. Three orthogonal concepts:

* ``frame_mode`` (single / range / all): which input batch frames to
  write. Default ``all`` = write the full incoming batch.
* ``first_frame`` / ``last_frame``: 1-based input batch indices used
  in ``range`` mode. ``last_frame=-1`` means "auto = batch length".
* ``start_frame``: the OUTPUT frame number for the first written file.
  Default ``1001`` (VFX convention — leaves a 1000-frame leader).
* ``frame_padding``: digit width for the substituted token. Default ``5``.

So a 40-frame batch from a Read at 1001..1040, written with
``frame_mode=all`` + ``start_frame=1001`` produces files
``..._01001.exr`` ... ``..._01040.exr``. To write only the middle ten
frames as ``..._01015.exr`` ... ``..._01024.exr``: ``frame_mode=range``
+ ``first_frame=15`` + ``last_frame=24`` + ``start_frame=1015``.

The ``use_frame_numbers`` toggle (default True) decides whether the
filename embeds a frame digit. When True, per-frame substitution
produces ``...img.01001.png``. When False, the writer collapses every
frame onto the verbatim path (``...img.png``) — useful for single-image
AI saves where the frame number adds noise. Multi-frame batches with
the toggle off log a warning because every frame collapses onto the
same path.

The ``use_batch`` toggle (default ``False``) controls a fourth axis on
top of frame-numbering: queue-iteration uniqueness. ComfyUI's "Batch
count" runs the same workflow N times, producing N execute() calls.
Without a per-execute() suffix, every iteration's writes overwrite the
previous one. When ``use_batch=True``, the node scans the output
folder for existing files matching the rendered stem with a varying
``_bNNNN`` segment, picks ``max+1``, and uses that for every frame
written by this execute() call. So a 40-frame batch run twice produces
``...img_b0001.01001.exr``..``...img_b0001.01040.exr`` then
``...img_b0002.01001.exr``..``...img_b0002.01040.exr``. Mirrors stock
ComfyUI's ``folder_paths.get_save_image_path`` counter; the integer is
runtime-discovered, NOT a workflow widget — there's no ``batch`` INT
knob (was removed 2026-04-28). The slot format ``_b{N:04d}`` — lowercase
``_b`` followed by four digits, no key/value separator. When
``use_batch=False`` (default), no suffix is added and queue iterations
overwrite each other; the toggle is opt-in for queue Batch-count >1
runs.

The ``seed`` parameter (default ``-1``) is metadata-only — it never
flows into the filename. When set to ``-1``, the node scans the
active ``prompt`` for ``AMSeed`` class entries and looks each one up
in the process-global ``_core.seed_registry`` (populated by AM Seed's
``IS_CHANGED`` before any node's ``execute()`` runs — so the value
is guaranteed available regardless of execution order between AM
Seed and this write node). With no AM Seed in the graph the lookup
returns ``None`` and the seed key is omitted entirely from the
file's metadata. Any non-``-1`` value
(typed into the widget OR delivered by a wire from convert-widget-to-
input) wins outright over the registry — the wire path is also how
artists guarantee execution-order pinning in graphs where the AM Seed
node would otherwise be unreachable from the write node.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

import numpy as np
import torch  # noqa: F401  (kept for type-compatibility w/ ComfyUI passthrough)

from ._core import (
    batch_suffix, color, image_backend, preview, reformat, sequence, seed_registry,
)
from ._core import video_lazy

# ComfyUI's global ``--disable-metadata`` flag — same kill-switch that gates
# the stock SaveImage/SaveVideo nodes. Imported defensively so the module
# still loads under the dcc-core test runner where comfy is absent.
try:
    from comfy.cli_args import args as _comfy_args  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    _comfy_args = None

# Native ComfyUI VIDEO type — see docs/media-io-sync-rule.md invariant 28.
# Streaming per-frame branch: peak RAM stays at ONE frame.
try:
    from comfy_api.v0_0_2 import InputImpl as _ComfyInputImpl  # type: ignore[import-not-found]
    _VIDEO_TYPE_AVAILABLE = True
except ImportError:
    _ComfyInputImpl = None  # type: ignore[assignment]
    _VIDEO_TYPE_AVAILABLE = False

# PyAV — already a ComfyUI dependency. Defensive in case it's missing.
try:
    import av as _av  # type: ignore[import-not-found]
    _PYAV_AVAILABLE = True
except ImportError:
    _av = None  # type: ignore[assignment]
    _PYAV_AVAILABLE = False

log = logging.getLogger("am_vfx_tools.write_image")


FRAME_MODE_SINGLE = "single"
FRAME_MODE_RANGE  = "range"
FRAME_MODE_ALL    = "all"
_FRAME_MODES = [FRAME_MODE_SINGLE, FRAME_MODE_RANGE, FRAME_MODE_ALL]

# Saved workflows from before 2026-04-29 carry the unprefixed
# compression values. Translated to the new prefixed form at execute()
# time so existing workflows keep working without manual re-pick.
_LEGACY_COMPRESSION_MAP = {
    "zip":  "exr/zip",
    "zips": "exr/zips",
    "piz":  "exr/piz",
    "dwaa": "exr/dwaa",
    "dwab": "exr/dwab",
    "rle":  "exr/rle",
    "none": "exr/none",
}


def _strip_compression_prefix(value: str) -> Optional[str]:
    """Translate the dropdown value to the bare backend token.

    `exr/zips` → `zips`. Legacy unprefixed values (saved workflows from
    before the 2026-04-29 prefix rework) translate via the legacy map.
    `exr/none` → ``None`` (no compression attribute set on the OIIO
    spec) — matches the pre-prefix sentinel behaviour.
    """
    if value is None:
        return None
    canon = _LEGACY_COMPRESSION_MAP.get(value, value)
    bare = canon.split("/", 1)[1] if "/" in canon else canon
    return None if bare == "none" else bare


# Default padding for the {frame} token when the user-supplied path
# uses ``####`` style placeholders without an explicit width.
_DEFAULT_FRAME_PADDING = 5

# VFX convention: sequences start at 1001 (leaves a 1000-frame leader for
# handles / version bumps / re-times without wrapping the numbering).
_DEFAULT_START_FRAME = 1001


# Only counts EXPLICIT tokens (#### or %0Nd) — never trailing literal
# digits like "_v001.png", which would otherwise be misread as a frame
# spec on the write side.
_EXPLICIT_FRAME_TOKEN_RE = re.compile(r"#+|%0?\d*d")


def _has_frame_token(path: str) -> bool:
    return bool(_EXPLICIT_FRAME_TOKEN_RE.search(path))


def _suffix_with_frame(path: str, frame: int, padding: int) -> str:
    """For paths without an explicit frame token: append ``.NNNNN`` before
    the extension so a multi-frame batch doesn't clobber itself.

    Separator is ``.`` (period) to match the convention used by every
    other template's ``<.{frame}>`` segment.
    """
    base, ext = os.path.splitext(path)
    return f"{base}.{int(frame):0{max(1, padding)}d}{ext}"


class AMImageWrite:
    """ComfyUI node — write IMAGE batch via OIIO + OCIO."""

    @classmethod
    def INPUT_TYPES(cls):
        cs = color.color_space_choices()

        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Absolute output path (directory + base filename)",
                    "tooltip": (
                        "Absolute output path. The `ext` widget below picks "
                        "the file format. Use the 📂 Browse button for the "
                        "native dialog."
                    ),
                }),
                "ext": (["exr", "png", "jpg", "tif", "dpx", "hdr", "webp"], {
                    "default": "exr",
                    "tooltip": (
                        "Output file format. Drives the on-disk extension "
                        "and the OIIO writer plugin selected for the encode."
                    ),
                }),
                # `seed` is unconditional — no `use_seed` toggle (removed
                # 2026-04-28). It does NOT flow into the filename. Sole
                # purpose: metadata embed via the `embed_workflow` toggle
                # (parallel branch). Sentinel `-1` = "unset" → look up the
                # process-global seed_registry by id(prompt) (an AM Seed
                # node in the graph publishes under the same key);
                # registry hit emits `comfyui/seed`, registry miss omits
                # it. Any other value (typed widget OR wired-in via
                # convert-widget-to-input) wins outright over the
                # registry. The frontend extension splices out the
                # auto-generated `control_after_generate` widget for this
                # node — it's obsolete now that AM Seed owns mode/
                # increment/decrement/randomize semantics server-side.
                "seed": ("INT", {
                    "default": -1, "min": -2**63, "max": 2**63 - 1,
                    "tooltip": (
                        "Generation seed — metadata-only, never in the filename. "
                        "-1 = look up the AM Seed registry by id(prompt). "
                        "Any other value (typed or wired) wins over the registry."
                    ),
                }),
                # use_batch toggles a runtime-discovered `_bNNNN` slot
                # (per-execute() unique suffix; mirrors stock ComfyUI's
                # filename counter). Default False — opt-in for queue
                # Batch-count >1 runs. There is intentionally NO paired
                # `batch` INT widget — the value is scanned from the
                # output folder, not artist-set. See module docstring.
                "use_batch": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When On, append a runtime-discovered `_bNNNN` suffix "
                        "(queue-iteration counter, scanned from the output dir). "
                        "Off = no suffix; queue iterations overwrite each other."
                    ),
                }),
                "frame_mode": (_FRAME_MODES, {
                    "default": FRAME_MODE_ALL,
                    "tooltip": (
                        "Which input batch frames to write. "
                        "single = only `first_frame`. "
                        "range = `first_frame`..`last_frame` inclusive. "
                        "all = every frame in the input batch."
                    ),
                }),
                # Frame rate sits directly under `frame_mode` and above
                # `first_frame` — locked across the family (see
                # media-io-sync-rule.md invariant 14b). Sentinel `-1`
                # (the default) = "do not write any frame-rate
                # metadata" — keeps deliverables clean when a sequence
                # has no meaningful frame rate. Any other value (typed
                # widget OR delivered by a wire from convert-widget-to-
                # input — e.g. an upstream AM Read Image's `frame_rate`
                # output) writes the canonical EXR Rational
                # `framesPerSecond` AND the Nuke-pipeline-friendly
                # `input/frame_rate` STRING into the file's header,
                # closing the read→write→re-read roundtrip for image
                # sequences (the read side already probes both keys).
                # PNG also gets these tags; TIFF/JPEG carry them via
                # the JSON-packed ImageDescription strategy.
                "frame_rate": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 480.0,
                    "tooltip": (
                        "Frame rate metadata for the written sequence. "
                        "-1 = don't write any fps metadata (default). "
                        "Any other value writes `framesPerSecond` (EXR canonical "
                        "Rational) AND `input/frame_rate` (Nuke convention) into "
                        "the file header. Wire AM Read Image's `frame_rate` "
                        "output here for full sequence roundtrip."
                    ),
                }),
                # `use_frame_numbers` sits between `frame_rate` and
                # `first_frame` — together with the start_frame /
                # frame_padding knobs below it, governs how the input
                # batch's frames map to filenames.
                "use_frame_numbers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "When On, the filename embeds a frame digit — produces a "
                        "numbered sequence (e.g. `...img.01001.exr`). Off = single-"
                        "image save, no digit; multi-frame batches will collapse "
                        "onto the same path (warning logged)."
                    ),
                }),
                "first_frame": ("INT", {
                    "default": 1, "min": 1, "max": 999999,
                    "tooltip": (
                        "Input batch index (1-based) for single mode; lower bound "
                        "for range mode. Ignored in all mode."
                    ),
                }),
                "last_frame": ("INT", {
                    "default": -1, "min": -1, "max": 999999,
                    "tooltip": (
                        "Input batch upper bound (1-based, inclusive) for range mode. "
                        "-1 = auto = batch length."
                    ),
                }),
                "start_frame": ("INT", {
                    "default": _DEFAULT_START_FRAME,
                    "min": -999999, "max": 999999,
                    "tooltip": (
                        "First OUTPUT frame number embedded in the filename. "
                        "Default 1001 (VFX convention — leaves a 1000-frame leader "
                        "for handles / re-times). Ignored when `use_frame_numbers` "
                        "is Off."
                    ),
                }),
                "frame_padding": ("INT", {
                    "default": _DEFAULT_FRAME_PADDING,
                    "min": 1, "max": 12,
                    "tooltip": (
                        "Digit width of the frame token (default 5 → `01001`). "
                        "Applied when `use_frame_numbers` is On."
                    ),
                }),
                "bit_depth": (["8", "16", "16f", "32f"], {
                    "default": "16f",
                    "tooltip": (
                        "Output numeric type per channel. "
                        "8 = uint8 (LDR formats). 16 = uint16 (TIFF/PNG hi-bit). "
                        "16f = half-float (EXR default — recommended for VFX). "
                        "32f = full float (rare; very large files)."
                    ),
                }),
                # Compression options prefixed `exr/` to flag the
                # OpenEXR-specific applicability (mirrors how AM Write
                # Video's `codec_profile` carries the `prores/422` etc.
                # codec prefix; lowercase to align with the prefix's
                # role as a key/group rather than a brand name). The
                # `exr/` prefix is stripped at execute() time before the
                # value is passed to the backend. Pre-2026-04-29 saved
                # workflows carrying the unprefixed forms still load
                # via a _LEGACY_COMPRESSION_MAP shim. Default `exr/dwaa`
                # — compact lossy compression appropriate for AI-render
                # output (high-frequency noise compresses well; the
                # ~3-4× size reduction is worth the perceptually-
                # invisible loss for proxy/preview deliverables).
                "compression": ([
                    "exr/zip", "exr/zips", "exr/piz",
                    "exr/dwaa", "exr/dwab", "exr/rle", "exr/none",
                ], {
                    "default": "exr/dwaa",
                    "tooltip": (
                        "EXR compression algorithm — other formats ignore this. "
                        "zip/zips/piz = lossless. dwaa/dwab = lossy (~3-4× smaller, "
                        "perceptually invisible for AI/proxy work). rle = run-length. "
                        "none = uncompressed (largest)."
                    ),
                }),
                "working_colorspace": (cs, {
                    "default": color.default_working_colorspace(cs),
                    "tooltip": (
                        "Source colorspace — the space the upstream IMAGE tensor "
                        "is in. The OCIO transform converts from this to "
                        "`output_colorspace`."
                    ),
                }),
                "raw_data": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When On, skip the OCIO transform — pixels written verbatim. "
                        "`working_colorspace` and `output_colorspace` are ignored."
                    ),
                }),
                "output_colorspace": (cs, {
                    "default": color.pick_default(
                        cs, ("ACES2065-1", "ACEScg", "sRGB - Display"),
                    ),
                    "tooltip": (
                        "Destination colorspace. The OCIO transform converts to this, "
                        "and the value is written into the file as the "
                        "`oiio:ColorSpace` tag for downstream readers."
                    ),
                }),
                # When True, embed the workflow's API graph (`prompt`)
                # into the file as ComfyUI-style metadata. Default True
                # mirrors stock ComfyUI's SaveImage. Backend silently
                # skips formats that don't preserve arbitrary metadata
                # via OIIO (only PNG and EXR carry it cleanly today;
                # JPG/TIFF/WebP/DPX/HDR drop it). Artists who don't want
                # workflow JSON in deliverables flip this off.
                "embed_workflow": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Embed the API graph (`comfyui/prompt`) as file metadata "
                        "so the file is round-tripped via AM-Pipe drag-drop. "
                        "Off = clean deliverable without workflow JSON. Honors "
                        "ComfyUI's global `--disable-metadata` flag."
                    ),
                }),
                # ----- Reformat block (synced across all four IO nodes; see
                # media-io-sync-rule.md invariants 15a–15e). Eight widgets,
                # always visible regardless of mode (ComfyUI doesn't support
                # conditional widget hiding). On a Write node the reformat
                # is applied to the input batch BEFORE encoding — common
                # case: AI-generated 1024×576 → deliver as 1920×1080.
                "reformat_mode": (reformat.REFORMAT_MODES, {
                    "default": reformat.MODE_OFF,
                    "tooltip": reformat.TOOLTIP_MODE,
                }),
                "scale": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01,
                    "tooltip": reformat.TOOLTIP_SCALE,
                }),
                "preset": (reformat.PRESET_CHOICES, {
                    "default": reformat.PRESET_WH,
                    "tooltip": reformat.TOOLTIP_PRESET,
                }),
                "target_width": ("INT", {
                    "default": 1920, "min": 1, "max": 16384,
                    "tooltip": reformat.TOOLTIP_TARGET_W,
                }),
                "target_height": ("INT", {
                    "default": 1080, "min": 1, "max": 16384,
                    "tooltip": reformat.TOOLTIP_TARGET_H,
                }),
                "resize_type": (reformat.RESIZE_CHOICES, {
                    "default": reformat.RESIZE_FIT,
                    "tooltip": reformat.TOOLTIP_RESIZE_TYPE,
                }),
                "filter": (reformat.FILTER_CHOICES, {
                    "default": reformat.FILTER_CUBIC,
                    "tooltip": reformat.TOOLTIP_FILTER,
                }),
                "output_dtype": (reformat.DTYPE_CHOICES, {
                    "default": reformat.DEFAULT_DTYPE,
                    "tooltip": reformat.TOOLTIP_DTYPE,
                }),
                # ----- end reformat block
                "show_preview": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show a thumbnail of the first written frame on the node.",
                }),
                # Round-trip / read-only toggle. Two-mode contract:
                #
                #   OFF (default) — node writes the upstream IMAGE batch to
                #   disk. IMAGE + MASK output sockets emit the in-memory
                #   encoder buffer (post-reformat, post-OCIO). Standard
                #   write-node behavior.
                #
                #   ON — node SKIPS the write entirely and re-reads from
                #   disk the files at the same paths it WOULD have written
                #   to. The IMAGE + MASK output sockets emit the
                #   disk-loaded pixels with the INVERSE OCIO transform
                #   applied (output_colorspace -> working_colorspace).
                #   Upstream nodes feeding `image`/`mask` are NOT
                #   evaluated (the inputs are declared `lazy: True` and
                #   `check_lazy_status` returns `[]` when this toggle is
                #   on) — so the color-correction / sampling chain that
                #   feeds this node is bypassed entirely. Effectively
                #   turns this Write node into a Read node sourcing from
                #   the same path the Write would resolve.
                #
                # Typical workflow: run with toggle OFF once to write the
                # files, then flip ON to switch to read-only mode for
                # downstream iteration without re-running the upstream
                # graph. With no on-disk files yet AND toggle ON the read
                # will fail and the node returns empty — flip OFF for the
                # first run.
                #
                # POSITION (append-only): widgets MUST be added at the
                # END of INPUT_TYPES so the positional `widgets_values`
                # array in saved workflows isn't shifted. ComfyUI
                # persists widget values by index; mid-list inserts
                # silently corrupt every existing workflow on disk. The
                # visual position on canvas lands at the bottom of the
                # node's required block — accept that as the cost of
                # saved-workflow safety. See media-io-sync-rule invariant
                # 32 (and the new "append-only widgets" rule) for the
                # full contract.
                "load_saved_from_disk": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "OFF (default): write upstream IMAGE batch to disk. "
                        "ON: skip the write entirely, re-read from disk the "
                        "files at the same paths the Write would resolve, "
                        "and apply the INVERSE OCIO transform to land back "
                        "in `working_colorspace`. Upstream nodes feeding the "
                        "image/mask inputs are NOT evaluated when ON — the "
                        "Write node behaves as a Read node. IMAGE output "
                        "is in `working_colorspace` in BOTH modes (the OCIO "
                        "transform is scoped to the disk write; downstream "
                        "nodes always see working-cs). Honors `raw_data` "
                        "(skips OCIO both ways)."
                    ),
                }),
            },
            "optional": {
                # `image` / `mask` are declared `lazy: True` so ComfyUI
                # gates upstream evaluation through `check_lazy_status` —
                # when `load_saved_from_disk=True`, the lazy hook returns
                # an empty list and the upstream nodes feeding these
                # inputs are skipped entirely (read-only mode). When
                # `load_saved_from_disk=False`, the hook requests the
                # wired inputs and ComfyUI evaluates the upstream chain
                # as normal.
                "image": ("IMAGE", {
                    "tooltip": "Image batch to write.",
                    "lazy": True,
                }),
                "mask": ("MASK", {
                    "tooltip": reformat.TOOLTIP_MASK_IN_WRITE + (
                        "\n\n"
                        "Per-format support: RGBA writes natively to EXR / PNG / TIFF. "
                        "JPG / HDR silently strip the alpha channel at the OIIO writer "
                        "(but the MASK output still carries the right data)."
                    ),
                    "lazy": True,
                }),
                # VIDEO input — invariant 28. Non-lazy so check_lazy_status
                # can see whether it's wired and skip image/mask.
                "video": ("VIDEO", {
                    "tooltip": (
                        "Optional VIDEO input. When wired, iterates the "
                        "source frame-by-frame via PyAV and writes each EXR "
                        "— peak RAM stays at ONE frame regardless of "
                        "source length. Per-frame OCIO + Reformat + dtype "
                        "apply (driven by this node's widgets). Upstream "
                        "image/mask are lazy-skipped. MASK is dropped — "
                        "write RGB only; wire `image` if you need mask "
                        "handling. `VideoFromComponents` upstream works but "
                        "with reduced RAM benefit (batch already in memory)."
                    ),
                }),
            },
            "hidden": {
                # `prompt` feeds comfyui/prompt (the API graph); `extra_pnginfo`
                # carries the editor `workflow` graph → comfyui/workflow, which
                # is what drag-drop loadback rebuilds the canvas from. Declared
                # hidden so ComfyUI supplies them at execute() time. (These were
                # dropped when the public pack was stripped of Auto mode —
                # without them the embed never runs and saved files carry no
                # workflow metadata.)
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    # Output socket order — kept symmetric across the four media-IO nodes
    # Output socket order — see media-io-sync-rule.md invariants 14a + 28.
    # `video` appended as a passthrough socket (mirrors AM Video Write).
    RETURN_TYPES = (
        "IMAGE", "MASK", "STRING", "STRING", "INT", "INT", "FLOAT", "INT",
        "VIDEO",
    )
    RETURN_NAMES = (
        "image", "mask", "resolved_path", "info",
        "width", "height", "frame_rate", "frame_count",
        "video",
    )
    OUTPUT_TOOLTIPS = (
        "Sliced IMAGE passthrough — the input image post-frame-slice / "
        "post-mask-fold / post-reformat, in `working_colorspace`. The OCIO "
        "transform is scoped to the disk write only; downstream nodes see "
        "the same colorspace as the upstream chain. RGB only.",
        reformat.TOOLTIP_MASK_OUT_WRITE,
        "Newline-joined absolute paths of every file written this execute.",
        "Human-readable summary: dimensions, bit depth, output colorspace, count.",
        "Frame width in pixels.",
        "Frame height in pixels.",
        "Frame rate widget passthrough — the value written into file metadata "
        "(-1 = no fps metadata was emitted).",
        "Number of files written.",
        "VIDEO passthrough — when the IMAGE branch fired, wraps the IMAGE "
        "batch in a `VideoFromComponents` (zero copy). When the VIDEO "
        "branch fired, emits the input VIDEO as-is. None on no-op. "
        "Lets graphs chain post-write VIDEO downstream without re-reading.",
    )
    FUNCTION = "execute"
    CATEGORY = "AM VFX Tools"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Cache invalidation hook.

        - When `load_saved_from_disk=False` (write mode), upstream
          changes drive re-execution as usual (return NaN sentinel so
          ComfyUI always re-runs the write — output is side-effect-
          producing, never cacheable).
        - When `load_saved_from_disk=True` (read-only mode), no upstream
          is fetched. We hash the resolved-path's mtime so the read
          re-runs when the on-disk files change. Best-effort: if the
          path can't be resolved at IS_CHANGED time we still return a
          fresh sentinel so ComfyUI tries the read and surfaces the
          error in execute().
        """
        if not kwargs.get("load_saved_from_disk"):
            return float("nan")
        # Read-only mode: try to find the on-disk file(s) to hash mtime.
        try:
            fp = kwargs.get("file_path") or ""
            if fp:
                fp = os.path.expandvars(os.path.expanduser(fp))
                parent = os.path.dirname(fp) or "."
                if os.path.isdir(parent):
                    mtimes = sorted(
                        os.path.getmtime(os.path.join(parent, n))
                        for n in os.listdir(parent)
                        if os.path.isfile(os.path.join(parent, n))
                    )
                    return str(mtimes)
        except Exception:
            pass
        return float("nan")

    def check_lazy_status(self, **kwargs):
        """ComfyUI lazy-input gate. Returns the list of lazy input
        names that need upstream evaluation BEFORE execute() runs.

        - `load_saved_from_disk=True` (read-only) → return `[]`. No
          upstream is needed; execute() will source pixels from disk.
          Color-correction / sampler chains feeding this node are
          skipped, matching the artist's mental model of "behave as a
          Read node".
        - `video` wired (non-None) → return `[]`. The VIDEO streaming
          branch handles the encode by iterating the source — IMAGE/MASK
          are not needed. `video` is non-lazy so this kwarg is populated
          by the time check_lazy_status fires.
        - `load_saved_from_disk=False` (write mode) → return whichever
          of `image`/`mask` is wired and not yet evaluated. ComfyUI
          fetches them, then calls execute() with the populated values.
        """
        if kwargs.get("load_saved_from_disk"):
            return []
        if kwargs.get("video") is not None:
            return []
        needed = []
        # Only request inputs that are still None — already-evaluated
        # ones don't need re-fetching. `image` MUST be wired for the
        # write to succeed; `mask` is optional.
        if kwargs.get("image") is None:
            needed.append("image")
        if kwargs.get("mask") is None and "mask" in kwargs:
            needed.append("mask")
        return needed

    def execute(
        self,
        file_path,
        ext,
        seed,
        use_batch,
        frame_mode,
        frame_rate,
        use_frame_numbers,
        first_frame,
        last_frame,
        start_frame,
        frame_padding,
        bit_depth,
        compression,
        working_colorspace,
        raw_data,
        output_colorspace,
        embed_workflow,
        # Reformat block — synced across all four IO nodes.
        reformat_mode: str = reformat.MODE_OFF,
        scale: float = 1.0,
        preset: str = reformat.PRESET_WH,
        target_width: int = 1920,
        target_height: int = 1080,
        resize_type: str = reformat.RESIZE_FIT,
        filter: str = reformat.FILTER_CUBIC,
        output_dtype: str = reformat.DEFAULT_DTYPE,
        show_preview: bool = True,
        # ``load_saved_from_disk`` is appended at the END of INPUT_TYPES
        # (saved-workflow safety — see media-io-sync-rule "append-only
        # widgets" rule) and matches that position here with a default
        # value so older cached node classes calling execute() without
        # the kwarg don't TypeError on a missing positional arg.
        load_saved_from_disk: bool = False,
        image=None,
        mask=None,
        # VIDEO input — non-lazy. See check_lazy_status() and the
        # INPUT_TYPES tooltip. When wired, `_execute_video_streaming`
        # handles the encode frame-by-frame and the IMAGE/MASK branch
        # below is bypassed entirely.
        video=None,
        prompt: Optional[Dict[str, Any]] = None,
        extra_pnginfo: Optional[Dict[str, Any]] = None,
    ):
        log.info(
            "[am_vfx_tools/write_image] execute() entered — "
            "ext=%s frame_mode=%s use_batch=%s load_saved_from_disk=%s "
            "image_wired=%s",
            ext, frame_mode, use_batch,
            load_saved_from_disk, image is not None,
        )
        padding = max(1, int(frame_padding) or _DEFAULT_FRAME_PADDING)

        # Read-only branch — when On, skip the write entirely and source
        # pixels from disk. The lazy-input gate (`check_lazy_status`)
        # has already prevented upstream evaluation, so `image` is None
        # by design here. Frame-range semantics are documented inside
        # `_execute_read_only`; the same path-resolution code that the
        # write branch uses is reused via `_resolve_path_template`.
        if load_saved_from_disk:
            return self._execute_read_only(
                file_path=file_path, ext=ext, use_batch=use_batch,
                frame_mode=frame_mode, frame_rate=frame_rate,
                use_frame_numbers=use_frame_numbers,
                first_frame=first_frame, last_frame=last_frame,
                start_frame=start_frame, padding=padding,
                working_colorspace=working_colorspace,
                raw_data=raw_data, output_colorspace=output_colorspace,
                output_dtype=output_dtype, show_preview=show_preview,
            )

        # VIDEO streaming branch — see docs/media-io-sync-rule.md invariant 28.
        # When a native VIDEO is wired, iterate the source frame-by-frame and
        # write each EXR. Peak RAM stays at one frame regardless of source
        # length. check_lazy_status already short-circuited the image/mask
        # upstream eval, so those are None here by design.
        if video is not None:
            return self._execute_video_streaming(
                video=video,
                file_path=file_path, ext=ext,
                seed=seed, use_batch=use_batch,
                frame_rate=frame_rate,
                use_frame_numbers=use_frame_numbers,
                start_frame=start_frame, padding=padding,
                bit_depth=bit_depth, compression=compression,
                working_colorspace=working_colorspace,
                raw_data=raw_data, output_colorspace=output_colorspace,
                embed_workflow=embed_workflow,
                reformat_mode=reformat_mode, scale=scale, preset=preset,
                target_width=target_width, target_height=target_height,
                resize_type=resize_type, filter=filter,
                output_dtype=output_dtype, show_preview=show_preview,
                prompt=prompt,
                extra_pnginfo=extra_pnginfo,
            )

        if image is None:
            log.warning(
                "[am_vfx_tools/write_image] neither `image` nor `video` input is "
                "wired — write skipped"
            )
            return self._noop(None)

        # Seed sentinel resolution. `seed == -1` means "unset" — scan
        # `prompt` for AM Seed nodes and look up the seed each one
        # published to the process-global seed_registry during its own
        # IS_CHANGED (the publish runs BEFORE any execute(), so the
        # value is guaranteed available regardless of execution order
        # between AM Seed and this Write node). Registry hit → use the
        # published value for metadata; miss → leave `seed` at -1 and
        # `_build_workflow_metadata` will omit the key. Any other value
        # (typed widget OR delivered by a wire from convert-widget-to-
        # input) wins outright over the registry — wiring is the
        # explicit-override path, also useful for multi-AMSeed
        # workflows where the artist wants a specific source.
        if int(seed) == -1:
            looked_up = seed_registry.find_seed_for_prompt(prompt)
            if looked_up is not None:
                seed = int(looked_up)

        # 1. Resolve path. When use_batch=True, the path's parent dir is
        #    scanned for existing `_bNNNN` slots and max+1 is injected
        #    before the extension. Mirrors stock ComfyUI's
        #    `folder_paths.get_save_image_path` counter — runtime
        #    discovery, NOT a workflow-time integer. See module docstring +
        #    _core/batch_suffix.py.
        # ``_batch_n`` carries the runtime-discovered _bNNNN integer for
        # the metadata embed (``comfyui/batch``); set None when the artist
        # opted out via ``use_batch=False`` so the embed helper skips it.
        _batch_n: Optional[int] = None
        if not file_path:
            log.warning("[am_vfx_tools/write_image] empty file_path")
            return self._noop(image)
        base_path = os.path.expandvars(os.path.expanduser(file_path))
        # If the base path lacks an extension, append the one chosen
        # via the `ext` widget. If it already has an extension, leave
        # it as-is (the artist's literal path wins).
        _stem, _ext = os.path.splitext(base_path)
        if not _ext and ext:
            base_path = f"{base_path}.{ext.lstrip('.')}"
        if use_batch:
            _batch_n, full_path_template = batch_suffix.resolve_for_manual_path(base_path)
        else:
            full_path_template = base_path

        # 2. Determine the input batch slice.
        if image.ndim == 3:
            image = image[None, ...]
        n_in = int(image.shape[0])

        slice_lo, slice_hi = self._slice_indices(
            frame_mode, int(first_frame), int(last_frame), n_in,
        )
        count = max(0, slice_hi - slice_lo + 1)
        if count <= 0:
            log.warning(
                "[am_vfx_tools/write_image] empty input slice (mode=%s, first=%s, last=%s, n_in=%d)",
                frame_mode, first_frame, last_frame, n_in,
            )
            return self._noop(image)

        # 3. Build the workflow-embed payload once, shared across frames.
        # All keys land under the ``comfyui/`` namespace so an artist
        # inspecting via exiftool / ffprobe / oiiotool sees a clean
        # grouped block (``comfyui/prompt``, ``comfyui/seed``, ...)
        # rather than top-level entries that clash with container-native
        # tags. Width/height are NOT emitted — already native to image
        # dimensions / video stream metadata, duplicating adds noise.
        workflow_meta = self._build_workflow_metadata(
            embed_workflow, prompt, extra_pnginfo,
            seed=seed,
            batch_no=_batch_n,
        )

        # 4. OCIO transform — built once, shared across frames.
        wcs = color.resolve_choice_to_cs(working_colorspace)
        ocs = color.resolve_choice_to_cs(output_colorspace)
        try:
            proc = color.ColorProcessor(wcs, ocs, raw_data=raw_data)
        except Exception as e:
            log.warning(
                "[am_vfx_tools/write_image] cannot build OCIO %s -> %s (%s); "
                "writing pixels untouched", wcs, ocs, e,
            )
            proc = None

        # 4. Per-frame write loop.
        written: list = []
        # Accumulators for the IMAGE + MASK output sockets. Every iteration
        # appends its post-OCIO + post-reformat pixels here; the loop end
        # stacks them into tensors matching what was actually written.
        # Sized to `count` so downstream nodes see exactly the encoded frames.
        reformatted_frames: list = []
        # Mask passthrough — extracted from the per-frame post-reformat
        # buffer right alongside the RGB strip. Nuke convention: mask =
        # alpha. Empty mask (ones, fully opaque) when the per-frame
        # buffer ended up as RGB only.
        mask_frames: list = []
        has_token = _has_frame_token(full_path_template)
        # When the rendered path has no frame token AND the toggle is off,
        # multi-frame batches all collapse onto the same path. That's the
        # artist's explicit choice (toggle override) but it's worth logging
        # so they don't lose work to a typo.
        if not use_frame_numbers and not has_token and count > 1:
            log.warning(
                "[am_vfx_tools/write_image] use_frame_numbers=False with %d-frame batch "
                "and no frame token in path — every frame writes to %r; only "
                "the last frame survives.",
                count, full_path_template,
            )
        out_first = int(start_frame)
        for offset in range(count):
            in_idx = slice_lo + offset      # 0-based input batch index
            out_frame_no = out_first + offset

            if has_token:
                # Path has a frame token — always substitute, regardless
                # of toggle. If the artist put `####` in their literal
                # path they want the substitution.
                target = sequence.expand_frame_pattern(
                    full_path_template, out_frame_no, padding,
                )
            elif use_frame_numbers:
                # Path has no frame token AND the artist asked for frame
                # numbers — append `.NNNNN` before the extension.
                target = _suffix_with_frame(full_path_template, out_frame_no, padding)
            else:
                # Toggle is off — write the path verbatim. For multi-frame
                # batches that means every frame collapses onto the same
                # path; the warning above flags this.
                target = full_path_template

            pixels = image[in_idx].cpu().numpy().astype(np.float32, copy=True)
            # Fold an optionally-wired MASK into the alpha channel BEFORE
            # OCIO (alpha is preserved by OCIO's RGBA path) and BEFORE
            # reformat (so the geometry applies uniformly to RGB+alpha).
            # Stock-ComfyUI convention: alpha = 1 - mask. Mask whose H,W
            # don't match the image is auto-resized (cubic).
            if mask is not None:
                try:
                    mask_frame = mask[in_idx].cpu().numpy().astype(np.float32, copy=False)
                    pixels = reformat.combine_image_mask(pixels, mask_frame)
                except Exception as e:
                    log.warning(
                        "[am_vfx_tools/write_image] mask combine failed for frame %d (%s); "
                        "writing without mask", out_frame_no, e,
                    )

            # Per-frame reformat (geometry only, in working_colorspace).
            # Reordered 2026-05-02: reformat now runs BEFORE OCIO so we
            # can stash a working_colorspace snapshot for the IMAGE output
            # socket between reformat and OCIO. Reformat (cv2.resize on
            # fp32) and OCIO (per-pixel transform) commute mathematically
            # — the on-disk pixels are equivalent within fp32 epsilon
            # either order. `output_dtype` is forced to fp32 here so
            # `image_backend.write_image` gets the fp32 buffer it expects
            # (its OIIO quantization step would just upcast a fp16 input
            # back, wasting work). The widget-level fp32 → fp16 cast
            # lands once on the IMAGE output tensor after the loop,
            # where it actually saves memory.
            if reformat_mode != reformat.MODE_OFF:
                try:
                    pixels = reformat.reformat_array(
                        pixels,
                        mode=reformat_mode,
                        scale=float(scale),
                        preset=preset,
                        target_w=int(target_width),
                        target_h=int(target_height),
                        resize_type=resize_type,
                        filter_name=filter,
                        output_dtype=reformat.DTYPE_FP32,
                    )
                except Exception as e:
                    log.warning(
                        "[am_vfx_tools/write_image] reformat failed for frame %d (%s); "
                        "writing original pixels", out_frame_no, e,
                    )

            # Snapshot for the IMAGE + MASK output sockets, BEFORE OCIO.
            # The IMAGE socket emits the input image (in
            # `working_colorspace`) post-frame-slice / post-mask-fold /
            # post-reformat — i.e. the artist's image as it flows through
            # the node, NOT the OCIO-transformed encoded pixels that
            # land on disk. The OCIO transform is scoped to the disk
            # write only; downstream nodes continue to see
            # `working_colorspace`. Matches the AM Read Image convention
            # (input_cs → working_cs on the IMAGE output) and the Nuke
            # Write-node passthrough semantic. Pre-2026-05-02 the IMAGE
            # socket emitted post-OCIO pixels (output_colorspace), which
            # was inconsistent with the read-only branch (which emits
            # working_colorspace via inverse OCIO) and surprised artists
            # who expected a passthrough.
            _output_socket_pixels = pixels.copy()

            if proc is not None and not proc.is_identity:
                try:
                    proc.apply_inplace(pixels)
                except Exception as e:
                    log.warning(
                        "[am_vfx_tools/write_image] OCIO apply failed for frame %d (%s); "
                        "writing untransformed pixels", out_frame_no, e,
                    )

            per_frame_meta = workflow_meta if workflow_meta else None
            try:
                image_backend.write_image(
                    target,
                    pixels,
                    bit_depth=bit_depth,
                    compression=_strip_compression_prefix(compression),
                    color_space_tag=(
                        ocs if ocs and ocs != color.PASSTHROUGH else None
                    ),
                    metadata={"Software": "comfyui"},
                    workflow_metadata=per_frame_meta,
                    # Frame-rate sentinel `-1` = "don't write any fps
                    # metadata" — passes through as None so the backend
                    # skips the keys entirely. Any other value writes
                    # both `framesPerSecond` (Rational, EXR canonical)
                    # AND `input/frame_rate` (Nuke convention) into
                    # the file header.
                    frame_rate=(float(frame_rate) if float(frame_rate) > 0 else None),
                )
            except Exception as e:
                log.warning("[am_vfx_tools/write_image] write failed for %s: %s", target, e)
                continue

            written.append(target)
            # Stash for the IMAGE + MASK output sockets — uses the
            # pre-OCIO snapshot taken above so the sockets emit pixels
            # in `working_colorspace` (NOT the post-OCIO output_cs that
            # landed on disk). Strip alpha for IMAGE (RGB-only); invert
            # alpha to stock-ComfyUI MASK convention (mask = 1 - alpha)
            # so downstream nodes get a MASK that interops cleanly with
            # stock SD-inpainting nodes.
            if _output_socket_pixels.shape[-1] >= 4:
                reformatted_frames.append(_output_socket_pixels[..., :3])
                mask_frames.append(1.0 - _output_socket_pixels[..., 3])
            else:
                reformatted_frames.append(_output_socket_pixels)
                # Empty mask = zeros (stock ComfyUI: nothing to inpaint).
                mask_frames.append(
                    np.zeros(
                        _output_socket_pixels.shape[:2],
                        dtype=_output_socket_pixels.dtype,
                    )
                )
            log.info("[am_vfx_tools/write_image] wrote %s", target)

        paths_str = "\n".join(written)

        # IMAGE + MASK outputs reflect the SLICED range (the frames the
        # artist configured to write via frame_mode/first_frame/last_frame),
        # at POST-REFORMAT dimensions and POST-CAST dtype — i.e. exactly
        # what landed on disk in shape, with the artist's chosen output
        # dtype for the in-memory tensor. Falls back to the original input
        # slice if reformat is off and no frame succeeded (so the IMAGE
        # output is never empty when the input wasn't).
        if reformatted_frames:
            stacked = np.stack(reformatted_frames, axis=0)
            if output_dtype == reformat.DTYPE_FP16:
                stacked = stacked.astype(np.float16, copy=False)
            sliced = torch.from_numpy(np.ascontiguousarray(stacked))
            mask_stacked = np.stack(mask_frames, axis=0)
            if output_dtype == reformat.DTYPE_FP16:
                mask_stacked = mask_stacked.astype(np.float16, copy=False)
            mask_out = torch.from_numpy(np.ascontiguousarray(mask_stacked))
        else:
            sliced = image[slice_lo:slice_hi + 1]
            # No frames written — emit a shape-consistent empty MASK
            # (zeros, stock ComfyUI: "nothing to inpaint").
            mask_out = torch.zeros(
                (int(sliced.shape[0]), int(sliced.shape[1]), int(sliced.shape[2])),
                dtype=torch.float32,
            )

        ui_payload = self._ui_payload(
            sliced, written, paths_str, show_preview, working_colorspace,
        )

        # Metadata outputs — derived from the OUTPUT tensor shape so they
        # match what the IMAGE socket emits. Source dimensions stay in the
        # info string for forensic context via the appended reformat fragment.
        src_h = int(image.shape[1])
        src_w = int(image.shape[2])
        out_h = int(sliced.shape[1])
        out_w = int(sliced.shape[2])
        info_str = (
            f"{out_w}x{out_h} {bit_depth} {ocs or '(no-cs)'}, "
            f"{len(written)}/{count} frames written"
        )
        rf_frag = reformat.info_fragment(
            mode=reformat_mode,
            scale=float(scale),
            preset=preset,
            target_w=int(target_width),
            target_h=int(target_height),
            resize_type=resize_type,
            filter_name=filter,
            output_dtype=output_dtype,
            src_w=src_w, src_h=src_h,
        )
        if rf_frag:
            info_str = f"{info_str} | {rf_frag}"

        # VIDEO passthrough output — wrap the written IMAGE batch in a
        # VideoFromComponents so downstream nodes can chain the result.
        video_passthrough = self._build_video_passthrough(sliced, float(frame_rate))

        return {
            "ui": ui_payload,
            "result": (
                # image, mask, resolved_path, info, width, height,
                # frame_rate, frame_count, video
                sliced, mask_out, paths_str, info_str,
                int(out_w), int(out_h), float(frame_rate), int(len(written)),
                video_passthrough,
            ),
        }

    @staticmethod
    def _build_video_passthrough(images, fps: float):
        """Wrap the IMAGE batch as a VIDEO passthrough output.

        Mirrors AM Read Image's `_make_video_socket` — zero-copy
        VideoFromComponents reference. Returns None when comfy_api is
        unavailable or fps is non-positive (sentinel "no fps metadata").
        """
        if not _VIDEO_TYPE_AVAILABLE or images is None:
            return None
        try:
            from fractions import Fraction as _Fraction
            rate = _Fraction(fps if fps > 0 else 1).limit_denominator(1_000_000)
            from comfy_api.v0_0_2 import Types as _ComfyTypes  # local import to keep top tidy
            return _ComfyInputImpl.VideoFromComponents(
                _ComfyTypes.VideoComponents(images=images, frame_rate=rate),
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[am_vfx_tools/write_image] VIDEO passthrough build failed (%s); None", e,
            )
            return None

    @staticmethod
    def _build_workflow_metadata(
        embed_workflow: bool,
        prompt: Optional[Dict[str, Any]],
        extra_pnginfo: Optional[Dict[str, Any]] = None,
        *,
        seed: int = -1,
        batch_no: Optional[int] = None,
    ) -> Optional[Dict[str, str]]:
        """Render the prompt plus per-knob generation params into the
        string→string dict embedded in the saved file. Returns ``None``
        when embedding is disabled (per-node toggle OR ComfyUI's global
        ``--disable-metadata`` flag).

        All keys land under the ``comfyui/`` namespace so inspection
        tools (exiftool, ffprobe, oiiotool) show a clean grouped block
        instead of top-level entries that compete with container-native
        tags. The namespace also distinguishes AM-Pipe metadata from
        any random tagger already present in the file.

        Per-knob emit rules:

        * ``seed != -1`` -> ``comfyui/seed``
        * ``batch_no is not None`` -> ``comfyui/batch`` (the
          runtime-discovered ``_bNNNN`` integer the caller resolved
          via ``_core.batch_suffix``).
        """
        if not embed_workflow:
            return None
        if _comfy_args is not None and getattr(_comfy_args, "disable_metadata", False):
            return None

        out: Dict[str, str] = {}

        if prompt is not None:
            try:
                out["comfyui/prompt"] = json.dumps(prompt)
            except Exception:
                pass
        # Editor graph(s) from EXTRA_PNGINFO — most importantly the
        # `workflow` entry → comfyui/workflow, which is what drag-drop
        # loadback reconstructs the canvas from.
        if extra_pnginfo:
            for key, value in extra_pnginfo.items():
                try:
                    out[f"comfyui/{key}"] = json.dumps(value)
                except Exception:
                    continue

        if int(seed) != -1:
            out["comfyui/seed"] = str(int(seed))
        if batch_no is not None:
            out["comfyui/batch"] = str(int(batch_no))

        return out or None

    # VIDEO streaming mode — invariant 28.
    #
    # Iterates the wired VIDEO source one frame at a time and writes each
    # frame to its target EXR. Peak RAM = one frame. Per-frame OCIO +
    # Reformat + dtype apply (same widgets as the IMAGE branch). MASK is
    # dropped (no MASK input in this branch). frame_mode/first_frame/
    # last_frame are NOT consulted — every source frame is written
    # starting at `start_frame`. Slice upstream if you need a subset.
    #
    # Two source subtypes:
    #   * VideoFromFile: open via PyAV, iterate decoded frames.
    #   * VideoFromComponents: iterate the already-decoded tensor slice
    #     by slice (no extra full-batch copy added on top).

    def _execute_video_streaming(
        self, *, video, file_path, ext,
        seed, use_batch, frame_rate, use_frame_numbers,
        start_frame, padding,
        bit_depth, compression,
        working_colorspace, raw_data, output_colorspace, embed_workflow,
        reformat_mode, scale, preset, target_width, target_height,
        resize_type, filter, output_dtype, show_preview,
        prompt,
        extra_pnginfo,
    ):
        if not _VIDEO_TYPE_AVAILABLE:
            log.warning(
                "[am_vfx_tools/write_image] VIDEO streaming requested but "
                "comfy_api.v0_0_2 is not importable — falling back to "
                "no-op. Pin ComfyUI >= 0.3.48."
            )
            return self._noop(None)
        if not _PYAV_AVAILABLE:
            log.warning(
                "[am_vfx_tools/write_image] VIDEO streaming: PyAV is not importable. "
                "Falling back to no-op."
            )
            return self._noop(None)

        # 1. Resolve path. Manual-only — mirrors the IMAGE-branch logic
        #    (expandvars/expanduser, ext-append, batch slot). This pack
        #    has no Auto / template-rendering mode.
        _batch_n: Optional[int] = None
        if not file_path:
            log.warning(
                "[am_vfx_tools/write_image] VIDEO streaming: empty file_path"
            )
            return self._noop(None)
        base_path = os.path.expandvars(os.path.expanduser(file_path))
        _stem, _ext = os.path.splitext(base_path)
        if not _ext and ext:
            base_path = f"{base_path}.{ext.lstrip('.')}"
        if use_batch:
            _batch_n, full_path_template = batch_suffix.resolve_for_manual_path(base_path)
        else:
            full_path_template = base_path

        # 2. Seed sentinel resolution (matches IMAGE branch).
        if int(seed) == -1:
            looked_up = seed_registry.find_seed_for_prompt(prompt)
            if looked_up is not None:
                seed = int(looked_up)

        # 3. Workflow-metadata builder (matches IMAGE branch).
        disable_meta = bool(_comfy_args and getattr(_comfy_args, "disable_metadata", False))
        if embed_workflow and not disable_meta:
            workflow_meta = self._build_workflow_metadata(
                embed_workflow=True,
                prompt=prompt,
                extra_pnginfo=extra_pnginfo,
                seed=int(seed),
                batch_no=_batch_n,
            )
        else:
            workflow_meta = None

        # 4. OCIO processor (working_cs → output_cs). Matches IMAGE branch.
        wcs_resolved = color.resolve_choice_to_cs(working_colorspace)
        ocs_resolved = color.resolve_choice_to_cs(output_colorspace)
        proc: Optional[color.ColorProcessor] = None
        if not raw_data:
            try:
                proc = color.ColorProcessor(wcs_resolved, ocs_resolved, raw_data=raw_data)
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "[am_vfx_tools/write_image] VIDEO streaming: OCIO build failed "
                    "(%s); writing untransformed", e,
                )

        # 5. Make parent dir of the first output (handles fresh-shot paths).
        first_target = sequence.expand_frame_pattern(
            full_path_template, int(start_frame), padding,
        )
        try:
            os.makedirs(os.path.dirname(first_target) or ".", exist_ok=True)
        except OSError as e:
            log.warning(
                "[am_vfx_tools/write_image] VIDEO streaming: mkdir failed for "
                "%s: %s", first_target, e,
            )
            return self._noop(None)

        compression_value = _strip_compression_prefix(compression)
        color_space_tag = (
            ocs_resolved if ocs_resolved and ocs_resolved != color.PASSTHROUGH else None
        )
        frame_rate_metadata = (
            float(frame_rate) if float(frame_rate) > 0 else None
        )

        written: list[str] = []
        out_width = out_height = 0
        is_lazy_transform = isinstance(video, video_lazy.LazyVideoTransform)
        is_from_file = (
            not is_lazy_transform
            and isinstance(video, _ComfyInputImpl.VideoFromFile)
        )

        # 6. Iterate the source — three branches:
        #    * LazyVideoTransform: call iter_frames() so any chained AM
        #      transforms (Reformat, future Grade/OCIO) apply per-frame
        #      without materialising the IMAGE batch. See invariant 28
        #      and `_core/video_lazy.py`.
        #    * VideoFromFile: PyAV-decode directly.
        #    * VideoFromComponents: already-decoded tensor; slice it.
        try:
            if is_lazy_transform:
                for i, (np_frame, _alpha) in enumerate(video.iter_frames()):
                    # The lazy chain already applied any cascaded
                    # transforms (Reformat, etc.). This node's own
                    # reformat / OCIO / dtype still apply in
                    # _write_streamed_frame (per the node's widgets) —
                    # they compose on top of the lazy chain's transforms.
                    target = sequence.expand_frame_pattern(
                        full_path_template, int(start_frame) + i, padding,
                    )
                    np_frame = np_frame.astype(np.float32, copy=False)
                    wrote_target = self._write_streamed_frame(
                        np_frame, target,
                        proc=proc,
                        reformat_mode=reformat_mode, scale=scale,
                        preset=preset,
                        target_width=target_width, target_height=target_height,
                        resize_type=resize_type, filter_name=filter,
                        bit_depth=bit_depth, compression=compression_value,
                        color_space_tag=color_space_tag,
                        workflow_meta=workflow_meta,
                        frame_rate=frame_rate_metadata,
                    )
                    if wrote_target:
                        written.append(wrote_target)
                        if not out_width:
                            out_height, out_width = np_frame.shape[0], np_frame.shape[1]
            elif is_from_file:
                src = video.get_stream_source()
                container = _av.open(src)
                try:
                    if not container.streams.video:
                        log.warning(
                            "[am_vfx_tools/write_image] VIDEO streaming: no video "
                            "stream in source"
                        )
                        return self._noop(None)
                    vstream = container.streams.video[0]
                    for i, av_frame in enumerate(container.decode(vstream)):
                        np_frame = (
                            av_frame.to_ndarray(format="rgb24").astype(np.float32)
                            / 255.0
                        )
                        target = sequence.expand_frame_pattern(
                            full_path_template, int(start_frame) + i, padding,
                        )
                        wrote_target = self._write_streamed_frame(
                            np_frame, target,
                            proc=proc,
                            reformat_mode=reformat_mode, scale=scale,
                            preset=preset,
                            target_width=target_width, target_height=target_height,
                            resize_type=resize_type, filter_name=filter,
                            bit_depth=bit_depth, compression=compression_value,
                            color_space_tag=color_space_tag,
                            workflow_meta=workflow_meta,
                            frame_rate=frame_rate_metadata,
                        )
                        if wrote_target:
                            written.append(wrote_target)
                            if not out_width:
                                out_height, out_width = np_frame.shape[0], np_frame.shape[1]
                finally:
                    container.close()
            else:
                # VideoFromComponents — already-decoded tensor in RAM.
                components = video.get_components()
                images_tensor = components.images
                n_frames = int(images_tensor.shape[0])
                for i in range(n_frames):
                    np_frame = images_tensor[i].detach().cpu().numpy()
                    if np_frame.shape[-1] >= 4:
                        np_frame = np.ascontiguousarray(np_frame[..., :3])
                    np_frame = np_frame.astype(np.float32, copy=False)
                    target = sequence.expand_frame_pattern(
                        full_path_template, int(start_frame) + i, padding,
                    )
                    wrote_target = self._write_streamed_frame(
                        np_frame, target,
                        proc=proc,
                        reformat_mode=reformat_mode, scale=scale,
                        preset=preset,
                        target_width=target_width, target_height=target_height,
                        resize_type=resize_type, filter_name=filter,
                        bit_depth=bit_depth, compression=compression_value,
                        color_space_tag=color_space_tag,
                        workflow_meta=workflow_meta,
                        frame_rate=frame_rate_metadata,
                    )
                    if wrote_target:
                        written.append(wrote_target)
                        if not out_width:
                            out_height, out_width = np_frame.shape[0], np_frame.shape[1]
        except Exception as e:  # noqa: BLE001 — log + soft fail
            log.exception(
                "[am_vfx_tools/write_image] VIDEO streaming: iteration failed (%s); "
                "wrote %d frames before error", e, len(written),
            )
            if not written:
                return self._noop(None)

        # 7. Build info string + placeholder IMAGE/MASK outputs (the VIDEO
        #    branch consciously avoids materialising the IMAGE batch).
        info_str = (
            f"{int(out_width)}x{int(out_height)} {bit_depth} {ocs_resolved} "
            f"via VIDEO streaming, {len(written)} frames written"
        )
        rf_frag = reformat.info_fragment(
            mode=reformat_mode,
            scale=float(scale),
            preset=preset,
            target_w=int(target_width),
            target_h=int(target_height),
            resize_type=resize_type,
            filter_name=filter,
            output_dtype=output_dtype,
            src_w=int(out_width), src_h=int(out_height),
        )
        if rf_frag:
            info_str = f"{info_str} | {rf_frag}"

        placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        paths_str = "\n".join(written) if written else ""

        if written:
            ui_payload = self._ui_payload(
                placeholder, written, paths_str, show_preview, working_colorspace,
            )
        else:
            ui_payload = {"text": ["(no frames written)"]}

        out_fps = float(frame_rate) if float(frame_rate) > 0 else 0.0

        return {
            "ui": ui_payload,
            "result": (
                # image, mask, resolved_path, info, width, height,
                # frame_rate, frame_count, video
                placeholder, empty_mask, paths_str, info_str,
                int(out_width), int(out_height),
                out_fps, len(written),
                video,  # passthrough — the input VIDEO
            ),
        }

    def _write_streamed_frame(
        self, pixels_np: np.ndarray, target: str, *,
        proc: Optional[Any],
        reformat_mode: str, scale: float, preset: str,
        target_width: int, target_height: int,
        resize_type: str, filter_name: str,
        bit_depth: str, compression: Optional[str],
        color_space_tag: Optional[str],
        workflow_meta: Optional[Dict[str, str]],
        frame_rate: Optional[float],
    ) -> Optional[str]:
        """Apply reformat + OCIO + write a single frame to *target*.

        Returns the resolved target path on success, ``None`` on failure
        (per-frame failures are logged + skipped — the VIDEO streaming
        loop continues with the remaining frames).
        """
        if reformat_mode != reformat.MODE_OFF:
            try:
                pixels_np = reformat.reformat_array(
                    pixels_np,
                    mode=reformat_mode,
                    scale=float(scale),
                    preset=preset,
                    target_w=int(target_width),
                    target_h=int(target_height),
                    resize_type=resize_type,
                    filter_name=filter_name,
                    output_dtype=reformat.DTYPE_FP32,
                )
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "[am_vfx_tools/write_image] VIDEO streaming: reformat failed "
                    "for %s (%s); writing original pixels", target, e,
                )

        if proc is not None and not proc.is_identity:
            try:
                proc.apply_inplace(pixels_np)
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "[am_vfx_tools/write_image] VIDEO streaming: OCIO apply failed "
                    "for %s (%s); writing untransformed pixels", target, e,
                )

        per_frame_meta = workflow_meta if workflow_meta else None

        try:
            image_backend.write_image(
                target,
                pixels_np,
                bit_depth=bit_depth,
                compression=compression,
                color_space_tag=color_space_tag,
                metadata={"Software": "comfyui"},
                workflow_metadata=per_frame_meta,
                frame_rate=frame_rate,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[am_vfx_tools/write_image] VIDEO streaming: write failed for %s: %s",
                target, e,
            )
            return None

        log.info("[am_vfx_tools/write_image] VIDEO streaming: wrote %s", target)
        return target

    @staticmethod
    def _slice_indices(
        frame_mode: str, first_frame: int, last_frame: int, n_in: int,
    ) -> tuple:
        """Return ``(lo, hi)`` 0-based inclusive input-batch indices.

        * ``single`` — one frame at index ``first_frame - 1`` (clamped).
        * ``range``  — ``first_frame..last_frame`` (1-based, inclusive,
          clamped to batch). ``last_frame == -1`` means "auto = batch end".
        * ``all``    — the whole batch.

        Returns ``(0, -1)`` when the resulting slice is empty.
        """
        if n_in <= 0:
            return (0, -1)
        if frame_mode == FRAME_MODE_SINGLE:
            idx = max(1, int(first_frame)) - 1
            idx = min(idx, n_in - 1)
            return (idx, idx)
        if frame_mode == FRAME_MODE_RANGE:
            lo = max(1, int(first_frame)) - 1
            if int(last_frame) <= 0:
                hi = n_in - 1
            else:
                hi = min(int(last_frame) - 1, n_in - 1)
            if hi < lo:
                return (0, -1)
            return (lo, hi)
        # FRAME_MODE_ALL
        return (0, n_in - 1)

    @staticmethod
    def _ui_payload(image, written, paths_str, show_preview, working_colorspace):
        if not show_preview or not written:
            return {"text": [paths_str or "(no write)"]}
        try:
            payload = preview.create_single_preview(
                image,
                frame_index=0,
                working_colorspace=working_colorspace,
                filename_hint=written[0],
            )
        except Exception as e:
            log.warning("[am_vfx_tools/write_image] preview generation failed: %s", e)
            return {"text": [paths_str]}
        if not payload.get("images"):
            return {"text": [paths_str]}
        return payload

    def _noop(self, image):
        # image is a tensor (already promoted to (N, H, W, C) by the
        # caller) — or None when both image and video inputs are unwired,
        # or when read-only mode failed to load any frames.
        # Result tuple shape:
        #   image, mask, resolved_path, info, width, height, frame_rate, frame_count
        if hasattr(image, "shape") and len(image.shape) >= 3:
            h = int(image.shape[1] if image.ndim == 4 else image.shape[0])
            w = int(image.shape[2] if image.ndim == 4 else image.shape[1])
            n = int(image.shape[0]) if image.ndim == 4 else 1
        else:
            # Provide a real placeholder IMAGE so downstream consumers (e.g.
            # stock SaveImage at nodes.py save_images, which slices
            # `images[0].shape[1]`) don't crash with TypeError on None.
            # 1x64x64x3 matches the read-side `_empty_result` convention.
            h, w, n = 64, 64, 1
            image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        # Empty MASK = zeros (stock ComfyUI: nothing to inpaint).
        mask_stub = torch.zeros(
            (n, max(h, 1), max(w, 1)), dtype=torch.float32,
        )
        return {
            "ui": {"text": ["(no write)"]},
            "result": (
                image, mask_stub, "", "(no write)", w, h, 0.0, 0, None,
            ),
        }

    # ------------------------------------------------------------------
    # Read-only mode (load_saved_from_disk = True)
    # ------------------------------------------------------------------
    #
    # When the artist flips the toggle, the lazy-input gate prevents
    # ComfyUI from evaluating the `image`/`mask` upstream chain. execute()
    # then routes to `_execute_read_only` (below) which:
    #
    #   1. Resolves the same path the write branch would render. When
    #      `use_batch=True`, picks the LATEST `_bNNNN` slot on disk
    #      (max), not the next-after-max that the write branch would
    #      pick — read-side wants "what's already there", write-side
    #      wants "fresh slot".
    #
    #   2. Determines which output frame numbers to load:
    #        * single → just `start_frame` (the first output frame).
    #        * range with explicit last_frame → start_frame + 0 ..
    #          start_frame + (last_frame - first_frame).
    #        * range with last_frame=-1 OR all → scandir the parent
    #          directory and load every file matching the rendered
    #          stem pattern (uses _core.sequence.detect_sequence_range).
    #
    #   3. Reads each file via image_backend.read_image, applies the
    #      INVERSE OCIO transform (ocs -> wcs) so pixels land back in
    #      working space. raw_data=True short-circuits to identity.
    #
    #   4. Splits 4-ch loaded buffers into IMAGE (RGB) + MASK (1-alpha)
    #      per the family's stock-ComfyUI mask convention.
    #
    # Failure handling: per-file read errors are logged and the slot is
    # skipped; if all reads fail OR the path can't be resolved, the
    # node returns _noop with an empty IMAGE/MASK.

    def _execute_read_only(
        self, *, file_path, ext, use_batch, frame_mode, frame_rate,
        use_frame_numbers, first_frame, last_frame, start_frame, padding,
        working_colorspace, raw_data, output_colorspace,
        output_dtype, show_preview,
    ):
        # 1. Resolve path — read-side picks max (latest existing
        #    slot) for use_batch instead of max+1.
        resolved = self._resolve_path_template(
            file_path=file_path, ext=ext, use_batch=use_batch, for_read=True,
        )
        if resolved is None:
            log.warning("[am_vfx_tools/write_image] read-only mode: path resolution failed")
            return self._noop(None)
        full_path_template = resolved
        log.info("[am_vfx_tools/write_image] read-only mode: resolved template %r", full_path_template)

        # 2. Determine which output frame numbers to load.
        out_frames = self._enumerate_read_frames(
            full_path_template,
            frame_mode=frame_mode, first_frame=int(first_frame),
            last_frame=int(last_frame), start_frame=int(start_frame),
            padding=padding,
        )
        if not out_frames:
            log.warning(
                "[am_vfx_tools/write_image] read-only mode: no on-disk frames found "
                "for template %r (frame_mode=%s)",
                full_path_template, frame_mode,
            )
            return self._noop(None)

        # 3. Build inverse OCIO once.
        wcs = color.resolve_choice_to_cs(working_colorspace)
        ocs = color.resolve_choice_to_cs(output_colorspace)
        try:
            inv_proc = color.ColorProcessor(ocs, wcs, raw_data=raw_data)
        except Exception as e:
            log.warning(
                "[am_vfx_tools/write_image] read-only mode: cannot build inverse OCIO "
                "%s -> %s (%s); loading pixels untransformed", ocs, wcs, e,
            )
            inv_proc = None

        # 4. Per-frame read.
        rgb_frames: list = []
        mask_frames: list = []
        loaded_paths: list = []
        for fno in out_frames:
            target = sequence.expand_frame_pattern(
                full_path_template, fno, padding,
            ) if _has_frame_token(full_path_template) else full_path_template
            if not os.path.exists(target):
                log.warning("[am_vfx_tools/write_image] read-only mode: missing %s", target)
                continue
            try:
                rr = image_backend.read_image(target)
                buf = np.asarray(rr.pixels, dtype=np.float32)
            except Exception as e:
                log.warning("[am_vfx_tools/write_image] read-only mode: read failed for %s: %s", target, e)
                continue
            if inv_proc is not None and not inv_proc.is_identity:
                try:
                    inv_proc.apply_inplace(buf)
                except Exception as e:
                    log.warning(
                        "[am_vfx_tools/write_image] read-only mode: inverse OCIO failed "
                        "for %s (%s); using untransformed pixels", target, e,
                    )
            if buf.ndim == 2:
                buf = buf[..., None]
            if buf.shape[-1] >= 4:
                rgb_frames.append(buf[..., :3])
                mask_frames.append(1.0 - buf[..., 3])
            else:
                rgb = (
                    buf[..., :3] if buf.shape[-1] == 3
                    else np.repeat(buf, 3, axis=-1)
                )
                rgb_frames.append(rgb)
                mask_frames.append(np.zeros(buf.shape[:2], dtype=np.float32))
            loaded_paths.append(target)
            log.info("[am_vfx_tools/write_image] read-only mode: loaded %s", target)

        if not rgb_frames:
            return self._noop(None)

        # 5. Stack into output tensors with the artist's chosen dtype.
        stacked = np.stack(rgb_frames, axis=0)
        mask_stacked = np.stack(mask_frames, axis=0)
        if output_dtype == reformat.DTYPE_FP16:
            stacked = stacked.astype(np.float16, copy=False)
            mask_stacked = mask_stacked.astype(np.float16, copy=False)
        out_image = torch.from_numpy(np.ascontiguousarray(stacked))
        out_mask = torch.from_numpy(np.ascontiguousarray(mask_stacked))

        out_h = int(out_image.shape[1])
        out_w = int(out_image.shape[2])
        paths_str = "\n".join(loaded_paths)
        info_str = (
            f"{out_w}x{out_h} read-only {ocs or '(no-cs)'}, "
            f"{len(loaded_paths)} frames loaded from disk"
        )
        ui_payload = self._ui_payload(
            out_image, loaded_paths, paths_str, show_preview, working_colorspace,
        )
        # VIDEO passthrough — wrap the loaded IMAGE batch (read-only mode
        # decoded files from disk; this is what was "written" earlier and
        # is now being re-emitted).
        out_fps = float(frame_rate) if float(frame_rate) > 0 else 0.0
        video_passthrough = self._build_video_passthrough(out_image, out_fps)
        return {
            "ui": ui_payload,
            "result": (
                out_image, out_mask, paths_str, info_str,
                int(out_w), int(out_h),
                out_fps,
                int(len(loaded_paths)),
                video_passthrough,
            ),
        }

    def _resolve_path_template(
        self, *, file_path, ext, use_batch, for_read: bool = False,
    ) -> Optional[str]:
        """Render the Write node's full path (the same string the write
        loop uses, may contain a frame token).

        When ``for_read=True`` and ``use_batch=True``, returns the
        LATEST existing _bNNNN slot on disk (read-side semantic) rather
        than max+1 (write-side semantic). Returns ``None`` when
        ``file_path`` is empty.
        """
        if not file_path:
            return None
        base_path = os.path.expandvars(os.path.expanduser(file_path))
        # Append the chosen extension if the artist didn't include one.
        _stem, _ext = os.path.splitext(base_path)
        if not _ext and ext:
            base_path = f"{base_path}.{ext.lstrip('.')}"
        if use_batch:
            if for_read:
                # Read side: pick max, not max+1.
                return batch_suffix.resolve_latest_existing(base_path) or base_path
            _batch_n, full_path = batch_suffix.resolve_for_manual_path(base_path)
            return full_path
        return base_path

    @staticmethod
    def _enumerate_read_frames(
        full_path_template: str, *,
        frame_mode: str, first_frame: int, last_frame: int,
        start_frame: int, padding: int,
    ) -> list:
        """Decide which OUTPUT frame numbers the read-only branch loads.

        Mirrors the write branch's frame-range knobs adapted for the
        no-input case. See `_execute_read_only` docstring for the
        per-mode semantic.
        """
        # If the template has no frame token, only one file is on disk.
        if not _has_frame_token(full_path_template):
            return [start_frame]

        if frame_mode == FRAME_MODE_SINGLE:
            return [start_frame]

        if frame_mode == FRAME_MODE_RANGE and int(last_frame) > 0:
            count = max(1, int(last_frame) - int(first_frame) + 1)
            return [start_frame + i for i in range(count)]

        # range with last_frame=-1 OR all → scandir.
        try:
            info = sequence.detect_sequence_range(
                sequence.expand_frame_pattern(full_path_template, start_frame, padding),
                scan_dir=True,
            )
        except Exception as e:
            log.warning(
                "[am_vfx_tools/write_image] read-only mode: scandir failed for %s: %s",
                full_path_template, e,
            )
            return []
        if not info.present_set:
            return []
        return sorted(info.present_set)
