"""AM Read Image — ComfyUI node.

Reads an image or image sequence from disk via OpenImageIO, with OCIO 2.x
color management and an optional reformat / dtype-cast pass. Use the
``Browse`` button (or paste a path into ``file_path``) to point at any
file ImageOIIO can read; sequences are detected by ``####`` / ``%05d`` /
``$F4`` style frame tokens.

Three frame modes:

* ``single`` — read the one frame named by ``first_frame`` (no disk
  scan, fastest path).
* ``range`` — read every frame between ``first_frame`` and ``last_frame``
  inclusive, applying ``before`` / ``after`` extrapolation outside the
  available on-disk range and the ``missing_frames`` policy for gaps
  inside the range.
* ``all`` — single ``os.scandir`` of the parent directory, then load
  every present frame.

Image OIIO + OCIO core is shared with AM Write Image — see
:mod:`._core`.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ._core import color, image_backend, preview, reformat, sequence

log = logging.getLogger("am_vfx_tools.read_image")


FRAME_MODE_SINGLE = "single"
FRAME_MODE_RANGE  = "range"
FRAME_MODE_ALL    = "all"
_FRAME_MODES = [FRAME_MODE_SINGLE, FRAME_MODE_RANGE, FRAME_MODE_ALL]

MISSING_ERROR        = "error"
MISSING_BLACK        = "black"
MISSING_HOLD         = "hold"
MISSING_NEAREST      = "nearest"
MISSING_CHECKERBOARD = "checkerboard"
_MISSING_OPTIONS = [
    MISSING_ERROR, MISSING_BLACK, MISSING_HOLD, MISSING_NEAREST, MISSING_CHECKERBOARD,
]

EDGE_HOLD   = "hold"
EDGE_LOOP   = "loop"
EDGE_BOUNCE = "bounce"
EDGE_BLACK  = "black"
_EDGE_OPTIONS = [EDGE_HOLD, EDGE_LOOP, EDGE_BOUNCE, EDGE_BLACK]

# Default padding for the {frame} token when the user-supplied path
# uses ``####`` style placeholders without an explicit width.
_DEFAULT_FRAME_PADDING = 5
_DEFAULT_FRAME_TOKEN = "#" * _DEFAULT_FRAME_PADDING


class _UnresolvedPath(Exception):
    """Path could not be resolved (e.g. ``file_path`` left empty)."""


class AMImageRead:
    """ComfyUI node — read image / sequence with OCIO color management."""

    @classmethod
    def INPUT_TYPES(cls):
        cs = color.color_space_choices()
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Absolute path to the file or sequence",
                    "tooltip": (
                        "Absolute path to the file or image sequence. "
                        "For sequences, use a frame token: ``####`` / ``%05d`` / "
                        "``$F4`` style. Use the 📂 Browse button to populate "
                        "from the native file dialog."
                    ),
                }),
                # Default `all` (matches Read Video) — most artists
                # want every present frame; range/single are explicit
                # overrides.
                "frame_mode": (_FRAME_MODES, {
                    "default": FRAME_MODE_ALL,
                    "tooltip": (
                        "Which frames to load. "
                        "single = only `first_frame`. "
                        "range = `first_frame`..`last_frame` inclusive (with `before`/`after` policy at edges). "
                        "all = every present frame in the resolved directory."
                    ),
                }),
                # Frame rate sits directly under `frame_mode` and above
                # `first_frame` — locked across the family (see
                # media-io-sync-rule.md invariant 14b). Drives the
                # `frame_rate` FLOAT output socket. Sentinel `-1` =
                # auto: probe the first frame's OIIO header for
                # `framesPerSecond` / `input/framesPerSecond` /
                # `frameRate` / `framerate` / `fps`; fall back to 25.0
                # if nothing's tagged. Any other value is an explicit
                # override. Wire to AM Image Write's `frame_rate` for
                # full sequence-roundtrip metadata preservation.
                "frame_rate": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 480.0,
                    "tooltip": (
                        "Frame rate for the `frame_rate` output socket. "
                        "-1 = auto: probe EXR/OIIO metadata "
                        "(framesPerSecond, input/framesPerSecond, frameRate, framerate, fps); "
                        "fallback 25 fps. Any other value = explicit override."
                    ),
                }),
                "first_frame": ("INT", {
                    "default": 1, "min": -999999, "max": 999999,
                    "tooltip": (
                        "Frame index in single mode; lower bound in range mode. "
                        "Ignored in all mode. The 🔍 Detect Range button auto-fills this."
                    ),
                }),
                "last_frame":  ("INT", {
                    "default": 1, "min": -999999, "max": 999999,
                    "tooltip": (
                        "Range upper bound (inclusive). Used in range mode only. "
                        "The 🔍 Detect Range button auto-fills this from the on-disk scan."
                    ),
                }),
                "missing_frames": (_MISSING_OPTIONS, {
                    "default": MISSING_BLACK,
                    "tooltip": (
                        "Policy when a frame inside the requested set is missing on disk. "
                        "error = abort the load. "
                        "black / checkerboard = synthesize a placeholder. "
                        "hold = repeat the last successful frame. "
                        "nearest = use the nearest existing frame number."
                    ),
                }),
                "before": (_EDGE_OPTIONS, {
                    "default": EDGE_HOLD,
                    "tooltip": (
                        "Edge policy below the on-disk range when `frame_mode=range`. "
                        "hold = clamp to first frame; loop = wrap; bounce = ping-pong; "
                        "black = synthesize black."
                    ),
                }),
                "after": (_EDGE_OPTIONS, {
                    "default": EDGE_HOLD,
                    "tooltip": (
                        "Edge policy above the on-disk range when `frame_mode=range`. "
                        "Mirrors `before` — same options, applied at the upper edge."
                    ),
                }),
                "input_colorspace": (cs, {
                    "default": color.pick_default(
                        cs, ("ACES2065-1", "ACEScg", "sRGB - Display"),
                    ),
                    "tooltip": (
                        "Source colorspace of the file. The OCIO transform converts "
                        "from this to `working_colorspace`. Pick `raw` to honor the "
                        "file's own tagged colorspace; pick a specific value to override."
                    ),
                }),
                "raw_data": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When On, skip the OCIO transform — pixels pass through unchanged. "
                        "`input_colorspace` and `working_colorspace` are ignored."
                    ),
                }),
                "working_colorspace": (cs, {
                    "default": color.default_working_colorspace(cs),
                    "tooltip": (
                        "Target colorspace for the IMAGE output — the space downstream "
                        "nodes (Grade, samplers, ...) will see."
                    ),
                }),
                # ----- Reformat block (synced across all four IO nodes; see
                # media-io-sync-rule.md invariants 15a–15e). Eight widgets,
                # always visible regardless of mode (ComfyUI doesn't support
                # conditional widget hiding). Tooltips document precedence.
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
                    "tooltip": "Show a thumbnail of the loaded frame on the node.",
                }),
            },
            "optional": {},
        }

    # Output socket order — kept symmetric across the four media-IO nodes
    # (see media-io-sync-rule.md invariant 14a). Reads share an
    # `image[, audio]` prefix (Read Video splices `audio` after `image`)
    # then the same `resolved_path, info, width, height, frame_rate,
    # frame_count` tail.
    RETURN_TYPES = (
        "IMAGE", "MASK", "STRING", "STRING", "INT", "INT", "FLOAT", "INT",
    )
    RETURN_NAMES = (
        "image", "mask", "resolved_path", "info",
        "width", "height", "frame_rate", "frame_count",
    )
    OUTPUT_TOOLTIPS = (
        "Frame batch as IMAGE (N×H×W×3 float in [0,1]). RGB only — alpha is "
        "split out to the `mask` socket per stock-ComfyUI convention.",
        reformat.TOOLTIP_MASK_OUT_READ,
        "Resolved on-disk path of the last successfully read frame.",
        "Human-readable summary: dimensions, bit depth, source colorspace, frame count.",
        "Frame width in pixels.",
        "Frame height in pixels.",
        "Effective fps. From the `frame_rate` widget when set, else probed from "
        "EXR/OIIO metadata, else 25 fallback.",
        "Number of frames in the IMAGE batch.",
    )
    FUNCTION = "execute"
    CATEGORY = "AM VFX Tools"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        # Per-frame load is cheap; force re-execution on every queue.
        return float("nan")

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #

    def execute(
        self,
        file_path: str,
        frame_mode: str,
        frame_rate: float,
        first_frame: int,
        last_frame: int,
        missing_frames: str,
        before: str,
        after: str,
        input_colorspace: str,
        raw_data: bool,
        working_colorspace: str,
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
    ):
        # 1. Build the printf pattern + the list of frames the user requested.
        try:
            printf_pattern, frames_to_load, range_first, range_last, padding = (
                self._build_load_plan(
                    file_path=file_path,
                    frame_mode=frame_mode,
                    first_frame=first_frame,
                    last_frame=last_frame,
                )
            )
        except _UnresolvedPath as e:
            return self._empty_result(str(e))

        if not frames_to_load:
            return self._empty_result("(no frames requested)")

        # 2. Resolve dropdowns; build OCIO processor lazily.
        src_choice = color.resolve_choice_to_cs(input_colorspace)
        dst_choice = color.resolve_choice_to_cs(working_colorspace)
        proc = self._build_processor(src_choice, dst_choice, raw_data)
        # Honor file's tagged colorspace when src == "raw" — we'll lazily
        # build a per-tag processor on first successful read.
        honor_tag = (src_choice == color.PASSTHROUGH and not raw_data
                     and dst_choice != color.PASSTHROUGH)
        tag_proc: Optional[color.ColorProcessor] = None
        tag_seen: Optional[str] = None

        # 3. Loop frames.
        pixels_list: List[np.ndarray] = []
        # Pre-flight: cheap OIIO-spec probe so leading missing frames produce
        # correctly-shaped black/checkerboard placeholders (otherwise the
        # default (512, 512, 3) clashes with later real reads in np.stack).
        reference_shape: Optional[Tuple[int, int, int]] = (
            None if frame_mode == FRAME_MODE_SINGLE
            else self._preflight_shape(printf_pattern, frames_to_load, padding)
        )
        last_pixels: Optional[np.ndarray] = None
        last_resolved_path = ""
        # Captured from the first successful read; drives the info string
        # + width/height outputs.
        last_meta: Optional[Dict[str, Any]] = None
        # Lazy directory scan for `nearest` policy. None = not yet scanned.
        present_set: Optional[frozenset] = None

        for req_frame in frames_to_load:
            # Apply before/after extrapolation (range mode only).
            if frame_mode == FRAME_MODE_RANGE and range_first is not None and range_last is not None:
                actual_frame, force_black = self._extrapolate(
                    req_frame, range_first, range_last, before, after,
                )
            else:
                actual_frame, force_black = req_frame, False

            # Resolve target on-disk path.
            if frame_mode == FRAME_MODE_SINGLE:
                target_path = printf_pattern  # already concrete
            elif force_black:
                target_path = None
            else:
                target_path = sequence.expand_frame_pattern(
                    printf_pattern, actual_frame, padding,
                )

            # Surface the path the node WAS asked for, even when nothing
            # exists on disk yet — useful for debugging missing-frame issues.
            if target_path:
                last_resolved_path = target_path

            pixels: Optional[np.ndarray] = None

            if force_black:
                pixels = self._make_black(reference_shape)
            elif target_path and os.path.exists(target_path):
                try:
                    res = image_backend.read_image(target_path)
                    pixels = np.asarray(res.pixels, dtype=np.float32, copy=False)

                    # Capture file metadata from the FIRST successful read; the
                    # rest of the sequence is assumed to share the same spec.
                    # `oiio_attribs` carries the full extra-attribs dict (used
                    # downstream for the auto-fps probe).
                    if last_meta is None:
                        last_meta = {
                            "width":        int(res.width),
                            "height":       int(res.height),
                            "bit_depth":    str(res.bit_depth),
                            "src_cs":       res.color_space,
                            "oiio_attribs": dict(res.metadata or {}),
                        }

                    # Apply OCIO.
                    if proc is not None and not raw_data:
                        try:
                            proc.apply_inplace(pixels)
                        except Exception as e:
                            log.warning("[am_vfx_tools/read_image] OCIO apply failed for %s: %s",
                                        target_path, e)
                    elif honor_tag:
                        file_tag = res.color_space
                        if file_tag and file_tag != color.PASSTHROUGH:
                            if file_tag != tag_seen:
                                tag_proc = self._build_processor(
                                    file_tag, dst_choice, raw_data,
                                )
                                tag_seen = file_tag
                            if tag_proc is not None:
                                try:
                                    tag_proc.apply_inplace(pixels)
                                except Exception as e:
                                    log.warning(
                                        "[am_vfx_tools/read_image] OCIO file-tag %s -> %s "
                                        "failed (%s)", file_tag, dst_choice, e,
                                    )
                except Exception as e:
                    log.warning("[am_vfx_tools/read_image] read failed for %s: %s", target_path, e)
                    pixels = None

            if pixels is None and not force_black:
                # Frame missing on disk OR read raised — apply policy.
                if missing_frames == MISSING_ERROR:
                    log.warning("[am_vfx_tools/read_image] missing frame %d (%s)",
                                actual_frame, target_path or "?")
                    return self._empty_result(
                        f"missing frame {actual_frame}: {target_path or '?'}"
                    )
                if missing_frames == MISSING_NEAREST:
                    if present_set is None and frame_mode != FRAME_MODE_SINGLE:
                        seq_info = sequence.detect_sequence_range(
                            printf_pattern, scan_dir=True,
                        )
                        present_set = seq_info.present_set
                    pixels = self._missing_nearest(
                        actual_frame, present_set, printf_pattern, padding,
                        reference_shape, proc, raw_data, honor_tag,
                        dst_choice,
                    )
                elif missing_frames == MISSING_HOLD:
                    pixels = (last_pixels.copy() if last_pixels is not None
                              else self._make_black(reference_shape))
                elif missing_frames == MISSING_CHECKERBOARD:
                    pixels = self._make_checkerboard(reference_shape)
                else:  # MISSING_BLACK or anything else
                    pixels = self._make_black(reference_shape)

            pixels = self._normalize_channels(pixels)
            if reference_shape is None:
                # Capture pre-reformat shape so subsequent missing-frame
                # placeholders match the source dimensions; reformat is
                # then applied to the placeholder as well, yielding a
                # uniform post-reformat batch.
                reference_shape = pixels.shape
            last_pixels = pixels
            # Per-frame reformat (in-loop = streaming): peak RAM during read
            # is one full-res frame + accumulating already-shrunk list.
            # OCIO has already run on `pixels` at fp32, which the helper
            # requires; the optional fp32 → fp16 down-cast is the helper's
            # last step.
            if reformat_mode != reformat.MODE_OFF or output_dtype != reformat.DTYPE_FP32:
                pixels = reformat.reformat_array(
                    pixels,
                    mode=reformat_mode,
                    scale=float(scale),
                    preset=preset,
                    target_w=int(target_width),
                    target_h=int(target_height),
                    resize_type=resize_type,
                    filter_name=filter,
                    output_dtype=output_dtype,
                )
            pixels_list.append(pixels)

        # 4. Stack + tensor. When fp16 was requested, the per-frame
        # reformat step already cast each frame to float16 — preserve
        # that dtype here instead of upcasting back to float32.
        batch = np.stack(pixels_list, axis=0)
        if output_dtype == reformat.DTYPE_FP32 and batch.dtype != np.float32:
            batch = batch.astype(np.float32, copy=False)
        # Split into IMAGE (RGB) + MASK (alpha-derived). Stock-ComfyUI
        # convention: mask = 1 - alpha. RGB-only sources emit a zero MASK
        # ("nothing to inpaint" — the source is fully opaque).
        image_arr, mask_arr = reformat.split_image_mask(batch)
        tensor = torch.from_numpy(np.ascontiguousarray(image_arr))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_arr))

        # 5. Preview index — frame-relative offset for range/all modes.
        # In single mode the batch has length 1, so always show idx 0.
        # In range/all the user's "current" frame is first_frame.
        preview_idx = self._preview_index(
            frame_mode, first_frame, range_first, len(pixels_list),
        )

        # 6. Metadata outputs — width/height/frame_count/info.
        # `out_width`/`out_height` reflect the POST-reformat tensor shape
        # (what downstream nodes actually see). The info string keeps the
        # original on-disk dimensions for forensic context via the
        # appended reformat fragment.
        src_width  = last_meta["width"]  if last_meta is not None else int(tensor.shape[2])
        src_height = last_meta["height"] if last_meta is not None else int(tensor.shape[1])
        if last_meta is not None:
            out_bdepth = last_meta["bit_depth"]
            out_srccs  = last_meta["src_cs"] or "(untagged)"
            oiio_attribs = last_meta.get("oiio_attribs") or {}
        else:
            out_bdepth = "(no-read)"
            out_srccs  = "(no-read)"
            oiio_attribs = {}
        out_height = int(tensor.shape[1])
        out_width  = int(tensor.shape[2])
        frame_count = int(tensor.shape[0])

        # 6b. Effective frame rate: -1 widget = auto-from-metadata with 25
        # fps fallback; any other value = explicit override.
        effective_fps = self._effective_fps(frame_rate, oiio_attribs)

        info_str = self._build_info(
            frame_mode=frame_mode,
            first_frame=int(first_frame),
            last_frame=int(last_frame) if frame_mode == FRAME_MODE_RANGE else None,
            range_first=range_first, range_last=range_last,
            frame_count=frame_count,
            width=src_width, height=src_height,
            bit_depth=out_bdepth, src_cs=out_srccs,
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
            src_w=src_width, src_h=src_height,
        )
        if rf_frag:
            info_str = f"{info_str} | {rf_frag}"

        return {
            "ui": self._ui_payload(
                tensor, last_resolved_path, show_preview,
                working_colorspace, preview_idx,
            ),
            "result": (
                tensor, mask_tensor, last_resolved_path, info_str,
                int(out_width), int(out_height),
                float(effective_fps), int(frame_count),
            ),
        }

    @staticmethod
    def _effective_fps(widget_value: float, oiio_attribs: Dict[str, Any]) -> float:
        """Resolve the artist's frame-rate widget against on-disk metadata.

        Sentinel ``-1`` means "auto: probe the OIIO header attributes
        captured from the first successful read; fall back to 25 fps".
        Any other widget value is taken verbatim.

        Public so the symmetry rule can point at one canonical resolver
        when (eventually) AM Read Video gains an artist override.
        """
        try:
            v = float(widget_value)
        except (TypeError, ValueError):
            v = -1.0
        if v != -1.0:
            return v
        detected = image_backend.extract_frame_rate(oiio_attribs)
        if detected is not None:
            return float(detected)
        log.debug(
            "[am_vfx_tools/read_image] no fps in OIIO header attribs (keys=%s); "
            "falling back to 25 fps",
            sorted(oiio_attribs.keys()) if oiio_attribs else [],
        )
        return 25.0

    # ------------------------------------------------------------------ #
    #  Plan: figure out the printf pattern + which frames to load.
    # ------------------------------------------------------------------ #

    def _build_load_plan(
        self, *, file_path, frame_mode, first_frame, last_frame,
    ) -> Tuple[str, List[int], Optional[int], Optional[int], int]:
        """Resolve the printf pattern + frame list. Returns
        ``(printf_pattern, frames_to_load, range_first, range_last, padding)``.

        For ``frame_mode=single`` the printf_pattern is already a concrete
        path (frame number substituted) and ``padding`` is informational.
        """
        if not file_path:
            raise _UnresolvedPath("(no file)")

        single = (frame_mode == FRAME_MODE_SINGLE)
        single_frame = int(first_frame)
        expanded = os.path.expandvars(os.path.expanduser(file_path))

        if single:
            # Substitute the frame in the literal user path; respect any
            # token (####, %0Nd) the artist already typed.
            target_for_parse = sequence.expand_frame_pattern(
                expanded, single_frame, _DEFAULT_FRAME_PADDING,
            )
            # Already concrete. Padding is irrelevant; report 0 to signal
            # "no scan needed."
            return target_for_parse, [single_frame], None, None, 0

        target_for_parse = expanded
        printf_pattern, frame_spec, padding = sequence.parse_frame_pattern(
            target_for_parse,
        )
        if frame_spec is None or padding == 0:
            # User asked for range/all but path has no frame token. Treat
            # as a single literal file.
            return target_for_parse, [single_frame], None, None, 0

        if frame_mode == FRAME_MODE_RANGE:
            lo, hi = sorted([int(first_frame), int(last_frame)])
            frames_to_load = list(range(lo, hi + 1))
            return printf_pattern, frames_to_load, lo, hi, padding

        # FRAME_MODE_ALL
        info = sequence.detect_sequence_range(printf_pattern, scan_dir=True)
        if info.first is None:
            log.warning("[am_vfx_tools/read] all-mode: no frames matched %s", printf_pattern)
            return printf_pattern, [], None, None, padding
        frames = sorted(info.present_set)
        return info.pattern, frames, info.first, info.last, info.padding

    # ------------------------------------------------------------------ #
    #  Per-frame helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_processor(src: str, dst: str, raw_data: bool):
        if raw_data:
            return None
        if src == color.PASSTHROUGH or dst == color.PASSTHROUGH or src == dst:
            return None
        try:
            cand = color.ColorProcessor(src, dst, raw_data=raw_data)
            if cand.is_identity:
                return None
            return cand
        except Exception as e:
            log.warning("[am_vfx_tools/read_image] cannot build OCIO %s -> %s: %s", src, dst, e)
            return None

    @staticmethod
    def _extrapolate(
        req_frame: int, first: int, last: int, before: str, after: str,
    ) -> Tuple[int, bool]:
        """Map *req_frame* to an actual frame to load.

        Returns ``(actual_frame, force_black)``. For policies other than
        ``black``, ``force_black`` is ``False`` and the caller resolves
        the frame normally (which may then trip ``missing_frames`` if the
        mapped frame isn't on disk).
        """
        if first <= req_frame <= last:
            return req_frame, False
        policy = before if req_frame < first else after
        if policy == EDGE_BLACK:
            return req_frame, True
        if policy == EDGE_HOLD:
            return (first if req_frame < first else last), False
        span = max(1, last - first + 1)
        if policy == EDGE_LOOP:
            return first + ((req_frame - first) % span), False
        if policy == EDGE_BOUNCE:
            cycle = max(1, span * 2 - 2) if span > 1 else 1
            offset = (req_frame - first) % cycle
            if offset < span:
                return first + offset, False
            return last - (offset - (span - 1)), False
        return req_frame, False

    @staticmethod
    def _missing_nearest(
        frame: int,
        present_set: Optional[frozenset],
        printf_pattern: str,
        padding: int,
        reference_shape: Optional[Tuple[int, int, int]],
        proc,
        raw_data: bool,
        honor_tag: bool,
        dst_choice: str,
    ) -> np.ndarray:
        """Locate the nearest available frame and return its pixels (color-
        managed identically to the regular read path). Falls back to black
        when no frames are available."""
        if not present_set:
            return AMImageRead._make_black(reference_shape)
        nearest = min(present_set, key=lambda x: abs(x - frame))
        path = sequence.expand_frame_pattern(printf_pattern, nearest, padding)
        try:
            res = image_backend.read_image(path)
            pixels = np.asarray(res.pixels, dtype=np.float32, copy=False).copy()
            if proc is not None and not raw_data:
                try:
                    proc.apply_inplace(pixels)
                except Exception as e:
                    log.warning("[am_vfx_tools/read_image] OCIO on nearest frame failed: %s", e)
            elif honor_tag:
                file_tag = res.color_space
                if file_tag and file_tag != color.PASSTHROUGH:
                    try:
                        tp = color.ColorProcessor(
                            file_tag, dst_choice, raw_data=raw_data,
                        )
                        if not tp.is_identity:
                            tp.apply_inplace(pixels)
                    except Exception as e:
                        log.warning(
                            "[am_vfx_tools/read_image] OCIO nearest file-tag %s -> %s failed: %s",
                            file_tag, dst_choice, e,
                        )
            return pixels
        except Exception as e:
            log.warning("[am_vfx_tools/read_image] nearest read failed for %s: %s", path, e)
            return AMImageRead._make_black(reference_shape)

    @staticmethod
    def _preflight_shape(
        printf_pattern: str,
        frames_to_load: List[int],
        padding: int,
    ) -> Optional[Tuple[int, int, int]]:
        """Find the first frame on disk and return its (H, W, C) shape via
        a cheap OIIO header read. Used so leading missing frames can be
        rendered at the right resolution. Returns ``None`` when nothing
        could be probed.
        """
        try:
            import OpenImageIO as oiio  # type: ignore
        except ImportError:
            return None
        for f in frames_to_load:
            path = sequence.expand_frame_pattern(printf_pattern, f, padding)
            if not os.path.exists(path):
                continue
            inp = oiio.ImageInput.open(path)
            if inp is None:
                continue
            try:
                spec = inp.spec()
                channels = max(3, spec.nchannels)
                return (spec.height, spec.width, channels)
            finally:
                inp.close()
        return None

    @staticmethod
    def _make_black(reference_shape: Optional[Tuple[int, int, int]]) -> np.ndarray:
        if reference_shape is None:
            return np.zeros((512, 512, 3), dtype=np.float32)
        return np.zeros(reference_shape, dtype=np.float32)

    @staticmethod
    def _make_checkerboard(reference_shape: Optional[Tuple[int, int, int]]) -> np.ndarray:
        """Nuke-style 8x8 magenta-grey checkerboard."""
        h, w, c = (reference_shape if reference_shape is not None
                   else (512, 512, 3))
        c = max(c, 3)
        grid_y = (np.arange(h) // 8) % 2
        grid_x = (np.arange(w) // 8) % 2
        mask = (grid_y[:, None] ^ grid_x[None, :]).astype(np.float32)
        magenta = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        grey    = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        rgb = magenta[None, None, :] * mask[..., None] + grey[None, None, :] * (1.0 - mask[..., None])
        out = np.zeros((h, w, c), dtype=np.float32)
        out[..., :3] = rgb
        return out

    @staticmethod
    def _normalize_channels(pixels: np.ndarray) -> np.ndarray:
        """Promote to 3-channel minimum; broadcast 1-channel -> RGB."""
        if pixels.ndim == 2:
            pixels = pixels[..., None]
        if pixels.shape[-1] == 1:
            pixels = np.repeat(pixels, 3, axis=-1)
        if pixels.shape[-1] < 3:
            pad = 3 - pixels.shape[-1]
            pixels = np.concatenate(
                [pixels, np.zeros((*pixels.shape[:-1], pad), dtype=np.float32)],
                axis=-1,
            )
        return np.ascontiguousarray(pixels.astype(np.float32, copy=False))

    @staticmethod
    def _preview_index(
        frame_mode: str, current_frame: int,
        range_first: Optional[int], n: int,
    ) -> int:
        if frame_mode == FRAME_MODE_SINGLE or n <= 0:
            return 0
        if range_first is None:
            return 0
        idx = int(current_frame) - int(range_first)
        return max(0, min(idx, n - 1))

    # ------------------------------------------------------------------ #
    #  UI / empty result
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ui_payload(
        tensor, path: str, show_preview: bool,
        working_colorspace: str, preview_idx: int,
    ):
        if not show_preview:
            return {"text": [path]}
        try:
            payload = preview.create_single_preview(
                tensor,
                frame_index=preview_idx,
                working_colorspace=working_colorspace,
                filename_hint=path,
            )
        except Exception as e:
            log.warning("[am_vfx_tools/read_image] preview generation failed: %s", e)
            return {"text": [path]}
        if not payload.get("images"):
            return {"text": [path]}
        return payload

    def _empty_result(self, label: str):
        return {
            "ui": {"text": [label]},
            "result": (
                # image, mask, resolved_path, info, width, height, frame_rate, frame_count
                # Empty MASK = zeros (stock ComfyUI convention: nothing to inpaint).
                torch.zeros((1, 64, 64, 3), dtype=torch.float32),
                torch.zeros((1, 64, 64), dtype=torch.float32),
                label, label, 64, 64, 0.0, 0,
            ),
        }

    @staticmethod
    def _build_info(
        *, frame_mode: str,
        first_frame: int, last_frame: Optional[int],
        range_first: Optional[int], range_last: Optional[int],
        frame_count: int,
        width: int, height: int, bit_depth: str, src_cs: str,
    ) -> str:
        """One-line summary of what was loaded, for the `info` output socket."""
        prefix = f"{width}x{height} {bit_depth} {src_cs}"
        if frame_mode == FRAME_MODE_SINGLE:
            return f"{prefix}, frame {first_frame}"
        # range / all share the same frame-range descriptor.
        lo = range_first if range_first is not None else first_frame
        hi = range_last  if range_last  is not None else (last_frame or first_frame)
        return f"{prefix}, {frame_count} frames {lo}-{hi}"
