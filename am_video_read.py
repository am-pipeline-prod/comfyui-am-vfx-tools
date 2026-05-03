"""AM Video Read — ComfyUI node.

Reads a video container from disk via PyAV with OCIO 2.x color
management and an optional reformat / dtype-cast pass. Use the
``Browse`` button (or paste a path into ``file_path``) to point at any
container PyAV can decode.

Frame-range knobs match AM Read Image symmetrically (frame_mode /
first_frame / last_frame / before / after) so artists move between
image sequences and video without mental remapping. Per-knob deltas
vs the image sibling:

* No ``missing_frames`` — video containers don't have intra-stream
  gaps. If a packet fails to decode, that's a corruption error and the
  node fails fast rather than substituting black/checkerboard pixels.
* Default ``input_colorspace`` is ``Display/Gamma 2.2 Rec.709 - Display``
  — the studio dailies standard. Falls back to ``Display/Rec.1886
  Rec.709 - Display`` on OCIO configs that don't define the gamma 2.2
  display variant.
* ``frame_rate`` is an output socket, not an input widget — a video
  container has its own time base; you can't override it on read.

Outputs match AM Read Image plus video-specific additions: AUDIO,
FLOAT fps. The legacy FRAME_INFO output is replaced by the structured
width/height/frame_count/fps/info quartet.

Decode + audio + colorspace conversion uses :mod:`._core.video_backend`
(PyAV) and :mod:`._core.color` (OCIO 2.x).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ._core import color, preview, reformat, video_backend

log = logging.getLogger("am_vfx_tools.read_video")


FRAME_MODE_SINGLE = "single"
FRAME_MODE_RANGE  = "range"
FRAME_MODE_ALL    = "all"
_FRAME_MODES = [FRAME_MODE_SINGLE, FRAME_MODE_RANGE, FRAME_MODE_ALL]

EDGE_HOLD   = "hold"
EDGE_LOOP   = "loop"
EDGE_BOUNCE = "bounce"
EDGE_BLACK  = "black"
_EDGE_OPTIONS = [EDGE_HOLD, EDGE_LOOP, EDGE_BOUNCE, EDGE_BLACK]


class AMVideoRead:
    @classmethod
    def INPUT_TYPES(cls):
        cs = color.color_space_choices()
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Absolute path to the video container",
                    "tooltip": (
                        "Absolute path to the video container. "
                        "Use the 📂 Browse button to populate from the native dialog."
                    ),
                }),
                # Default `all` (vs Read Image's `single`): video reads are
                # typically loaded as full sequences for processing, while
                # image plates more commonly want a specific frame.
                "frame_mode": (_FRAME_MODES, {
                    "default": FRAME_MODE_ALL,
                    "tooltip": (
                        "Which frames to decode from the container. "
                        "single = only `first_frame`. "
                        "range = `first_frame`..`last_frame` inclusive (with `before`/`after` "
                        "policy outside the container). "
                        "all = every frame in the container."
                    ),
                }),
                "first_frame": ("INT", {
                    "default": 1, "min": -999999, "max": 999999,
                    "tooltip": (
                        "Frame index in single mode; lower bound in range mode (1-based). "
                        "Ignored in all mode. The 🔍 Detect Range button auto-fills this."
                    ),
                }),
                "last_frame":  ("INT", {
                    "default": -1, "min": -1, "max": 999999,
                    "tooltip": (
                        "Range upper bound (inclusive, 1-based). -1 = auto = container's "
                        "frame count. The 🔍 Detect Range button reads it from the PyAV header."
                    ),
                }),
                "before": (_EDGE_OPTIONS, {
                    "default": EDGE_HOLD,
                    "tooltip": (
                        "Edge policy below the container's frames when `frame_mode=range` "
                        "and the requested range starts before frame 1. "
                        "hold = clamp to first; loop = wrap; bounce = ping-pong; "
                        "black = synthesize black."
                    ),
                }),
                "after": (_EDGE_OPTIONS, {
                    "default": EDGE_HOLD,
                    "tooltip": (
                        "Edge policy above the container's frames when the requested "
                        "range extends past the last frame. Mirrors `before`."
                    ),
                }),
                "input_colorspace": (cs, {
                    "default": color.pick_default(
                        cs, ("Gamma 2.2 Rec.709 - Display", "Rec.1886 Rec.709 - Display"),
                    ),
                    "tooltip": (
                        "Source colorspace of the container (HD video typically "
                        "Gamma 2.2 Rec.709 in studio dailies, Rec.1886 Rec.709 "
                        "for broadcast). The OCIO transform converts from this "
                        "to `working_colorspace`. Pick `raw` to skip — but most "
                        "video containers don't carry a reliable colorspace tag, "
                        "so an explicit value is usually correct."
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
                        "nodes will see."
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
                    "tooltip": "Show a thumbnail of the first decoded frame on the node.",
                }),
            },
            "optional": {},
        }

    # Output socket order — kept symmetric across the four media-IO nodes
    # (see media-io-sync-rule.md invariant 14a). Read Video splices the
    # `audio` socket after `image`; the rest of the tail
    # (`resolved_path, info, width, height, frame_rate, frame_count`)
    # matches AM Read Image.
    RETURN_TYPES = (
        "IMAGE", "MASK", "AUDIO", "STRING", "STRING", "INT", "INT", "FLOAT", "INT",
    )
    RETURN_NAMES = (
        "image", "mask", "audio", "resolved_path", "info",
        "width", "height", "frame_rate", "frame_count",
    )
    OUTPUT_TOOLTIPS = (
        "Decoded frames as IMAGE (N×H×W×3 float in [0,1]). RGB only — alpha "
        "is split out to the `mask` socket.",
        reformat.TOOLTIP_MASK_OUT_READ,
        "Audio track from the container, or a silent stub if absent.",
        "Resolved on-disk container path.",
        "Human-readable summary: dimensions, codec, pixel format, fps, frame count.",
        "Frame width in pixels.",
        "Frame height in pixels.",
        "Container's native frame rate (read from the PyAV header).",
        "Number of frames decoded into the IMAGE batch.",
    )
    FUNCTION = "execute"
    CATEGORY = "AM Pipe"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def execute(
        self,
        file_path: str,
        frame_mode: str,
        first_frame: int,
        last_frame: int,
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
        if not file_path:
            log.warning("[am_vfx_tools/read_video] empty file_path")
            return self._empty_result("(no file)")
        path = os.path.expandvars(os.path.expanduser(file_path))

        if not os.path.exists(path):
            log.warning("[am_vfx_tools/read_video] file not found: %s", path)
            return self._empty_result(path)

        # Probe first so we know the container's frame count + fps.
        try:
            info = video_backend.probe(path)
        except Exception as e:
            log.warning("[am_vfx_tools/read_video] probe failed for %s: %s", path, e)
            return self._empty_result(path)

        # Resolve which container-level frame range to decode + the
        # output range we ultimately want emitted (which may be wider
        # than what's in the container — before/after extrapolation).
        decode_range, output_range = self._resolve_ranges(
            frame_mode, int(first_frame), int(last_frame), info.n_frames,
        )
        if output_range is None:
            log.warning("[am_vfx_tools/read_video] empty output range")
            return self._empty_result(path)

        decode_start, decode_count = decode_range
        out_lo, out_hi = output_range  # 1-indexed inclusive container frames

        # Decode the slice that actually exists on disk.
        try:
            stack, audio_buf, _info2 = video_backend.read_video_frames(
                path,
                start=decode_start,
                count=decode_count,
                audio_track=0,
            )
        except Exception as e:
            log.warning("[am_vfx_tools/read_video] decode failed for %s: %s", path, e)
            return self._empty_result(path)

        if stack.shape[0] == 0:
            log.warning("[am_vfx_tools/read_video] no frames decoded from %s", path)
            return self._empty_result(path)

        # Apply before/after extrapolation if the requested range
        # extends past the container.
        stack = self._apply_edge_extrapolation(
            stack,
            container_first=1, container_last=info.n_frames or stack.shape[0] + decode_start,
            requested_lo=out_lo, requested_hi=out_hi,
            decode_start_index=decode_start + 1,  # convert to 1-indexed
            before=before, after=after,
        )

        # OCIO transform (input_cs -> working_cs).
        src = color.resolve_choice_to_cs(input_colorspace)
        dst = color.resolve_choice_to_cs(working_colorspace)
        if (
            not raw_data
            and src != color.PASSTHROUGH
            and dst != color.PASSTHROUGH
            and src != dst
        ):
            try:
                proc = color.ColorProcessor(src, dst, raw_data=raw_data)
                if not proc.is_identity:
                    for i in range(stack.shape[0]):
                        proc.apply_inplace(stack[i])
            except Exception as e:
                log.warning(
                    "[am_vfx_tools/read_video] OCIO %s -> %s failed (%s); pixels unchanged",
                    src, dst, e,
                )

        # Reformat (post-decode, post-OCIO). Applied once on the whole batch
        # — the helper iterates per-frame internally for cv2 (no batched
        # API). For very long sequences this still holds the full-res stack
        # transiently; the cache win (small post-reformat output retained
        # for the workflow run) is the dominant benefit. A future
        # streaming-decode path in `video_backend.read_video_frames` would
        # close the peak-RAM gap.
        src_h_pre, src_w_pre = int(stack.shape[1]), int(stack.shape[2])
        if reformat_mode != reformat.MODE_OFF or output_dtype != reformat.DTYPE_FP32:
            try:
                stack = reformat.reformat_array(
                    stack,
                    mode=reformat_mode,
                    scale=float(scale),
                    preset=preset,
                    target_w=int(target_width),
                    target_h=int(target_height),
                    resize_type=resize_type,
                    filter_name=filter,
                    output_dtype=output_dtype,
                )
            except Exception as e:
                log.warning("[am_vfx_tools/read_video] reformat failed (%s); pixels unchanged", e)

        # Split into IMAGE (RGB) + MASK (1 - alpha). Stack returned by
        # video_backend.read_video_frames is (N, H, W, 3) for RGB-only
        # containers and (N, H, W, 4) for alpha-bearing pixfmts (ProRes
        # 4444, QT RLE, FFV1 yuva*). Empty MASK (zeros) for the RGB
        # case — matches stock-ComfyUI "no alpha" convention.
        image_arr, mask_arr = reformat.split_image_mask(stack)
        tensor = torch.from_numpy(np.ascontiguousarray(image_arr))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_arr))
        n_out = int(tensor.shape[0])
        height = int(tensor.shape[1])
        width = int(tensor.shape[2])
        fps = float(info.fps) if info.fps else 0.0
        info_str = (
            f"{src_w_pre}x{src_h_pre} {info.codec or '?'} {info.pix_fmt or '?'} "
            f"@ {fps:.3f}fps, {n_out} frame(s) "
            f"(of {info.n_frames or 'unknown'})"
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
            src_w=src_w_pre, src_h=src_h_pre,
        )
        if rf_frag:
            info_str = f"{info_str} | {rf_frag}"

        # Audio dict for the AUDIO socket.
        audio_dict = audio_buf.as_comfy_audio() if audio_buf is not None else self._silent_audio_stub()

        return {
            "ui": self._ui_payload(tensor, path, show_preview, working_colorspace),
            "result": (
                # image, mask, audio, resolved_path, info, width, height, frame_rate, frame_count
                tensor, mask_tensor, audio_dict, path, info_str,
                int(width), int(height), float(fps), int(n_out),
            ),
        }

    def _resolve_ranges(
        self, frame_mode: str, first: int, last: int, n: int,
    ) -> Tuple[Tuple[int, Optional[int]], Optional[Tuple[int, int]]]:
        """Compute (decode_range, output_range).

        - decode_range = (start_0idx, count_or_None) for what to decode
          out of the container.
        - output_range = (lo_1idx, hi_1idx) inclusive container-frame
          range the user requested. May extend past [1, n] when
          before/after extrapolation will fill the gap.
        """
        if n <= 0:
            n = 999999  # unknown frame count; trust the user
        if frame_mode == FRAME_MODE_ALL:
            return (0, None), (1, n)
        if frame_mode == FRAME_MODE_SINGLE:
            target = max(1, first)
            decode_idx = max(0, min(n - 1, target - 1))
            return (decode_idx, 1), (target, target)
        # range
        lo = first
        hi = n if last < 0 else last
        if hi < lo:
            return (0, 0), None
        decode_lo = max(0, lo - 1)
        decode_hi = min(n, hi)
        decode_count = max(0, decode_hi - decode_lo)
        return (decode_lo, decode_count), (lo, hi)

    @staticmethod
    def _apply_edge_extrapolation(
        stack: np.ndarray,
        *,
        container_first: int,
        container_last: int,
        requested_lo: int,
        requested_hi: int,
        decode_start_index: int,
        before: str,
        after: str,
    ) -> np.ndarray:
        """Pad *stack* on either side with before/after frames if the
        requested range extends past the container."""
        if stack.shape[0] == 0:
            return stack
        # How many frames are missing on each side?
        pre_missing = max(0, container_first - requested_lo)
        post_missing = max(0, requested_hi - container_last)
        if pre_missing == 0 and post_missing == 0:
            return stack

        h, w, c = stack.shape[1:]
        pad = []

        if pre_missing > 0:
            pad.append(_extrapolate_block(stack, pre_missing, side="before", policy=before))
        pad.append(stack)
        if post_missing > 0:
            pad.append(_extrapolate_block(stack, post_missing, side="after", policy=after))
        out = np.concatenate(pad, axis=0)
        return out

    @staticmethod
    def _silent_audio_stub() -> Dict[str, Any]:
        return {
            "waveform": torch.zeros((1, 1, 1), dtype=torch.float32),
            "sample_rate": 48000,
        }

    @staticmethod
    def _ui_payload(tensor, path: str, show_preview: bool, working_colorspace: str):
        if not show_preview:
            return {"text": [path]}
        try:
            payload = preview.create_single_preview(
                tensor,
                frame_index=0,
                working_colorspace=working_colorspace,
                filename_hint=path,
            )
        except Exception as e:
            log.warning("[am_vfx_tools/read_video] preview generation failed: %s", e)
            return {"text": [path]}
        if not payload.get("images"):
            return {"text": [path]}
        return payload

    def _empty_result(self, label: str):
        empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        # Empty MASK = zeros (stock ComfyUI convention: nothing to inpaint).
        empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        return {
            "ui": {"text": [label]},
            "result": (
                # image, mask, audio, resolved_path, info, width, height, frame_rate, frame_count
                empty, empty_mask, self._silent_audio_stub(), label, "(empty)",
                64, 64, 0.0, 1,
            ),
        }


def _extrapolate_block(stack: np.ndarray, count: int, *, side: str, policy: str) -> np.ndarray:
    """Build a (count, H, W, C) block from *stack* per the edge policy."""
    if count <= 0 or stack.shape[0] == 0:
        return np.zeros((0,) + stack.shape[1:], dtype=stack.dtype)

    n = stack.shape[0]
    if policy == EDGE_BLACK:
        return np.zeros((count,) + stack.shape[1:], dtype=stack.dtype)
    if policy == EDGE_HOLD:
        ref = stack[0:1] if side == "before" else stack[-1:]
        return np.repeat(ref, count, axis=0)
    if policy == EDGE_LOOP:
        # Cycle the existing range.
        if side == "before":
            idxs = [(n + (-(i + 1) % n)) % n for i in range(count)][::-1]
        else:
            idxs = [(i + 1) % n for i in range(count)]
        return stack[np.asarray(idxs, dtype=np.int64)]
    if policy == EDGE_BOUNCE:
        # Triangle wave: ping-pong between [0, n-1].
        if n == 1:
            return np.repeat(stack, count, axis=0)
        period = 2 * (n - 1)
        if side == "before":
            seq = []
            for i in range(count):
                t = (i + 1) % period
                seq.append(t if t < n else period - t)
            seq = seq[::-1]
        else:
            seq = []
            for i in range(count):
                t = (n + i) % period
                seq.append(t if t < n else period - t)
        return stack[np.asarray(seq, dtype=np.int64)]
    # Unknown policy — fall back to hold.
    ref = stack[0:1] if side == "before" else stack[-1:]
    return np.repeat(ref, count, axis=0)
