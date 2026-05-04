"""AM Frame Range — select a frame, a range, or pass through an IMAGE batch.

Mirrors AM Read Image / AM Read Video frame semantics on an in-memory
IMAGE tensor:

* ``single`` — output a single frame at index ``first_frame``.
* ``range``  — output frames ``first_frame``..``last_frame`` (inclusive).
* ``all``    — pass the batch through unchanged.

Frame indices are **1-based** (matches AM Read Image / AM Read Video).
``last_frame = -1`` in range mode means "to the end of the batch".
Out-of-range indices are clamped to ``[1, N]``.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

log = logging.getLogger("am_vfx_tools.media-io.frame-range")

FRAME_MODE_SINGLE = "single"
FRAME_MODE_RANGE  = "range"
FRAME_MODE_ALL    = "all"
_FRAME_MODES = [FRAME_MODE_SINGLE, FRAME_MODE_RANGE, FRAME_MODE_ALL]


class AMFrameRange:
    """ComfyUI node — select frame(s) from an IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_mode": (_FRAME_MODES, {
                    "default": FRAME_MODE_SINGLE,
                    "tooltip": (
                        "Which frames to emit from the input batch. "
                        "single = only `first_frame`. "
                        "range  = `first_frame`..`last_frame` inclusive. "
                        "all    = pass the batch through unchanged."
                    ),
                }),
                "first_frame": ("INT", {
                    "default": 1, "min": -999999, "max": 999999,
                    "tooltip": (
                        "Frame index in single mode; lower bound in range mode "
                        "(1-based). Ignored in all mode. Clamped to [1, N]."
                    ),
                }),
                "last_frame": ("INT", {
                    "default": -1, "min": -1, "max": 999999,
                    "tooltip": (
                        "Range upper bound (inclusive, 1-based). "
                        "-1 = end of batch. Ignored in single / all modes. "
                        "Clamped to [first_frame, N]."
                    ),
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Image batch to slice.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = (
        "Selected image batch. Shape (1,H,W,C) in single mode, "
        "(N,H,W,C) in range mode, identical to the input in all mode.",
    )
    FUNCTION = "execute"
    CATEGORY = "AM VFX Tools/Util"

    def execute(
        self,
        frame_mode: str,
        first_frame: int,
        last_frame: int,
        image: Optional[torch.Tensor] = None,
    ):
        if image is None:
            log.warning("[am-vfx-tools/frame-range] `image` input is not wired — passing through black")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

        if image.ndim == 3:
            image = image[None, ...]

        n = int(image.shape[0])
        if n == 0:
            return (image,)

        if frame_mode == FRAME_MODE_ALL:
            return (image,)

        # Clamp first_frame to [1, n], convert to 0-based.
        f = max(1, min(int(first_frame), n)) - 1

        if frame_mode == FRAME_MODE_SINGLE:
            return (image[f:f + 1],)

        # range
        if int(last_frame) < 0:
            l = n - 1
        else:
            l = max(1, min(int(last_frame), n)) - 1
        if l < f:
            l = f
        return (image[f:l + 1],)


__all__ = ["AMFrameRange"]
