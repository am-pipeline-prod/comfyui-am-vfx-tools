"""AM Frame Range — select a frame, a range, or pass through an IMAGE batch.

Mirrors AM Read Image / AM Read Video frame semantics on an in-memory
IMAGE tensor:

* ``single`` — output a single frame at index ``first_frame``.
* ``range``  — output frames ``first_frame``..``last_frame`` (inclusive).
* ``all``    — pass the batch through unchanged.

Frame indices are **1-based** (matches AM Read Image / AM Read Video).
``last_frame = -1`` in range mode means "to the end of the batch".
Out-of-range indices are clamped to ``[1, N]``.

Also accepts an optional ``video`` input — when wired, returns a lazy
:class:`._core.video_lazy.FrameRangeVideo` wrapper that filters frames
on consumption. With a ``VideoFromFile`` source, PyAV decodes only the
frames in the requested range and early-terminates — peak RAM stays at
one frame and the source's tail frames are never touched. See
docs/media-io-sync-rule.md invariant 28.
"""
from __future__ import annotations

import logging
from fractions import Fraction
from typing import Optional

import torch

from ._core import video_lazy

# Native ComfyUI VIDEO type — see docs/media-io-sync-rule.md invariant 28.
try:
    from comfy_api.v0_0_2 import InputImpl as _ComfyInputImpl  # type: ignore[import-not-found]
    from comfy_api.v0_0_2 import Types as _ComfyTypes  # type: ignore[import-not-found]
    _VIDEO_TYPE_AVAILABLE = True
except ImportError:
    _ComfyInputImpl = None  # type: ignore[assignment]
    _ComfyTypes = None  # type: ignore[assignment]
    _VIDEO_TYPE_AVAILABLE = False

log = logging.getLogger("am_vfx_tools.media-io.frame-range")

FRAME_MODE_SINGLE = "single"
FRAME_MODE_RANGE  = "range"
FRAME_MODE_ALL    = "all"
_FRAME_MODES = [FRAME_MODE_SINGLE, FRAME_MODE_RANGE, FRAME_MODE_ALL]


class AMFrameRange:
    """ComfyUI node — select frame(s) from an IMAGE batch or VIDEO."""

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
                # VIDEO input — appended at the end so saved-workflow
                # widget/socket indexing isn't disturbed.
                "video": ("VIDEO", {
                    "tooltip": (
                        "Optional VIDEO input. When wired, returns a lazy "
                        "`FrameRangeVideo` wrapper that filters frames on "
                        "consumption — no IMAGE materialisation here. With "
                        "a `VideoFromFile` source, PyAV decodes ONLY the "
                        "frames in the requested range and early-terminates "
                        "after the last requested frame is yielded. Combined "
                        "with a downstream AM consumer, peak RAM stays at "
                        "one frame and the source's unwanted tail frames are "
                        "never decoded. `image` is ignored when `video` is "
                        "wired. See invariant 28."
                    ),
                }),
            },
        }

    # Output sockets — VIDEO appended at the end.
    RETURN_TYPES = ("IMAGE", "VIDEO")
    RETURN_NAMES = ("image", "video")
    OUTPUT_TOOLTIPS = (
        "Selected image batch. Shape (1,H,W,C) in single mode, "
        "(N,H,W,C) in range mode, identical to the input in all mode.",
        "Lazy VIDEO output — emits a `FrameRangeVideo` wrapper when "
        "`video` is wired, else a zero-copy `VideoFromComponents` around "
        "the sliced IMAGE batch. None when no input is wired.",
    )
    FUNCTION = "execute"
    CATEGORY = "AM VFX Tools/Util"

    def execute(
        self,
        frame_mode: str,
        first_frame: int,
        last_frame: int,
        image: Optional[torch.Tensor] = None,
        video=None,
    ):
        # VIDEO branch — return a lazy FrameRangeVideo wrapper. No
        # materialisation here; downstream AM consumer iterates and the
        # wrapper filters / early-terminates the source decode.
        if video is not None:
            if image is not None:
                log.warning(
                    "[am_vfx_tools/frame-range] both VIDEO and IMAGE inputs "
                    "wired — VIDEO wins; IMAGE ignored"
                )
            wrapped = video_lazy.FrameRangeVideo(
                video,
                mode=frame_mode,
                first_frame=int(first_frame),
                last_frame=int(last_frame),
            )
            # Placeholder IMAGE — the artist wired VIDEO precisely to
            # avoid materialisation. Real data flows through the VIDEO
            # output socket as the lazy wrapper.
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder, wrapped)

        if image is None:
            log.warning(
                "[am_vfx_tools/frame-range] no input wired — passing through black"
            )
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty, None)

        if image.ndim == 3:
            image = image[None, ...]

        n = int(image.shape[0])
        if n == 0:
            return (image, _build_video_socket(image))

        if frame_mode == FRAME_MODE_ALL:
            return (image, _build_video_socket(image))

        # Clamp first_frame to [1, n], convert to 0-based.
        f = max(1, min(int(first_frame), n)) - 1

        if frame_mode == FRAME_MODE_SINGLE:
            sliced = image[f:f + 1]
            return (sliced, _build_video_socket(sliced))

        # range
        if int(last_frame) < 0:
            l = n - 1
        else:
            l = max(1, min(int(last_frame), n)) - 1
        if l < f:
            l = f
        sliced = image[f:l + 1]
        return (sliced, _build_video_socket(sliced))


def _build_video_socket(images):
    """Zero-copy VideoFromComponents wrapper for the VIDEO output socket
    when the IMAGE branch fires. Returns None when comfy_api is
    unavailable. No fps context on this node — defaults to 25 (matches
    AM Reformat / Grade / Reverse convention).
    """
    if not _VIDEO_TYPE_AVAILABLE or images is None:
        return None
    try:
        return _ComfyInputImpl.VideoFromComponents(
            _ComfyTypes.VideoComponents(
                images=images, frame_rate=Fraction(25, 1),
            ),
        )
    except Exception as e:  # noqa: BLE001
        log.warning(
            "[am_vfx_tools/frame-range] VIDEO socket build failed (%s); None", e,
        )
        return None


__all__ = ["AMFrameRange"]
