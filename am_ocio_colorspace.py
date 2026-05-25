"""AM OCIO Colorspace — single-purpose colorspace transform utility.

Mirrors the colorspace mode of Nuke's ``OCIOColorSpace`` node. Takes an
IMAGE batch in *input_colorspace* and converts it to *working_colorspace*
via a single OCIO ColorProcessor — same code path the AM Read / AM Write
nodes use.

Also accepts an optional ``video`` input — when wired, returns a lazy
:class:`._core.video_lazy.OCIOTransformVideo` wrapper that applies the
OCIO transform per-frame on consumption. See docs/media-io-sync-rule.md
invariant 28.

The colorspace-mode utility; the future AM OCIO Display Transform will
mirror Nuke's display+view ``transform_type=display`` mode (out of scope
for now).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from ._core import color
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

log = logging.getLogger("am_vfx_tools.media-io.ocio-colorspace")


class AMOCIOColorspace:
    """ComfyUI node — apply an OCIO colorspace transform to an IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        cs = color.color_space_choices()
        return {
            "required": {
                "input_colorspace": (cs, {
                    "default": color.pick_default(
                        cs, ("ACES2065-1", "ACEScg", "sRGB - Display"),
                    ),
                    "tooltip": (
                        "Source colorspace — the space the input pixels are in. "
                        "The OCIO transform converts from this to "
                        "`working_colorspace`."
                    ),
                }),
                "working_colorspace": (cs, {
                    "default": color.default_working_colorspace(cs),
                    "tooltip": (
                        "Destination colorspace — the space the output pixels "
                        "will be in."
                    ),
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Image batch to transform.",
                }),
                "video": ("VIDEO", {
                    "tooltip": (
                        "Optional VIDEO input. When wired, returns a lazy "
                        "`OCIOTransformVideo` wrapper applying the OCIO "
                        "transform per-frame on consumption — no IMAGE "
                        "materialisation here. Alpha (when present) passes "
                        "through untouched. `image` is ignored when "
                        "`video` is wired. See invariant 28."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO")
    RETURN_NAMES = ("image", "video")
    OUTPUT_TOOLTIPS = (
        "Transformed image batch (same shape and channels as the input).",
        "Lazy VIDEO output — emits an `OCIOTransformVideo` wrapper when "
        "`video` is wired, else a zero-copy `VideoFromComponents` around "
        "the IMAGE batch. None when no input is wired.",
    )
    FUNCTION = "execute"
    CATEGORY = "AM VFX Tools/Color"

    def execute(
        self,
        input_colorspace: str,
        working_colorspace: str,
        image: Optional[torch.Tensor] = None,
        video=None,
    ):
        src = color.resolve_choice_to_cs(input_colorspace)
        dst = color.resolve_choice_to_cs(working_colorspace)

        # VIDEO branch — return a lazy OCIOTransformVideo wrapper.
        if video is not None:
            if image is not None:
                log.warning(
                    "[am_vfx_tools/ocio-cs] both VIDEO and IMAGE inputs wired "
                    "— VIDEO wins; IMAGE ignored"
                )
            wrapped = video_lazy.OCIOTransformVideo(video, src=src, dst=dst)
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder, wrapped)

        if image is None:
            log.warning("[am_vfx_tools/ocio-cs] no input wired — passing through")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), None)

        out_image = image
        if (
            src != color.PASSTHROUGH
            and dst != color.PASSTHROUGH
            and src != dst
        ):
            try:
                proc = color.ColorProcessor(src, dst)
                if not proc.is_identity:
                    batch = image
                    if batch.ndim == 3:
                        batch = batch[None, ...]
                    out_np = batch.detach().cpu().numpy().astype(np.float32, copy=True)
                    for i in range(out_np.shape[0]):
                        try:
                            proc.apply_inplace(out_np[i])
                        except Exception as e:
                            log.warning(
                                "[am_vfx_tools/ocio-cs] OCIO apply failed on frame %d "
                                "(%s); leaving frame untransformed",
                                i, e,
                            )
                    out_image = torch.from_numpy(out_np).to(image.device)
            except Exception as e:
                log.warning(
                    "[am_vfx_tools/ocio-cs] cannot build %s -> %s (%s); pixels unchanged",
                    src, dst, e,
                )

        return (out_image, _build_video_socket(out_image))


def _build_video_socket(images):
    """Zero-copy VideoFromComponents wrapper for the VIDEO output socket."""
    if not _VIDEO_TYPE_AVAILABLE or images is None:
        return None
    try:
        from fractions import Fraction as _Fraction
        return _ComfyInputImpl.VideoFromComponents(
            _ComfyTypes.VideoComponents(
                images=images, frame_rate=_Fraction(25, 1),
            ),
        )
    except Exception as e:  # noqa: BLE001
        log.warning("[am_vfx_tools/ocio-cs] VIDEO socket build failed (%s); None", e)
        return None
