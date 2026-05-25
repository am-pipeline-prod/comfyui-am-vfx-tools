"""AM Color Correct — Nuke-style ColorCorrect node.

Scalar knobs for saturation / contrast / gamma / gain / offset /
hue_rotation applied in scene-linear space. Sockets mirror AM Grade:
optional IMAGE + VIDEO inputs, IMAGE + VIDEO outputs (the VIDEO output
is a lazy :class:`._core.video_lazy.ColorCorrectedVideo` wrapper when
``video`` is wired). See docs/media-io-sync-rule.md invariant 28 for the
lazy-VIDEO contract.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from ._core.color_correct import color_correct_apply
from ._core import video_lazy

log = logging.getLogger("am_vfx_tools.media-io.color-correct")


_VIDEO_TOOLTIP = (
    "Optional VIDEO input. When wired, returns a lazy "
    "`ColorCorrectedVideo` wrapper applying the correction per-frame on "
    "consumption — no IMAGE materialisation here. Alpha (when present) "
    "passes through untouched. `image` is ignored when `video` is wired. "
    "See invariant 28."
)
_VIDEO_OUT_TOOLTIP = (
    "Lazy VIDEO output — emits a `ColorCorrectedVideo` wrapper when "
    "`video` is wired, else a zero-copy `VideoFromComponents` around "
    "the IMAGE batch. None when no input is wired."
)


def _split_alpha(image: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if image.ndim == 3:
        image = image[None, ...]
    if image.shape[-1] >= 4:
        return image[..., :3], image[..., 3:]
    return image, None


# Native ComfyUI VIDEO type — see docs/media-io-sync-rule.md invariant 28.
try:
    from comfy_api.v0_0_2 import InputImpl as _ComfyInputImpl  # type: ignore[import-not-found]
    from comfy_api.v0_0_2 import Types as _ComfyTypes  # type: ignore[import-not-found]
    _VIDEO_TYPE_AVAILABLE = True
except ImportError:
    _ComfyInputImpl = None  # type: ignore[assignment]
    _ComfyTypes = None  # type: ignore[assignment]
    _VIDEO_TYPE_AVAILABLE = False


def _build_video_socket(images):
    """Wrap a corrected IMAGE batch in a VideoFromComponents (zero copy)."""
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
        log.warning("[am_vfx_tools/color-correct] VIDEO socket build failed (%s); None", e)
        return None


class AMColorCorrect:
    """Nuke-style ColorCorrect — single scalar per knob."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "saturation":   ("FLOAT", {
                    "default": 1.0, "step": 0.001,
                    "tooltip": (
                        "Saturation multiplier around Rec.709 luma. "
                        "0 = grayscale, 1 = identity, > 1 boosts saturation."
                    ),
                }),
                "contrast":     ("FLOAT", {
                    "default": 1.0, "step": 0.001,
                    "tooltip": (
                        "Contrast around scene-linear mid-gray (pivot 0.18). "
                        "0 = flat mid-gray, 1 = identity, > 1 increases contrast."
                    ),
                }),
                "gamma":        ("FLOAT", {
                    "default": 1.0, "step": 0.001,
                    "tooltip": (
                        "Gamma (sign-preserving `pow(x, 1/gamma)`). "
                        "> 1 brightens midtones, < 1 darkens, 1 = identity."
                    ),
                }),
                "gain":         ("FLOAT", {
                    "default": 1.0, "step": 0.001,
                    "tooltip": (
                        "Multiplicative gain — applied after `offset`, before "
                        "`gamma`. 1 = identity."
                    ),
                }),
                "offset":       ("FLOAT", {
                    "default": 0.0, "step": 0.001,
                    "tooltip": (
                        "Additive offset — applied first (before gain / gamma / "
                        "contrast / saturation / hue). 0 = identity."
                    ),
                }),
                "hue_rotation": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1,
                    "tooltip": (
                        "Hue rotation in degrees around the gray axis [1,1,1]. "
                        "0 = identity; ±180 spans the full hue wheel (equivalent "
                        "on the loop). Applied last."
                    ),
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Image batch to color-correct.",
                }),
                "video": ("VIDEO", {"tooltip": _VIDEO_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO")
    RETURN_NAMES = ("image", "video")
    OUTPUT_TOOLTIPS = (
        "Corrected image batch (same shape and channels as the input).",
        _VIDEO_OUT_TOOLTIP,
    )
    FUNCTION = "execute"
    CATEGORY = "AM VFX Tools/Color"

    def execute(
        self,
        saturation: float,
        contrast: float,
        gamma: float,
        gain: float,
        offset: float,
        hue_rotation: float,
        image: Optional[torch.Tensor] = None,
        video=None,
    ):
        # VIDEO branch — lazy wrapper; no materialisation here.
        if video is not None:
            if image is not None:
                log.warning(
                    "[am_vfx_tools/color-correct] both VIDEO and IMAGE inputs wired "
                    "— VIDEO wins; IMAGE ignored"
                )
            wrapped = video_lazy.ColorCorrectedVideo(
                video,
                saturation=saturation,
                contrast=contrast,
                gamma=gamma,
                gain=gain,
                offset=offset,
                hue_degrees=hue_rotation,
            )
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder, wrapped)

        if image is None:
            log.warning("[am_vfx_tools/color-correct] no input wired — passing through")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), None)

        rgb, alpha = _split_alpha(image)
        out_rgb = color_correct_apply(
            rgb,
            saturation=saturation,
            contrast=contrast,
            gamma=gamma,
            gain=gain,
            offset=offset,
            hue_degrees=hue_rotation,
        )

        out_image = (
            torch.cat([out_rgb, alpha], dim=-1) if alpha is not None else out_rgb
        )
        return (out_image, _build_video_socket(out_image))


__all__ = ["AMColorCorrect"]
