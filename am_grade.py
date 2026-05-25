"""AM Grade / AM Grade RGB — Nuke-style color grade nodes.

Two ComfyUI nodes that mirror the native Nuke Grade node UI:

* :class:`AMGrade`     — one float per knob (blackpoint, whitepoint,
  lift, gain, multiply, offset, gamma).
* :class:`AMGradeRGB`  — per-channel floats (``..._r/_g/_b``); the three
  booleans (reverse, black_clamp, white_clamp) stay scalar.

Both call :func:`._core.grade.grade_apply` for the IMAGE branch. Both
also accept an optional ``video`` input — when wired, return a lazy
:class:`._core.video_lazy.GradedVideo` wrapper that defers the grade
until a downstream AM consumer iterates frames. See
docs/media-io-sync-rule.md invariant 28.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from ._core.grade import grade_apply
from ._core import video_lazy

log = logging.getLogger("am_vfx_tools.media-io.grade")


_VIDEO_TOOLTIP = (
    "Optional VIDEO input. When wired, returns a lazy `GradedVideo` "
    "wrapper applying the grade per-frame on consumption — no IMAGE "
    "materialisation here. Alpha (when present) passes through "
    "untouched. `image` is ignored when `video` is wired. See "
    "invariant 28."
)
_VIDEO_OUT_TOOLTIP = (
    "Lazy VIDEO output — emits a `GradedVideo` wrapper when `video` is "
    "wired, else a zero-copy `VideoFromComponents` around the IMAGE "
    "batch. None when no input is wired."
)


def _split_alpha(image: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if image.ndim == 3:
        image = image[None, ...]
    if image.shape[-1] >= 4:
        return image[..., :3], image[..., 3:]
    return image, None


def _vec3(r: float, g: float, b: float, ref: torch.Tensor) -> torch.Tensor:
    return torch.tensor([r, g, b], device=ref.device, dtype=ref.dtype)


# Native ComfyUI VIDEO type — see docs/media-io-sync-rule.md invariant 28.
# Used for the zero-copy VIDEO output socket when the IMAGE branch fires.
try:
    from comfy_api.v0_0_2 import InputImpl as _ComfyInputImpl  # type: ignore[import-not-found]
    from comfy_api.v0_0_2 import Types as _ComfyTypes  # type: ignore[import-not-found]
    _VIDEO_TYPE_AVAILABLE = True
except ImportError:
    _ComfyInputImpl = None  # type: ignore[assignment]
    _ComfyTypes = None  # type: ignore[assignment]
    _VIDEO_TYPE_AVAILABLE = False


def _build_video_socket(images):
    """Wrap a graded IMAGE batch in a VideoFromComponents (zero copy).

    No fps context on grade nodes — defaults to 25 (matches AM Reformat).
    Returns None when comfy_api is unavailable.
    """
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
        log.warning("[am_vfx_tools/grade] VIDEO socket build failed (%s); None", e)
        return None


class AMGrade:
    """Nuke-style Grade — one float per knob, three booleans."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blackpoint":  ("FLOAT", {"default": 0.0, "step": 0.001}),
                "whitepoint":  ("FLOAT", {"default": 1.0, "step": 0.001}),
                "lift":        ("FLOAT", {"default": 0.0, "step": 0.001}),
                "gain":        ("FLOAT", {"default": 1.0, "step": 0.001}),
                "multiply":    ("FLOAT", {"default": 1.0, "step": 0.001}),
                "offset":      ("FLOAT", {"default": 0.0, "step": 0.001}),
                "gamma":       ("FLOAT", {"default": 1.0, "step": 0.001}),
                "reverse":     ("BOOLEAN", {"default": False}),
                "black_clamp": ("BOOLEAN", {"default": True}),
                "white_clamp": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Image batch to grade.",
                }),
                "video": ("VIDEO", {"tooltip": _VIDEO_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO")
    RETURN_NAMES = ("image", "video")
    OUTPUT_TOOLTIPS = (
        "Graded image batch (same shape and channels as the input).",
        _VIDEO_OUT_TOOLTIP,
    )
    FUNCTION = "execute"
    CATEGORY = "AM VFX Tools/Color"

    def execute(
        self,
        blackpoint: float,
        whitepoint: float,
        lift: float,
        gain: float,
        multiply: float,
        offset: float,
        gamma: float,
        reverse: bool,
        black_clamp: bool,
        white_clamp: bool,
        image: Optional[torch.Tensor] = None,
        video=None,
    ):
        # VIDEO branch — return a lazy GradedVideo wrapper. No
        # materialisation here; downstream AM consumer iterates.
        if video is not None:
            if image is not None:
                log.warning(
                    "[am_vfx_tools/grade] both VIDEO and IMAGE inputs wired — "
                    "VIDEO wins; IMAGE ignored"
                )
            wrapped = video_lazy.GradedVideo(
                video,
                blackpoint=(blackpoint, blackpoint, blackpoint),
                whitepoint=(whitepoint, whitepoint, whitepoint),
                lift=(lift, lift, lift),
                gain=(gain, gain, gain),
                multiply=(multiply, multiply, multiply),
                offset=(offset, offset, offset),
                gamma=(gamma, gamma, gamma),
                reverse=reverse,
                black_clamp=black_clamp, white_clamp=white_clamp,
            )
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder, wrapped)

        if image is None:
            log.warning("[am_vfx_tools/grade] no input wired — passing through")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), None)

        rgb, alpha = _split_alpha(image)
        out_rgb = grade_apply(
            rgb,
            blackpoint=_vec3(blackpoint, blackpoint, blackpoint, rgb),
            whitepoint=_vec3(whitepoint, whitepoint, whitepoint, rgb),
            lift=_vec3(lift, lift, lift, rgb),
            gain=_vec3(gain, gain, gain, rgb),
            multiply=_vec3(multiply, multiply, multiply, rgb),
            offset=_vec3(offset, offset, offset, rgb),
            gamma=_vec3(gamma, gamma, gamma, rgb),
            reverse=reverse,
            black_clamp=black_clamp,
            white_clamp=white_clamp,
        )

        out_image = (
            torch.cat([out_rgb, alpha], dim=-1) if alpha is not None else out_rgb
        )
        # VIDEO output — zero-copy VideoFromComponents wrapping the graded
        # batch. Same pattern as AM Reformat / AM Read Image.
        return (out_image, _build_video_socket(out_image))


class AMGradeRGB:
    """Nuke-style Grade — per-channel R/G/B knobs, three booleans."""

    @classmethod
    def INPUT_TYPES(cls):
        f0 = {"default": 0.0, "step": 0.001}
        f1 = {"default": 1.0, "step": 0.001}
        return {
            "required": {
                "blackpoint_r": ("FLOAT", f0),
                "blackpoint_g": ("FLOAT", f0),
                "blackpoint_b": ("FLOAT", f0),
                "whitepoint_r": ("FLOAT", f1),
                "whitepoint_g": ("FLOAT", f1),
                "whitepoint_b": ("FLOAT", f1),
                "lift_r":       ("FLOAT", f0),
                "lift_g":       ("FLOAT", f0),
                "lift_b":       ("FLOAT", f0),
                "gain_r":       ("FLOAT", f1),
                "gain_g":       ("FLOAT", f1),
                "gain_b":       ("FLOAT", f1),
                "multiply_r":   ("FLOAT", f1),
                "multiply_g":   ("FLOAT", f1),
                "multiply_b":   ("FLOAT", f1),
                "offset_r":     ("FLOAT", f0),
                "offset_g":     ("FLOAT", f0),
                "offset_b":     ("FLOAT", f0),
                "gamma_r":      ("FLOAT", f1),
                "gamma_g":      ("FLOAT", f1),
                "gamma_b":      ("FLOAT", f1),
                "reverse":      ("BOOLEAN", {"default": False}),
                "black_clamp":  ("BOOLEAN", {"default": True}),
                "white_clamp":  ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Image batch to grade.",
                }),
                "video": ("VIDEO", {"tooltip": _VIDEO_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO")
    RETURN_NAMES = ("image", "video")
    OUTPUT_TOOLTIPS = (
        "Graded image batch (same shape and channels as the input).",
        _VIDEO_OUT_TOOLTIP,
    )
    FUNCTION = "execute"
    CATEGORY = "AM VFX Tools/Color"

    def execute(
        self,
        blackpoint_r: float, blackpoint_g: float, blackpoint_b: float,
        whitepoint_r: float, whitepoint_g: float, whitepoint_b: float,
        lift_r: float, lift_g: float, lift_b: float,
        gain_r: float, gain_g: float, gain_b: float,
        multiply_r: float, multiply_g: float, multiply_b: float,
        offset_r: float, offset_g: float, offset_b: float,
        gamma_r: float, gamma_g: float, gamma_b: float,
        reverse: bool,
        black_clamp: bool,
        white_clamp: bool,
        image: Optional[torch.Tensor] = None,
        video=None,
    ):
        if video is not None:
            if image is not None:
                log.warning(
                    "[am_vfx_tools/grade-rgb] both VIDEO and IMAGE inputs wired "
                    "— VIDEO wins; IMAGE ignored"
                )
            wrapped = video_lazy.GradedVideo(
                video,
                blackpoint=(blackpoint_r, blackpoint_g, blackpoint_b),
                whitepoint=(whitepoint_r, whitepoint_g, whitepoint_b),
                lift=(lift_r, lift_g, lift_b),
                gain=(gain_r, gain_g, gain_b),
                multiply=(multiply_r, multiply_g, multiply_b),
                offset=(offset_r, offset_g, offset_b),
                gamma=(gamma_r, gamma_g, gamma_b),
                reverse=reverse,
                black_clamp=black_clamp, white_clamp=white_clamp,
            )
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder, wrapped)

        if image is None:
            log.warning("[am_vfx_tools/grade-rgb] no input wired — passing through")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), None)

        rgb, alpha = _split_alpha(image)
        out_rgb = grade_apply(
            rgb,
            blackpoint=_vec3(blackpoint_r, blackpoint_g, blackpoint_b, rgb),
            whitepoint=_vec3(whitepoint_r, whitepoint_g, whitepoint_b, rgb),
            lift=_vec3(lift_r, lift_g, lift_b, rgb),
            gain=_vec3(gain_r, gain_g, gain_b, rgb),
            multiply=_vec3(multiply_r, multiply_g, multiply_b, rgb),
            offset=_vec3(offset_r, offset_g, offset_b, rgb),
            gamma=_vec3(gamma_r, gamma_g, gamma_b, rgb),
            reverse=reverse,
            black_clamp=black_clamp,
            white_clamp=white_clamp,
        )

        out_image = (
            torch.cat([out_rgb, alpha], dim=-1) if alpha is not None else out_rgb
        )
        return (out_image, _build_video_socket(out_image))


__all__ = ["AMGrade", "AMGradeRGB"]
