"""AM Grade / AM Grade RGB — Nuke-style color grade nodes.

Two ComfyUI nodes that mirror the native Nuke Grade node UI:

* :class:`AMGrade`     — one float per knob (blackpoint, whitepoint,
  lift, gain, multiply, offset, gamma).
* :class:`AMGradeRGB`  — per-channel floats (``..._r/_g/_b``); the three
  booleans (reverse, black_clamp, white_clamp) stay scalar.

Both call :func:`._core.grade.grade_apply`.

Code-import baseline: ``custom-nodes/nuke-nodes/grade_nodes.py::NukeGrade``
— but the parameter set, formula, and reverse path are rewritten to
match Nuke's published Grade math (the upstream version uses a
simplified lift/gamma/gain formula and has no blackpoint/whitepoint or
reverse).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from ._core.grade import grade_apply

log = logging.getLogger("am_vfx_tools.media-io.grade")


def _split_alpha(image: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if image.ndim == 3:
        image = image[None, ...]
    if image.shape[-1] >= 4:
        return image[..., :3], image[..., 3:]
    return image, None


def _vec3(r: float, g: float, b: float, ref: torch.Tensor) -> torch.Tensor:
    return torch.tensor([r, g, b], device=ref.device, dtype=ref.dtype)


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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = (
        "Graded image batch (same shape and channels as the input).",
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
    ):
        if image is None:
            log.warning("[am-vfx-tools/grade] `image` input is not wired — passing through")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

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
        return (out_image,)


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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = (
        "Graded image batch (same shape and channels as the input).",
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
    ):
        if image is None:
            log.warning("[am-vfx-tools/grade-rgb] `image` input is not wired — passing through")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

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
        return (out_image,)


__all__ = ["AMGrade", "AMGradeRGB"]
