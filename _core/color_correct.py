"""am-vfx-tools-media-io._core.color_correct — Color Correct math (pure torch).

Shared by :class:`AMColorCorrect` (image branch) and
:class:`._video_lazy.ColorCorrectedVideo` (lazy video branch).

Pipeline (Nuke ColorCorrect-style order):

    1. offset      : rgb' = rgb + offset
    2. gain        : rgb' = rgb' * gain
    3. gamma       : rgb' = sign(x) * pow(|x|, 1/gamma)
    4. contrast    : rgb' = (rgb' - PIVOT) * contrast + PIVOT
    5. saturation  : rgb' = luma + (rgb' - luma) * sat        (Rec.709 luma)
    6. hue         : rotate around the [1,1,1] gray axis by `hue_degrees`

Contrast pivot is 0.18 — scene-linear mid-gray (18% gray card),
consistent with the rest of this pack's linear-space assumption.

Hue rotation uses the rotation matrix around the gray axis
[1,1,1]/sqrt(3) (Rodrigues), so neutral values map to themselves and
rotating by 360° is the identity.

Sign-preserving pow on the gamma step mirrors the same trick AM Grade
uses — without it, gamma != 1 produces NaN as soon as the channel value
goes negative (common after a negative offset on a scene-linear plate).
"""
from __future__ import annotations

import math

import torch

_EPS = 1e-7
# Scene-linear mid-gray pivot — same value Nuke's ColorCorrect uses for
# contrast in linear workflows. Acts as the fixed point of the contrast
# step: pixels at PIVOT are unchanged regardless of `contrast`.
_PIVOT = 0.18
# Rec.709 luma weights — standard for HD pipelines and what matches our
# OCIO configs' default display chain. Sums to 1.0.
_LUMA_R = 0.2126
_LUMA_G = 0.7152
_LUMA_B = 0.0722


def color_correct_apply(
    rgb: torch.Tensor,
    *,
    saturation: float,
    contrast: float,
    gamma: float,
    gain: float,
    offset: float,
    hue_degrees: float,
) -> torch.Tensor:
    """Apply the Color Correct pipeline to *rgb* (last dim = 3).

    All knobs are scalars and operate uniformly on R/G/B. Returns a new
    tensor of the same shape, dtype and device.
    """
    out = rgb

    if offset != 0.0:
        out = out + offset

    if gain != 1.0:
        out = out * gain

    g = max(float(gamma), _EPS)
    if g != 1.0:
        out = torch.sign(out) * torch.pow(torch.abs(out), 1.0 / g)

    if contrast != 1.0:
        out = (out - _PIVOT) * contrast + _PIVOT

    if saturation != 1.0:
        luma = (
            out[..., 0:1] * _LUMA_R
            + out[..., 1:2] * _LUMA_G
            + out[..., 2:3] * _LUMA_B
        )
        out = luma + (out - luma) * saturation

    if hue_degrees != 0.0:
        out = _hue_rotate(out, hue_degrees)

    return out


def _hue_rotate(rgb: torch.Tensor, hue_degrees: float) -> torch.Tensor:
    """Rotate RGB around the gray axis [1,1,1]/sqrt(3) by *hue_degrees*.

    Neutral (gray) values are preserved; saturated colors cycle through
    the hue wheel. ±180° flip the hue; 360° is the identity.

    Derivation: Rodrigues' rotation formula with the unit axis n =
    (1,1,1)/sqrt(3). Substituting gives a 3x3 matrix where the diagonal
    is c + (1-c)/3 and the off-diagonals are (1-c)/3 ± sqrt(1/3)*s,
    arranged in a circulant pattern so R/G/B map cyclically at 120°.
    """
    theta = math.radians(hue_degrees)
    c = math.cos(theta)
    s = math.sin(theta)
    rt3 = math.sqrt(1.0 / 3.0)
    one_minus_c_over_3 = (1.0 - c) / 3.0
    diag = c + one_minus_c_over_3
    a = one_minus_c_over_3 - rt3 * s
    b = one_minus_c_over_3 + rt3 * s
    matrix = torch.tensor(
        [[diag, a,    b   ],
         [b,    diag, a   ],
         [a,    b,    diag]],
        dtype=rgb.dtype, device=rgb.device,
    )
    # Row-vector convention: out_row = rgb_row @ M.T  ==  M @ rgb_col.
    return rgb @ matrix.T


__all__ = ["color_correct_apply"]
