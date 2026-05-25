"""am-vfx-tools-media-io._core.grade — Nuke Grade math (pure torch).

Shared by :class:`AMGrade` (single float per knob) and :class:`AMGradeRGB`
(per-channel floats). Both call :func:`grade_apply` with parameter
tensors of shape ``(3,)``; the mono node simply broadcasts a single value
to all three channels.

Formula (matches native Nuke Grade):

    A = multiply * (gain - lift) / (whitepoint - blackpoint)
    B = offset + lift - A * blackpoint
    forward:  out = sign(A*in + B) * pow(|A*in + B|, 1/gamma)
    reverse:  out = (sign(in) * pow(|in|, gamma) - B) / A

The sign-preserving pow mirrors Nuke's behavior on negative intermediate
values — without it, gamma != 1 produces NaN as soon as ``A*in + B``
goes negative (common with non-default lift/offset/blackpoint).

Round-trip identity: forward then reverse with the same parameters
returns the input bit-exact for default params and within float pow
precision (~1e-5) for non-default params, EXCEPT where ``black_clamp``
on the forward pass clipped negatives to 0 (not analytically
recoverable; same caveat as Nuke).
"""
from __future__ import annotations

import torch

_EPS = 1e-7


def grade_apply(
    rgb: torch.Tensor,
    blackpoint: torch.Tensor,
    whitepoint: torch.Tensor,
    lift: torch.Tensor,
    gain: torch.Tensor,
    multiply: torch.Tensor,
    offset: torch.Tensor,
    gamma: torch.Tensor,
    *,
    reverse: bool,
    black_clamp: bool,
    white_clamp: bool,
) -> torch.Tensor:
    """Apply the Nuke Grade transform to *rgb* (shape ``(..., 3)``).

    All parameter tensors must be broadcast-compatible with the last dim
    of *rgb* — typically shape ``(3,)``.
    """
    wp_minus_bp = torch.clamp(whitepoint - blackpoint, min=_EPS)
    A = multiply * (gain - lift) / wp_minus_bp
    B = offset + lift - A * blackpoint

    g = torch.clamp(gamma, min=_EPS)

    if reverse:
        powed = torch.sign(rgb) * torch.pow(torch.abs(rgb), g)
        # If gain == lift, A collapses to 0 and the forward is a constant —
        # no inverse exists. Clamp |A| away from 0 so we return a finite
        # (meaningless) value instead of inf/nan.
        A_safe = torch.where(torch.abs(A) < _EPS, torch.full_like(A, _EPS), A)
        out = (powed - B) / A_safe
    else:
        inner = A * rgb + B
        out = torch.sign(inner) * torch.pow(torch.abs(inner), 1.0 / g)

    if black_clamp:
        out = torch.clamp(out, min=0.0)
    if white_clamp:
        out = torch.clamp(out, max=1.0)

    return out


__all__ = ["grade_apply"]
