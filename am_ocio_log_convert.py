"""AM OCIO Log Convert — Nuke-style OCIO_LogConvert utility.

Mirrors the native Nuke ``OCIO_LogConvert`` node. Uses the active OCIO
config's standard roles:

* ``scene_linear``    — typically ACEScg or Linear Rec.709
* ``compositing_log`` — typically Cineon-like log encoding

Default direction is lin → log (``reverse=False``); ``reverse=True``
gives log → lin. The actual encoding follows whatever the active OCIO
config binds those roles to, so behavior is config-driven (same code
works across Studio, CG, and any custom config that defines the
standard roles).

Code-import baseline: same OCIO processor pattern as
:class:`AMOCIOColorspace`, but with fixed role-based src/dst instead of
a user-pickable colorspace dropdown — see the family-sync rule in
``docs/media-io-sync-rule.md`` for why this node is exempt from the
dropdown / raw_data / default-picker invariants.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from ._core import color

log = logging.getLogger("am_pipe.media-io.ocio-log-convert")

_LINEAR_ROLE = "scene_linear"
_LOG_ROLE = "compositing_log"


class AMOCIOLogConvert:
    """AM OCIO Log Convert — Lin↔Log via OCIO scene_linear/compositing_log roles.

    Default reverse=False = Lin → Log. Input is treated as scene_linear
    (ACEScg in ACES configs); output is the same primaries with a
    Cineon-like log curve.

    Recommended chain for AI inference (squash HDR into 0–1 cleanly):
        AM Read Image (input=ACES2065-1, working=ACEScg)
        AM OCIO Colorspace (ACEScg → Linear Rec.709)   primaries swap
        AM OCIO Log Convert (reverse=False)            log curve
        AM Grade                                       contrast in 0–1
        → inference

    Don't follow Log Convert with `ACEScg → sRGB - Display`: that bakes
    an ACES tone-mapper on top of the log curve. Do the gamut swap with
    `Linear Rec.709 (sRGB)` (a primaries-only transform) BEFORE the log
    convert, then add Log + Grade for the encoding curve.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reverse": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Off (default): scene_linear → compositing_log. "
                        "On: compositing_log → scene_linear. "
                        "The exact curve is whatever the active OCIO config binds "
                        "those roles to — config-driven, same code works across "
                        "Studio / CG / custom configs."
                    ),
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": (
                        "Image batch to transform. Default direction expects "
                        "linear pixels in the active OCIO config's "
                        "`scene_linear` role — typically ACEScg in ACES "
                        "configs, Linear Rec.709 in legacy configs. Reverse "
                        "direction expects the `compositing_log` role "
                        "(Cineon-like)."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = (
        "Log/Lin-converted IMAGE — same primaries as the input, different encoding "
        "curve (linear ↔ Cineon-like log per the active OCIO config).",
    )
    FUNCTION = "execute"
    CATEGORY = "AM Pipe/Color"

    def execute(
        self,
        reverse: bool,
        image: Optional[torch.Tensor] = None,
    ):
        if image is None:
            log.warning("[am_pipe/ocio-logconv] `image` input is not wired — passing through")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

        src = _LOG_ROLE if reverse else _LINEAR_ROLE
        dst = _LINEAR_ROLE if reverse else _LOG_ROLE

        out_image = image
        try:
            proc = color.ColorProcessor(src, dst)
        except Exception as e:
            log.warning(
                "[am_pipe/ocio-logconv] cannot build %s -> %s (%s); "
                "pixels unchanged",
                src, dst, e,
            )
            proc = None

        if proc is not None and not proc.is_identity:
            batch = image
            if batch.ndim == 3:
                batch = batch[None, ...]
            out_np = batch.detach().cpu().numpy().astype(np.float32, copy=True)
            for i in range(out_np.shape[0]):
                try:
                    proc.apply_inplace(out_np[i])
                except Exception as e:
                    log.warning(
                        "[am_pipe/ocio-logconv] OCIO apply failed on frame %d "
                        "(%s); leaving frame untransformed",
                        i, e,
                    )
            out_image = torch.from_numpy(out_np).to(image.device)

        return (out_image,)


__all__ = ["AMOCIOLogConvert"]
