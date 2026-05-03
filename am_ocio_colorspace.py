"""AM OCIO Colorspace — single-purpose colorspace transform utility.

Mirrors the colorspace mode of Nuke's ``OCIOColorSpace`` node. Takes an
IMAGE batch in *input_colorspace* and converts it to *working_colorspace*
via a single OCIO ColorProcessor — same code path the AM Read / AM Write
nodes use.

This is the colorspace-mode utility; the future AM OCIO Display Transform
will mirror Nuke's display+view two-stage transform_type=display mode (out
of scope for now — see the rework plan §10).

Code-import baseline: ``custom-nodes/nuke-nodes/colorspace_nodes.py::
NukeOCIOColorSpace`` — but the implementation is rewritten on top of our
``_core.color.ColorProcessor`` (in-place buffer transform, no per-frame
config rebuild) and our family-grouped dropdown.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from ._core import color

log = logging.getLogger("am_pipe.media-io.ocio-colorspace")


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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = (
        "Transformed image batch (same shape and channels as the input).",
    )
    FUNCTION = "execute"
    CATEGORY = "AM Pipe/Color"

    def execute(
        self,
        input_colorspace: str,
        working_colorspace: str,
        image: Optional[torch.Tensor] = None,
    ):
        if image is None:
            log.warning("[am_pipe/ocio-cs] `image` input is not wired — passing through")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

        src = color.resolve_choice_to_cs(input_colorspace)
        dst = color.resolve_choice_to_cs(working_colorspace)

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
                                "[am_pipe/ocio-cs] OCIO apply failed on frame %d "
                                "(%s); leaving frame untransformed",
                                i, e,
                            )
                    out_image = torch.from_numpy(out_np).to(image.device)
            except Exception as e:
                log.warning(
                    "[am_pipe/ocio-cs] cannot build %s -> %s (%s); pixels unchanged",
                    src, dst, e,
                )

        return (out_image,)
