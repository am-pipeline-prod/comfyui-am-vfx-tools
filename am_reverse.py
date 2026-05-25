"""AM Reverse Sequence — frame-order reverse on IMAGE / VIDEO.

Mirrors KJNodes' ``Reverse Image Batch`` but with native VIDEO socket
support so the reversed sequence can feed a lazy AM transform chain
downstream without further materialisations.

Also accepts an optional ``video`` input. **Important caveat** — reverse
is a random-access operation on the time dimension; PyAV can't decode
in reverse without keyframe seek. The VIDEO branch therefore
**materialises** the source via ``video.get_components()``, flips the
in-memory tensors, and returns a new ``VideoFromComponents``. Peak RAM
during the flip is ~2× source. The lazy win this node provides is for
the chain *after* the reverse — downstream AM transforms can consume
the wrapped output without each one allocating a full batch.

For very long / high-resolution sequences where 2× peak doesn't fit in
RAM, reverse BEFORE the upscale (in the smaller resolution) rather than
after.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from ._core import preview

# Native ComfyUI VIDEO type — see docs/media-io-sync-rule.md invariant 28.
try:
    from comfy_api.v0_0_2 import InputImpl as _ComfyInputImpl  # type: ignore[import-not-found]
    from comfy_api.v0_0_2 import Types as _ComfyTypes  # type: ignore[import-not-found]
    _VIDEO_TYPE_AVAILABLE = True
except ImportError:
    _ComfyInputImpl = None  # type: ignore[assignment]
    _ComfyTypes = None  # type: ignore[assignment]
    _VIDEO_TYPE_AVAILABLE = False

from fractions import Fraction as _Fraction

log = logging.getLogger("am_vfx_tools.media-io.reverse")


_VIDEO_IN_TOOLTIP = (
    "Optional VIDEO input. When wired, this node materialises the "
    "source via `get_components()` to access frames in random order, "
    "flips the in-memory tensors, and returns a fresh "
    "`VideoFromComponents`. Peak RAM during the flip is ~2× source — "
    "reverse can't be lazy because PyAV doesn't decode backward. The "
    "wrapped output IS consumable by downstream lazy AM chains. "
    "`image` and `mask` are ignored when `video` is wired. See "
    "invariant 28."
)
_VIDEO_OUT_TOOLTIP = (
    "Lazy-friendly VIDEO output — wraps the reversed IMAGE batch in a "
    "zero-copy `VideoFromComponents`. Downstream AM transforms can "
    "consume this lazily (one frame at a time) without further "
    "materialisations. None when no input is wired."
)


class AMReverseSequence:
    """ComfyUI node — reverse the frame order of an IMAGE batch / VIDEO."""

    DESCRIPTION = (
        "Reverse the frame order of an IMAGE batch or VIDEO. Drop-in "
        "replacement for KJNodes' Reverse Image Batch, with an added "
        "VIDEO output socket so the reversed sequence can feed a lazy "
        "AM transform chain (Reformat / Grade / OCIO / Image Write) "
        "without materialising the batch again at each node."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "show_preview": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Show a thumbnail of the FIRST frame of the "
                        "reversed sequence (= LAST frame of the input) "
                        "on the node — visual confirmation the reverse "
                        "fired."
                    ),
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": (
                        "Image batch to reverse along the frame "
                        "dimension. Cheap — `torch.flip` per the time "
                        "axis. Peak RAM during the flip is 2× the batch."
                    ),
                }),
                "mask": ("MASK", {
                    "tooltip": (
                        "Optional MASK batch to reverse alongside the "
                        "IMAGE. If unwired and IMAGE has no alpha, the "
                        "output MASK is zeros (stock-ComfyUI 'nothing "
                        "to inpaint' convention)."
                    ),
                }),
                "video": ("VIDEO", {"tooltip": _VIDEO_IN_TOOLTIP}),
            },
        }

    # Output socket order — image, mask, video. Matches the
    # IMAGE/MASK pair convention from the AM IO family + the appended
    # lazy VIDEO output added by invariant 28.
    RETURN_TYPES = ("IMAGE", "MASK", "VIDEO")
    RETURN_NAMES = ("image", "mask", "video")
    OUTPUT_TOOLTIPS = (
        "Reversed image batch (same shape / dtype as input). RGB only.",
        "Reversed mask batch (same shape / dtype as input mask). Zeros "
        "when no mask was wired and the source had no alpha.",
        _VIDEO_OUT_TOOLTIP,
    )
    FUNCTION = "execute"
    CATEGORY = "AM VFX Tools"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def execute(
        self,
        show_preview: bool = True,
        image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        video=None,
    ):
        # VIDEO branch — materialise via get_components, flip, re-wrap.
        # The reverse itself is unavoidably 2× peak (random access).
        if video is not None:
            if image is not None or mask is not None:
                log.warning(
                    "[am_vfx_tools/reverse] both VIDEO and IMAGE/MASK inputs "
                    "wired — VIDEO wins; IMAGE/MASK ignored"
                )
            return self._execute_video_branch(video, show_preview)

        if image is None:
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return {
                "ui": {"text": ["(no input)"]},
                "result": (empty, empty_mask, None),
            }

        # IMAGE branch — torch.flip along the batch axis. Cheap (2× peak
        # briefly, no per-pixel work).
        if image.ndim == 3:
            image = image[None, ...]
        reversed_image = torch.flip(image, dims=[0])

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, ...]
            reversed_mask = torch.flip(mask, dims=[0])
        else:
            n, h, w = (
                int(reversed_image.shape[0]),
                int(reversed_image.shape[1]),
                int(reversed_image.shape[2]),
            )
            reversed_mask = torch.zeros((n, h, w), dtype=torch.float32)

        video_socket = self._build_video_socket(reversed_image, reversed_mask)
        ui = self._ui_payload(reversed_image, show_preview)
        return {
            "ui": ui,
            "result": (reversed_image, reversed_mask, video_socket),
        }

    # ------------------------------------------------------------------ #
    #  VIDEO branch — materialise + flip + wrap.
    # ------------------------------------------------------------------ #

    def _execute_video_branch(self, video, show_preview: bool):
        if not _VIDEO_TYPE_AVAILABLE:
            log.warning(
                "[am_vfx_tools/reverse] VIDEO input wired but comfy_api.v0_0_2 "
                "is not importable — falling back to no-op"
            )
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return {
                "ui": {"text": ["(comfy_api unavailable)"]},
                "result": (empty, empty_mask, None),
            }

        try:
            components = video.get_components()
        except Exception as e:  # noqa: BLE001
            log.exception("[am_vfx_tools/reverse] get_components failed: %s", e)
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return {
                "ui": {"text": ["(materialise failed)"]},
                "result": (empty, empty_mask, None),
            }

        images = components.images
        alpha = components.alpha
        if images is None or int(images.shape[0]) == 0:
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return {
                "ui": {"text": ["(empty video)"]},
                "result": (empty, empty_mask, None),
            }

        reversed_images = torch.flip(images, dims=[0])
        reversed_alpha = (
            torch.flip(alpha, dims=[0]) if alpha is not None else None
        )

        # Build the IMAGE + MASK output sockets. MASK uses the stock
        # ComfyUI convention `mask = 1 - alpha`.
        out_image = reversed_images
        if reversed_alpha is not None:
            out_mask = (1.0 - reversed_alpha).clamp(0.0, 1.0)
        else:
            n, h, w = (
                int(reversed_images.shape[0]),
                int(reversed_images.shape[1]),
                int(reversed_images.shape[2]),
            )
            out_mask = torch.zeros((n, h, w), dtype=torch.float32)

        # Build the VIDEO output — new VideoFromComponents wrapping the
        # reversed batches. Preserves frame_rate + audio + metadata
        # from the source.
        new_components = _ComfyTypes.VideoComponents(
            images=reversed_images,
            alpha=reversed_alpha,
            frame_rate=components.frame_rate,
            audio=components.audio,
            metadata=components.metadata,
        )
        video_socket = _ComfyInputImpl.VideoFromComponents(new_components)

        ui = self._ui_payload(out_image, show_preview)
        return {
            "ui": ui,
            "result": (out_image, out_mask, video_socket),
        }

    # ------------------------------------------------------------------ #
    #  VIDEO output socket (IMAGE branch).
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_video_socket(
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Wrap the reversed IMAGE batch in a VideoFromComponents.

        When *mask* is provided, encodes it as alpha (alpha = 1 - mask)
        so downstream lazy AM transforms carry alpha natively. Returns
        None when comfy_api is unavailable.
        """
        if not _VIDEO_TYPE_AVAILABLE or images is None:
            return None
        try:
            alpha = None
            if mask is not None:
                try:
                    alpha = (1.0 - mask).clamp(0.0, 1.0)
                except Exception:  # noqa: BLE001
                    alpha = None
            # No fps context on this node — default to 25 (matches AM
            # Reformat / Grade convention).
            rate = _Fraction(25, 1)
            return _ComfyInputImpl.VideoFromComponents(
                _ComfyTypes.VideoComponents(
                    images=images, alpha=alpha, frame_rate=rate,
                ),
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[am_vfx_tools/reverse] VIDEO socket build failed (%s); None", e,
            )
            return None

    # ------------------------------------------------------------------ #
    #  UI preview helper.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ui_payload(image: torch.Tensor, show_preview: bool):
        if not show_preview:
            return {"text": [f"reversed {int(image.shape[0])} frames"]}
        try:
            payload = preview.create_single_preview(
                image, frame_index=0,
                working_colorspace="sRGB - Display",
                filename_hint="reverse",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("[am_vfx_tools/reverse] preview generation failed: %s", e)
            return {"text": [f"reversed {int(image.shape[0])} frames"]}
        if not payload.get("images"):
            return {"text": [f"reversed {int(image.shape[0])} frames"]}
        return payload


__all__ = ["AMReverseSequence"]
