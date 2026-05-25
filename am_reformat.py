"""AM Reformat — ComfyUI node.

Standalone Nuke-flavored reformat / dtype-cast utility. Identical
widget block + defaults to the reformat section embedded in AM
Read/Write nodes — drop this node anywhere in a graph to apply the
same transform on a wired IMAGE or VIDEO without touching another
node's settings.

Pure geometry + dtype: no OCIO, no path resolution, no batch/seed.
The shared implementation lives in :mod:`._core.reformat`.

VIDEO input — see docs/media-io-sync-rule.md invariant 28. When the
`video` socket is wired, this node emits a :class:`._core.video_lazy.ReformatVideo`
wrapper that defers the actual resize until a downstream consumer
iterates frames. Multiple AM transforms can chain via VIDEO without
each materialising the IMAGE batch.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

from ._core import preview, reformat
from ._core import video_lazy

# Native ComfyUI VIDEO type — see docs/media-io-sync-rule.md invariant 28.
try:
    from comfy_api.v0_0_2 import InputImpl as _ComfyInputImpl  # type: ignore[import-not-found]
    _VIDEO_TYPE_AVAILABLE = True
except ImportError:
    _ComfyInputImpl = None  # type: ignore[assignment]
    _VIDEO_TYPE_AVAILABLE = False

log = logging.getLogger("am_vfx_tools.media-io.reformat-node")


class AMReformat:
    """ComfyUI node — Nuke-style reformat + optional fp32→fp16 cast."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reformat_mode": (reformat.REFORMAT_MODES, {
                    "default": reformat.MODE_OFF,
                    "tooltip": reformat.TOOLTIP_MODE,
                }),
                "scale": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01,
                    "tooltip": reformat.TOOLTIP_SCALE,
                }),
                "preset": (reformat.PRESET_CHOICES, {
                    "default": reformat.PRESET_WH,
                    "tooltip": reformat.TOOLTIP_PRESET,
                }),
                "target_width": ("INT", {
                    "default": 1920, "min": 1, "max": 16384,
                    "tooltip": reformat.TOOLTIP_TARGET_W,
                }),
                "target_height": ("INT", {
                    "default": 1080, "min": 1, "max": 16384,
                    "tooltip": reformat.TOOLTIP_TARGET_H,
                }),
                "resize_type": (reformat.RESIZE_CHOICES, {
                    "default": reformat.RESIZE_FIT,
                    "tooltip": reformat.TOOLTIP_RESIZE_TYPE,
                }),
                "filter": (reformat.FILTER_CHOICES, {
                    "default": reformat.FILTER_CUBIC,
                    "tooltip": reformat.TOOLTIP_FILTER,
                }),
                "output_dtype": (reformat.DTYPE_CHOICES, {
                    "default": reformat.DEFAULT_DTYPE,
                    "tooltip": reformat.TOOLTIP_DTYPE,
                }),
                "show_preview": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show a thumbnail of the reformatted result on the node.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Image batch to reformat (N×H×W×C float).",
                }),
                "mask": ("MASK", {
                    "tooltip": reformat.TOOLTIP_MASK_IN_REFORMAT,
                }),
                # VIDEO input — invariant 28. Appended after image/mask.
                "video": ("VIDEO", {
                    "tooltip": (
                        "Optional VIDEO input. When wired, returns a lazy "
                        "`ReformatVideo` wrapper applying the resize "
                        "per-frame on consumption — no IMAGE materialisation "
                        "here. Alpha (when present) is resized alongside "
                        "the image. `image` and `mask` are ignored when "
                        "`video` is wired. See invariant 28."
                    ),
                }),
            },
        }

    # Output order — `video` appended (invariant 28).
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING", "VIDEO")
    RETURN_NAMES = ("image", "mask", "width", "height", "info", "video")
    OUTPUT_TOOLTIPS = (
        "Reformatted image batch (N×H×W×3, dtype per `output_dtype`).",
        reformat.TOOLTIP_MASK_OUT_REFORMAT,
        "Output width in pixels (post-reformat).",
        "Output height in pixels (post-reformat).",
        "One-line summary of the reformat applied.",
        "Lazy VIDEO output — emits a `ReformatVideo` wrapper when `video` "
        "is wired, else a zero-copy `VideoFromComponents` around the IMAGE "
        "batch. None when no input is wired.",
    )
    FUNCTION = "execute"
    CATEGORY = "AM VFX Tools"
    OUTPUT_NODE = True

    def execute(
        self,
        reformat_mode: str,
        scale: float,
        preset: str,
        target_width: int,
        target_height: int,
        resize_type: str,
        filter: str,
        output_dtype: str,
        show_preview: bool = True,
        image=None,
        mask=None,
        video=None,
    ):
        # VIDEO branch — wrap the source in a lazy ReformatVideo and emit.
        # IMAGE/MASK outputs are placeholders since no materialisation
        # happens here. See docs/media-io-sync-rule.md invariant 28.
        if video is not None:
            if image is not None or mask is not None:
                log.warning(
                    "[am_vfx_tools/reformat] both VIDEO and IMAGE/MASK inputs "
                    "wired — VIDEO wins; IMAGE/MASK ignored"
                )
            return self._execute_video_branch(
                video=video,
                reformat_mode=reformat_mode, scale=scale, preset=preset,
                target_width=target_width, target_height=target_height,
                resize_type=resize_type, filter_name=filter,
                show_preview=show_preview,
            )

        if image is None:
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            # Empty MASK = zeros (stock ComfyUI: nothing to inpaint).
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return {
                "ui": {"text": ["(no input)"]},
                "result": (empty, empty_mask, 64, 64, "(no input)", None),
            }

        # Pull to numpy fp32 for the cv2-backed helper. Tensor → contiguous
        # numpy is a single host-side copy; the downstream cast back to
        # torch is a zero-copy view when dtypes match.
        arr = image.detach().cpu().numpy()
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)

        src_h = int(arr.shape[1]) if arr.ndim == 4 else int(arr.shape[0])
        src_w = int(arr.shape[2]) if arr.ndim == 4 else int(arr.shape[1])

        # Combine MASK into alpha BEFORE reformat so geometry applies
        # uniformly to RGB+alpha. If MASK isn't wired but the input
        # IMAGE is already 4-channel, that embedded alpha rides through
        # the helper as-is. If neither: helper sees 3-channel input and
        # split_image_mask emits a zero MASK at the end.
        if mask is not None:
            try:
                mask_arr = mask.detach().cpu().numpy().astype(np.float32, copy=False)
                arr = reformat.combine_image_mask(arr, mask_arr)
            except Exception as e:
                log.warning("[am_vfx_tools/reformat] mask combine failed: %s; ignoring mask", e)

        try:
            out_arr = reformat.reformat_array(
                arr,
                mode=reformat_mode,
                scale=float(scale),
                preset=preset,
                target_w=int(target_width),
                target_h=int(target_height),
                resize_type=resize_type,
                filter_name=filter,
                output_dtype=output_dtype,
            )
        except Exception as e:
            log.warning("[am_vfx_tools/reformat] reformat failed: %s; passing input through", e)
            out_arr = arr

        # Split post-reformat into IMAGE (RGB) + MASK (1 - alpha) for the
        # output sockets — symmetric with the Read nodes.
        image_part, mask_part = reformat.split_image_mask(out_arr)
        out_tensor = torch.from_numpy(np.ascontiguousarray(image_part))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_part))
        out_h = int(out_tensor.shape[1])
        out_w = int(out_tensor.shape[2])

        info_str = reformat.info_fragment(
            mode=reformat_mode,
            scale=float(scale),
            preset=preset,
            target_w=int(target_width),
            target_h=int(target_height),
            resize_type=resize_type,
            filter_name=filter,
            output_dtype=output_dtype,
            src_w=src_w, src_h=src_h,
        )
        if not info_str:
            info_str = f"{out_w}x{out_h} (no-op)"

        ui = self._ui_payload(out_tensor, show_preview, info_str)
        # VIDEO output — wrap the reformatted IMAGE batch in a
        # VideoFromComponents (zero-copy convenience socket so downstream
        # nodes can take VIDEO without an intermediate Create Video).
        video_socket = self._build_video_socket(out_tensor)
        return {
            "ui": ui,
            "result": (
                out_tensor, mask_tensor, int(out_w), int(out_h),
                info_str, video_socket,
            ),
        }

    # ------------------------------------------------------------------ #
    #  VIDEO branch — wrap the source in a lazy ReformatVideo.
    # ------------------------------------------------------------------ #

    def _execute_video_branch(
        self, *, video, reformat_mode, scale, preset, target_width,
        target_height, resize_type, filter_name, show_preview,
    ):
        wrapped = video_lazy.ReformatVideo(
            video,
            mode=reformat_mode,
            scale=float(scale),
            preset=preset,
            target_w=int(target_width),
            target_h=int(target_height),
            resize_type=resize_type,
            filter_name=filter_name,
        )
        # Compute output dimensions from the source via the lazy wrapper
        # for the metadata sockets (cheap — just inspects source headers).
        try:
            out_w, out_h = wrapped.get_dimensions()
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[am_vfx_tools/reformat] VIDEO dimension probe failed (%s)", e,
            )
            out_w, out_h = 0, 0
        info_str = (
            f"{int(out_w)}x{int(out_h)} VIDEO-lazy "
            f"({reformat_mode}/{resize_type}/{filter_name})"
        )
        # Placeholder IMAGE/MASK — the artist wired VIDEO so no
        # materialisation here. Downstream AM consumer iterates the
        # lazy wrapper to actually run the resize.
        placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        return {
            "ui": {"text": [info_str]},
            "result": (
                placeholder, empty_mask, int(out_w), int(out_h),
                info_str, wrapped,
            ),
        }

    @staticmethod
    def _build_video_socket(images):
        """Wrap the reformatted IMAGE batch in a VideoFromComponents.

        Zero-copy convenience socket — same as AM Read Image's pattern.
        Returns None when comfy_api is unavailable or images is empty.
        """
        if not _VIDEO_TYPE_AVAILABLE or images is None:
            return None
        try:
            from fractions import Fraction as _Fraction
            from comfy_api.v0_0_2 import Types as _ComfyTypes  # local import
            # No fps context here — use 25 as a sane default; the node
            # is geometry-only so downstream consumers should override
            # if they care about timing.
            return _ComfyInputImpl.VideoFromComponents(
                _ComfyTypes.VideoComponents(
                    images=images, frame_rate=_Fraction(25, 1),
                ),
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[am_vfx_tools/reformat] VIDEO socket build failed (%s); None", e,
            )
            return None

    @staticmethod
    def _ui_payload(tensor, show_preview: bool, info_str: str):
        if not show_preview:
            return {"text": [info_str]}
        try:
            payload = preview.create_single_preview(
                tensor,
                frame_index=0,
                # No working colorspace context here — preview falls back
                # to the tensor as-given (already display-referred for
                # most reformat-only graphs). Matches ImageScale's
                # behavior in stock ComfyUI.
                working_colorspace="sRGB - Display",
                filename_hint="reformat",
            )
        except Exception as e:
            log.warning("[am_vfx_tools/reformat] preview generation failed: %s", e)
            return {"text": [info_str]}
        if not payload.get("images"):
            return {"text": [info_str]}
        return payload
