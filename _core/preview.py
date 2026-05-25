"""am-vfx-tools-media-io._core.preview — single-frame thumbnail to ComfyUI temp/.

Used by AM Read Image / Write Image / Read Video / Write Video when their
``show_preview`` widget is on. Writes one 256-px sRGB-encoded PNG into
ComfyUI's temp directory and returns a UI dict shaped for the standard
``{"ui": {"images": [...]}}`` channel.

Single-frame, not sequence — the caller picks the frame index that
matches the user's intent (Read Image: ``frame - first_loaded``;
Read Video / Write Image / Write Video: 0).

If the tensor is in a working colorspace that's not display-referred sRGB,
an OCIO transform is applied so the thumbnail looks right in the browser.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from . import color as _color

log = logging.getLogger("am_vfx_tools.media-io.preview")

_THUMB_MAX_EDGE = 256
_TARGET_DISPLAY = "sRGB - Display"


def _temp_dir() -> str:
    """Return ComfyUI's temp directory (created on demand)."""
    try:
        from folder_paths import get_temp_directory  # type: ignore
        d = get_temp_directory()
    except Exception:
        # Fallback — module load order or running outside ComfyUI.
        d = os.path.join(os.getcwd(), "temp")
    os.makedirs(d, exist_ok=True)
    return d


def _resize_to_thumb(rgb: np.ndarray) -> np.ndarray:
    """Downscale (H, W, 3) to a max edge of _THUMB_MAX_EDGE via cv2 area filter."""
    h, w = rgb.shape[:2]
    long = max(h, w)
    if long <= _THUMB_MAX_EDGE:
        return rgb
    scale = _THUMB_MAX_EDGE / long
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    try:
        import cv2  # type: ignore
        return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except ImportError:
        # Naive box downsample — slow but correct enough for thumbnails.
        ys = (np.linspace(0, h - 1, new_h)).astype(np.int32)
        xs = (np.linspace(0, w - 1, new_w)).astype(np.int32)
        return rgb[np.ix_(ys, xs)]


def _to_display_referred_uint8(
    pixels: np.ndarray, working_colorspace: str
) -> np.ndarray:
    """Convert (H, W, 3..4) float to sRGB-Display uint8 ready for PNG."""
    src = _color.resolve_choice_to_cs(working_colorspace)

    # Drop alpha for the thumbnail; PNG doesn't need it for a preview.
    if pixels.ndim == 3 and pixels.shape[-1] >= 4:
        pixels = pixels[..., :3]
    elif pixels.ndim == 3 and pixels.shape[-1] == 1:
        pixels = np.repeat(pixels, 3, axis=-1)
    elif pixels.ndim == 2:
        pixels = np.repeat(pixels[..., None], 3, axis=-1)

    pixels = np.ascontiguousarray(pixels.astype(np.float32, copy=True))

    # Apply OCIO if working_cs is not already sRGB-Display.
    if (
        src != _color.PASSTHROUGH
        and src != _TARGET_DISPLAY
        and _color.is_available()
    ):
        try:
            proc = _color.ColorProcessor(src, _TARGET_DISPLAY)
            if not proc.is_identity:
                proc.apply_inplace(pixels)
        except Exception as e:
            log.warning(
                "[am_vfx_tools/preview] OCIO %s -> %s failed (%s); thumbnail "
                "may look wrong",
                src, _TARGET_DISPLAY, e,
            )

    pixels = _resize_to_thumb(pixels)
    pixels = np.clip(pixels, 0.0, 1.0)
    return (pixels * 255.0 + 0.5).astype(np.uint8)


def create_single_preview(
    tensor,
    frame_index: int = 0,
    *,
    working_colorspace: str = "sRGB - Display",
    filename_hint: str = "",
) -> Dict[str, Any]:
    """Write a 256-px sRGB PNG thumbnail of ``tensor[frame_index]`` and return
    the ComfyUI UI-channel dict for it.

    *tensor* is a torch tensor or numpy array shaped ``(N, H, W, C)`` or
    ``(H, W, C)``. *working_colorspace* is the colorspace the pixels are
    in (so the thumbnail can be transformed into sRGB-Display for the
    browser). *filename_hint* — if given, used to seed a stable hash so
    re-rendering the same source yields the same temp filename (browser
    cache-friendly).

    Returns ``{"images": [{"filename": ..., "subfolder": "", "type":
    "temp"}]}`` — caller wraps with ``{"ui": <this>, "result": ...}``.
    """
    try:
        import torch  # type: ignore
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(tensor)
    except ImportError:
        arr = np.asarray(tensor)

    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        log.warning("[am_vfx_tools/preview] unexpected tensor shape %s", arr.shape)
        return {"images": []}

    n = arr.shape[0]
    if n == 0:
        return {"images": []}

    idx = int(np.clip(frame_index, 0, n - 1))
    frame = arr[idx]

    rgb_u8 = _to_display_referred_uint8(frame, working_colorspace)

    # Stable-ish filename: hash of (hint, frame index, frame bytes).
    sha = hashlib.sha1()
    sha.update(filename_hint.encode("utf-8", errors="replace"))
    sha.update(str(idx).encode("ascii"))
    sha.update(rgb_u8.tobytes())
    digest = sha.hexdigest()[:12]
    filename = f"am_vfx_tools_preview_{digest}.png"

    out_dir = _temp_dir()
    out_path = os.path.join(out_dir, filename)

    try:
        # Use OIIO so we don't add a Pillow dependency for what's already
        # in the venv. Tag as sRGB so any downstream OCIO-aware viewer
        # treats it correctly.
        from . import image_backend as _img
        _img.write_image(
            out_path, rgb_u8.astype(np.float32) / 255.0,
            bit_depth="uint8",
            compression=None,
            color_space_tag="sRGB - Display",
            create_directories=True,
        )
    except Exception as e:
        log.warning("[am_vfx_tools/preview] OIIO write failed (%s); trying PIL", e)
        try:
            from PIL import Image  # type: ignore
            Image.fromarray(rgb_u8, mode="RGB").save(out_path, format="PNG")
        except Exception as e2:
            log.error("[am_vfx_tools/preview] PIL write also failed: %s", e2)
            return {"images": []}

    return {
        "images": [
            {"filename": filename, "subfolder": "", "type": "temp"},
        ],
    }


__all__ = ["create_single_preview"]
