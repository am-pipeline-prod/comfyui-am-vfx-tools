"""am-pipe-media-io._core.reformat — shared reformat / dtype-cast helper.

Used by the standalone ``AMReformat`` node and embedded in the four
media-IO nodes (AM Read Image / Read Video / Write Image / Write Video)
to provide identical Nuke-flavored reformat semantics.

Single public entry point: :func:`reformat_array`. Accepts either a
single frame ``(H, W, C)`` or a batch ``(N, H, W, C)`` numpy array; the
batch path internally loops per-frame because cv2 has no batched API.

Design choices (see media-io-sync-rule.md invariants 15a–15e):

* **Backend = cv2.** Pixels live as numpy fp32 at every call site
  (post-decode for reads, pre-encode for writes); cv2.resize is the
  fastest correct option and supports all five filters in our enum
  with one mapping. No torch round-trip, no GPU upload — keeps the
  resize on the CPU path the rest of the IO pipeline runs on.
* **OCIO stays fp32.** The optional fp32 → fp16 down-cast happens at
  the very end of this helper, after every geometric op. Callers wire
  OCIO before the reformat call so the dtype cast never lands inside
  a color-managed step.
* **Centering is implicit.** Nuke's `center` checkbox is dropped — we
  always center on `fit` (letterbox) and `none` (crop/pad).
* **No flip / flop / turn / clamp / black-outside / preserve-bbox.**
  These are dropped per the design discussion; reformat is a
  geometric utility, not a full Nuke Reformat replacement.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger("am_vfx_tools.media-io.reformat")


# ---------------------------------------------------------------------------
#  Public widget enums — kept in sync across every call site so the AM IO
#  nodes and the standalone node can reference one canonical value list.
# ---------------------------------------------------------------------------

MODE_OFF    = "off"
MODE_SCALE  = "scale"
MODE_TO_BOX = "to_box"
REFORMAT_MODES = [MODE_OFF, MODE_SCALE, MODE_TO_BOX]

PRESET_WH = "Width/Height"
# Order matters — the first entry is "use the explicit width/height widgets",
# everything below is a named preset that overrides them. Resolutions chosen
# to cover the common AI-inference + delivery formats:
PRESETS: Dict[str, Tuple[int, int]] = {
    PRESET_WH:                       (0, 0),       # sentinel — read from widgets
    "HD 1920x1080":                  (1920, 1080),
    "2K DCI 2048x1080":              (2048, 1080),
    "2K Academy 2048x1556":          (2048, 1556),
    # Same pixel count as 2K Academy but a distinct Nuke-named format —
    # full Super 35 negative scan (no academy crop). Listed alongside
    # Academy so artists see both labels and pick the one matching their
    # source. Mirrors Nuke's `2K_Super_35(full-ap)` builtin format.
    "2K Super 35 (full-ap) 2048x1556": (2048, 1556),
    "UHD 3840x2160":                 (3840, 2160),
    "4K DCI 4096x2160":              (4096, 2160),
    "720p 1280x720":                 (1280, 720),
    "540p 960x540":                  (960, 540),
    "Square 512x512":                (512, 512),
    "Square 1024x1024":              (1024, 1024),
}
PRESET_CHOICES = list(PRESETS.keys())

# Nuke-named filters. Mapped to cv2 INTER_* enums in :func:`_filter_to_cv2`.
# `area` is not a Nuke filter — added because it's the cv2-canonical pick
# for downscaling (anti-aliased mean pooling).
FILTER_IMPULSE  = "impulse"
FILTER_LINEAR   = "linear"
FILTER_CUBIC    = "cubic"
FILTER_LANCZOS4 = "Lanczos4"
FILTER_AREA     = "area"
FILTER_CHOICES = [
    FILTER_IMPULSE, FILTER_LINEAR, FILTER_CUBIC, FILTER_LANCZOS4, FILTER_AREA,
]

# Nuke `resize_type` values verbatim. `none` means "crop/pad without scale".
RESIZE_WIDTH   = "width"
RESIZE_HEIGHT  = "height"
RESIZE_FIT     = "fit"
RESIZE_FILL    = "fill"
RESIZE_DISTORT = "distort"
RESIZE_NONE    = "none"
RESIZE_CHOICES = [
    RESIZE_WIDTH, RESIZE_HEIGHT, RESIZE_FIT, RESIZE_FILL, RESIZE_DISTORT, RESIZE_NONE,
]

DTYPE_FP32 = "fp32"
DTYPE_FP16 = "fp16"
DTYPE_CHOICES = [DTYPE_FP32, DTYPE_FP16]
# Default applied across every node's `output_dtype` widget. Centralized
# here so a flip-back is one edit. Currently fp32 — reverted from fp16
# 2026-05-01 after running into downstream nodes that assume fp32 IMAGE
# tensors. Feature stays intact; artists who want the memory win flip
# the per-node widget to fp16 explicitly.
DEFAULT_DTYPE = DTYPE_FP32


# ---------------------------------------------------------------------------
#  Tooltips — exposed so each node's INPUT_TYPES can reference one canonical
#  string per widget. Keeps the eight tooltips identical across all five
#  call sites without a copy-paste hazard.
# ---------------------------------------------------------------------------

TOOLTIP_MODE = (
    "Reformat mode. "
    "off = bypass, output matches input. "
    "scale = uniform scale by `scale` (other widgets ignored). "
    "to_box = resize/crop to a target W×H from `preset` or `target_width`/`target_height`."
)
TOOLTIP_SCALE = (
    "Uniform scale factor. Used when `reformat_mode=scale`; ignored otherwise. "
    "Output dimensions are round(input × scale)."
)
TOOLTIP_PRESET = (
    "Named output format. Used when `reformat_mode=to_box`. "
    "`Width/Height` = use the `target_width` / `target_height` widgets below. "
    "Any other entry overrides those widgets with the preset's resolution."
)
TOOLTIP_TARGET_W = (
    "Target output width in pixels. Used when `reformat_mode=to_box` AND "
    "`preset=Width/Height`; ignored when a named preset is selected."
)
TOOLTIP_TARGET_H = (
    "Target output height in pixels. Used when `reformat_mode=to_box` AND "
    "`preset=Width/Height`; ignored when a named preset is selected."
)
TOOLTIP_RESIZE_TYPE = (
    "How input maps into the target box. Used when `reformat_mode=to_box`. "
    "width/height = scale uniformly to match that edge. "
    "fit = scale to fit inside the box (letterbox; black where the box exceeds the scaled image). "
    "fill = scale to cover the box (crops the overflow). "
    "distort = scale W and H independently to exactly match the box (changes aspect). "
    "none = no scale; place input centered in the box (crop if larger, pad if smaller). "
    "Cropped-away/padded regions are TRANSPARENT — RGB sources are promoted to RGBA "
    "with alpha=0 in the padded area so downstream compositing is clean."
)
TOOLTIP_FILTER = (
    "Pixel filter for resampling. "
    "impulse = nearest-neighbor (mask passes, exact pixel preservation). "
    "linear = bilinear (cheap, smooth). "
    "cubic = bicubic (default; the safe Nuke-equivalent). "
    "Lanczos4 = sharpest; for high-quality stills / final delivery. "
    "area = best for downscaling — anti-aliased mean pooling, softer but artifact-free."
)
TOOLTIP_DTYPE = (
    "Output tensor dtype. fp32 = ComfyUI default (4 bytes/sample). "
    "fp16 = half memory + half VRAM (2 bytes/sample). EXR-native precision; "
    "fits the [0,1] LDR + scene-linear range with headroom up to ~65504. "
    "Some downstream nodes assume fp32 — flip back to fp32 if you hit dtype errors."
)


# ---------------------------------------------------------------------------
#  MASK convention — stock ComfyUI (mask = 1 - alpha). Tooltips here are the
#  single source of truth; every node embeds (or composes) these strings so
#  the convention is documented identically everywhere an artist looks.
# ---------------------------------------------------------------------------

# Shared paragraph explaining the convention. Embedded in all five MASK
# tooltips (Read outputs, Write inputs, Write outputs, Reformat in/out).
TOOLTIP_MASK_CONVENTION = (
    "MASK CONVENTION (stock ComfyUI):\n"
    "  mask = 1 - alpha\n"
    "  white (1.0) = 'area to inpaint' (source was transparent)\n"
    "  black (0.0) = 'keep'            (source was opaque)\n"
    "  empty mask  = all zeros          (source has no alpha = fully visible)\n"
    "\n"
    "This is the SD-inpainting convention every stock ComfyUI mask-using node "
    "expects (LoadImage, MaskComposite, SetLatentNoiseMask, ImpactPack mask "
    "pipeline, etc.). Drop-in compatible with all of them.\n"
    "\n"
    "If you want NUKE-STYLE natural alpha (mask = alpha, where 1.0 = opaque), "
    "wire a MaskInvert node between this socket and your downstream consumer."
)

# Read-side mask output tooltip. Used on AM Read Image + AM Read Video.
TOOLTIP_MASK_OUT_READ = (
    "Alpha channel as MASK (N×H×W float in [0,1]).\n"
    "\n"
    + TOOLTIP_MASK_CONVENTION + "\n"
    "\n"
    "Populated when the source carries alpha:\n"
    "  * Image: EXR / PNG / TIFF with α channel\n"
    "  * Video: ProRes 4444 / 4444 XQ, QuickTime RLE, FFV1 (yuva*)\n"
    "Otherwise emits the empty mask (zeros = fully opaque source)."
)

# Write-side mask input tooltip. Same convention as outputs — what the
# artist sees on the input is the same MASK they got from a Read or
# upstream node, no inversion in between.
TOOLTIP_MASK_IN_WRITE = (
    "Optional alpha channel as MASK (N×H×W float in [0,1]).\n"
    "\n"
    + TOOLTIP_MASK_CONVENTION + "\n"
    "\n"
    "When wired, the file is encoded with alpha = 1 - mask. Mismatched "
    "mask dimensions are auto-resized (cubic) to match the image's H,W. "
    "When the input IMAGE is already 4-channel and a MASK is also wired, "
    "MASK overrides the embedded alpha."
)

# Write-side mask OUTPUT tooltip — passthrough socket so graphs can chain
# downstream of a Write without re-extracting alpha from disk.
TOOLTIP_MASK_OUT_WRITE = (
    "MASK passthrough — the alpha that was written to disk (or would have "
    "been, if the codec supports alpha).\n"
    "\n"
    + TOOLTIP_MASK_CONVENTION + "\n"
    "\n"
    "Sources, in priority order:\n"
    "  1. The wired `mask` input (if any).\n"
    "  2. Embedded alpha from a 4-channel IMAGE input.\n"
    "  3. Empty mask (zeros) — input was 3-channel with no MASK wired.\n"
    "Sized + reformatted to match the IMAGE output. Useful when chaining "
    "another node after a Write without re-reading from disk."
)

# Reformat (standalone) — input is a passthrough that gets the same
# geometry applied as the IMAGE; output is the post-reformat mask.
TOOLTIP_MASK_IN_REFORMAT = (
    "Optional MASK input (N×H×W float in [0,1]).\n"
    "\n"
    + TOOLTIP_MASK_CONVENTION + "\n"
    "\n"
    "When wired, the same geometric reformat is applied to the mask in "
    "lockstep with the image. Mismatched mask dimensions are auto-resized "
    "(cubic) to the image's H,W before reformat."
)
TOOLTIP_MASK_OUT_REFORMAT = (
    "Reformatted MASK (N×H×W float in [0,1]).\n"
    "\n"
    + TOOLTIP_MASK_CONVENTION + "\n"
    "\n"
    "Sources, in priority order:\n"
    "  1. The wired `mask` input (post-reformat geometry).\n"
    "  2. Extracted from a 4-channel IMAGE input's alpha (mask = 1 - alpha).\n"
    "  3. Empty mask (zeros) — IMAGE was 3-channel with no MASK wired."
)


# ---------------------------------------------------------------------------
#  Filter mapping — done lazily so this module doesn't hard-fail at import
#  time on environments without cv2 (the helper itself errors at call time
#  if cv2 is missing AND a non-off mode is requested).
# ---------------------------------------------------------------------------

def _filter_to_cv2(filter_name: str):
    import cv2  # type: ignore
    return {
        FILTER_IMPULSE:  cv2.INTER_NEAREST,
        FILTER_LINEAR:   cv2.INTER_LINEAR,
        FILTER_CUBIC:    cv2.INTER_CUBIC,
        FILTER_LANCZOS4: cv2.INTER_LANCZOS4,
        FILTER_AREA:     cv2.INTER_AREA,
    }.get(filter_name, cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def resolve_target_size(
    *,
    mode: str,
    scale: float,
    preset: str,
    target_w: int,
    target_h: int,
    src_h: int,
    src_w: int,
) -> Tuple[int, int]:
    """Compute the (out_w, out_h) the reformat block will produce given the
    widget values and the source's H,W.

    Returned pair is (W, H) — matches cv2 / Nuke convention. For
    ``mode=off`` returns ``(src_w, src_h)``.

    Centralized so callers (mainly the IO nodes' info-string builders and
    the standalone node's width/height output sockets) all agree on what
    the block "would" produce without doing the actual resize.
    """
    if mode == MODE_OFF:
        return int(src_w), int(src_h)
    if mode == MODE_SCALE:
        s = max(1e-6, float(scale))
        return max(1, int(round(src_w * s))), max(1, int(round(src_h * s)))
    # to_box: preset wins if not the sentinel.
    if preset != PRESET_WH and preset in PRESETS:
        pw, ph = PRESETS[preset]
        return int(pw), int(ph)
    return max(1, int(target_w)), max(1, int(target_h))


def reformat_array(
    arr: np.ndarray,
    *,
    mode: str,
    scale: float,
    preset: str,
    target_w: int,
    target_h: int,
    resize_type: str,
    filter_name: str,
    output_dtype: str,
) -> np.ndarray:
    """Apply the reformat block to *arr* and (optionally) down-cast its dtype.

    Accepts a single frame ``(H, W, C)`` or a batch ``(N, H, W, C)``;
    returns the same rank. For batches the per-frame loop runs inside
    this helper (cv2 has no batched API) so the call site stays clean.

    The dtype cast is the LAST step — geometric ops always run in fp32
    even when ``output_dtype == "fp16"`` so OCIO and downstream-precision
    behavior is unaffected by the memory-saving knob.

    ``mode == "off"`` short-circuits the geometric step entirely; the
    only work done is the dtype cast (if requested). Cheap no-op when
    both `mode=off` and `output_dtype=fp32`.
    """
    if arr is None:
        return arr

    is_batch = (arr.ndim == 4)
    if mode == MODE_OFF:
        return _maybe_cast(arr, output_dtype)

    out_w, out_h = resolve_target_size(
        mode=mode, scale=scale, preset=preset,
        target_w=target_w, target_h=target_h,
        src_h=int(arr.shape[-3]), src_w=int(arr.shape[-2]),
    )

    if is_batch:
        frames = [
            _reformat_frame(
                arr[i], mode=mode, out_w=out_w, out_h=out_h,
                resize_type=resize_type, filter_name=filter_name,
            )
            for i in range(arr.shape[0])
        ]
        result = np.stack(frames, axis=0) if frames else arr
    else:
        result = _reformat_frame(
            arr, mode=mode, out_w=out_w, out_h=out_h,
            resize_type=resize_type, filter_name=filter_name,
        )

    return _maybe_cast(result, output_dtype)


# ---------------------------------------------------------------------------
#  Private — single-frame geometry
# ---------------------------------------------------------------------------

def _reformat_frame(
    frame: np.ndarray,
    *,
    mode: str,
    out_w: int,
    out_h: int,
    resize_type: str,
    filter_name: str,
) -> np.ndarray:
    """Apply geometry to ONE (H, W, C) fp32 frame. Returns (out_h, out_w, C)."""
    import cv2  # type: ignore

    src_h, src_w = int(frame.shape[0]), int(frame.shape[1])
    if src_h <= 0 or src_w <= 0:
        return frame

    # Ensure float32 contiguous — cv2 wants this and downstream nodes assume it.
    if frame.dtype != np.float32:
        frame = frame.astype(np.float32, copy=False)
    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)

    interp = _filter_to_cv2(filter_name)

    if mode == MODE_SCALE:
        # Uniform scale; resize_type doesn't apply (no aspect mismatch).
        return cv2.resize(frame, (out_w, out_h), interpolation=interp)

    # mode == MODE_TO_BOX — resize_type decides how src maps into (out_w, out_h).
    if resize_type == RESIZE_DISTORT:
        return cv2.resize(frame, (out_w, out_h), interpolation=interp)

    if resize_type == RESIZE_NONE:
        # Nuke "crop": no scale, place src centered in the (out_h, out_w) canvas.
        # Padded regions are transparent (alpha=0) — RGB sources are
        # promoted to RGBA so the cropped-away area composites cleanly
        # against whatever the artist puts behind. See _center_paste.
        return _center_paste(frame, out_w, out_h, transparent_pad=True)

    # Compute the uniform scale factor s from src→box, per resize_type.
    sx = out_w / src_w
    sy = out_h / src_h
    if resize_type == RESIZE_WIDTH:
        s = sx
    elif resize_type == RESIZE_HEIGHT:
        s = sy
    elif resize_type == RESIZE_FILL:
        s = max(sx, sy)
    else:  # RESIZE_FIT (default)
        s = min(sx, sy)

    new_w = max(1, int(round(src_w * s)))
    new_h = max(1, int(round(src_h * s)))
    scaled = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    # `width` and `height` modes use the scaled image as-is — let it overflow
    # or leave gaps relative to the box; that's the documented Nuke behavior.
    # For these modes we still emit the scaled-image dims, NOT (out_w, out_h),
    # because Nuke's `width`/`height` resize_types intentionally produce a
    # non-target-sized output (height follows aspect, etc).
    if resize_type in (RESIZE_WIDTH, RESIZE_HEIGHT):
        return scaled

    # `fit` letterboxes inside the box; `fill` crops the overflow. Both
    # produce exactly (out_h, out_w, C).
    return _center_paste(scaled, out_w, out_h)


def _center_paste(
    src: np.ndarray,
    out_w: int,
    out_h: int,
    *,
    transparent_pad: bool = False,
) -> np.ndarray:
    """Place *src* centered inside an (out_h, out_w, C) canvas; crop if larger,
    pad if smaller. Used by both `none` (no-scale) and `fit` (letterbox).

    When ``transparent_pad=False`` (default — used by `fit`), padded regions
    are zero-filled across all channels (black for RGB, fully-transparent
    black for RGBA — but the latter is rarely artist-intended for letterbox).

    When ``transparent_pad=True`` (used by `none`/Nuke crop), padded regions
    are transparent:
    * RGB (3-ch) source is promoted to RGBA (4-ch). Source region gets
      alpha=1 (fully opaque); padded region keeps the canvas zeros (alpha=0,
      transparent). The artist gets a clean alpha matte for compositing.
    * RGBA (4-ch) source: source region keeps its native alpha; padded
      region is the canvas zeros (alpha=0).
    """
    src_h, src_w, c = int(src.shape[0]), int(src.shape[1]), int(src.shape[2])

    out_c = c
    promote_alpha = False
    if transparent_pad and c == 3:
        # Promote RGB → RGBA so the padded region can be transparent.
        out_c = 4
        promote_alpha = True

    canvas = np.zeros((out_h, out_w, out_c), dtype=src.dtype)

    # Source region (cropped if src is larger than canvas).
    src_x0 = max(0, (src_w - out_w) // 2)
    src_y0 = max(0, (src_h - out_h) // 2)
    src_x1 = min(src_w, src_x0 + out_w)
    src_y1 = min(src_h, src_y0 + out_h)

    # Destination region (centered if src is smaller than canvas).
    dst_x0 = max(0, (out_w - src_w) // 2)
    dst_y0 = max(0, (out_h - src_h) // 2)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if promote_alpha:
        # Copy RGB into channels 0-2; mark the source region opaque (alpha=1).
        # Padded region keeps zeros across all four channels (alpha=0 → transparent).
        canvas[dst_y0:dst_y1, dst_x0:dst_x1, :3] = src[src_y0:src_y1, src_x0:src_x1, :]
        canvas[dst_y0:dst_y1, dst_x0:dst_x1, 3] = 1.0
    else:
        canvas[dst_y0:dst_y1, dst_x0:dst_x1, :] = src[src_y0:src_y1, src_x0:src_x1, :]
    return canvas


def split_image_mask(rgba_or_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a packed (H,W,C) or (N,H,W,C) array into (image_RGB, mask).

    Used by Read nodes after OCIO + reformat to produce the (IMAGE, MASK)
    output socket pair. **Convention matches stock ComfyUI ``LoadImage``:
    ``mask = 1 - alpha``** — the SD-inpainting convention where white
    (1.0) marks "area to regenerate" and black (0.0) marks "keep". This
    is the inverse of natural Nuke-style alpha. Direct interop with stock
    SD inpainting nodes; VFX artists wanting Nuke-style ``mask = alpha``
    wire a ``MaskInvert`` between AM and their downstream nodes.

    * 4-channel input (RGBA): IMAGE = ``arr[..., :3]`` (RGB),
      MASK = ``1 - arr[..., 3]`` (alpha-inverted).
    * 3-channel input (RGB): IMAGE = arr unchanged, MASK = zeros
      ("empty mask" = nothing to inpaint, the natural value when the
      source has no alpha info / is fully visible).

    Mask dtype matches input dtype so fp16 reads emit fp16 masks.
    """
    if rgba_or_rgb.ndim == 4:
        # Batch: (N, H, W, C)
        if rgba_or_rgb.shape[-1] >= 4:
            image = rgba_or_rgb[..., :3]
            mask = (np.ones_like(rgba_or_rgb[..., 3]) - rgba_or_rgb[..., 3])
        else:
            image = rgba_or_rgb
            mask = np.zeros(rgba_or_rgb.shape[:3], dtype=rgba_or_rgb.dtype)
    elif rgba_or_rgb.ndim == 3:
        if rgba_or_rgb.shape[-1] >= 4:
            image = rgba_or_rgb[..., :3]
            mask = (np.ones_like(rgba_or_rgb[..., 3]) - rgba_or_rgb[..., 3])
        else:
            image = rgba_or_rgb
            mask = np.zeros(rgba_or_rgb.shape[:2], dtype=rgba_or_rgb.dtype)
    else:
        raise ValueError(f"split_image_mask: expected 3D/4D array, got shape {rgba_or_rgb.shape}")
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def combine_image_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    *,
    auto_resize_mask: bool = True,
) -> np.ndarray:
    """Combine an IMAGE (RGB or RGBA) with a MASK into a packed RGBA buffer.

    Used by Write nodes to fold a wired MASK back into the encode/write
    buffer. **Convention matches stock ComfyUI: ``alpha = 1 - mask``**
    (the inverse of :func:`split_image_mask`). White mask (1.0) →
    transparent alpha (0.0); black mask (0.0) → opaque alpha (1.0).

    Behavior:

    * ``mask is None`` AND image is 4-ch → returns image as-is (existing alpha).
    * ``mask is None`` AND image is 3-ch → returns image as-is (caller writes RGB).
    * ``mask`` wired AND image is 3-ch → concatenate alpha; returns 4-ch.
    * ``mask`` wired AND image is 4-ch → MASK overrides the embedded alpha;
      returns 4-ch with alpha = 1 - mask.

    *auto_resize_mask*: when True (default), a mask whose H,W differ from
    the image's H,W is resized to match using ``cv2.INTER_CUBIC``. When
    False, a size mismatch raises ValueError. Auto-resize is the artist-
    friendly default; raise when you need strict alignment guarantees.

    Accepts (H,W,C) image + (H,W) mask, or (N,H,W,C) image + (N,H,W) mask.
    The N must agree on the batch path; the per-frame path is also fine.
    """
    if mask is None:
        return image

    is_batch = (image.ndim == 4)
    img_h = int(image.shape[-3])
    img_w = int(image.shape[-2])

    # Reconcile dimensions. Mask comes in as (H,W) for single, (N,H,W) for batch.
    mask_h = int(mask.shape[-2])
    mask_w = int(mask.shape[-1])
    if (mask_h, mask_w) != (img_h, img_w):
        if not auto_resize_mask:
            raise ValueError(
                f"combine_image_mask: mask shape {mask.shape} != image H,W "
                f"({img_h}, {img_w}). Wire a same-size mask or set "
                f"auto_resize_mask=True."
            )
        import cv2  # type: ignore
        log.info(
            "[am-vfx-tools/reformat] auto-resizing mask from %dx%d to %dx%d (cubic) "
            "to match image dims",
            mask_w, mask_h, img_w, img_h,
        )
        if mask.ndim == 3:  # (N, H, W) batch
            resized = np.empty((mask.shape[0], img_h, img_w), dtype=mask.dtype)
            for i in range(mask.shape[0]):
                resized[i] = cv2.resize(
                    mask[i].astype(np.float32, copy=False),
                    (img_w, img_h),
                    interpolation=cv2.INTER_CUBIC,
                ).astype(mask.dtype, copy=False)
            mask = resized
        else:  # (H, W) single
            mask = cv2.resize(
                mask.astype(np.float32, copy=False),
                (img_w, img_h),
                interpolation=cv2.INTER_CUBIC,
            ).astype(mask.dtype, copy=False)

    # Mask-to-alpha inversion (matches stock ComfyUI convention). Clamp
    # into [0, 1] because upstream nodes occasionally drift outside that
    # range and we don't want negative or super-bright alpha.
    alpha = 1.0 - mask.astype(np.float32, copy=False)
    alpha = np.clip(alpha, 0.0, 1.0).astype(image.dtype, copy=False)

    if is_batch:
        # (N, H, W, C) image + (N, H, W) mask → (N, H, W, 4)
        if image.shape[-1] >= 4:
            out = image.copy()
            out[..., 3] = alpha
            return out
        return np.concatenate([image, alpha[..., None]], axis=-1)

    # (H, W, C) + (H, W) → (H, W, 4)
    if image.shape[-1] >= 4:
        out = image.copy()
        out[..., 3] = alpha
        return out
    return np.concatenate([image, alpha[..., None]], axis=-1)


def _maybe_cast(arr: np.ndarray, output_dtype: str) -> np.ndarray:
    """Down-cast to fp16 if requested. fp32 is a no-op (returns input).

    NOTE: callers that need to feed the result back into a fp32-only API
    (OCIO `apply_inplace`, OIIO write paths that allocate fp32 buffers,
    etc.) should call THOSE first, then pass the post-color-managed pixels
    through this helper. The dtype cast lives at the end of the geometric
    pipeline by design — see module docstring.
    """
    if output_dtype == DTYPE_FP16 and arr.dtype != np.float16:
        return arr.astype(np.float16, copy=False)
    return arr


# ---------------------------------------------------------------------------
#  Convenience: one-shot info-string fragment for the IO nodes' `info` socket.
# ---------------------------------------------------------------------------

def info_fragment(
    *,
    mode: str,
    scale: float,
    preset: str,
    target_w: int,
    target_h: int,
    resize_type: str,
    filter_name: str,
    output_dtype: str,
    src_w: int,
    src_h: int,
) -> str:
    """One-line summary suffix for the node's `info` socket. Empty string
    when reformat is off and dtype is fp32 (no-op case)."""
    if mode == MODE_OFF and output_dtype == DTYPE_FP32:
        return ""
    out_w, out_h = resolve_target_size(
        mode=mode, scale=scale, preset=preset,
        target_w=target_w, target_h=target_h,
        src_h=src_h, src_w=src_w,
    )
    parts = []
    if mode == MODE_SCALE:
        parts.append(f"scale {scale:g}")
    elif mode == MODE_TO_BOX:
        if preset != PRESET_WH:
            parts.append(f"preset {preset}")
        else:
            parts.append("to_box")
        parts.append(resize_type)
    if mode != MODE_OFF:
        parts.append(filter_name)
    if output_dtype != DTYPE_FP32:
        parts.append(output_dtype)
    if mode != MODE_OFF:
        parts.append(f"{src_w}x{src_h}→{out_w}x{out_h}")
    return f"reformat: {', '.join(parts)}"
