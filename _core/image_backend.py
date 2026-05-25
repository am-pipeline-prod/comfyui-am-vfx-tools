"""am-vfx-tools-media-io._core.image_backend — OpenImageIO read/write.

Pure I/O — no color management. Pair with :mod:`._core.color` to apply a
transform between read and downstream / before write.

Public surface:
  * :func:`read_image` — load a single file into a float32 ``(H, W, C)`` array.
  * :func:`write_image` — write a float32 array to disk via OIIO.
  * :func:`is_available` — True if PyOpenImageIO is importable.
  * :func:`format_supported` — quick compatibility probe.
  * :class:`ImageReadResult` — read return value (pixels + spec dict).
  * :class:`OIIONotInstalled` — raised when OIIO is required but missing.

Imported guarded so the test runner (no OIIO) can still import the module
to run grammar/templates/tokens tests; the actual read/write functions
raise :class:`OIIONotInstalled` when called without OIIO.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

log = logging.getLogger("am_vfx_tools.media-io.image")


try:
    import OpenImageIO as _oiio  # type: ignore[import-not-found]
    _OIIO_AVAILABLE = True
except ImportError:
    _oiio = None
    _OIIO_AVAILABLE = False


# Empirically-verified per-format metadata behavior (probe 2026-04-28 against
# OIIO 2.5+, libtiff, libjpeg-turbo). Two distinct strategies live below:
#
# * **Native multi-key** — PNG (tEXt/iTXt chunks) and EXR (native header
#   attribs) preserve arbitrary namespaced string keys with no practical
#   size limit. Each `comfyui/<key>` lands as its own attrib for clean
#   readback via `spec.extra_attribs`.
# * **JSON-packed ImageDescription** — TIFF (TIFFTAG_IMAGEDESCRIPTION,
#   ASCII) and JPEG (APP1/EXIF ImageDescription) preserve a single ASCII
#   payload up to ~65,500 bytes. Both libraries silently drop custom
#   namespaced keys but cleanly round-trip the standard ImageDescription
#   tag. We pack the workflow_metadata dict as JSON and write it there;
#   on readback, parsing the JSON string recovers the original dict.
#   Fallback ladder when oversize:
#     1) drop large keys (`comfyui/prompt`, `comfyui/workflow`); retry.
#     2) if still oversize, skip with a warning (the small structured
#        keys — seed/model/batch/source_path — are tiny and almost
#        always fit even after dropping the big ones).
#
# WebP / DPX / HDR / TGA / BMP drop everything at the writer level; no
# path exists for them via OIIO and they're excluded.
_NATIVE_METADATA_FORMATS = frozenset({"png", "exr"})
_JSON_PACKED_METADATA_FORMATS = frozenset({"tif", "tiff", "jpg", "jpeg"})

# TIFF ASCII tag and JPEG APP1 segment both cap around 65,536 bytes —
# probe shows truncation at 65,536 for TIFF and a libjpeg fatal "Bogus
# marker length" above ~65,500 for JPEG. We stay 500 bytes under the
# tighter limit so JSON expansion of the trailing closing-brace etc.
# never tips us over.
_PACKED_DESCRIPTION_LIMIT = 65_000

# Keys that are large by design — full prompt graph + editor-graph JSON.
# Dropped first when the JSON-packed payload exceeds the size limit so
# the small structured fields (seed/model/batch/source_path) survive.
_LARGE_METADATA_KEYS = ("comfyui/prompt", "comfyui/workflow")


def supports_workflow_metadata(filepath_or_ext: str) -> bool:
    """True if writing workflow metadata to *filepath* (or to a file with
    this bare extension) is preserved by OIIO. Covers both the native
    multi-key strategy (PNG/EXR) and the JSON-packed ImageDescription
    fallback (TIFF/JPEG). WebP/DPX/HDR/TGA/BMP return False.

    Accepts ``"png"`` / ``".png"`` / ``"/path/to/file.png"`` interchangeably.
    """
    ext = _ext_of(filepath_or_ext)
    return ext in _NATIVE_METADATA_FORMATS or ext in _JSON_PACKED_METADATA_FORMATS


def _ext_of(filepath_or_ext: str) -> str:
    s = (filepath_or_ext or "").lower()
    return os.path.splitext(s)[1].lstrip(".") if "." in s else s.lstrip(".")


# ---------------------------------------------------------------------------
# Frame-rate metadata extraction
# ---------------------------------------------------------------------------

# Header-attribute keys queried in priority order for an image's frame
# rate. EXR's spec-canonical attribute is ``framesPerSecond`` (an
# Imath.Rational); Nuke and some pipelines emit ``input/framesPerSecond``
# alongside it. ``frameRate``, ``framerate``, ``fps`` are the informal
# fallbacks seen across DCCs. Comparison is case-insensitive — OIIO
# normalizes attribute names per writer and we don't want to miss a
# variant on case alone.
_FRAME_RATE_KEYS = (
    "framesPerSecond",
    "input/framesPerSecond",
    "frameRate",
    "framerate",
    "fps",
)


def extract_frame_rate(metadata: Dict[str, Any]) -> Optional[float]:
    """Mine an image-header ``metadata`` dict (e.g. ``ImageReadResult.metadata``)
    for a frame rate. Returns the first hit converted to ``float``, or
    ``None`` when no recognized key is present / the value can't be
    coerced.

    Handles two value shapes:

    * **Imath.Rational** (the canonical EXR ``framesPerSecond`` shape) —
      OIIO surfaces this as a tuple/sequence ``(num, den)`` or as an
      object exposing ``.n`` / ``.d``. Both are converted via
      ``num / den``.
    * **Plain numeric** (float / int / numeric string) — coerced via
      ``float()``.

    A non-positive result (``<= 0``) is treated as missing and returns
    ``None`` so the caller's auto-fallback fires instead of emitting
    bogus ``0`` fps to downstream nodes.
    """
    if not metadata:
        return None
    # Build a lower-cased lookup so the priority-ordered scan is case-insensitive.
    lookup = {str(k).lower(): v for k, v in metadata.items()}
    for key in _FRAME_RATE_KEYS:
        value = lookup.get(key.lower())
        if value is None:
            continue
        fps = _coerce_rational_to_float(value)
        if fps is not None and fps > 0:
            return fps
    return None


def _coerce_rational_to_float(value: Any) -> Optional[float]:
    """Convert OIIO's various Rational / numeric attribute shapes to float."""
    # Tuple/list shape: (numerator, denominator).
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            num, den = float(value[0]), float(value[1])
            if den != 0:
                return num / den
        except (TypeError, ValueError):
            return None
        return None
    # Object with .n/.d (Imath.Rational from PyOpenEXR / OIIO bindings).
    n = getattr(value, "n", None)
    d = getattr(value, "d", None)
    if n is not None and d is not None:
        try:
            num, den = float(n), float(d)
            if den != 0:
                return num / den
        except (TypeError, ValueError):
            return None
        return None
    # Plain numeric / numeric string.
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class OIIONotInstalled(RuntimeError):
    """OIIO is required for this operation but not importable."""


def is_available() -> bool:
    return _OIIO_AVAILABLE


def _require_oiio() -> None:
    if not _OIIO_AVAILABLE:
        raise OIIONotInstalled(
            "OpenImageIO Python bindings (`OpenImageIO`) are required for "
            "image I/O. Install via your DCC's bundled venv or pip."
        )


@dataclass
class ImageReadResult:
    """Float32 pixel array plus the read spec metadata that artists may want."""
    pixels: Any                 # numpy.ndarray, shape (H, W, C), dtype float32
    width: int
    height: int
    n_channels: int
    bit_depth: str              # "uint8" / "uint16" / "half" / "float" / etc.
    color_space: Optional[str]  # OCIO ColorSpace string from the file, if any
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_image(filepath: str) -> ImageReadResult:
    """Read *filepath* via OIIO; return float32 ``(H, W, C)`` pixels + spec.

    Raises :class:`FileNotFoundError` if the file is absent;
    :class:`OIIONotInstalled` if the OIIO Python bindings aren't loaded;
    :class:`RuntimeError` for any OIIO-side read failure.
    """
    _require_oiio()
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)

    import numpy as np

    inp = _oiio.ImageInput.open(filepath)
    if inp is None:
        raise RuntimeError(f"OIIO open() failed for {filepath}: {_oiio.geterror()}")
    try:
        spec = inp.spec()
        pixels = inp.read_image("float")
        if pixels is None:
            raise RuntimeError(
                f"OIIO read_image() returned None for {filepath}: {inp.geterror()}"
            )
    finally:
        inp.close()

    pixels = np.asarray(pixels, dtype=np.float32).reshape(
        spec.height, spec.width, spec.nchannels
    )

    color_space = spec.get_string_attribute("oiio:ColorSpace") or None
    metadata: Dict[str, Any] = {}
    for attr in spec.extra_attribs:
        try:
            metadata[attr.name] = attr.value
        except Exception:
            continue

    return ImageReadResult(
        pixels=pixels,
        width=spec.width,
        height=spec.height,
        n_channels=spec.nchannels,
        bit_depth=str(spec.format),
        color_space=color_space,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


_BIT_DEPTH_TO_OIIO = {
    # Map artist-friendly labels to OIIO type tokens.
    "uint8":  "uint8",
    "uint16": "uint16",
    "half":   "half",
    "float":  "float",
    # Aliases:
    "8":      "uint8",
    "16":     "uint16",
    "16f":    "half",
    "32f":    "float",
}


def _normalize_bit_depth(label: str) -> str:
    label = (label or "").strip().lower()
    if label in _BIT_DEPTH_TO_OIIO:
        return _BIT_DEPTH_TO_OIIO[label]
    raise ValueError(
        f"Unknown bit_depth {label!r} — supported: "
        f"{sorted(_BIT_DEPTH_TO_OIIO)}"
    )


def _quantize_for(pixels, oiio_format: str):
    """Return the pixel buffer in the right dtype for *oiio_format*."""
    import numpy as np
    if oiio_format == "uint8":
        return (np.clip(pixels, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if oiio_format == "uint16":
        return (np.clip(pixels, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    if oiio_format == "half":
        return pixels.astype(np.float16)
    if oiio_format == "float":
        return pixels.astype(np.float32)
    return pixels


def _format_attr_name():
    """OIIO's TypeDesc enum; returns the matching constant for the format."""
    return {
        "uint8": _oiio.UINT8,
        "uint16": _oiio.UINT16,
        "half": _oiio.HALF,
        "float": _oiio.FLOAT,
    }


def _embed_workflow_metadata(
    spec: Any, filepath: str, workflow_metadata: Dict[str, str],
) -> None:
    """Embed *workflow_metadata* into *spec* using the right strategy
    for *filepath*'s extension.

    PNG/EXR  → write each k/v as a separate spec.attribute (OIIO maps
               them to tEXt/iTXt chunks for PNG and native header
               attribs for EXR). No size limit in practice.
    TIF/JPG  → pack the dict as JSON and write the result as
               ``ImageDescription``. Falls back to dropping the large
               ``comfyui/prompt`` / ``comfyui/workflow`` keys when the
               JSON exceeds ~65 KB; if still oversize, skips with a
               warning. The small structured fields (seed/model/batch/
               source_path) almost always survive the fallback.
    Other    → silently skip; no path exists via OIIO.
    """
    import json

    ext = _ext_of(filepath)

    if ext in _NATIVE_METADATA_FORMATS:
        for k, v in workflow_metadata.items():
            try:
                spec.attribute(str(k), str(v))
            except Exception:
                log.warning("[am_vfx_tools/image] skipped workflow attr %r", k)
        return

    if ext in _JSON_PACKED_METADATA_FORMATS:
        # Try the full payload first.
        payload = json.dumps(workflow_metadata, separators=(",", ":"))
        if len(payload) > _PACKED_DESCRIPTION_LIMIT:
            # Drop the large keys; retry.
            trimmed = {k: v for k, v in workflow_metadata.items()
                       if k not in _LARGE_METADATA_KEYS}
            payload = json.dumps(trimmed, separators=(",", ":"))
            if len(payload) > _PACKED_DESCRIPTION_LIMIT:
                log.warning(
                    "[am_vfx_tools/image] workflow metadata too large for %s "
                    "ImageDescription tag (%d B > %d B even after dropping "
                    "%s); skipping embed for this file. Switch to PNG/EXR "
                    "to keep the full workflow.",
                    ext, len(payload), _PACKED_DESCRIPTION_LIMIT,
                    _LARGE_METADATA_KEYS,
                )
                return
            log.info(
                "[am_vfx_tools/image] %s ImageDescription too small for full "
                "payload — embedded structured fields only (dropped %s)",
                ext, _LARGE_METADATA_KEYS,
            )
        try:
            spec.attribute("ImageDescription", payload)
        except Exception:
            log.warning(
                "[am_vfx_tools/image] could not set ImageDescription on %s",
                filepath,
            )
        return

    # Unsupported extension — nothing to do.
    log.debug(
        "[am_vfx_tools/image] workflow metadata not embedded — extension %r "
        "is not in the supported set (PNG/EXR/TIF/JPG)",
        ext,
    )


def _embed_frame_rate(spec: Any, filepath: str, fps: float) -> None:
    """Write frame-rate metadata into *spec* per the format's conventions.

    Two keys are emitted across formats so a downstream reader can pick
    up either:

    * **EXR** — ``framesPerSecond`` as the canonical OpenEXR
      ``Imath::Rational`` (every spec-compliant DCC reads this; Resolve,
      Nuke, Houdini, Fusion all surface it). PLUS ``input/frame_rate``
      as a STRING — Nuke's metadata-bus convention; the value Nuke
      shows in its own metadata viewer when an EXR carries the tag.
      Both keys read symmetric to :func:`extract_frame_rate`.
    * **PNG** — ``frameRate`` and ``input/frame_rate`` as plain string
      attributes; OIIO maps both to tEXt/iTXt chunks. PNG has no
      Rational native type, so a stringified float is the round-trip-
      safe choice.
    * **TIFF / JPEG** — best-effort string attributes via OIIO; whether
      they survive depends on the underlying libtiff/libjpeg writer's
      tolerance for non-standard tags. The structured workflow embed
      uses an entirely different strategy (JSON-packed
      ``ImageDescription`` — see :func:`_embed_workflow_metadata`), so
      these are NOT folded into that payload — they go in as direct
      attributes and may or may not be preserved. EXR is the canonical
      VFX image-sequence format, so this is acceptable.
    * **Others** (WebP / DPX / HDR / TGA / BMP) — silently skipped.

    *fps* must be positive; the caller is responsible for filtering out
    the ``-1`` / ``None`` "unset" sentinels before calling.
    """
    from fractions import Fraction

    ext = _ext_of(filepath)
    fps_str = f"{fps:.6g}"

    if ext == "exr":
        # OpenEXR canonical attribute (Imath::Rational). OIIO's Python
        # binding accepts a (num, den) tuple paired with TypeRational;
        # limit_denominator(1_000_000) keeps cinema-pulldown rates
        # exact (24000/1001 → 23.976) without unbounded precision.
        try:
            r = Fraction(fps).limit_denominator(1_000_000)
            type_rational = getattr(_oiio, "TypeRational", None)
            if type_rational is not None:
                spec.attribute(
                    "framesPerSecond", type_rational,
                    (r.numerator, r.denominator),
                )
            else:
                # Fallback for older OIIO bindings that lack TypeRational
                # — write as a plain string. Spec compliance suffers but
                # the read side's extract_frame_rate accepts numeric
                # strings too.
                spec.attribute("framesPerSecond", fps_str)
                log.debug(
                    "[am_vfx_tools/image] OIIO has no TypeRational — wrote "
                    "framesPerSecond as STRING %r (%s)", fps_str, filepath,
                )
        except Exception as e:
            log.warning(
                "[am_vfx_tools/image] failed to write framesPerSecond on %s: %s",
                filepath, e,
            )
        # Nuke pipeline convention — string-valued.
        try:
            spec.attribute("input/frame_rate", fps_str)
        except Exception as e:
            log.warning(
                "[am_vfx_tools/image] failed to write input/frame_rate on %s: %s",
                filepath, e,
            )
        return

    if ext in ("png", "tif", "tiff", "jpg", "jpeg"):
        for key in ("frameRate", "input/frame_rate"):
            try:
                spec.attribute(key, fps_str)
            except Exception as e:
                log.warning(
                    "[am_vfx_tools/image] failed to write %s on %s: %s",
                    key, filepath, e,
                )
        return

    # Other formats — silently skip.
    log.debug(
        "[am_vfx_tools/image] frame_rate not embedded — extension %r is not "
        "in the supported set (EXR/PNG/TIF/JPG)",
        ext,
    )


def write_image(
    filepath: str,
    pixels,
    *,
    bit_depth: str = "16f",
    compression: Optional[str] = "zip",
    metadata: Optional[Dict[str, Any]] = None,
    create_directories: bool = True,
    color_space_tag: Optional[str] = None,
    workflow_metadata: Optional[Dict[str, str]] = None,
    frame_rate: Optional[float] = None,
) -> None:
    """Write *pixels* (HxWxC float32) to *filepath* via OIIO.

    *bit_depth* controls the on-disk numeric type (``uint8`` / ``uint16``
    / ``half`` / ``float``; aliases ``8`` / ``16`` / ``16f`` / ``32f``
    accepted).

    *compression* is set as the OIIO ``compression`` attribute when the
    container supports it (EXR, TIFF). Pass ``None`` to skip.

    *color_space_tag* — if given, written into ``oiio:ColorSpace``.
    Conventionally set to the color space the pixels are *in* at write
    time (after any OCIO transform).

    Extra OIIO attributes can be passed via *metadata*.

    *workflow_metadata* — dict of string→string entries to embed alongside
    the image (typically ``{"comfyui/prompt": json_str,
    "comfyui/workflow": json_str, ...}`` matching the AM VFX Tools drag-drop
    convention). Embed strategy is per-format (see
    :func:`_embed_workflow_metadata`):
      * PNG / EXR  — each k/v as a separate native attribute; no size cap.
      * TIFF / JPEG — JSON-packed into ``ImageDescription`` (~65 KB cap;
        falls back to dropping ``comfyui/prompt`` / ``comfyui/workflow``
        before giving up).
      * WebP / DPX / HDR / TGA / BMP — silently skipped (no path via OIIO).
    The caller can always pass *workflow_metadata* without a per-format
    guard.

    *frame_rate* — when set (positive float), writes frame-rate metadata
    into the file header per :func:`_embed_frame_rate`. EXR gets the
    spec-canonical ``framesPerSecond`` Imath::Rational AND the Nuke-
    pipeline-friendly ``input/frame_rate`` STRING. PNG gets ``frameRate``
    + ``input/frame_rate`` as plain string attributes. TIFF/JPEG get
    them folded into the JSON-packed ``ImageDescription`` payload.
    Pass ``None`` (or omit) to skip — the read side's
    :func:`extract_frame_rate` already accepts every key written here,
    closing the read→write→re-read roundtrip.
    """
    _require_oiio()
    import numpy as np

    if create_directories:
        out_dir = os.path.dirname(filepath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    pixels = np.asarray(pixels)
    if pixels.ndim == 2:
        pixels = pixels[..., None]
    height, width, channels = pixels.shape

    oiio_format = _normalize_bit_depth(bit_depth)

    pixels_out = _quantize_for(pixels.astype(np.float32, copy=False), oiio_format)
    pixels_out = np.ascontiguousarray(pixels_out)

    type_desc = _format_attr_name()[oiio_format]
    spec = _oiio.ImageSpec(width, height, channels, type_desc)

    ext = os.path.splitext(filepath)[1].lower()
    if compression and ext in (".exr", ".tif", ".tiff"):
        spec.attribute("compression", compression)

    if color_space_tag:
        spec.attribute("oiio:ColorSpace", color_space_tag)

    if metadata:
        for k, v in metadata.items():
            try:
                spec.attribute(k, v)
            except Exception:
                log.warning("[am_vfx_tools/image] skipped metadata attr %r=%r", k, v)

    if workflow_metadata:
        _embed_workflow_metadata(spec, filepath, workflow_metadata)

    if frame_rate is not None and frame_rate > 0:
        _embed_frame_rate(spec, filepath, float(frame_rate))

    out = _oiio.ImageOutput.create(filepath)
    if out is None:
        raise RuntimeError(
            f"OIIO ImageOutput.create() failed for {filepath}: {_oiio.geterror()}"
        )
    if not out.open(filepath, spec):
        msg = out.geterror()
        out.close()
        raise RuntimeError(f"OIIO open({filepath}) failed: {msg}")

    if not out.write_image(pixels_out):
        msg = out.geterror()
        out.close()
        raise RuntimeError(f"OIIO write_image({filepath}) failed: {msg}")
    out.close()


def format_supported(ext: str, *, write: bool = False) -> bool:
    """Quick check whether OIIO can read/write a given extension."""
    if not _OIIO_AVAILABLE:
        return False
    e = ext.lower().lstrip(".")
    # Simple positive list — OIIO covers all of these.
    common = {
        "exr", "png", "jpg", "jpeg", "tif", "tiff", "dpx", "hdr",
        "tga", "bmp", "webp", "pnm", "pbm", "pgm", "ppm",
    }
    if e in common:
        return True
    # Ask OIIO whether it has a plugin for the extension.
    try:
        if write:
            return bool(_oiio.ImageOutput.create("test." + e))
        return bool(_oiio.ImageInput.create("test." + e, ""))
    except Exception:
        return False


__all__ = [
    "ImageReadResult",
    "OIIONotInstalled",
    "read_image",
    "write_image",
    "is_available",
    "format_supported",
    "supports_workflow_metadata",
]
