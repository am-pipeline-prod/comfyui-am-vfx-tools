"""Lazy per-frame VIDEO transforms — see docs/media-io-sync-rule.md invariant 28.

A :class:`LazyVideoTransform` wraps a source :class:`VideoInput` and
applies a per-frame transform on consumption. Memory profile: peak ≈
source + one in-flight frame. Composes by stacking source references:
``OCIOVideo(GradedVideo(ReformatVideo(VideoFromFile(...))))``.

The streaming win materialises only when the consumer iterates frame-by-
frame via :meth:`LazyVideoTransform.iter_frames` (preferred for AM
consumers) or :meth:`LazyVideoTransform.save_to`. Consumers that call
:meth:`get_components` or :meth:`get_stream_source` force the full chain
to materialise — both are supported but log a warning.

Subclassing recipe::

    class FooVideo(LazyVideoTransform):
        def __init__(self, source, foo_param):
            super().__init__(source)
            self._foo = foo_param

        def _transform_frame(self, image, alpha):
            return apply_foo(image, self._foo), alpha   # alpha untouched

        def _rewrap(self, new_source):
            return FooVideo(new_source, self._foo)

Override :meth:`_transform_dims` too if the transform changes output
dimensions (only Reformat does this in our pack today).
"""
from __future__ import annotations

import io
import logging
from fractions import Fraction
from typing import Iterator, Optional, Tuple

import numpy as np
import torch

try:
    import av  # type: ignore[import-not-found]
    _PYAV_AVAILABLE = True
except ImportError:
    av = None  # type: ignore[assignment]
    _PYAV_AVAILABLE = False

try:
    from comfy_api.v0_0_2 import InputImpl as _ComfyInputImpl  # type: ignore[import-not-found]
    from comfy_api.v0_0_2 import Types as _ComfyTypes  # type: ignore[import-not-found]
    # Pull the abstract base class for proper isinstance subtyping.
    _VideoInputBase = _ComfyInputImpl.VideoFromFile.__bases__[0]
    _VIDEO_TYPE_AVAILABLE = True
except ImportError:
    _ComfyInputImpl = None  # type: ignore[assignment]
    _ComfyTypes = None  # type: ignore[assignment]
    _VideoInputBase = object
    _VIDEO_TYPE_AVAILABLE = False

from . import color as _color
from . import color_correct as _color_correct
from . import grade as _grade
from . import reformat as _reformat

log = logging.getLogger("am_vfx_tools.video_lazy")


# Type alias — one decoded frame: (image HWC fp32 in [0,1], optional alpha HW fp32 in [0,1])
Frame = Tuple[np.ndarray, Optional[np.ndarray]]


class LazyVideoTransform(_VideoInputBase):  # type: ignore[misc]
    """Wraps a source VideoInput; applies a per-frame transform on consumption.

    See module docstring for the subclassing recipe.
    """

    def __init__(self, source):
        self._source = source

    # ------------------------------------------------------------------ #
    #  Subclass hooks
    # ------------------------------------------------------------------ #

    def _transform_frame(
        self, image_np: np.ndarray, alpha_np: Optional[np.ndarray],
    ) -> Frame:
        """Apply this transform to one frame. Default: passthrough."""
        return image_np, alpha_np

    def _transform_dims(self, src_w: int, src_h: int) -> Tuple[int, int]:
        """Output (width, height) given source dims. Default: passthrough."""
        return src_w, src_h

    def _rewrap(self, new_source) -> "LazyVideoTransform":
        """Rebuild this wrapper around *new_source* (for ``as_trimmed``)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Streaming consumer API (preferred path for AM consumers)
    # ------------------------------------------------------------------ #

    def iter_frames(self) -> Iterator[Frame]:
        """Yield ``(image_fp32, alpha_fp32_or_None)`` per frame.

        Streams through the source one frame at a time, applying this
        wrapper's transform inline. No full-batch buffer is built.
        """
        for img, alpha in self._iter_source_frames():
            yield self._transform_frame(img, alpha)

    # ------------------------------------------------------------------ #
    #  VideoInput interface (for non-AM consumers + bridging)
    # ------------------------------------------------------------------ #

    def get_components(self):
        """Materialise the chain — logs a warning. Streaming consumers
        should call :meth:`iter_frames` instead."""
        log.info(
            "[am_vfx_tools/video_lazy] %s.get_components() materialises the "
            "lazy chain — RAM peaks at full batch. Use iter_frames() or "
            "save_to() for streaming.", type(self).__name__,
        )
        if not _VIDEO_TYPE_AVAILABLE:
            return None
        images_list = []
        alpha_list = []
        for img, alpha in self.iter_frames():
            images_list.append(torch.from_numpy(np.ascontiguousarray(img)))
            if alpha is not None:
                alpha_list.append(torch.from_numpy(np.ascontiguousarray(alpha)))
        images = (
            torch.stack(images_list) if images_list
            else torch.zeros((0, 0, 0, 3), dtype=torch.float32)
        )
        alpha = torch.stack(alpha_list) if alpha_list else None
        # Forward audio + metadata + frame_rate from the original source.
        src_components = self._source.get_components()
        return _ComfyTypes.VideoComponents(
            images=images,
            alpha=alpha,
            frame_rate=src_components.frame_rate,
            audio=src_components.audio,
            metadata=src_components.metadata,
        )

    def save_to(self, path, format=None, codec=None, metadata=None):
        """Encode this lazy chain to *path* via PyAV. MP4 + H264 only.

        Matches the native ``VideoFromComponents.save_to`` constraint.
        Streams the encode frame-by-frame — peak RAM stays at one frame.
        """
        if not _VIDEO_TYPE_AVAILABLE or not _PYAV_AVAILABLE:
            raise RuntimeError(
                f"{type(self).__name__}.save_to needs comfy_api + av"
            )
        VideoContainer = _ComfyTypes.VideoContainer
        VideoCodec = _ComfyTypes.VideoCodec
        if (format is not None
                and format != VideoContainer.AUTO
                and format != VideoContainer.MP4):
            raise ValueError(
                f"LazyVideoTransform.save_to only supports MP4 "
                f"(requested format={format!r})"
            )
        if (codec is not None
                and codec != VideoCodec.AUTO
                and codec != VideoCodec.H264):
            raise ValueError(
                f"LazyVideoTransform.save_to only supports H264 "
                f"(requested codec={codec!r})"
            )

        out_w, out_h = self.get_dimensions()
        fps_fraction = self.get_frame_rate()
        rate = (
            Fraction(round(float(fps_fraction) * 1000), 1000)
            if fps_fraction else Fraction(25, 1)
        )

        open_kwargs = {"mode": "w", "options": {"movflags": "use_metadata_tags"}}
        if isinstance(path, io.BytesIO):
            open_kwargs["format"] = "mp4"

        with av.open(path, **open_kwargs) as out:
            if metadata:
                for k, v in metadata.items():
                    out.metadata[k] = str(v) if not isinstance(v, str) else v

            video_stream = out.add_stream("h264", rate=rate)
            video_stream.width = int(out_w)
            video_stream.height = int(out_h)
            video_stream.pix_fmt = "yuv420p"

            for img, _alpha in self.iter_frames():
                # img is HWC fp32 in [0,1]. Encode to uint8 yuv420p
                # via PyAV's reformat (matches native VideoFromComponents).
                img_uint8 = (
                    np.clip(img, 0.0, 1.0) * 255.0
                ).astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(img_uint8, format="rgb24")
                frame = frame.reformat(format="yuv420p")
                for packet in video_stream.encode(frame):
                    out.mux(packet)
            # Flush the encoder.
            for packet in video_stream.encode(None):
                out.mux(packet)

    def get_stream_source(self):
        """Forces materialisation to a BytesIO buffer. **Warns.**

        AM consumers should call :meth:`iter_frames` to keep the chain
        lazy. Non-AM consumers (Topaz, stock SaveVideo, etc.) call this
        expecting a file path or buffer; we accommodate but lose the
        streaming win at this boundary.
        """
        log.warning(
            "[am_vfx_tools/video_lazy] %s.get_stream_source() encodes the "
            "lazy chain to a BytesIO buffer — streaming win lost. Wire "
            "to an AM consumer (AM Image Write VIDEO input) for the "
            "frame-streaming path.",
            type(self).__name__,
        )
        buf = io.BytesIO()
        self.save_to(buf)
        buf.seek(0)
        return buf

    def get_dimensions(self):
        src_w, src_h = self._source.get_dimensions()
        return self._transform_dims(int(src_w), int(src_h))

    def get_frame_count(self) -> int:
        return int(self._source.get_frame_count())

    def get_frame_rate(self) -> Fraction:
        return self._source.get_frame_rate()

    def get_container_format(self) -> str:
        return self._source.get_container_format()

    def as_trimmed(
        self,
        start_time: float | None = None,
        duration: float | None = None,
        strict_duration: bool = False,
    ):
        trimmed = self._source.as_trimmed(start_time, duration, strict_duration)
        if trimmed is None:
            return None
        return self._rewrap(trimmed)

    # ------------------------------------------------------------------ #
    #  Source iteration — dispatches on source subtype.
    # ------------------------------------------------------------------ #

    def _iter_source_frames(self) -> Iterator[Frame]:
        if isinstance(self._source, LazyVideoTransform):
            # Chain through the source's transform.
            yield from self._source.iter_frames()
            return
        if not _VIDEO_TYPE_AVAILABLE:
            return
        if isinstance(self._source, _ComfyInputImpl.VideoFromFile):
            yield from self._iter_videofromfile(self._source)
            return
        if isinstance(self._source, _ComfyInputImpl.VideoFromComponents):
            yield from self._iter_videofromcomponents(self._source)
            return
        # Unknown VideoInput subclass — materialise.
        log.warning(
            "[am_vfx_tools/video_lazy] unknown VideoInput subclass %s — "
            "materialising via get_components()",
            type(self._source).__name__,
        )
        yield from self._iter_components_dataclass(self._source.get_components())

    @staticmethod
    def _iter_videofromfile(source) -> Iterator[Frame]:
        if not _PYAV_AVAILABLE:
            raise RuntimeError("PyAV unavailable — cannot stream VideoFromFile")
        src = source.get_stream_source()
        with av.open(src) as container:
            if not container.streams.video:
                return
            stream = container.streams.video[0]
            # Detect alpha pix_fmt (mirrors _core/video_backend.read_video_frames).
            has_alpha = False
            try:
                pf = stream.codec_context.pix_fmt or ""
                has_alpha = (
                    "yuva" in pf
                    or pf.startswith(("argb", "rgba", "abgr", "bgra"))
                )
            except Exception:  # noqa: BLE001
                pass
            decode_format = "rgba" if has_alpha else "rgb24"
            for av_frame in container.decode(stream):
                arr = (
                    av_frame.to_ndarray(format=decode_format).astype(np.float32)
                    / 255.0
                )
                if has_alpha:
                    img = np.ascontiguousarray(arr[..., :3])
                    alpha = np.ascontiguousarray(arr[..., 3])
                    yield img, alpha
                else:
                    yield arr, None

    @staticmethod
    def _iter_videofromcomponents(source) -> Iterator[Frame]:
        yield from LazyVideoTransform._iter_components_dataclass(
            source.get_components(),
        )

    @staticmethod
    def _iter_components_dataclass(components) -> Iterator[Frame]:
        if components is None or components.images is None:
            return
        images = components.images
        alpha_t = components.alpha
        n = int(images.shape[0])
        for i in range(n):
            img = images[i].detach().cpu().numpy().astype(np.float32, copy=False)
            # Source IMAGE may itself be 4-ch (rare — most reads split it
            # out to a MASK socket, but the lazy chain may receive a
            # 4-ch buffer if anything upstream packed it).
            if img.shape[-1] >= 4:
                yield (
                    np.ascontiguousarray(img[..., :3]),
                    np.ascontiguousarray(img[..., 3]),
                )
                continue
            alpha = None
            if alpha_t is not None and i < int(alpha_t.shape[0]):
                alpha = (
                    alpha_t[i].detach().cpu().numpy().astype(np.float32, copy=False)
                )
            yield img, alpha


# ---------------------------------------------------------------------- #
#  Concrete subclass: Reformat
# ---------------------------------------------------------------------- #


class ReformatVideo(LazyVideoTransform):
    """Lazy reformat — applies cv2.resize per frame to image AND alpha.

    Matches the existing eager ``_core/reformat.reformat_array`` semantics.
    The only AM transform that changes output dimensions.
    """

    def __init__(
        self,
        source,
        *,
        mode: str,
        scale: float,
        preset: str,
        target_w: int,
        target_h: int,
        resize_type: str,
        filter_name: str,
    ):
        super().__init__(source)
        self._mode = mode
        self._scale = float(scale)
        self._preset = preset
        self._target_w = int(target_w)
        self._target_h = int(target_h)
        self._resize_type = resize_type
        self._filter_name = filter_name

    def _transform_frame(
        self, image_np: np.ndarray, alpha_np: Optional[np.ndarray],
    ) -> Frame:
        if self._mode == _reformat.MODE_OFF:
            return image_np, alpha_np
        # cv2.resize handles 4-channel arrays natively, so concatenate
        # image + alpha into one buffer and resize in a single call.
        if alpha_np is not None:
            alpha_3d = alpha_np[..., None] if alpha_np.ndim == 2 else alpha_np
            combined = np.concatenate(
                [image_np, alpha_3d], axis=-1,
            ).astype(np.float32, copy=False)
            resized = _reformat.reformat_array(
                combined,
                mode=self._mode, scale=self._scale, preset=self._preset,
                target_w=self._target_w, target_h=self._target_h,
                resize_type=self._resize_type, filter_name=self._filter_name,
                output_dtype=_reformat.DTYPE_FP32,
            )
            return (
                np.ascontiguousarray(resized[..., :3]),
                np.ascontiguousarray(resized[..., 3]),
            )
        resized = _reformat.reformat_array(
            image_np,
            mode=self._mode, scale=self._scale, preset=self._preset,
            target_w=self._target_w, target_h=self._target_h,
            resize_type=self._resize_type, filter_name=self._filter_name,
            output_dtype=_reformat.DTYPE_FP32,
        )
        return resized, None

    def _transform_dims(self, src_w: int, src_h: int) -> Tuple[int, int]:
        return _reformat.resolve_target_size(
            mode=self._mode, scale=self._scale, preset=self._preset,
            target_w=self._target_w, target_h=self._target_h,
            src_h=src_h, src_w=src_w,
        )

    def _rewrap(self, new_source) -> "ReformatVideo":
        return ReformatVideo(
            new_source,
            mode=self._mode, scale=self._scale, preset=self._preset,
            target_w=self._target_w, target_h=self._target_h,
            resize_type=self._resize_type, filter_name=self._filter_name,
        )


# ---------------------------------------------------------------------- #
#  Concrete subclass: Grade (Nuke-style, per-channel vectors)
# ---------------------------------------------------------------------- #


class GradedVideo(LazyVideoTransform):
    """Lazy Nuke-style grade — applies per-frame ``grade_apply`` via torch.

    Both AM Grade (scalar knobs broadcast to ``(X, X, X)``) and AM Grade RGB
    (per-channel ``(X_r, X_g, X_b)``) wire to this single wrapper — the
    params are always 3-tuples here. Alpha (when present) passes through
    untouched.
    """

    def __init__(
        self,
        source,
        *,
        blackpoint: Tuple[float, float, float],
        whitepoint: Tuple[float, float, float],
        lift: Tuple[float, float, float],
        gain: Tuple[float, float, float],
        multiply: Tuple[float, float, float],
        offset: Tuple[float, float, float],
        gamma: Tuple[float, float, float],
        reverse: bool,
        black_clamp: bool,
        white_clamp: bool,
    ):
        super().__init__(source)
        self._blackpoint = tuple(blackpoint)
        self._whitepoint = tuple(whitepoint)
        self._lift = tuple(lift)
        self._gain = tuple(gain)
        self._multiply = tuple(multiply)
        self._offset = tuple(offset)
        self._gamma = tuple(gamma)
        self._reverse = bool(reverse)
        self._black_clamp = bool(black_clamp)
        self._white_clamp = bool(white_clamp)

    def _transform_frame(
        self, image_np: np.ndarray, alpha_np: Optional[np.ndarray],
    ) -> Frame:
        # grade_apply operates on torch tensors — convert per frame.
        # Cost: ~25 MB transfer for a 4K frame; negligible vs decode work.
        rgb_t = torch.from_numpy(image_np)
        # Build vec3 tensors matching the rgb dtype/device.
        def _v3(t):
            return torch.tensor(t, dtype=rgb_t.dtype, device=rgb_t.device)
        out_rgb = _grade.grade_apply(
            rgb_t,
            blackpoint=_v3(self._blackpoint),
            whitepoint=_v3(self._whitepoint),
            lift=_v3(self._lift),
            gain=_v3(self._gain),
            multiply=_v3(self._multiply),
            offset=_v3(self._offset),
            gamma=_v3(self._gamma),
            reverse=self._reverse,
            black_clamp=self._black_clamp,
            white_clamp=self._white_clamp,
        )
        return (
            np.ascontiguousarray(out_rgb.cpu().numpy().astype(np.float32, copy=False)),
            alpha_np,
        )

    def _rewrap(self, new_source) -> "GradedVideo":
        return GradedVideo(
            new_source,
            blackpoint=self._blackpoint, whitepoint=self._whitepoint,
            lift=self._lift, gain=self._gain, multiply=self._multiply,
            offset=self._offset, gamma=self._gamma,
            reverse=self._reverse,
            black_clamp=self._black_clamp,
            white_clamp=self._white_clamp,
        )


# ---------------------------------------------------------------------- #
#  Concrete subclass: Color Correct (Nuke-style sat/contrast/hue/...)
# ---------------------------------------------------------------------- #


class ColorCorrectedVideo(LazyVideoTransform):
    """Lazy Nuke-style ColorCorrect — applies per-frame
    :func:`color_correct.color_correct_apply` via torch.

    Same scalar knobs as :class:`AMColorCorrect`. Alpha (when present)
    passes through untouched.
    """

    def __init__(
        self,
        source,
        *,
        saturation: float,
        contrast: float,
        gamma: float,
        gain: float,
        offset: float,
        hue_degrees: float,
    ):
        super().__init__(source)
        self._saturation = float(saturation)
        self._contrast = float(contrast)
        self._gamma = float(gamma)
        self._gain = float(gain)
        self._offset = float(offset)
        self._hue_degrees = float(hue_degrees)

    def _transform_frame(
        self, image_np: np.ndarray, alpha_np: Optional[np.ndarray],
    ) -> Frame:
        rgb_t = torch.from_numpy(image_np)
        out_rgb = _color_correct.color_correct_apply(
            rgb_t,
            saturation=self._saturation,
            contrast=self._contrast,
            gamma=self._gamma,
            gain=self._gain,
            offset=self._offset,
            hue_degrees=self._hue_degrees,
        )
        return (
            np.ascontiguousarray(out_rgb.cpu().numpy().astype(np.float32, copy=False)),
            alpha_np,
        )

    def _rewrap(self, new_source) -> "ColorCorrectedVideo":
        return ColorCorrectedVideo(
            new_source,
            saturation=self._saturation,
            contrast=self._contrast,
            gamma=self._gamma,
            gain=self._gain,
            offset=self._offset,
            hue_degrees=self._hue_degrees,
        )


# ---------------------------------------------------------------------- #
#  Concrete subclass: OCIO transform (Colorspace OR LogConvert)
# ---------------------------------------------------------------------- #


class OCIOTransformVideo(LazyVideoTransform):
    """Lazy OCIO src→dst transform — applies ``ColorProcessor.apply_inplace``
    per frame to the image buffer. Alpha passes through untouched (OCIO is
    RGB-only). Used by both AM OCIO Colorspace (user-picked src/dst) and
    AM OCIO Log Convert (fixed scene_linear/compositing_log roles).
    """

    def __init__(self, source, *, src: str, dst: str, raw_data: bool = False):
        super().__init__(source)
        self._src = src
        self._dst = dst
        self._raw_data = bool(raw_data)
        # Build the processor once at construction — passes through identity
        # for any of the no-op cases (raw, src=dst, etc.).
        self._proc = None
        try:
            if (
                not raw_data
                and src != _color.PASSTHROUGH
                and dst != _color.PASSTHROUGH
                and src != dst
            ):
                proc = _color.ColorProcessor(src, dst)
                if not proc.is_identity:
                    self._proc = proc
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[am_vfx_tools/video_lazy] OCIO build %s -> %s failed (%s); "
                "wrapper will pass pixels through unchanged", src, dst, e,
            )

    def _transform_frame(
        self, image_np: np.ndarray, alpha_np: Optional[np.ndarray],
    ) -> Frame:
        if self._proc is None:
            return image_np, alpha_np
        # apply_inplace mutates the buffer; ensure we own a fp32 copy.
        out = image_np.astype(np.float32, copy=True)
        try:
            self._proc.apply_inplace(out)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[am_vfx_tools/video_lazy] OCIO apply failed mid-stream "
                "(%s); leaving frame untransformed", e,
            )
            return image_np, alpha_np
        return out, alpha_np

    def _rewrap(self, new_source) -> "OCIOTransformVideo":
        return OCIOTransformVideo(
            new_source, src=self._src, dst=self._dst, raw_data=self._raw_data,
        )


# ---------------------------------------------------------------------- #
#  Concrete subclass: Frame Range (filter, not per-pixel transform)
# ---------------------------------------------------------------------- #


# Frame-mode constants mirror am_frame_range.py to avoid a cross-module
# import dependency. Kept in lockstep — if the node renames a mode, this
# wrapper has to update too.
_FRANGE_MODE_SINGLE = "single"
_FRANGE_MODE_RANGE = "range"
_FRANGE_MODE_ALL = "all"


class FrameRangeVideo(LazyVideoTransform):
    """Lazy frame-range filter — yields a subset of source frames.

    Unlike the other lazy wrappers, this one **filters** the frame
    sequence rather than transforming pixel values. Three modes mirror
    AM Frame Range's IMAGE-branch semantics (1-based indexing):

    * ``single`` — yield exactly one frame at ``first_frame``; early-
      terminate the source iteration after that frame is yielded.
    * ``range``  — skip frames before ``first_frame``; yield frames in
      ``[first_frame, last_frame]``; early-terminate after the last
      requested frame is yielded. ``last_frame=-1`` means "to end".
    * ``all``    — pass through unchanged.

    The big win versus the eager IMAGE-branch slice: with a
    ``VideoFromFile`` source, PyAV decodes only the frames the wrapper
    actually needs. Frames past the requested range are never touched.
    Combined with the per-frame streaming consumption pattern, peak RAM
    stays at one frame regardless of source length.
    """

    def __init__(
        self,
        source,
        *,
        mode: str,
        first_frame: int,
        last_frame: int,
    ):
        super().__init__(source)
        self._mode = mode
        # Stored 1-based to match the node widgets; converted to 0-based
        # inside iter_frames() and get_frame_count().
        self._first_1b = int(first_frame)
        self._last_1b = int(last_frame)

    # --- Lazy iteration: override directly (we filter, not transform) --- #

    def iter_frames(self) -> Iterator[Frame]:
        if self._mode == _FRANGE_MODE_ALL:
            yield from self._iter_source_frames()
            return

        n = self._source_frame_count()
        if n <= 0:
            return

        f0 = max(0, min(self._first_1b - 1, n - 1))  # 0-based first

        if self._mode == _FRANGE_MODE_SINGLE:
            target = f0
            for i, frame in enumerate(self._iter_source_frames()):
                if i == target:
                    yield frame
                    return  # early terminate — saves decoding the rest
            return

        # range mode
        if self._last_1b < 0:
            l0 = n - 1
        else:
            l0 = max(0, min(self._last_1b - 1, n - 1))
        if l0 < f0:
            l0 = f0

        for i, frame in enumerate(self._iter_source_frames()):
            if i < f0:
                continue   # skip prefix
            if i > l0:
                return     # early terminate — saves decoding the rest
            yield frame

    # --- Metadata for output sockets / API consumers --- #

    def get_frame_count(self) -> int:
        if self._mode == _FRANGE_MODE_ALL:
            return self._source_frame_count()
        if self._mode == _FRANGE_MODE_SINGLE:
            return 1 if self._source_frame_count() > 0 else 0
        # range
        n = self._source_frame_count()
        if n <= 0:
            return 0
        f0 = max(0, min(self._first_1b - 1, n - 1))
        if self._last_1b < 0:
            l0 = n - 1
        else:
            l0 = max(0, min(self._last_1b - 1, n - 1))
        if l0 < f0:
            l0 = f0
        return l0 - f0 + 1

    def _source_frame_count(self) -> int:
        try:
            return int(self._source.get_frame_count())
        except Exception:  # noqa: BLE001 — defensive: unknown VideoInput subclass
            return 0

    # --- as_trimmed support --- #

    def _rewrap(self, new_source) -> "FrameRangeVideo":
        return FrameRangeVideo(
            new_source,
            mode=self._mode,
            first_frame=self._first_1b,
            last_frame=self._last_1b,
        )


__all__ = [
    "LazyVideoTransform",
    "ReformatVideo",
    "GradedVideo",
    "OCIOTransformVideo",
    "FrameRangeVideo",
    "Frame",
]
