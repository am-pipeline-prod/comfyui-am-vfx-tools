"""am-pipe-media-io._core.video_backend — PyAV decode/encode wrapper.

Pure I/O — no color management. Pair with :mod:`._core.color` to apply
``input_cs -> working_cs`` on read and ``working_cs -> output_cs`` on
write. Mirrors :mod:`._core.image_backend` in spirit.

Public surface:
  * :func:`is_available`         — True if PyAV is importable.
  * :func:`probe`                — quick header read; returns :class:`VideoInfo`.
  * :func:`read_video_frames`    — decode N contiguous video frames + optional audio.
  * :func:`write_video`          — encode an iterable of float32 RGB frames + optional audio.
  * :data:`CODECS`               — codec registry per the rework plan §6.2.
  * :func:`resolve_pixel_format` — ``(auto)`` -> codec-canonical pixfmt.
  * :func:`encoder_available`    — `av.codec.Codec(name, "w")` probe.
  * :class:`VideoInfo`           — width/height/fps/duration/codec/pixfmt/n_frames.
  * :class:`AudioBuffer`         — ``{waveform tensor, sample_rate}`` — matches ComfyUI's AUDIO socket dict shape.

PyAV 17.0.1 wraps system ffmpeg 5.1.8 in the workstation venv. All five
codecs in :data:`CODECS` (libx264 / libx265 / prores_ks / dnxhd /
libvpx-vp9) are probed once at module load; any missing encoder is
removed from the registry and a warning is logged. Container/codec
mismatch is enforced at queue-time with a clear error.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("am_vfx_tools.media-io.video")


try:
    import av  # type: ignore[import-not-found]
    _PYAV_AVAILABLE = True
except ImportError:
    av = None  # type: ignore[assignment]
    _PYAV_AVAILABLE = False


def is_available() -> bool:
    return _PYAV_AVAILABLE


# ---------------------------------------------------------------------------
# Codec registry (rework plan §6.2)
# ---------------------------------------------------------------------------

CODECS: Dict[str, Dict[str, Any]] = {
    "h264":   {"encoder": "libx264",
               "container_default": "mp4",
               "containers": ("mp4", "mov", "mkv"),
               "profiles": ["baseline", "main", "high", "high10"]},
    "h265":   {"encoder": "libx265",
               "container_default": "mp4",
               "containers": ("mp4", "mov", "mkv"),
               "profiles": ["main", "main10"]},
    "prores": {"encoder": "prores_ks",
               "container_default": "mov",
               "containers": ("mov",),
               "profiles": ["proxy", "lt", "422", "422hq", "4444", "4444xq"]},
    "dnxhr":  {"encoder": "dnxhd",
               "container_default": "mov",
               "containers": ("mov", "mxf"),
               "profiles": ["lb", "sq", "hq", "hqx", "444"]},
    "vp9":    {"encoder": "libvpx-vp9",
               "container_default": "webm",
               "containers": ("webm", "mkv"),
               "profiles": ["0", "2"]},
}


# (auto) pixel-format resolver — rework plan §6.2 table.
_AUTO_PIXFMT: Dict[Tuple[str, str], str] = {
    ("h264", "baseline"): "yuv420p",
    ("h264", "main"):     "yuv420p",
    ("h264", "high"):     "yuv420p",
    ("h264", "high10"):   "yuv420p10le",
    ("h265", "main"):     "yuv420p",
    ("h265", "main10"):   "yuv420p10le",
    ("prores", "proxy"):  "yuv422p10le",
    ("prores", "lt"):     "yuv422p10le",
    ("prores", "422"):    "yuv422p10le",
    ("prores", "422hq"):  "yuv422p10le",
    ("prores", "4444"):   "yuva444p10le",
    ("prores", "4444xq"): "yuva444p10le",
    ("dnxhr", "lb"):      "yuv422p",
    ("dnxhr", "sq"):      "yuv422p",
    ("dnxhr", "hq"):      "yuv422p",
    ("dnxhr", "hqx"):     "yuv422p10le",
    ("dnxhr", "444"):     "yuv444p10le",
    ("vp9", "0"):         "yuv420p",
    ("vp9", "2"):         "yuv420p10le",
}


# Numeric profile codes ffmpeg's prores_ks encoder expects.
# 0=proxy 1=lt 2=422 3=422hq 4=4444 5=4444xq.
_PRORES_PROFILE_CODE = {
    "proxy": "0", "lt": "1", "422": "2", "422hq": "3",
    "4444": "4", "4444xq": "5",
}

# DNxHR encoder (`dnxhd`) expects the literal `dnxhr_<profile>` family name
# via the ``profile`` option.
_DNXHR_PROFILE_CODE = {
    "lb": "dnxhr_lb", "sq": "dnxhr_sq", "hq": "dnxhr_hq",
    "hqx": "dnxhr_hqx", "444": "dnxhr_444",
}


def encoder_available(name: str) -> bool:
    """True if PyAV can build a write-side codec by name."""
    if not _PYAV_AVAILABLE:
        return False
    try:
        av.codec.Codec(name, "w")
        return True
    except Exception:
        return False


# Trim CODECS at import time to whatever this ffmpeg actually has.
if _PYAV_AVAILABLE:
    _missing = [k for k, v in CODECS.items() if not encoder_available(v["encoder"])]
    for k in _missing:
        log.warning(
            "[am-vfx-tools/video] encoder %r missing in this ffmpeg build; "
            "dropping codec %r from the registry",
            CODECS[k]["encoder"], k,
        )
        CODECS.pop(k, None)


def resolve_pixel_format(codec: str, profile: str, requested: Optional[str]) -> str:
    """Resolve auto to the codec/profile-canonical pixfmt.

    Empty / None / the literal ``(auto)`` token all trigger auto resolution.
    Any other value is passed through with a quick sanity check that the
    format exists in PyAV.
    """
    auto_tokens = ("", "(auto)", "auto", None)
    norm = requested.strip() if isinstance(requested, str) else requested
    if norm not in auto_tokens:
        if _PYAV_AVAILABLE:
            try:
                av.video.format.VideoFormat(norm)
            except Exception as e:
                raise ValueError(
                    f"Unknown pixel_format {requested!r}: {e}. "
                    f"Leave empty for (auto), or use a standard FFmpeg pixfmt name."
                )
        return norm
    key = (codec, profile)
    if key not in _AUTO_PIXFMT:
        raise ValueError(
            f"No (auto) pixel format mapping for codec={codec!r} "
            f"profile={profile!r}. Set pixel_format explicitly."
        )
    return _AUTO_PIXFMT[key]


def codec_profile_supports_alpha(codec: str, profile: str) -> bool:
    """Return True if the codec/profile combination encodes an alpha channel.

    Resolves the auto pixel format for ``(codec, profile)`` and checks
    whether the resolved pixfmt is alpha-bearing (``yuva*`` / ``rgba``-
    family). Used by AM Video Write to decide whether to honor a wired
    MASK input vs log a "mask dropped" warning.

    Currently true for: ProRes 4444 / 4444 XQ. False for everything
    else in the active codec table.
    """
    key = (codec, profile)
    if key not in _AUTO_PIXFMT:
        return False
    pf = _AUTO_PIXFMT[key].lower()
    return "yuva" in pf or pf.startswith(("argb", "rgba", "abgr", "bgra"))


def validate_container_codec(filepath: str, codec: str) -> None:
    """Raise a clear error when *filepath*'s extension can't host *codec*.

    Matches the rework plan §6.2 example error.
    """
    if codec not in CODECS:
        raise ValueError(
            f"Unknown codec {codec!r}. Known: {sorted(CODECS)}."
        )
    ext = os.path.splitext(filepath)[1].lstrip(".").lower()
    allowed = CODECS[codec]["containers"]
    if ext and ext not in allowed:
        default_ext = CODECS[codec]["container_default"]
        raise ValueError(
            f"{codec} requires .{default_ext} container; got .{ext}. "
            f"Change ext to {default_ext}, or change codec."
        )


# ---------------------------------------------------------------------------
# Dataclasses — light-weight return types
# ---------------------------------------------------------------------------


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float                 # seconds; 0 if unknown
    codec: str                      # ffmpeg codec name (e.g. "prores", "h264")
    pix_fmt: Optional[str]
    n_frames: int                   # 0 if unknown / streaming
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioBuffer:
    """Mirrors ComfyUI's AUDIO socket dict shape.

    Per ComfyUI convention the waveform is a ``torch.Tensor`` of shape
    ``(batch=1, channels, samples)`` in [-1, 1] float32. We hold a numpy
    array internally; :meth:`as_comfy_audio` wraps it as a torch tensor
    for socket emission.
    """
    waveform: Any                   # numpy.ndarray (1, C, N) float32
    sample_rate: int

    def as_comfy_audio(self) -> Dict[str, Any]:
        try:
            import torch  # type: ignore
            wf = torch.as_tensor(self.waveform, dtype=torch.float32)
        except ImportError:
            wf = self.waveform
        return {"waveform": wf, "sample_rate": int(self.sample_rate)}


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def _require_pyav() -> None:
    if not _PYAV_AVAILABLE:
        raise RuntimeError(
            "PyAV (`av`) is required for video I/O. The ComfyUI venv ships "
            "PyAV 17.x; system Python may not."
        )


def probe(filepath: str) -> VideoInfo:
    """Quick header read."""
    _require_pyav()
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    container = av.open(filepath)
    try:
        if not container.streams.video:
            raise RuntimeError(f"No video stream in {filepath}")
        vs = container.streams.video[0]
        try:
            fps = float(vs.average_rate) if vs.average_rate else 0.0
        except Exception:
            fps = 0.0
        try:
            dur = float(container.duration) / 1_000_000.0 if container.duration else 0.0
        except Exception:
            dur = 0.0
        return VideoInfo(
            width=int(vs.width or 0),
            height=int(vs.height or 0),
            fps=fps,
            duration=dur,
            codec=vs.codec_context.name if vs.codec_context else "",
            pix_fmt=str(vs.codec_context.pix_fmt) if vs.codec_context and vs.codec_context.pix_fmt else None,
            n_frames=int(vs.frames or 0),
            metadata=dict(container.metadata) if container.metadata else {},
        )
    finally:
        container.close()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_video_frames(
    filepath: str,
    *,
    start: int = 0,
    count: Optional[int] = None,
    audio_track: Optional[int] = 0,
) -> Tuple[Any, Optional[AudioBuffer], VideoInfo]:
    """Decode a contiguous range of frames into a ``(N, H, W, C)`` float32
    array in [0, 1].

    *C* is 3 for RGB-only sources and 4 for sources whose pixel format
    carries alpha (ProRes 4444 / 4444 XQ → ``yuva*``, QuickTime RLE →
    ``argb``/``rgba``, FFV1 → ``yuva*``, etc). The caller checks
    ``stack.shape[-1]`` to decide whether to emit a populated MASK
    socket. Detection is via ``VideoFormat.has_alpha`` (PyAV 14+); legacy
    PyAV without that attribute falls back to a substring match on the
    format name (``"a"`` in ``"yuva444p10le"``, etc).

    *start* — frame index (0-based) to begin emitting at.
    *count* — number of frames to emit; ``None`` = decode through EOF.
    *audio_track* — audio stream index to also decode; ``None`` or ``-1``
        for "no audio".

    OCIO transforms are NOT applied here — the caller does that.
    """
    _require_pyav()
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)

    import numpy as np

    info = probe(filepath)
    container = av.open(filepath)
    frames: List[Any] = []
    audio: Optional[AudioBuffer] = None
    try:
        if not container.streams.video:
            raise RuntimeError(f"No video stream in {filepath}")

        # Probe whether the source carries alpha. PyAV exposes this via
        # the codec context's `format.has_alpha` property; fall back to
        # a name-substring heuristic for older PyAV builds.
        has_alpha = False
        try:
            vstream = container.streams.video[0]
            fmt = getattr(vstream.codec_context, "format", None) or getattr(vstream, "format", None)
            if fmt is not None:
                has_alpha = bool(getattr(fmt, "has_alpha", False))
                if not has_alpha:
                    name = (getattr(fmt, "name", "") or "").lower()
                    # Common alpha-bearing pixfmts: yuva444p10le / yuva444p / argb / rgba / bgra / etc.
                    has_alpha = (
                        "yuva" in name or name.startswith(("argb", "rgba", "abgr", "bgra"))
                    )
        except Exception:  # noqa: BLE001 — defensive: keep going as RGB on any probe error
            has_alpha = False

        decode_format = "rgba" if has_alpha else "rgb24"
        out_channels = 4 if has_alpha else 3

        emitted = 0
        seen = 0
        for frame in container.decode(video=0):
            if seen < start:
                seen += 1
                continue
            if count is not None and emitted >= count:
                break
            arr = frame.to_ndarray(format=decode_format)  # (H, W, 3 or 4) uint8
            frames.append(arr.astype(np.float32) / 255.0)
            emitted += 1
            seen += 1

        if not frames:
            stack = np.zeros(
                (0, info.height or 1, info.width or 1, out_channels), dtype=np.float32,
            )
        else:
            stack = np.stack(frames, axis=0)

        if audio_track is not None and audio_track >= 0 and container.streams.audio:
            try:
                audio = _decode_audio(container, audio_track)
            except Exception as e:
                log.warning("[am-vfx-tools/video] audio decode failed (%s); audio dropped", e)
                audio = None
    finally:
        container.close()

    return stack, audio, info


def _decode_audio(container, track_index: int) -> Optional[AudioBuffer]:
    """Decode the requested audio track into a packed ``(1, C, N)`` float32 array."""
    import numpy as np

    if track_index >= len(container.streams.audio):
        log.warning(
            "[am-vfx-tools/video] requested audio_track=%d but only %d audio stream(s)",
            track_index, len(container.streams.audio),
        )
        return None

    astream = container.streams.audio[track_index]
    sample_rate = int(astream.rate or 48000)

    chunks: List[Any] = []
    n_chan = 0
    container.seek(0)
    for frame in container.decode(audio=track_index):
        nd = frame.to_ndarray()
        # PyAV returns (channels, samples) for planar formats and
        # (samples * channels,) interleaved for packed formats. Normalize.
        if nd.ndim == 1:
            chans = frame.layout.channels if frame.layout else 1
            try:
                chans = len(chans) if hasattr(chans, "__len__") else int(chans)
            except Exception:
                chans = 1
            if chans > 0:
                nd = nd.reshape(-1, chans).T
            else:
                nd = nd[None, :]
        if nd.ndim == 2 and n_chan == 0:
            n_chan = nd.shape[0]
        # Convert to float32 [-1, 1]
        if nd.dtype.kind == "i":
            denom = float(np.iinfo(nd.dtype).max)
            nd = nd.astype(np.float32) / denom
        elif nd.dtype.kind == "u":
            denom = float(np.iinfo(nd.dtype).max)
            nd = (nd.astype(np.float32) - denom / 2.0) / (denom / 2.0)
        else:
            nd = nd.astype(np.float32)
        chunks.append(nd)

    if not chunks:
        return None

    waveform = np.concatenate(chunks, axis=-1)
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    waveform = waveform[None, ...]  # (1, C, N) for ComfyUI
    return AudioBuffer(waveform=waveform, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def _apply_codec_options(
    stream, codec: str, profile: str, bitrate_or_crf: str, gop_size: int
) -> None:
    """Push codec-specific options onto *stream*.codec_context.options."""
    opts: Dict[str, str] = {}

    if codec == "h264" or codec == "h265":
        if profile:
            opts["profile"] = profile
        if bitrate_or_crf:
            br = bitrate_or_crf.strip()
            if br.lower().startswith("crf="):
                opts["crf"] = br.split("=", 1)[1]
            elif br.isdigit() or any(br.lower().endswith(s) for s in ("k", "m")):
                stream.bit_rate = _parse_bitrate(br)
            else:
                opts["crf"] = br  # bare number = CRF for x264/x265
    elif codec == "prores":
        opts["profile"] = _PRORES_PROFILE_CODE.get(profile, "2")
        opts.setdefault("vendor", "ap10")
        if bitrate_or_crf:
            br = bitrate_or_crf.strip()
            if br.lower().startswith("crf"):
                log.warning(
                    "[am-vfx-tools/video] codec=prores ignores %r — ProRes is fixed-bitrate "
                    "per profile. Use a bitrate like '120M' or leave empty.", bitrate_or_crf,
                )
            else:
                stream.bit_rate = _parse_bitrate(br)
    elif codec == "dnxhr":
        opts["profile"] = _DNXHR_PROFILE_CODE.get(profile, "dnxhr_hq")
        if bitrate_or_crf:
            br = bitrate_or_crf.strip()
            if br.lower().startswith("crf"):
                log.warning(
                    "[am-vfx-tools/video] codec=dnxhr ignores %r — DNxHR is fixed-bitrate "
                    "per profile/resolution. Use a bitrate like '120M' or leave empty.",
                    bitrate_or_crf,
                )
            else:
                stream.bit_rate = _parse_bitrate(br)
    elif codec == "vp9":
        if profile:
            opts["profile"] = profile
        if bitrate_or_crf:
            br = bitrate_or_crf.strip()
            if br.lower().startswith("crf="):
                opts["crf"] = br.split("=", 1)[1]
                opts["b"] = "0"  # CRF mode for VP9 needs explicit zero target bitrate
            else:
                stream.bit_rate = _parse_bitrate(br)

    if gop_size and gop_size > 0:
        stream.codec_context.gop_size = int(gop_size)

    if opts:
        stream.codec_context.options = opts


def _parse_bitrate(s: str) -> int:
    """`8M` -> 8_000_000, `500k` -> 500_000, `8000000` -> 8_000_000."""
    s = s.strip().lower()
    if not s:
        return 0
    if s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    if s.endswith("k"):
        return int(float(s[:-1]) * 1_000)
    return int(float(s))


def write_video(
    filepath: str,
    frames_iter: Iterable[Any],
    *,
    codec: str,
    codec_profile: str,
    pixel_format: Optional[str],
    frame_rate: float,
    bitrate_or_crf: str = "",
    gop_size: int = 0,
    audio_buffer: Optional[AudioBuffer] = None,
    color_space_tag: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    workflow_metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Encode an iterable of ``(H, W, 3)`` float32 RGB frames in [0, 1].

    Validates the container/codec match before opening. Frames are
    converted to uint8 RGB24 then reformatted into the resolved pixfmt
    by libswscale via PyAV's ``frame.reformat`` — same path PyAV's own
    examples use.

    *workflow_metadata* — dict of string→string entries (typically
    ``{"prompt": json_str, "workflow": json_str}`` matching ComfyUI's
    drag-drop convention) embedded as container-level metadata. For
    ISOBMFF containers (mp4/mov) this needs ``movflags=use_metadata_tags``
    or FFmpeg silently drops user-defined keys — handled below. MKV/WebM
    accept arbitrary tags natively (Matroska normalizes the key names to
    UPPERCASE on round-trip; loaders should be case-insensitive).
    """
    _require_pyav()
    if codec not in CODECS:
        raise ValueError(
            f"Unknown codec {codec!r}. Known: {sorted(CODECS)}."
        )

    validate_container_codec(filepath, codec)

    profiles = CODECS[codec]["profiles"]
    if codec_profile not in profiles:
        raise ValueError(
            f"{codec}: unknown profile {codec_profile!r}. "
            f"Allowed: {profiles}."
        )

    pix_fmt = resolve_pixel_format(codec, codec_profile, pixel_format)
    encoder_name = CODECS[codec]["encoder"]
    fps = float(frame_rate) if frame_rate and frame_rate > 0 else 25.0

    out_dir = os.path.dirname(filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    import numpy as np

    iterator = iter(frames_iter)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("write_video: empty frame iterator")

    first = np.asarray(first)
    if first.ndim != 3 or first.shape[-1] not in (3, 4):
        raise ValueError(
            f"write_video: expected (H,W,3|4) frames, got shape {first.shape}"
        )
    height, width = int(first.shape[0]), int(first.shape[1])

    # ISOBMFF (mp4/mov) needs ``use_metadata_tags`` or FFmpeg's mov muxer
    # silently drops any non-built-in metadata key. Mirrors ComfyUI's stock
    # SaveVideo/SaveWEBM nodes. Harmless for other containers (the option
    # is simply ignored by non-mov muxers) but we still gate by extension
    # to keep av.open's argv minimal.
    open_options: Dict[str, str] = {}
    container_ext = os.path.splitext(filepath)[1].lstrip(".").lower()
    if container_ext in ("mp4", "mov"):
        open_options["movflags"] = "use_metadata_tags"

    container = av.open(filepath, mode="w", options=open_options or None)
    if metadata:
        try:
            container.metadata.update({str(k): str(v) for k, v in metadata.items()})
        except Exception:
            pass
    if workflow_metadata:
        try:
            for k, v in workflow_metadata.items():
                container.metadata[str(k)] = str(v)
        except Exception:
            log.warning(
                "[am-vfx-tools/video] failed to set workflow metadata on %s",
                filepath,
            )

    # PyAV's add_stream wants a Fraction-compatible rate, not a float.
    from fractions import Fraction
    stream_rate = Fraction(fps).limit_denominator(1001)
    stream = container.add_stream(encoder_name, rate=stream_rate)
    stream.width = width
    stream.height = height
    stream.pix_fmt = pix_fmt
    if color_space_tag:
        try:
            stream.codec_context.options = {
                **(stream.codec_context.options or {}),
                "color_primaries": _ocio_to_color_primaries(color_space_tag),
            }
        except Exception:
            pass
    _apply_codec_options(stream, codec, codec_profile, bitrate_or_crf, gop_size)

    # Resolve whether the chosen pixfmt carries alpha. The pixfmt is the
    # ground truth — `yuva*` / `*rgba`-variants → alpha, anything else
    # → drop alpha at the encoder boundary (current behavior). Lets
    # ProRes 4444 / 4444 XQ / FFV1 / QT RLE pass alpha through cleanly
    # while keeping every other codec lossy-RGB.
    _pixfmt_lower = (pix_fmt or "").lower()
    _alpha_pixfmt = (
        "yuva" in _pixfmt_lower
        or _pixfmt_lower.startswith(("argb", "rgba", "abgr", "bgra"))
    )

    def _emit(arr) -> None:
        arr = np.asarray(arr)
        if arr.ndim != 3:
            raise ValueError(f"frame must be (H,W,C); got shape {arr.shape}")
        if _alpha_pixfmt:
            # Promote 3-ch → 4-ch with opaque alpha so a non-mask-wired
            # frame still encodes cleanly into an alpha-bearing pixfmt.
            if arr.shape[-1] == 3:
                opaque = np.ones(arr.shape[:2] + (1,), dtype=arr.dtype)
                arr = np.concatenate([arr, opaque], axis=-1)
            from_format = "rgba"
        else:
            # Strip alpha for non-alpha pixfmts (existing behavior).
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            from_format = "rgb24"
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0 + 0.5).astype(np.uint8)
        arr = np.ascontiguousarray(arr)
        frame = av.VideoFrame.from_ndarray(arr, format=from_format)
        frame = frame.reformat(format=pix_fmt, width=width, height=height)
        for packet in stream.encode(frame):
            container.mux(packet)

    try:
        _emit(first)
        for arr in iterator:
            _emit(arr)
        for packet in stream.encode():
            container.mux(packet)

        if audio_buffer is not None:
            try:
                _mux_audio(container, audio_buffer, container_ext=os.path.splitext(filepath)[1])
            except Exception as e:
                log.warning(
                    "[am-vfx-tools/video] audio mux failed (%s); video written without audio", e,
                )
    finally:
        container.close()


def _mux_audio(container, audio: AudioBuffer, *, container_ext: str) -> None:
    """Encode and mux an :class:`AudioBuffer` into *container*.

    Default audio codec depends on container: aac for mp4/mov/mkv,
    libopus for webm.
    """
    import numpy as np

    ext = container_ext.lstrip(".").lower()
    audio_codec = "libopus" if ext == "webm" else "aac"
    if not encoder_available(audio_codec):
        log.warning("[am-vfx-tools/video] audio encoder %s missing; skipping audio", audio_codec)
        return

    waveform = np.asarray(audio.waveform)
    # Accept (1, C, N), (C, N), or (N,)
    if waveform.ndim == 3 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    if waveform.ndim != 2:
        log.warning(
            "[am-vfx-tools/video] audio waveform shape %s unsupported; skipping",
            waveform.shape,
        )
        return

    n_chan, n_samp = waveform.shape
    if n_chan not in (1, 2):
        # Fold to stereo if more than 2; let single-channel pass.
        log.warning(
            "[am-vfx-tools/video] audio with %d channels — mixing down to stereo", n_chan,
        )
        waveform = waveform[:2]
        n_chan = 2

    layout = "mono" if n_chan == 1 else "stereo"
    sample_rate = int(audio.sample_rate)
    astream = container.add_stream(audio_codec, rate=sample_rate)
    astream.layout = layout
    # AAC/Opus are encoded as fltp (planar float). Build interleaved-then-frame.
    target_format = "fltp"

    # Slice into encoder-friendly chunks. Use a moderate chunk size; the
    # encoder will rebuffer to its frame_size internally.
    chunk = 4096
    for start in range(0, n_samp, chunk):
        block = waveform[:, start:start + chunk]
        if block.size == 0:
            continue
        # PyAV wants (channels, samples) for planar formats — already that shape.
        block = np.ascontiguousarray(block.astype(np.float32))
        af = av.AudioFrame.from_ndarray(block, format=target_format, layout=layout)
        af.sample_rate = sample_rate
        for packet in astream.encode(af):
            container.mux(packet)
    for packet in astream.encode():
        container.mux(packet)


def _ocio_to_color_primaries(cs_tag: str) -> str:
    """Best-effort OCIO -> ffmpeg color_primaries mapping.

    Only used as a metadata hint; failure is non-fatal upstream.
    """
    s = cs_tag.lower()
    if "rec.2020" in s or "rec2020" in s:
        return "bt2020"
    if "p3" in s:
        return "smpte432"
    return "bt709"


__all__ = [
    "AudioBuffer",
    "CODECS",
    "VideoInfo",
    "encoder_available",
    "is_available",
    "probe",
    "read_video_frames",
    "resolve_pixel_format",
    "validate_container_codec",
    "write_video",
]
