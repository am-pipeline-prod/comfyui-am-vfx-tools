"""AM Video Write — ComfyUI node.

Encodes an input IMAGE batch to a video container at the literal
``file_path``. Use the Browse button to populate the path from the
native dialog.

Frame-range knobs match AM Write Image symmetrically (frame_mode /
first_frame / last_frame), with two deletions:

* No ``start_frame`` / ``frame_padding`` — a video write produces a
  *single* container, not a numbered sequence; per-frame file labels
  don't exist.
* No ``bit_depth`` / ``compression`` — replaced by codec/profile/
  pixfmt/bitrate/GOP knobs because video encodes are codec-driven.

``frame_rate`` IS an input knob here (unlike AM Read Video where it's
an output socket) — on write, the artist chooses the OUTPUT
container's time base. The optional ``audio`` socket muxes an audio
track if connected.

Encode + audio mux uses :mod:`._core.video_backend` (PyAV) and
:mod:`._core.color` (OCIO 2.x).

The ``use_batch`` toggle (default ``False``) controls a runtime-discovered
``_bNNNN`` suffix on the output filename — same shape as on AM Image
Write. ComfyUI's queue "Batch count" produces N execute() calls; without
a per-iteration suffix every video would clobber the previous one. When
on, the node scans the output dir for files matching the rendered stem
with a varying ``_bNNNN`` segment, picks ``max+1``, and writes there.
The integer is NOT a workflow widget — it's discovered from disk per
execute(), mirroring stock ComfyUI's filename counter. The slot format
``_b{N:04d}`` matches the dcc-core grammar (``project-structure.md
§5.5.6``) — lowercase ``_b`` followed by four digits, no key/value
separator. When off (default), queue iterations overwrite each other
— the toggle is opt-in for queue Batch-count >1 runs.

The ``seed`` parameter (default ``-1``) is metadata-only — it never
flows into the filename. When set to ``-1``, the node scans the
active ``prompt`` for ``AMSeed`` class entries and looks each one up
in the process-global ``_core.seed_registry`` (populated by AM Seed's
``IS_CHANGED`` before any node's ``execute()`` runs — so the value
is guaranteed available regardless of execution order between AM
Seed and this write node). With no AM Seed in the graph the lookup
returns ``None`` and the seed key is omitted entirely from the
file's metadata. Any non-``-1`` value
(typed widget OR delivered by a wire from convert-widget-to-input)
wins outright over the registry — the wire path is also how artists
guarantee execution-order pinning in graphs where the AM Seed node
would otherwise be unreachable from this write node.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from ._core import (
    batch_suffix, color, preview, reformat, seed_registry, video_backend,
)

# ComfyUI's global ``--disable-metadata`` flag — same kill-switch that gates
# stock SaveImage/SaveVideo. Imported defensively so the module still loads
# under the dcc-core test runner where comfy is absent.
try:
    from comfy.cli_args import args as _comfy_args  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    _comfy_args = None

log = logging.getLogger("am_vfx_tools.write_video")


FRAME_MODE_SINGLE = "single"
FRAME_MODE_RANGE  = "range"
FRAME_MODE_ALL    = "all"
_FRAME_MODES = [FRAME_MODE_SINGLE, FRAME_MODE_RANGE, FRAME_MODE_ALL]


# Codec dropdown order matches the rework plan §6.2.
_CODEC_ORDER = ["h264", "h265", "prores", "dnxhr", "vp9"]


# Container ↔ codec dropdown grouping (2026-04-29 rework). The codec
# dropdown is now ``container/codec`` so the entry tells the artist
# which ``ext`` the codec naturally pairs with — same convention as
# the ``ext``-grouped compression dropdown on AM Write Image and the
# ``codec/profile`` dropdown directly below this one. Order in the
# tuple drives dropdown order on the canvas.
#
# Why one container per codec (rather than every valid combo):
#   FFmpeg accepts h264/h265 inside MOV / MKV / MP4 — listing every
#   permutation triples the menu without actionable benefit. The
#   guidance baked into the dropdown is "this is the canonical
#   container for this codec", not "this is the only legal container".
#   `validate_container_codec` is permissive and still accepts
#   alternative legal pairings (e.g. an artist who picks ``ext=mkv``
#   with ``codec=mp4/h265`` — bare ``h265`` is valid in MKV too).
#
# MKV pairing: H.265 (HEVC) is the most common MKV-native choice in
# the wild (HEVC remuxes, archival masters); listed as a duplicate of
# the mp4/h265 entry so both containers have a discoverable codec
# entry from the menu. The user can opt for ``mp4/h265`` if they're
# targeting MP4 instead. Future codec additions for MKV (FFV1, AV1)
# would slot in here once the backend supports them.
_CONTAINER_CODEC_PAIRS = (
    ("mov",  "prores"),
    ("mov",  "dnxhr"),
    ("mp4",  "h264"),
    ("mp4",  "h265"),
    ("mkv",  "h265"),
    ("webm", "vp9"),
)


# Pre-2026-04-29 saved workflows carry the unprefixed codec value.
# Translated at execute() time so existing workflows keep working.
_LEGACY_CODEC_MAP = {
    "h264":   "mp4/h264",
    "h265":   "mp4/h265",
    "prores": "mov/prores",
    "dnxhr":  "mov/dnxhr",
    "vp9":    "webm/vp9",
}


def _strip_codec_prefix(value: str) -> str:
    """Translate the dropdown value to the bare codec token.

    ``mov/prores`` → ``prores``. Legacy unprefixed values (saved
    workflows from before the 2026-04-29 prefix rework) translate
    via :data:`_LEGACY_CODEC_MAP`.
    """
    if value is None:
        return ""
    canon = _LEGACY_CODEC_MAP.get(value, value)
    return canon.split("/", 1)[1] if "/" in canon else canon


def _codec_choices():
    """Container-prefixed codec dropdown — ``mov/prores``, ``mp4/h264``, ...

    Filtered against the backend's actually-loaded codec set so a
    dropdown entry never points at a codec PyAV / FFmpeg can't reach.
    """
    out = []
    for container, codec in _CONTAINER_CODEC_PAIRS:
        if codec in video_backend.CODECS:
            out.append(f"{container}/{codec}")
    return out or [f"{c}/{k}" for c, k in _CONTAINER_CODEC_PAIRS]


def _profile_choices():
    """Codec-prefixed profile dropdown — ``prores/422``, ``h264/main``, ...

    Prefix tells the artist which codec each profile belongs to and
    prevents silently invalid combos like ``codec=h264 + profile=422``.
    The prefix is the BARE codec name (no container) — at execute()
    we strip the container prefix from `codec` and the bare prefix
    from `codec_profile` and verify they agree. Listed once per
    bare codec even if multiple containers point to it (e.g.
    ``h265/main`` covers both ``mp4/h265`` and ``mkv/h265``).
    """
    out: list = []
    for c in _CODEC_ORDER:
        spec = video_backend.CODECS.get(c)
        if not spec:
            continue
        for p in spec["profiles"]:
            out.append(f"{c}/{p}")
    return out


# Placeholder hints — surfaced as gray-text guidance when the field is
# empty, same UX pattern the ``template`` widget uses.
_PIXFMT_HINT = (
    "(empty = (auto) per codec/profile; or yuv420p / yuv420p10le / "
    "yuv422p / yuv422p10le / yuv444p10le / yuva444p10le / yuv444p / yuv420p12le)"
)
_BITRATE_HINT = (
    "(empty = codec default. For h264/h265/vp9: 'crf=18' (quality, lower=better) "
    "or '8M' / '500k' / '8000000' (bitrate). For prores/dnxhr: bitrate only, "
    "e.g. '120M' (CRF ignored)."
)
_GOP_HINT = "0 = codec default (typically 250 for h264/h265). Lower = more keyframes, larger files."


class AMVideoWrite:
    @classmethod
    def INPUT_TYPES(cls):
        cs = color.color_space_choices()
        codecs = _codec_choices()
        profiles = _profile_choices()

        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Absolute output container path",
                    "tooltip": (
                        "Absolute output container path. "
                        "Use the 📂 Browse button for the native dialog."
                    ),
                }),
                "ext": (["mov", "mp4", "mkv", "webm"], {
                    "default": "mov",
                    "tooltip": (
                        "Container format. The codec dropdown below is filtered to "
                        "codecs the chosen container natively pairs with."
                    ),
                }),
                # `seed` is unconditional — no `use_seed` toggle (removed
                # 2026-04-28). Mirrors AM Image Write. Filename never
                # carries seed; seed only feeds metadata embed via the
                # `embed_workflow` toggle. Sentinel `-1` = "unset" → look
                # up the process-global seed_registry by id(prompt) (an
                # AM Seed node in the graph publishes under the same
                # key); registry hit emits `comfyui/seed`, registry miss
                # omits it. Any other value (typed widget OR wired-in via
                # convert-widget-to-input) wins outright over the
                # registry. The frontend extension splices out the
                # auto-generated `control_after_generate` widget for this
                # node — it's obsolete now that AM Seed owns mode/
                # increment/decrement/randomize semantics server-side.
                "seed": ("INT", {
                    "default": -1, "min": -2**63, "max": 2**63 - 1,
                    "tooltip": (
                        "Generation seed — metadata-only, never in the filename. "
                        "-1 = look up the AM Seed registry by id(prompt). "
                        "Any other value (typed or wired) wins over the registry."
                    ),
                }),
                # use_batch toggles a runtime-discovered `_bNNNN` slot
                # (per-execute() unique suffix; mirrors stock ComfyUI's
                # filename counter). Default False — opt-in for queue
                # Batch-count >1 runs. There is intentionally NO paired
                # `batch` INT widget — the value is scanned from the
                # output folder, not artist-set. See module docstring.
                "use_batch": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When On, append a runtime-discovered `_bNNNN` suffix "
                        "(queue-iteration counter, scanned from the output dir). "
                        "Off = no suffix; queue iterations overwrite each other."
                    ),
                }),
                "frame_mode": (_FRAME_MODES, {
                    "default": FRAME_MODE_ALL,
                    "tooltip": (
                        "Which input batch frames to encode. "
                        "single = only `first_frame`. "
                        "range = `first_frame`..`last_frame` inclusive. "
                        "all = every frame in the input batch."
                    ),
                }),
                # Frame rate sits directly under `frame_mode` and above
                # `first_frame` — locked across the family (see
                # media-io-sync-rule.md invariant 14b). Video writes
                # always need a real fps for the encoded container.
                "frame_rate": ("FLOAT", {
                    "default": 25.0, "min": 0.1, "max": 480.0,
                    "tooltip": "Output container's encoded time base.",
                }),
                "first_frame": ("INT", {
                    "default": 1, "min": 1, "max": 999999,
                    "tooltip": (
                        "Input batch index (1-based) for single mode; lower bound "
                        "for range mode. Ignored in all mode."
                    ),
                }),
                "last_frame": ("INT", {
                    "default": -1, "min": -1, "max": 999999,
                    "tooltip": (
                        "Input batch upper bound (1-based, inclusive) for range mode. "
                        "-1 = auto = batch length."
                    ),
                }),
                "codec": (codecs, {
                    "default": "mov/prores" if "mov/prores" in codecs else codecs[0],
                    "tooltip": (
                        "Encoding codec, prefixed by its canonical container "
                        "(`mov/prores`, `mp4/h264`, ...). The container half is "
                        "guidance — `validate_container_codec` permits other legal "
                        "pairings if `ext` is set differently."
                    ),
                }),
                "codec_profile": (profiles, {
                    "default": "prores/422" if "prores/422" in profiles else profiles[0],
                    "tooltip": (
                        "Codec profile, prefixed by the bare codec name "
                        "(`prores/422`, `h264/main`). Must match the codec chosen "
                        "above — mismatched prefix raises at execute time."
                    ),
                }),
                "pixel_format": ("STRING", {
                    "default": "", "multiline": False,
                    "placeholder": _PIXFMT_HINT,
                    "tooltip": (
                        "Output pixel format. Empty = (auto) per codec/profile. "
                        "Override for specific subsampling / bit depth (e.g. "
                        "`yuv422p10le` for 10-bit 4:2:2)."
                    ),
                }),
                "bitrate_or_crf": ("STRING", {
                    "default": "", "multiline": False,
                    "placeholder": _BITRATE_HINT,
                    "tooltip": (
                        "Quality / bitrate setting. Empty = codec default. "
                        "h264/h265/vp9: `crf=18` (quality, lower=better) or `8M` / "
                        "`500k` (bitrate). prores/dnxhr: bitrate only (CRF ignored)."
                    ),
                }),
                "gop_size": ("INT", {
                    "default": 0, "min": 0, "max": 600,
                    "tooltip": _GOP_HINT,
                }),
                "working_colorspace": (cs, {
                    "default": color.default_working_colorspace(cs),
                    "tooltip": (
                        "Source colorspace — the space the upstream IMAGE tensor "
                        "is in. The OCIO transform converts from this to "
                        "`output_colorspace`."
                    ),
                }),
                "raw_data": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When On, skip the OCIO transform — pixels encoded verbatim. "
                        "`working_colorspace` and `output_colorspace` are ignored."
                    ),
                }),
                "output_colorspace": (cs, {
                    "default": color.pick_default(
                        cs, ("Gamma 2.2 Rec.709 - Display", "Rec.1886 Rec.709 - Display"),
                    ),
                    "tooltip": (
                        "Destination colorspace. The OCIO transform converts to this, "
                        "and the value is tagged into the encoded container metadata "
                        "(default `Gamma 2.2 Rec.709 - Display` — the studio dailies "
                        "standard, falling back to `Rec.1886 Rec.709 - Display` on "
                        "OCIO configs without the gamma 2.2 display variant)."
                    ),
                }),
                # When True, embed the workflow's API graph (`prompt`) AND
                # editor graph (`workflow`) into the container as
                # ComfyUI-style metadata. Default True mirrors stock
                # ComfyUI's SaveVideo. All four supported containers
                # (mov/mp4/mkv/webm) preserve the metadata cleanly — mp4/mov
                # via the FFmpeg ``movflags=use_metadata_tags`` option that
                # video_backend.write_video sets automatically.
                "embed_workflow": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Embed the API graph (`comfyui/prompt`) as container "
                        "metadata so the file is round-tripped via AM-Pipe "
                        "drag-drop. Off = clean deliverable. Honors ComfyUI's "
                        "global `--disable-metadata` flag."
                    ),
                }),
                # ----- Reformat block (synced across all four IO nodes; see
                # media-io-sync-rule.md invariants 15a–15e). Eight widgets,
                # always visible regardless of mode (ComfyUI doesn't support
                # conditional widget hiding). On a Write node the reformat
                # is applied to the input batch BEFORE encoding — common
                # case: AI-generated 1024×576 → deliver as 1920×1080.
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
                # ----- end reformat block
                "show_preview": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show a thumbnail of the first encoded frame on the node.",
                }),
                # Round-trip / read-only toggle — mirror of AM Image Write's
                # same-named widget. When ON, decode the just-written
                # container from disk instead of running the encoder, and
                # apply the INVERSE OCIO transform to the decoded frames.
                # Upstream nodes feeding `image` / `mask` / `audio` are NOT
                # evaluated (lazy inputs + check_lazy_status). Same
                # append-only widget-position rule as AM Image Write.
                "load_saved_from_disk": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "OFF (default): encode upstream IMAGE batch to a "
                        "video container. ON: skip the encode entirely, "
                        "decode the existing container at the same path "
                        "the Write would resolve, and apply the INVERSE "
                        "OCIO transform to land back in `working_colorspace`. "
                        "Upstream nodes feeding image/mask/audio are NOT "
                        "evaluated when ON. IMAGE output is in "
                        "`working_colorspace` in BOTH modes (the OCIO "
                        "transform is scoped to the encoded container; "
                        "downstream nodes always see working-cs). Honors "
                        "`raw_data` (skips OCIO both ways)."
                    ),
                }),
            },
            "optional": {
                # `image` / `mask` / `audio` are lazy: ComfyUI defers
                # upstream evaluation until `check_lazy_status` returns
                # the input names that actually need fetching. When
                # `load_saved_from_disk=True`, that hook returns `[]` and
                # all three upstream chains are skipped entirely.
                "image": ("IMAGE", {
                    "tooltip": "Image batch to encode.",
                    "lazy": True,
                }),
                "mask": ("MASK", {
                    "tooltip": reformat.TOOLTIP_MASK_IN_WRITE + (
                        "\n\n"
                        "Codec-aware: only ProRes 4444 / 4444 XQ in our codec table "
                        "actually encode the alpha channel into the container. Wiring "
                        "a MASK into a non-alpha codec (h264 / h265 / vp9 / dnxhr / "
                        "ProRes 422 etc.) logs a warning and drops the mask at the "
                        "ENCODER boundary — but the mask still appears on the `mask` "
                        "OUTPUT socket for downstream use."
                    ),
                    "lazy": True,
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Audio track to mux into the container. Optional.",
                    "lazy": True,
                }),
            },
        }

    # Output socket order — kept symmetric across the four media-IO nodes
    # (see media-io-sync-rule.md invariant 14a). Tail matches the Read
    # nodes: `resolved_path, info, width, height, frame_rate, frame_count`.
    # The legacy socket name `fps` was renamed to `frame_rate` 2026-04-29
    # for cross-node consistency.
    RETURN_TYPES = (
        "IMAGE", "MASK", "STRING", "STRING", "INT", "INT", "FLOAT", "INT",
    )
    RETURN_NAMES = (
        "image", "mask", "resolved_path", "info",
        "width", "height", "frame_rate", "frame_count",
    )
    OUTPUT_TOOLTIPS = (
        "Sliced IMAGE passthrough — the input image post-frame-slice / "
        "post-mask-fold / post-reformat, in `working_colorspace`. The OCIO "
        "transform is scoped to the encoded container only; downstream nodes "
        "see the same colorspace as the upstream chain. RGB only.",
        reformat.TOOLTIP_MASK_OUT_WRITE,
        "Absolute path of the written container.",
        "Human-readable summary: dimensions, codec/profile, fps, frame count.",
        "Encoded frame width in pixels.",
        "Encoded frame height in pixels.",
        "Container's encoded frame rate (the value used by the encoder).",
        "Number of frames encoded into the container.",
    )
    FUNCTION = "execute"
    CATEGORY = "AM Pipe"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Mirror of AM Image Write's IS_CHANGED. Returns NaN sentinel
        for write mode (always re-run) and an mtime-based hash for
        read-only mode when the path is resolvable from the file_path
        widget.
        """
        if not kwargs.get("load_saved_from_disk"):
            return float("nan")
        try:
            fp = kwargs.get("file_path") or ""
            if fp:
                fp = os.path.expandvars(os.path.expanduser(fp))
                if os.path.isfile(fp):
                    return str(os.path.getmtime(fp))
        except Exception:
            pass
        return float("nan")

    def check_lazy_status(self, **kwargs):
        """ComfyUI lazy-input gate. Returns the list of lazy inputs that
        need upstream evaluation before execute().

        When `load_saved_from_disk=True`, returns `[]` so upstream
        encoders / samplers / OCIO chains feeding `image`/`mask`/`audio`
        are skipped entirely — read-only mode sources from disk.
        Otherwise requests whichever of image/mask/audio is wired.
        """
        if kwargs.get("load_saved_from_disk"):
            return []
        needed = []
        if kwargs.get("image") is None:
            needed.append("image")
        if kwargs.get("mask") is None and "mask" in kwargs:
            needed.append("mask")
        if kwargs.get("audio") is None and "audio" in kwargs:
            needed.append("audio")
        return needed

    def execute(
        self,
        file_path,
        ext,
        seed,
        use_batch,
        frame_mode, frame_rate, first_frame, last_frame,
        codec, codec_profile, pixel_format, bitrate_or_crf, gop_size,
        working_colorspace, raw_data, output_colorspace,
        embed_workflow,
        # Reformat block — synced across all four IO nodes.
        reformat_mode: str = reformat.MODE_OFF,
        scale: float = 1.0,
        preset: str = reformat.PRESET_WH,
        target_width: int = 1920,
        target_height: int = 1080,
        resize_type: str = reformat.RESIZE_FIT,
        filter: str = reformat.FILTER_CUBIC,
        output_dtype: str = reformat.DEFAULT_DTYPE,
        show_preview: bool = True,
        # See AM Image Write's matching signature note.
        load_saved_from_disk: bool = False,
        image=None,
        mask=None,
        audio: Optional[Dict[str, Any]] = None,
        prompt: Optional[Dict[str, Any]] = None,
    ):
        log.info(
            "[am_vfx_tools/write_video] execute() entered — "
            "ext=%s frame_mode=%s use_batch=%s load_saved_from_disk=%s "
            "image_wired=%s",
            ext, frame_mode, use_batch,
            load_saved_from_disk, image is not None,
        )

        # Read-only branch — when On, decode the existing container
        # instead of running the encoder. The lazy-input gate has
        # already prevented upstream evaluation, so `image`/`mask`/
        # `audio` are None by design here.
        if load_saved_from_disk:
            return self._execute_read_only(
                file_path=file_path, use_batch=use_batch,
                frame_rate=frame_rate,
                working_colorspace=working_colorspace,
                raw_data=raw_data, output_colorspace=output_colorspace,
                output_dtype=output_dtype, show_preview=show_preview,
            )

        if image is None:
            log.warning(
                "[am_vfx_tools/write_video] `image` input is not wired — write skipped"
            )
            return self._noop(None)

        # Seed sentinel resolution. `seed == -1` means "unset" — scan
        # `prompt` for AM Seed nodes and look up the seed each one
        # published to the process-global seed_registry during its own
        # IS_CHANGED (the publish runs BEFORE any execute(), so the
        # value is guaranteed available regardless of execution order
        # between AM Seed and this Write node). Registry hit → use the
        # published value for metadata; miss → leave `seed` at -1 and
        # `_build_workflow_metadata` will omit the key. Any other value
        # (typed widget OR delivered by a wire from convert-widget-to-
        # input) wins outright over the registry — wiring is the
        # explicit-override path, also useful for multi-AMSeed
        # workflows where the artist wants a specific source.
        if int(seed) == -1:
            looked_up = seed_registry.find_seed_for_prompt(prompt)
            if looked_up is not None:
                seed = int(looked_up)

        # 1. Resolve output path. Manual mode injects `_bNNNN` before the
        #    extension via batch_suffix.resolve_for_manual_path when use_batch
        #    is on. When use_batch=False, queue iterations overwrite each
        #    other (explicit artist opt-out).
        # ``_batch_n`` carries the runtime-discovered _bNNNN integer for
        # the metadata embed (``comfyui/batch``); set None when the artist
        # opted out via ``use_batch=False`` so the embed helper skips it.
        _batch_n: Optional[int] = None
        if not file_path:
            log.warning("[am_vfx_tools/write_video] empty file_path")
            return self._noop(image)
        base_path = os.path.expandvars(os.path.expanduser(file_path))
        if use_batch:
            _batch_n, full_path = batch_suffix.resolve_for_manual_path(base_path)
        else:
            full_path = base_path

        # 2. Strip the container prefix from `codec` before any
        # downstream use. The dropdown shape is `container/codec`
        # (e.g. `mov/prores`); the backend wants the bare codec token.
        # Legacy unprefixed values (saved workflows from before the
        # 2026-04-29 prefix rework) translate via `_LEGACY_CODEC_MAP`
        # inside the helper.
        bare_codec = _strip_codec_prefix(codec)

        # 2b. Container/codec sanity check before any encode work.
        # `validate_container_codec` is permissive — it accepts the
        # FFmpeg-legal pairings, not just the canonical ones surfaced
        # in the dropdown — so an artist who picks ``ext=mkv`` with
        # ``codec=mp4/h265`` (h265 is also legal in MKV) still passes.
        try:
            video_backend.validate_container_codec(full_path, bare_codec)
        except ValueError as e:
            log.error("[am_vfx_tools/write_video] %s", e)
            raise

        # 2c. Codec/profile cross-check — the codec_profile prefix is
        # the BARE codec name (e.g. `prores/422`, `h265/main`), so we
        # compare against `bare_codec` rather than the container-
        # prefixed `codec` dropdown value.
        bare_profile = codec_profile
        if "/" in codec_profile:
            prefix, _, bare_profile = codec_profile.partition("/")
            if prefix != bare_codec:
                spec = video_backend.CODECS.get(bare_codec, {})
                allowed = [f"{bare_codec}/{p}" for p in spec.get("profiles", [])]
                raise ValueError(
                    f"codec={codec!r} (bare={bare_codec!r}) but "
                    f"codec_profile={codec_profile!r}; pick one of: {allowed}"
                )

        # 3. Slice the input batch per frame_mode.
        if image.ndim == 3:
            image = image[None, ...]
        n_input = int(image.shape[0])
        slice_lo, slice_hi = self._resolve_slice(
            frame_mode, int(first_frame), int(last_frame), n_input,
        )
        if slice_hi <= slice_lo:
            log.warning(
                "[am_vfx_tools/write_video] empty slice (frame_mode=%s, first=%d, last=%d, n_in=%d) — write skipped",
                frame_mode, first_frame, last_frame, n_input,
            )
            return self._noop(image)
        sliced = image[slice_lo:slice_hi]
        n_frames = int(sliced.shape[0])

        # 4. OCIO transform (working_cs -> output_cs), built once.
        wcs = color.resolve_choice_to_cs(working_colorspace)
        ocs = color.resolve_choice_to_cs(output_colorspace)
        try:
            proc = color.ColorProcessor(wcs, ocs, raw_data=raw_data)
        except Exception as e:
            log.warning(
                "[am_vfx_tools/write_video] cannot build OCIO %s -> %s (%s); "
                "writing untransformed pixels", wcs, ocs, e,
            )
            proc = None

        # Per-frame reformat lands inside the iterator so the encoder gets
        # post-reformat fp32 frames at the artist's chosen output dimensions.
        # Each yielded frame is also stashed in `_reformatted_frames` +
        # `_mask_frames` so that after `video_backend.write_video` drains
        # the iterator, we have the full post-reformat batch to emit on
        # the IMAGE + MASK output sockets (matching what was encoded into
        # the container — or what WOULD have been encoded if the codec
        # had carried alpha). The fp32 → fp16 cast is deferred to the
        # post-loop tensor build because the encoder needs fp32 either way.
        _reformatted_frames: list = []
        _mask_frames: list = []

        # Decide whether to ENCODE the mask based on codec/profile alpha
        # support. Currently true for ProRes 4444 / 4444 XQ. The mask
        # ALWAYS rides along on the output socket — even when the codec
        # drops it at encode time — so downstream nodes can chain on it.
        _alpha_capable = video_backend.codec_profile_supports_alpha(bare_codec, bare_profile)
        _encode_mask = mask is not None and _alpha_capable
        # Whether to FOLD mask into the encoder buffer at all. Even when
        # the codec drops alpha, we still combine for the output-socket
        # passthrough so the artist sees what they wired in.
        _combine_mask = mask is not None
        if mask is not None and not _alpha_capable:
            log.warning(
                "[am_vfx_tools/write_video] mask wired but codec=%s/%s does not "
                "carry alpha — mask dropped at encoder boundary (still "
                "available on the `mask` output socket). Use ProRes 4444 "
                "/ 4444 XQ (in MOV) for alpha-bearing video output.",
                bare_codec, bare_profile,
            )

        def _frame_iter():
            for i in range(n_frames):
                pixels = sliced[i].detach().cpu().numpy().astype(np.float32, copy=True)
                # Fold MASK into the alpha channel BEFORE OCIO and
                # reformat. We always combine when a mask is wired (so
                # the output-socket passthrough reflects the wired data);
                # whether the encoder actually keeps the alpha is decided
                # by `_encode_mask` at yield time.
                if _combine_mask:
                    try:
                        mask_frame = mask[i].detach().cpu().numpy().astype(np.float32, copy=False)
                        pixels = reformat.combine_image_mask(pixels, mask_frame)
                    except Exception as e:
                        log.warning(
                            "[am_vfx_tools/write_video] mask combine failed for frame %d (%s); "
                            "encoding without mask", i, e,
                        )

                # Per-frame reformat (geometry only, in working_colorspace).
                # Reordered 2026-05-02: reformat now runs BEFORE OCIO so
                # we can stash a working_colorspace snapshot for the
                # IMAGE output socket between reformat and OCIO. Reformat
                # and OCIO commute mathematically — the encoded pixels
                # are equivalent within fp32 epsilon either order.
                if reformat_mode != reformat.MODE_OFF:
                    try:
                        pixels = reformat.reformat_array(
                            pixels,
                            mode=reformat_mode,
                            scale=float(scale),
                            preset=preset,
                            target_w=int(target_width),
                            target_h=int(target_height),
                            resize_type=resize_type,
                            filter_name=filter,
                            output_dtype=reformat.DTYPE_FP32,
                        )
                    except Exception as e:
                        log.warning(
                            "[am_vfx_tools/write_video] reformat failed for frame %d (%s); "
                            "encoding original pixels", i, e,
                        )

                # Snapshot for the IMAGE + MASK output sockets, BEFORE
                # OCIO. The IMAGE socket emits `working_colorspace`
                # passthrough — the input image post-frame-slice / post-
                # mask-fold / post-reformat. The OCIO transform is
                # scoped to the encoded container only; downstream nodes
                # continue to see `working_colorspace`. Mirrors AM Image
                # Write (see its matching comment) and the Nuke Write
                # passthrough convention. Pre-2026-05-02 the IMAGE socket
                # emitted post-OCIO encoded pixels — inconsistent with
                # the read-only branch and surprised artists who expected
                # a passthrough.
                if pixels.shape[-1] >= 4:
                    _reformatted_frames.append(pixels[..., :3].copy())
                    _mask_frames.append((1.0 - pixels[..., 3]).copy())
                else:
                    _reformatted_frames.append(pixels.copy())
                    _mask_frames.append(np.zeros(pixels.shape[:2], dtype=pixels.dtype))

                if proc is not None and not proc.is_identity:
                    try:
                        proc.apply_inplace(pixels)
                    except Exception as e:
                        log.warning(
                            "[am_vfx_tools/write_video] OCIO apply failed for frame %d (%s); "
                            "writing untransformed pixels", i, e,
                        )

                # Yield the full buffer to the encoder; it strips alpha
                # at the encoder boundary when the pixfmt is RGB-only.
                yield pixels

        # 5. Audio passthrough.
        audio_buf = self._coerce_audio(audio) if audio is not None else None

        workflow_meta = self._build_workflow_metadata(
            embed_workflow, prompt,
            seed=seed,
            batch_no=_batch_n,
        )

        per_file_meta = workflow_meta if workflow_meta else None

        try:
            video_backend.write_video(
                full_path,
                _frame_iter(),
                codec=bare_codec,
                codec_profile=bare_profile,
                pixel_format=pixel_format,
                frame_rate=float(frame_rate),
                bitrate_or_crf=bitrate_or_crf,
                gop_size=int(gop_size),
                audio_buffer=audio_buf,
                color_space_tag=(ocs if ocs and ocs != color.PASSTHROUGH else None),
                metadata={"Software": "comfyui"},
                workflow_metadata=per_file_meta,
            )
        except Exception as e:
            log.warning("[am_vfx_tools/write_video] write failed for %s: %s", full_path, e)
            raise

        log.info(
            "[am_vfx_tools/write_video] wrote %s (%d frames, codec=%s/%s)",
            full_path, n_frames, bare_codec, bare_profile,
        )

        # IMAGE + MASK outputs reflect the post-reformat batch the encoder
        # actually consumed (matches AM Write Image's pattern). When
        # reformat is off and no mask is involved we fall back to the
        # original sliced tensor — no needless copy.
        src_h = int(sliced.shape[1])
        src_w = int(sliced.shape[2])
        if _reformatted_frames and (reformat_mode != reformat.MODE_OFF or _combine_mask):
            stacked = np.stack(_reformatted_frames, axis=0)
            if output_dtype == reformat.DTYPE_FP16:
                stacked = stacked.astype(np.float16, copy=False)
            out_image = torch.from_numpy(np.ascontiguousarray(stacked))
            mask_stacked = np.stack(_mask_frames, axis=0)
            if output_dtype == reformat.DTYPE_FP16:
                mask_stacked = mask_stacked.astype(np.float16, copy=False)
            out_mask = torch.from_numpy(np.ascontiguousarray(mask_stacked))
        else:
            if output_dtype == reformat.DTYPE_FP16:
                out_image = sliced.to(torch.float16)
            else:
                out_image = sliced
            # No reformat AND no mask wired — emit empty mask matching
            # IMAGE shape (stock ComfyUI sentinel = zeros, "nothing to
            # inpaint"). Source: `sliced` (the input batch slice); strip
            # the channel axis.
            out_mask = torch.zeros(
                (int(sliced.shape[0]), int(sliced.shape[1]), int(sliced.shape[2])),
                dtype=(torch.float16 if output_dtype == reformat.DTYPE_FP16 else torch.float32),
            )

        ui_payload = self._ui_payload(out_image, full_path, show_preview, working_colorspace)

        # Structured outputs — IMAGE socket is the post-reformat tensor; the
        # metadata quartet (width/height/frame_count/fps) + info string lets
        # graphs branch on size/duration/codec without poking at the IMAGE.
        out_h = int(out_image.shape[1])
        out_w = int(out_image.shape[2])
        info_str = (
            f"{out_w}x{out_h} {bare_codec}/{bare_profile} "
            f"{float(frame_rate):.4g}fps, {n_frames} frames written"
        )
        rf_frag = reformat.info_fragment(
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
        if rf_frag:
            info_str = f"{info_str} | {rf_frag}"

        return {
            "ui": ui_payload,
            "result": (
                # image, mask, resolved_path, info, width, height, frame_rate, frame_count
                out_image, out_mask, full_path, info_str,
                int(out_w), int(out_h), float(frame_rate), int(n_frames),
            ),
        }

    @staticmethod
    def _build_workflow_metadata(
        embed_workflow: bool,
        prompt: Optional[Dict[str, Any]],
        *,
        seed: int = -1,
        batch_no: Optional[int] = None,
    ) -> Optional[Dict[str, str]]:
        """Render the prompt plus per-field generation knobs into the
        string→string dict embedded in the saved file.

        Mirror of :meth:`AMImageWrite._build_workflow_metadata` — see
        that docstring for the namespacing rationale. Kept duplicated
        because the Write nodes don't share a base class; if a third
        Write node ever lands, lift this into ``_core``.
        """
        if not embed_workflow:
            return None
        if _comfy_args is not None and getattr(_comfy_args, "disable_metadata", False):
            return None

        out: Dict[str, str] = {}

        if prompt is not None:
            try:
                out["comfyui/prompt"] = json.dumps(prompt)
            except Exception:
                pass

        if int(seed) != -1:
            out["comfyui/seed"] = str(int(seed))
        if batch_no is not None:
            out["comfyui/batch"] = str(int(batch_no))

        return out or None

    @staticmethod
    def _resolve_slice(frame_mode: str, first: int, last: int, n: int) -> tuple:
        """Return ``(lo, hi)`` half-open Python slice into the input batch.

        first/last are 1-indexed inclusive in the UI (matching AM Write
        Image's ``frame_first`` / ``frame_last`` convention). ``last=-1``
        means "to the end of the batch".
        """
        if frame_mode == FRAME_MODE_ALL:
            return 0, n
        if frame_mode == FRAME_MODE_SINGLE:
            idx = max(0, min(n - 1, first - 1))
            return idx, idx + 1
        # range
        lo = max(0, first - 1)
        hi = n if last < 0 else min(n, last)
        return lo, hi

    @staticmethod
    def _coerce_audio(audio: Dict[str, Any]) -> Optional[video_backend.AudioBuffer]:
        try:
            wf = audio.get("waveform")
            sr = int(audio.get("sample_rate") or 0)
        except Exception:
            return None
        if wf is None or sr <= 0:
            return None
        try:
            arr = wf.detach().cpu().numpy() if hasattr(wf, "detach") else np.asarray(wf)
        except Exception:
            return None
        if arr.size <= 1:
            return None
        return video_backend.AudioBuffer(waveform=arr, sample_rate=sr)

    @staticmethod
    def _ui_payload(image, written_path: str, show_preview: bool, working_colorspace: str):
        if not show_preview:
            return {"text": [written_path]}
        try:
            payload = preview.create_single_preview(
                image, frame_index=0,
                working_colorspace=working_colorspace,
                filename_hint=written_path,
            )
        except Exception as e:
            log.warning("[am_vfx_tools/write_video] preview generation failed: %s", e)
            return {"text": [written_path]}
        if not payload.get("images"):
            return {"text": [written_path]}
        return payload

    @staticmethod
    def _noop(image=None):
        # Match the 8-output RETURN_TYPES shape so ComfyUI doesn't slot a
        # bare string into the IMAGE socket. Order mirrors AM Image
        # Write._noop and the live `result` tuple above:
        # image, mask, resolved_path, info, width, height, frame_rate, frame_count.
        if image is None or not hasattr(image, "shape") or image.ndim < 3:
            empty_mask = torch.zeros((1, 1, 1), dtype=torch.float32)
            return {
                "ui": {"text": ["(no write)"]},
                "result": (image, empty_mask, "", "(no write)", 0, 0, 0.0, 0),
            }
        # image is a tensor (already promoted to (N, H, W, C) by the caller).
        h = int(image.shape[1] if image.ndim == 4 else image.shape[0])
        w = int(image.shape[2] if image.ndim == 4 else image.shape[1])
        n = int(image.shape[0]) if image.ndim == 4 else 1
        # Empty MASK = zeros (stock ComfyUI: nothing to inpaint).
        empty_mask = torch.zeros((n, h, w), dtype=torch.float32)
        return {
            "ui": {"text": ["(no write)"]},
            "result": (image, empty_mask, "", "(no write)", w, h, 0.0, 0),
        }

    # ------------------------------------------------------------------
    # Read-only mode (load_saved_from_disk = True)
    # ------------------------------------------------------------------
    #
    # Decode the existing container at the literal file_path, apply
    # inverse OCIO, and emit the IMAGE + MASK + frame_rate sockets.
    # Simpler than AM Image Write's per-frame loop because a video is
    # one container — single decode covers the whole sequence.
    #
    # Frame range is set by the encoder when the container was originally
    # written; on read we just emit the entire decoded stream. The Write
    # node's frame_mode/first_frame/last_frame widgets are NOT consulted
    # in read mode for video — they were write-time slicing of the input
    # batch, and don't have a sensible read-side interpretation for an
    # already-encoded container.

    def _execute_read_only(
        self, *, file_path, use_batch, frame_rate,
        working_colorspace, raw_data, output_colorspace,
        output_dtype, show_preview,
    ):
        if not file_path:
            log.warning(
                "[am_vfx_tools/write_video] read-only mode: empty file_path",
            )
            return self._noop(None)
        base_path = os.path.expandvars(os.path.expanduser(file_path))
        if use_batch:
            full_path = batch_suffix.resolve_latest_existing(base_path) or base_path
        else:
            full_path = base_path
        if not full_path or not os.path.exists(full_path):
            log.warning(
                "[am_vfx_tools/write_video] read-only mode: container not found at %r",
                full_path,
            )
            return self._noop(None)
        log.info("[am_vfx_tools/write_video] read-only mode: decoding %s", full_path)

        try:
            loaded_stack, _audio, info = video_backend.read_video_frames(
                full_path, audio_track=None,
            )
            loaded_stack = np.asarray(loaded_stack, dtype=np.float32)
        except Exception as e:
            log.warning(
                "[am_vfx_tools/write_video] read-only mode: decode failed for %s: %s",
                full_path, e,
            )
            return self._noop(None)

        if loaded_stack.size == 0 or loaded_stack.shape[0] == 0:
            return self._noop(None)

        # Inverse OCIO.
        wcs = color.resolve_choice_to_cs(working_colorspace)
        ocs = color.resolve_choice_to_cs(output_colorspace)
        try:
            inv_proc = color.ColorProcessor(ocs, wcs, raw_data=raw_data)
        except Exception as e:
            log.warning(
                "[am_vfx_tools/write_video] read-only mode: cannot build inverse OCIO "
                "%s -> %s (%s); using untransformed pixels", ocs, wcs, e,
            )
            inv_proc = None
        if inv_proc is not None and not inv_proc.is_identity:
            for i in range(int(loaded_stack.shape[0])):
                try:
                    inv_proc.apply_inplace(loaded_stack[i])
                except Exception as e:
                    log.warning(
                        "[am_vfx_tools/write_video] read-only mode: inverse OCIO "
                        "failed for frame %d (%s); using untransformed pixels",
                        i, e,
                    )
                    break

        # Split RGB / mask, dtype-cast.
        if loaded_stack.shape[-1] >= 4:
            rgb = loaded_stack[..., :3]
            msk = 1.0 - loaded_stack[..., 3]
        else:
            rgb = loaded_stack
            msk = np.zeros(
                (int(loaded_stack.shape[0]),
                 int(loaded_stack.shape[1]),
                 int(loaded_stack.shape[2])),
                dtype=np.float32,
            )
        if output_dtype == reformat.DTYPE_FP16:
            rgb = rgb.astype(np.float16, copy=False)
            msk = msk.astype(np.float16, copy=False)
        out_image = torch.from_numpy(np.ascontiguousarray(rgb))
        out_mask = torch.from_numpy(np.ascontiguousarray(msk))

        out_h = int(out_image.shape[1])
        out_w = int(out_image.shape[2])
        n_frames = int(out_image.shape[0])
        # Container frame rate from the probe (falls back to widget if
        # missing — the widget is the artist's intended fps anyway).
        decoded_fps = float(getattr(info, "frame_rate", 0) or 0) or float(frame_rate)
        info_str = (
            f"{out_w}x{out_h} read-only {ocs or '(no-cs)'} "
            f"{decoded_fps:.4g}fps, {n_frames} frames decoded from disk"
        )
        ui_payload = self._ui_payload(out_image, full_path, show_preview, working_colorspace)
        return {
            "ui": ui_payload,
            "result": (
                out_image, out_mask, full_path, info_str,
                int(out_w), int(out_h), decoded_fps, int(n_frames),
            ),
        }
