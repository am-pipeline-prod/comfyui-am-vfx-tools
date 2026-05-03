"""am-pipe-media-io._core.color — OCIO 2.x ColorProcessor + family-grouped dropdown.

Single shared color core for AM Image / AM Video Read & Write nodes and the
AM OCIO Colorspace utility node.

Loads the OCIO config from (in order):

  1. ``$OCIO`` environment variable, if set and points to a readable file.
  2. OCIO 2.x builtin ``studio-config-v4.0.0_aces-v2.0_ocio-v2.5``
     (52+ colorspaces incl. ARRI/Sony/RED/Canon/Panasonic/Apple/BMD/DJI IDTs).
  3. OCIO 2.x builtin ``cg-config-v4.0.0_aces-v2.0_ocio-v2.5``
     (smaller; ACES roles + utility encodings, no camera IDTs).
  4. ``OCIO.GetCurrentConfig()`` — the "raw" identity-only stub.

Each layer logs which config loaded and the colorspace count. Cached
per-process after first successful load.

Public surface:
  * :class:`ColorProcessor` — wraps a CPU processor between two color spaces.
  * :func:`color_space_choices` — flat dropdown: ``raw`` sentinel + family-
    prefixed colorspace names (``Display/sRGB - Display``, ``Input/ARRI/...``).
    Roles are NOT listed as separate entries — they're aliases for actual
    colorspaces and listing them duplicates the menu.
  * :func:`resolve_choice_to_cs` — strip the family prefix to recover the
    bare OCIO colorspace name at execute time.
  * :func:`pick_default` — pick the highest-priority preferred name available
    in the dropdown, for portable workflows.
  * :func:`default_input_colorspace` / :func:`default_working_colorspace` /
    :func:`default_output_colorspace` — sensible per-knob defaults.
  * :func:`available_color_spaces` — flat list of color-space names from the
    active config.
  * :func:`config_source` — origin label for the active config.
  * :func:`is_available` — True if PyOpenColorIO is importable.
  * :data:`PASSTHROUGH` — sentinel meaning "do not touch pixels" (``"raw"``).

Designed so the rest of the codebase can import this module unconditionally
even if PyOpenColorIO isn't installed (the ComfyUI venv has it; system
Python may not). All public calls degrade to no-ops with clear errors when
the binding is missing.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

log = logging.getLogger("am_pipe.media-io.color")

PASSTHROUGH = "raw"

_BUILTIN_STUDIO = "studio-config-v4.0.0_aces-v2.0_ocio-v2.5"
_BUILTIN_CG     = "cg-config-v4.0.0_aces-v2.0_ocio-v2.5"


# ---------------------------------------------------------------------------
# Lazy OCIO import
# ---------------------------------------------------------------------------

try:
    import PyOpenColorIO as _ocio  # type: ignore[import-not-found]
    _OCIO_AVAILABLE = True
except ImportError:
    _ocio = None
    _OCIO_AVAILABLE = False


def is_available() -> bool:
    return _OCIO_AVAILABLE


# ---------------------------------------------------------------------------
# Config caching — layered loader
# ---------------------------------------------------------------------------

_CONFIG = None
_CONFIG_SOURCE: Optional[str] = None


def _try_builtin(name: str):
    """Return a builtin config by name or None if unavailable."""
    try:
        return _ocio.Config.CreateFromBuiltinConfig(name)
    except Exception as e:
        log.info("[am_pipe/color] builtin %s not available: %s", name, e)
        return None


def _load_config():
    """Layered OCIO config loader. See module docstring for the order."""
    global _CONFIG, _CONFIG_SOURCE
    if _CONFIG is not None:
        return _CONFIG
    if not _OCIO_AVAILABLE:
        raise RuntimeError(
            "PyOpenColorIO is not installed. Color-managed read/write requires "
            "the `opencolorio` Python package (typically shipped with the "
            "ComfyUI venv)."
        )

    # 1. $OCIO env
    ocio_env = os.environ.get("OCIO", "").strip()
    if ocio_env and os.path.isfile(ocio_env):
        try:
            cfg = _ocio.Config.CreateFromFile(ocio_env)
            _CONFIG = cfg
            _CONFIG_SOURCE = ocio_env
            log.info(
                "[am_pipe/color] loaded $OCIO=%s (%d colorspaces)",
                ocio_env, len(list(cfg.getColorSpaces())),
            )
            return _CONFIG
        except Exception as e:
            log.warning(
                "[am_pipe/color] $OCIO=%s failed to load (%s); falling back",
                ocio_env, e,
            )

    # 2. builtin Studio
    cfg = _try_builtin(_BUILTIN_STUDIO)
    if cfg is not None:
        _CONFIG = cfg
        _CONFIG_SOURCE = f"(builtin: {_BUILTIN_STUDIO})"
        log.info(
            "[am_pipe/color] loaded builtin %s (%d colorspaces)",
            _BUILTIN_STUDIO, len(list(cfg.getColorSpaces())),
        )
        return _CONFIG

    # 3. builtin CG
    cfg = _try_builtin(_BUILTIN_CG)
    if cfg is not None:
        _CONFIG = cfg
        _CONFIG_SOURCE = f"(builtin: {_BUILTIN_CG})"
        log.info(
            "[am_pipe/color] loaded builtin %s (%d colorspaces)",
            _BUILTIN_CG, len(list(cfg.getColorSpaces())),
        )
        return _CONFIG

    # 4. raw stub — identity-only
    try:
        cfg = _ocio.GetCurrentConfig()
        _CONFIG = cfg
        _CONFIG_SOURCE = "(raw stub)"
        log.warning(
            "[am_pipe/color] no usable OCIO config found — falling back to "
            "raw stub (identity transforms only)"
        )
        return _CONFIG
    except Exception as e:
        raise RuntimeError(
            f"Could not load any OCIO config (env unset/invalid, builtins "
            f"unavailable, raw stub failed: {e})"
        )


def config_source() -> Optional[str]:
    """Origin label for the active config — file path or builtin marker."""
    if _OCIO_AVAILABLE and _CONFIG is None:
        try:
            _load_config()
        except RuntimeError:
            return None
    return _CONFIG_SOURCE


def available_color_spaces() -> List[str]:
    """Flat list of color-space names from the active config."""
    if not _OCIO_AVAILABLE:
        return []
    cfg = _load_config()
    return [cs.getName() for cs in cfg.getColorSpaces()]


# ---------------------------------------------------------------------------
# Family-grouped dropdown — flat list, prefix grouping
# ---------------------------------------------------------------------------

# Hard fallback list used when OCIO isn't importable at all.
_FALLBACK_CHOICES = (
    PASSTHROUGH,
    "ACEScg",
    "sRGB - Display",
    "Linear Rec.709 (sRGB)",
    "sRGB Encoded Rec.709 (sRGB)",
)


def color_space_choices() -> List[str]:
    """Return the family-prefixed OCIO dropdown:

        raw                                                ← bypass sentinel (pos 0)
        ACES/ACES2065-1                                    ← family-prefixed list
        ACES/ACEScc
        ...
        Display/sRGB - Display
        ...
        Input/ARRI/ARRI LogC4
        ...
        Utility/sRGB Encoded Rec.709 (sRGB)
        ...

    Roles (``scene_linear``, ``aces_interchange``, etc.) are NOT listed
    as separate entries — they're config-level aliases for actual
    colorspaces, so they'd duplicate menu space without adding choices.
    """
    out = [PASSTHROUGH]
    if not _OCIO_AVAILABLE:
        return list(_FALLBACK_CHOICES)
    try:
        cfg = _load_config()
    except RuntimeError:
        return list(_FALLBACK_CHOICES)

    for cs in cfg.getColorSpaces():
        family = (cs.getFamily() or "").strip("/")
        name = cs.getName()
        out.append(f"{family}/{name}" if family else name)

    return out


def resolve_choice_to_cs(value: str) -> str:
    """Strip the family prefix to recover the bare OCIO colorspace name.

    Examples:
        ``"raw"``                                 → ``"raw"`` (sentinel)
        ``"Display/sRGB - Display"``              → ``"sRGB - Display"``
        ``"Input/ARRI/ARRI LogC4"``               → ``"ARRI LogC4"``
        ``"sRGB - Display"`` (already bare)       → ``"sRGB - Display"``
        ``"scene_linear"`` (role typed directly)  → ``"scene_linear"`` — OCIO
                                                     accepts role names as
                                                     processor sources.
    """
    if not value or value == PASSTHROUGH:
        return PASSTHROUGH
    return value.rsplit("/", 1)[-1]


def _strip_family_prefix(value: str) -> str:
    """Bare name minus any role wrapper — used by pick_default."""
    return resolve_choice_to_cs(value)


def pick_default(choices: List[str], preferred_names: Tuple[str, ...]) -> str:
    """Pick the highest-priority preferred name that exists in *choices*.

    Iterates *preferred_names* in order (priority), so if `sRGB - Display`
    is the top preference and the dropdown contains both `Display/sRGB -
    Display` and `[role] scene_linear (ACEScg)` (which also matches a
    later preference), the sRGB-Display entry wins.

    For ties — when a single preferred name is bound to both a role
    shortcut and a family entry — the role shortcut wins because it
    appears first in the dropdown order.
    """
    by_stripped: dict = {}
    for c in choices:
        by_stripped.setdefault(_strip_family_prefix(c), c)
    for name in preferred_names:
        if name in by_stripped:
            return by_stripped[name]
    return choices[0]


def default_input_colorspace(choices: Optional[List[str]] = None) -> str:
    return pick_default(choices or color_space_choices(), (PASSTHROUGH,))


def default_working_colorspace(choices: Optional[List[str]] = None) -> str:
    return pick_default(
        choices or color_space_choices(),
        ("sRGB - Display", "sRGB", "Linear Rec.709 (sRGB)", "ACEScg"),
    )


def default_output_colorspace(choices: Optional[List[str]] = None) -> str:
    return pick_default(choices or color_space_choices(), (PASSTHROUGH,))


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class ColorProcessor:
    """Cached OCIO processor between two color spaces.

    Both *src* and *dst* may be color-space names *or* OCIO role names
    (``scene_linear``, ``aces_interchange``, etc.). When either is
    :data:`PASSTHROUGH` (``"raw"``), or ``raw_data=True``, or src==dst,
    :meth:`apply_inplace` is a no-op (identity).
    """

    def __init__(self, src: str, dst: str, *, raw_data: bool = False):
        self.src = src
        self.dst = dst
        self._cpu = None  # built lazily

        if raw_data or src == PASSTHROUGH or dst == PASSTHROUGH or src == dst:
            self._is_identity = True
            return

        self._is_identity = False
        if not _OCIO_AVAILABLE:
            raise RuntimeError(
                f"PyOpenColorIO required for {src!r} → {dst!r} transform"
            )
        cfg = _load_config()
        try:
            proc = cfg.getProcessor(src, dst)
        except Exception as e:
            raise ValueError(
                f"OCIO: could not build processor {src!r} → {dst!r}: {e}"
            )
        self._cpu = proc.getDefaultCPUProcessor()

    @property
    def is_identity(self) -> bool:
        return self._is_identity

    def apply_inplace(self, pixels) -> None:
        """Apply transform in place to a packed float32 buffer (3 or 4 ch).

        *pixels* must follow the Python buffer protocol (numpy array
        recommended). ``applyRGB`` / ``applyRGBA`` modify the buffer in
        place — see https://opencolorio.readthedocs.io/.
        """
        if self._is_identity:
            return
        n_chan = pixels.shape[-1] if hasattr(pixels, "shape") else 3
        try:
            import numpy as np
            if not pixels.flags["C_CONTIGUOUS"]:
                pixels = np.ascontiguousarray(pixels)
        except ImportError:
            pass
        if n_chan == 4:
            self._cpu.applyRGBA(pixels)
        else:
            self._cpu.applyRGB(pixels)


__all__ = [
    "ColorProcessor",
    "PASSTHROUGH",
    "available_color_spaces",
    "color_space_choices",
    "config_source",
    "default_input_colorspace",
    "default_output_colorspace",
    "default_working_colorspace",
    "is_available",
    "pick_default",
    "resolve_choice_to_cs",
]
