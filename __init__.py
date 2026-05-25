"""comfyui-am-vfx-tools — ComfyUI custom-node pack for VFX I/O & color.

Thirteen nodes for image / video read+write (OpenImageIO + PyAV), OCIO
2.x color management, Nuke-style Grade + Color Correct, OpenCV-backed
reformat, frame-order reverse, render-farm-safe Seed, and frame-range
slicing — plus a workfile-io subsystem with native OS file dialogs for
save/load of workflow JSON to absolute paths. All pixel nodes carry
native ComfyUI VIDEO sockets for low-RAM streaming workflows.

The IO nodes are Manual-mode only — pick a path explicitly via the
``file_path`` widget, optionally with a ``####`` / ``%05d`` / ``$F4``
frame token for sequences. Buttons on each node:
  * 📂 Browse — native OS file dialog to pick a path under the configured
    sandbox roots (default: user home + ~/Documents; override via env
    ``AM_VFX_TOOLS_FILECHOOSER_ROOTS``).
  * 🔍 Detect Range (read nodes) — auto-fills first/last from on-disk scan.
  * 📁 Open in Explorer (read/write) — reveals path in OS file manager.
  * 📋 Copy File Path — copies the resolved path to clipboard.

The workfile-io menu adds Save / Open / Recent / Incremental Save
commands for managing workflow JSON files outside ComfyUI's default
folder. Sandbox-protected against the same configured roots.

Project home: https://github.com/am-pipeline-prod/comfyui-am-vfx-tools
"""
from __future__ import annotations

import logging
import os

# Side-effect imports: each routes module registers HTTP handlers against
# ``PromptServer.instance.routes`` at import time. URL prefixes are
# disjoint (`/am-vfx-tools/*` for media-IO + filechooser, plus
# `/am-vfx-tools/workfile-io/*` for workfile-IO) so registration order
# is irrelevant. Wrapped in try/except so a failure to register HTTP
# routes can never prevent NODE_CLASS_MAPPINGS (defined below) from being
# importable — e.g. in the Comfy Registry's node-extraction sandbox.
try:
    from . import routes  # noqa: F401
    from . import routes_workfile  # noqa: F401
except Exception as _route_err:  # noqa: BLE001
    logging.getLogger("am_vfx_tools").warning(
        "[am-vfx-tools] HTTP route registration skipped: %s", _route_err
    )

from .am_color_correct import AMColorCorrect
from .am_frame_range import AMFrameRange
from .am_grade import AMGrade, AMGradeRGB
from .am_image_read import AMImageRead
from .am_image_write import AMImageWrite
from .am_ocio_colorspace import AMOCIOColorspace
from .am_ocio_log_convert import AMOCIOLogConvert
from .am_reformat import AMReformat
from .am_reverse import AMReverseSequence
from .am_seed import AMSeed
from .am_video_read import AMVideoRead
from .am_video_write import AMVideoWrite

WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")

NODE_CLASS_MAPPINGS = {
    "AMImageRead":       AMImageRead,
    "AMImageWrite":      AMImageWrite,
    "AMColorCorrect":    AMColorCorrect,
    "AMFrameRange":      AMFrameRange,
    "AMGrade":           AMGrade,
    "AMGradeRGB":        AMGradeRGB,
    "AMOCIOColorspace":  AMOCIOColorspace,
    "AMOCIOLogConvert":  AMOCIOLogConvert,
    "AMReformat":        AMReformat,
    "AMReverseSequence": AMReverseSequence,
    "AMSeed":            AMSeed,
    "AMVideoRead":       AMVideoRead,
    "AMVideoWrite":      AMVideoWrite,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AMImageRead":       "AM Read Image",
    "AMImageWrite":      "AM Write Image",
    "AMColorCorrect":    "AM Color Correct",
    "AMFrameRange":      "AM Frame Range",
    "AMGrade":           "AM Grade",
    "AMGradeRGB":        "AM Grade RGB",
    "AMOCIOColorspace":  "AM OCIO Colorspace",
    "AMOCIOLogConvert":  "AM OCIO Log Convert",
    "AMReformat":        "AM Reformat",
    "AMReverseSequence": "AM Reverse Sequence",
    "AMSeed":            "AM Seed",
    "AMVideoRead":       "AM Read Video",
    "AMVideoWrite":      "AM Write Video",
}

logging.getLogger("am_vfx_tools").info("[am-vfx-tools] loaded (13 nodes)")

# Register node-replacement migrations for any breaking shape changes
# (currently empty — the pack ships at v0.1.0). Wrapped so a failure
# here can't break pack loading.
try:
    import asyncio
    from . import _node_replacements

    try:
        _loop = asyncio.get_event_loop()
        if _loop.is_running():
            _loop.create_task(_node_replacements.register_replacements())
        else:
            _loop.run_until_complete(_node_replacements.register_replacements())
    except RuntimeError:
        asyncio.run(_node_replacements.register_replacements())
except Exception as _e:
    logging.getLogger("am_vfx_tools").warning(
        "[am-vfx-tools] node-replacement registration skipped: %s", _e
    )

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
