"""comfyui-am-vfx-tools — ComfyUI custom-node pack for VFX I/O & color.

Eleven nodes for image / video read+write (OpenImageIO + PyAV), OCIO 2.x
color management, Nuke-style Grade, OpenCV-backed reformat, render-farm-
safe Seed, and frame-range slicing.

The IO nodes are Manual-mode only — pick a path explicitly via the
``file_path`` widget, optionally with a ``####`` / ``%05d`` / ``$F4``
frame token for sequences. The 🔍 Detect Range button on the read nodes
auto-fills first/last frame from an on-disk scan; the 📁 Open in Explorer
button on read/write nodes reveals the resolved path in your OS file
manager; dropping a media file onto the canvas spawns an AM Read node
configured to load it.

Project home: https://github.com/am-pipeline-prod/comfyui-am-vfx-tools
"""
from __future__ import annotations

import logging
import os

# Side-effect import: routes.py registers HTTP handlers against
# ``PromptServer.instance.routes`` at import time.
from . import routes  # noqa: F401

from .am_frame_range import AMFrameRange
from .am_grade import AMGrade, AMGradeRGB
from .am_image_read import AMImageRead
from .am_image_write import AMImageWrite
from .am_ocio_colorspace import AMOCIOColorspace
from .am_ocio_log_convert import AMOCIOLogConvert
from .am_reformat import AMReformat
from .am_seed import AMSeed
from .am_video_read import AMVideoRead
from .am_video_write import AMVideoWrite

WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")

NODE_CLASS_MAPPINGS = {
    "AMImageRead":       AMImageRead,
    "AMImageWrite":      AMImageWrite,
    "AMFrameRange":      AMFrameRange,
    "AMGrade":           AMGrade,
    "AMGradeRGB":        AMGradeRGB,
    "AMOCIOColorspace":  AMOCIOColorspace,
    "AMOCIOLogConvert":  AMOCIOLogConvert,
    "AMReformat":        AMReformat,
    "AMSeed":            AMSeed,
    "AMVideoRead":       AMVideoRead,
    "AMVideoWrite":      AMVideoWrite,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AMImageRead":       "AM Read Image",
    "AMImageWrite":      "AM Write Image",
    "AMFrameRange":      "AM Frame Range",
    "AMGrade":           "AM Grade",
    "AMGradeRGB":        "AM Grade RGB",
    "AMOCIOColorspace":  "AM OCIO Colorspace",
    "AMOCIOLogConvert":  "AM OCIO Log Convert",
    "AMReformat":        "AM Reformat",
    "AMSeed":            "AM Seed",
    "AMVideoRead":       "AM Read Video",
    "AMVideoWrite":      "AM Write Video",
}

logging.getLogger("am_vfx_tools").info("[am-vfx-tools] loaded (11 nodes)")

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
