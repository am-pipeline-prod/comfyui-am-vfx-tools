# comfyui-am-vfx-tools

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-7d3aed)](https://github.com/comfyanonymous/ComfyUI)

VFX I/O & color toolkit for ComfyUI: image + video read/write
(OpenImageIO + PyAV), OCIO 2.x color management, Nuke-style Grade,
OpenCV reformat, render-farm-safe Seed, and frame-range slicing.
**11 nodes**, all under the **AM Pipe** category in the node menu.

This pack is the public, generic subset of an internal studio pipeline.
The internal pack (`am-pipe-comfy`) adds an "Auto" mode that resolves
output paths from a studio-specific folder grammar; this public pack
ships **Manual mode only** — pick a path explicitly via the `file_path`
widget. Everything else (color management, frame ranges, codec coverage,
reformat) is identical.

## Nodes

### Image & video I/O

| Node | What it does |
|---|---|
| **AM Read Image** | Read still or sequence via OpenImageIO. EXR / DPX / TIFF / PNG / JPG / HDR / etc. Frame-token aware (`####` / `%05d` / `$F4`). Three frame modes (single / range / all). Per-frame missing-frame policies (error / black / hold / nearest / checkerboard). Edge-extrapolation policies for out-of-range reads. OCIO 2.x source→working colorspace transform. Optional reformat + dtype cast. Splits alpha to a MASK socket per stock-Comfy convention. |
| **AM Write Image** | Write still or sequence via OpenImageIO. Same format coverage. Per-format compression options (EXR DWAB/PIZ/ZIPS, PNG level, JPG quality, ...). Frame-numbered or zero-padded output. OCIO working→output colorspace transform. Optional reformat + dtype cast. Embeds the live ComfyUI workflow JSON into EXR / PNG metadata for round-trip recovery. |
| **AM Read Video** | Read video via PyAV. Full codec coverage (h264 / h265 / prores / dnxhr / vp9 / ...). Frame-accurate seek. Alpha-bearing pixel formats decoded losslessly when the codec supports them (prores 4444, prores 4444xq). OCIO source→working colorspace transform. Optional reformat. |
| **AM Write Video** | Write video via PyAV. Codec / profile / pixel-format / fps widgets. Alpha-channel encode for prores 4444. OCIO working→output colorspace transform. Optional reformat. |

### Color management

| Node | What it does |
|---|---|
| **AM OCIO Colorspace** | Apply an OCIO 2.x colorspace transform between any two roles in your active config. Loader hierarchy: `$OCIO` env → built-in **Studio** config → built-in **CG** config → identity stub. Works out of the box on any ComfyUI install with `opencolorio>=2.5`. |
| **AM OCIO Log Convert** | Round-trip between scene-linear and a configured log encoding (`compositing_log` / `scene_linear` roles). Useful as the boundary node when feeding scene-linear output into log-trained samplers and back. |

### Reformat / grade / frame-range / seed

| Node | What it does |
|---|---|
| **AM Reformat** | OpenCV-backed resize. Five filters (impulse / linear / cubic / Lanczos4 / area). Scale by factor, target W/H, or preset. Fit / fill / pad / crop. 4-channel alpha-preserving. Optional dtype cast on output. |
| **AM Grade** | Nuke-style Grade math (`(x - blackpoint) * (whitepoint - blackpoint)^-1 * (gain - lift) + lift`, then `gamma` and `multiply`/`offset`). Single luminance values for blackpoint / whitepoint / gain / lift. |
| **AM Grade RGB** | Same math as AM Grade but per-channel (separate R / G / B controls for each parameter). |
| **AM Frame Range** | Slice an IMAGE batch by `start_frame` / `end_frame` / `step`. Output a sub-batch. |
| **AM Seed** | Render-farm-safe seed node that fixes [comfyanonymous/ComfyUI#11905](https://github.com/comfyanonymous/ComfyUI/issues/11905) — the seed value is captured at queue-time and surfaces on the socket, so a workflow re-queued days later (or run on a different host) reproduces exactly. |

## Installation

### Via ComfyUI-Manager (once published to the Registry)

Search for `comfyui-am-vfx-tools` in ComfyUI-Manager and click Install.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/am-pipeline-prod/comfyui-am-vfx-tools.git
cd comfyui-am-vfx-tools
pip install -r requirements.txt
```

Restart ComfyUI. You should see the **AM Pipe** category in the node menu.

### Runtime dependencies

Installed via `requirements.txt`:

* **`openimageio`** (image read/write — EXR/DPX/TIFF/PNG/JPG/HDR; pip wheel name is lowercase `openimageio`, the import name is `OpenImageIO`).
* **`opencolorio>=2.5.0`** (OCIO 2.x — needed for the built-in Studio / CG config fallbacks).
* **`av>=14.0.0`** (PyAV — video read/write, alpha-bearing pixel formats, frame-accurate seek).
* **`imageio_ffmpeg>=0.5.0`** (bundled ffmpeg shim).
* **`opencv-python-headless>=4.10.0`** (reformat filters + preview thumbnailer).

### EXR support

If your existing ComfyUI environment doesn't already have EXR enabled in
OpenCV, set this before launch:

```bash
export OPENCV_IO_ENABLE_OPENEXR=1
```

(The pack also sets it in-process at import for safety, but earlier
custom nodes that touch OpenCV may already have committed to the
default.)

## OCIO config selection

The pack picks an OCIO config in this order at import time:

1. **`$OCIO` environment variable** — if set and resolvable, uses that.
2. **Built-in OCIO 2.5+ Studio config** — ACES, full reference primaries, log encodings.
3. **Built-in OCIO 2.5+ CG config** — the lighter ACES-derived CG-focused config.
4. **Identity stub** — last resort, single-colorspace passthrough so the nodes still load on environments without a config at all.

Override per-launch with `OCIO=/path/to/your.ocio`.

## File-path conventions

`file_path` accepts:

* An absolute path to a single file (any format OIIO / PyAV understands).
* A path with a frame token for sequences:
  * `####` (with N hashes for N-digit zero-padding)
  * `%05d` printf style
  * `$F4` Nuke style

Examples:

```
/work/shots/sh010/plate.0001.exr           # single frame
/work/shots/sh010/plate.####.exr           # sequence (4-digit padding)
/work/shots/sh010/plate.%05d.exr           # sequence (5-digit padding)
/work/shots/sh010/plate.$F4.exr            # sequence (Nuke style, 4-digit)
/work/shots/sh010/take2.mov                # video container
```

The 🔍 **Detect Range** button next to `first_frame` / `last_frame` on
the read nodes scans the directory and auto-fills the range.

## Drag-drop support

Drop a media file from your OS file manager onto the ComfyUI canvas:
the pack spawns the right AM Read node (image vs. video by extension)
already configured to load that file. A setting under
**Settings → AM VFX Tools → Drag-drop mode** lets you switch between
"create AM Read node" and "load workflow" (the latter for files with
embedded ComfyUI workflows in their metadata, like an EXR previously
saved by AM Write Image).

## 📁 Open in Explorer

Each AM Read / Write node has a 📁 button that reveals the resolved
`file_path` in your OS file manager (Explorer / Finder / Nautilus /
Dolphin). Walks up to the deepest existing parent if the resolved path
doesn't exist yet (useful for write nodes targeting a not-yet-created dir).

## Development

```bash
git clone https://github.com/am-pipeline-prod/comfyui-am-vfx-tools.git
cd comfyui-am-vfx-tools
pip install -r requirements.txt
python -m py_compile $(find . -name '*.py' -not -path './__pycache__/*')
```

Architecture notes: most of the heavy lifting lives in `_core/`
(`color.py`, `image_backend.py`, `video_backend.py`, `reformat.py`,
`sequence.py`, `grade.py`, `seed_registry.py`, `preview.py`,
`batch_suffix.py`, `_node_replacements.py`). The `am_*.py` node files
at the top level are mostly INPUT_TYPES + execute() shells over those
core modules.

## Contributing

Issues and PRs welcome on
[github.com/am-pipeline-prod/comfyui-am-vfx-tools](https://github.com/am-pipeline-prod/comfyui-am-vfx-tools).

* Small, focused PRs preferred.
* `python -m py_compile` should pass for any touched .py file.
* No new runtime dependencies without a matching update to BOTH
  `requirements.txt` AND the `dependencies` list in `pyproject.toml`.

## Credits

* Image-IO codebase descends from a code-port of
  [`sumitchatterjee13/nuke-nodes-comfyui`](https://github.com/sumitchatterjee13/nuke-nodes-comfyui)'s
  `io_nodes.py` — the OIIO read/write idioms and frame-pattern handling
  come from there. See `NOTICE`.
* The HDR / Bit-depth / OCIO config-loader work that powers the OCIO
  nodes is built on top of [OpenColorIO](https://opencolorio.org/) 2.5+.

## License

[MIT](LICENSE).
