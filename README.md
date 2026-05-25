# comfyui-am-vfx-tools

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-7d3aed)](https://github.com/comfyanonymous/ComfyUI)

VFX I/O & color toolkit for ComfyUI: image + video read/write
(OpenImageIO + PyAV), OCIO 2.x color management, Nuke-style Grade +
Color Correct, OpenCV reformat, frame-order reverse, render-farm-safe
Seed, and frame-range slicing. **13 nodes** under the **AM VFX Tools**
category, plus a workfile-io menu (Save / Open / Recent / Incremental
Save) for managing workflow JSON files outside ComfyUI's default folder.

Every pixel node also carries a native ComfyUI **`VIDEO` socket** for
low-RAM streaming — see [Video sockets](#video-sockets) below.

This pack is the public, generic subset of an internal studio pipeline.
The internal pack (`am-pipe-comfy`) adds an "Auto" mode that resolves
output paths from a studio-specific folder grammar; this public pack
ships **Manual mode only** — pick a path explicitly via the `file_path`
widget (the 📂 Browse button gives you the OS-native file dialog).
Everything else (color management, frame ranges, codec coverage,
reformat, workfile-io) is identical.

> **Maintenance:** This is a **self-serve** project. It works for me and
> I'm sharing it as a starting point — feel free to use, fork, or copy
> the code (MIT). I'm **not actively maintaining it**: bug reports + PRs
> may sit unanswered, and feature requests may not land. If you depend
> on it, plan to maintain your own fork.

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
| **AM Color Correct** | Nuke ColorCorrect-style grade: saturation, contrast, gamma, gain, offset, and hue-rotation controls. Pure-torch per-pixel math, alpha-preserving. |
| **AM Reverse Sequence** | Reverse the frame order of an IMAGE/MASK batch (or a wired VIDEO stream). |
| **AM Frame Range** | Slice an IMAGE batch by `start_frame` / `end_frame` / `step`. Output a sub-batch. |
| **AM Seed** | Render-farm-safe seed node that fixes [comfyanonymous/ComfyUI#11905](https://github.com/comfyanonymous/ComfyUI/issues/11905) — the seed value is captured at queue-time and surfaces on the socket, so a workflow re-queued days later (or run on a different host) reproduces exactly. |

## Video sockets

Every pixel node carries a native ComfyUI **`VIDEO`** socket (requires
ComfyUI ≥ 0.3.48 for `comfy_api.v0_0_2`; the nodes degrade gracefully on
older builds). These let you keep long sequences out of RAM:

* **AM Read Video → `video`** — a `VideoFromFile` referencing the source
  on disk. Wiring *only* this socket (not `image`) skips the PyAV decode
  entirely — peak RAM stays at file-handle level instead of an N-frame
  tensor. ⚠️ Raw passthrough: source colorspace/resolution/codec,
  unaffected by the node's OCIO/Reformat (those apply to `image`).
* **AM Read Image → `video`** — a `VideoFromComponents` wrapper around
  the already-decoded batch (convenience; no RAM benefit).
* **AM Write Image ← `video`** — the streaming branch. Wire a
  `VideoFromFile` (e.g. the output of an upstream API/upscaler node) and
  the writer iterates **one frame at a time** through the per-frame OCIO
  + Reformat + dtype pipeline, writing each EXR — peak RAM stays at one
  frame regardless of sequence length. This is the node that unblocks
  long-sequence pipelines that would otherwise OOM on a full IMAGE batch.
* **AM Write Video ← `video`** — packet-level remux (`save_to`) when
  source/dest containers match, decode+re-encode otherwise; no IMAGE
  materialization.
* **AM Reformat / Grade / Grade RGB / Color Correct / OCIO Colorspace /
  OCIO Log Convert / Frame Range / Reverse Sequence** — VIDEO in + out,
  returning a *lazy* wrapper that defers the per-pixel work until a
  downstream consumer iterates frames. Chains of these collapse from
  O(N) full-batch materializations to O(1) — peak RAM ≈ source batch +
  one frame's working buffers.

Mixing IMAGE and VIDEO branches in the same graph is valid (e.g. an
IMAGE branch for color-managed inference plus a VIDEO branch for an API
call that takes the original file).

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

Restart ComfyUI. You should see the **AM VFX Tools** category in the
node menu and an **AM VFX Tools** menu in the top menubar.

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

## File-path buttons (Browse / Open in Explorer / Copy)

Each AM Read / Write node has three buttons stacked above `file_path`:

* **📂 Browse** — opens the OS-native file dialog (zenity / kdialog /
  yad on Linux, PowerShell on Windows). Picked path is written into
  `file_path`. Opens at the configured default directory (user home +
  `~/Documents`; override via env `AM_VFX_TOOLS_FILECHOOSER_ROOTS` —
  see the "Configuring default dialog directories" section below).
  Falls back to an in-browser file browser when no native tool is
  available. **Path access is not restricted** — pick anywhere the
  OS lets you read/write.
* **📁 Open in Explorer** — reveals the resolved `file_path` in your
  OS file manager (Explorer / Finder / Nautilus / Dolphin). Walks up
  to the deepest existing parent if the resolved path doesn't exist
  yet (useful for write nodes targeting a not-yet-created dir).
* **📋 Copy File Path** — copies the resolved `file_path` to the
  system clipboard.

## Workfile-io menu

Adds an **AM VFX Tools** top-level menu replacing the stock File menu,
with:

* **Open…** (`Ctrl+Shift+O`)
* **Open Recent…** (`Ctrl+Alt+O`)
* **Save** (`Ctrl+Shift+S`)
* **Save As…**
* **Save Incremental** (`Ctrl+Alt+S`) — bumps the trailing `_v###`
* **Open Current Folder**
* **Copy Current Path**

Save / Open prefer the **native OS file dialog** when available, fall
back to a built-in browser dialog otherwise. Default starting directory
matches the Browse button (user home + `~/Documents`, or whatever
`AM_VFX_TOOLS_FILECHOOSER_ROOTS` points at). No path restrictions.

A toggle under **Settings → AM VFX Tools → Workfile IO → Prefer
native OS file dialogs** lets you force the in-browser dialog even when
native is available.

## Configuring default dialog directories

The Browse button + workfile-io dialogs need a sensible starting
directory. By default they open at:

* **Linux / Windows / macOS:** `~` + `~/Documents` (when it exists)

You can override these defaults by setting
`AM_VFX_TOOLS_FILECHOOSER_ROOTS` in your ComfyUI launch environment —
useful if your work lives somewhere else and you don't want to navigate
there every time:

* **Linux / macOS:** `AM_VFX_TOOLS_FILECHOOSER_ROOTS=/work/projects:/scratch:/mnt/nas`
* **Windows:** `AM_VFX_TOOLS_FILECHOOSER_ROOTS=D:\work;E:\projects;Z:\nas`

> **No path restrictions.** This pack does NOT sandbox path access —
> you can type or pick any absolute path the OS lets you read/write,
> regardless of the configured "roots." The roots are purely the
> *default starting directory* and the entries in the in-browser
> browser's roots dropdown. Same model as stock ComfyUI.

## Development

```bash
git clone https://github.com/am-pipeline-prod/comfyui-am-vfx-tools.git
cd comfyui-am-vfx-tools
pip install -r requirements.txt
python -m py_compile $(find . -name '*.py' -not -path './__pycache__/*')
```

Architecture notes: most of the heavy lifting lives in `_core/`
(`color.py`, `color_correct.py`, `image_backend.py`, `video_backend.py`,
`video_lazy.py`, `reformat.py`, `sequence.py`, `grade.py`,
`seed_registry.py`, `preview.py`, `batch_suffix.py`). The `am_*.py` node
files at the top level are mostly INPUT_TYPES + execute() shells over
those core modules. VIDEO-socket lazy-transform wrappers live in
`_core/video_lazy.py`.

## Contributing

Issues and PRs welcome on
[github.com/am-pipeline-prod/comfyui-am-vfx-tools](https://github.com/am-pipeline-prod/comfyui-am-vfx-tools).

* Small, focused PRs preferred.
* `python -m py_compile` should pass for any touched .py file.
* No new runtime dependencies without a matching update to BOTH
  `requirements.txt` AND the `dependencies` list in `pyproject.toml`.

## Credits

The OIIO read/write paths (`am_image_read.py` / `am_image_write.py` /
`_core/image_backend.py`) and the frame-token handling (`_core/sequence.py`)
started from
[**`sumitchatterjee13/nuke-nodes-comfyui`**](https://github.com/sumitchatterjee13/nuke-nodes-comfyui)
— that repo's `io_nodes.py` and its `parse_frame_pattern` /
`expand_frame_pattern` / `detect_sequence` helpers gave us the OIIO +
frame-token foundations. **Big thanks to Sumit Chatterjee** for putting
the original work out under MIT.

This pack has been **rewritten from scratch and extended substantially**
on top of those foundations: full OCIO 2.x ColorProcessor pipeline,
PyAV-backed video I/O with alpha-channel support, embedded ComfyUI
workflow metadata round-trip, OpenCV reformat with five filters,
Nuke-style Grade math, a render-farm-safe Seed registry, and the
workfile-io subsystem (native OS file dialogs + sandbox-free workflow
JSON management). See [`NOTICE`](NOTICE) for the full per-file
attribution detail.

Other dependencies powering the pack:
* [OpenColorIO](https://opencolorio.org/) 2.5+ — color management.
* [OpenImageIO](https://sites.google.com/site/openimageio/home) — image I/O.
* [PyAV](https://pyav.org/) — video I/O.
* [OpenCV](https://opencv.org/) — reformat filters.

## License

[MIT](LICENSE).
