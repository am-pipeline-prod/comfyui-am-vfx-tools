// AM VFX Tools — frontend extension.
//
// Adds:
//   * Width-patching for AM nodes — every AM VFX Tools class spawns at the
//     same fixed canvas width (350 px) so the node graph has a uniform
//     visual rhythm regardless of LiteGraph's per-widget auto-sizing.
//   * "🔍 Detect Range" button on AM Read Image AND AM Read Video. Posts
//     to /am-vfx-tools/detect-range — the backend picks the probe
//     backend from the resolved extension (os.scandir for image
//     sequences; PyAV header read for video containers). Result
//     populates `first_frame` / `last_frame` widgets.
//   * "📁 Open in Explorer" button on AM Read / AM Write nodes. Posts
//     to /am-vfx-tools/open-in-explorer — server walks up to the
//     deepest existing ancestor when the path itself doesn't exist.
//   * Drag-drop support: drop an image or video file onto the canvas
//     and it becomes an AM Image/Video Read node configured to load
//     the file. Posts to /am-vfx-tools/drop. Also supports the inverse
//     "load embedded workflow from media metadata" mode via a settings
//     toggle.
//   * Removes the auto-generated `control_after_generate` widget on
//     Write nodes — obsolete now that AM Seed owns mode/randomize/
//     increment/decrement semantics server-side. The seed widget on
//     Write nodes is metadata-only (default -1 = "look up the AM Seed
//     registry"), so the per-prompt mutation knob has no role there.
//
// Image formats handled by the drop handler: exr, png, jpg, jpeg, tif,
// tiff, dpx, hdr, webp, tga, bmp.
// Video formats handled by the drop handler: mov, mp4, mkv, webm, avi,
// m4v, mpg, mpeg.
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const DETECT_RANGE_URL   = "/am-vfx-tools/detect-range";
const DROP_URL           = "/am-vfx-tools/drop";
const OPEN_EXPLORER_URL  = "/am-vfx-tools/open-in-explorer";
const ROOTS_URL          = "/am-vfx-tools/roots";
const NATIVE_OPEN_URL    = "/am-vfx-tools/native-dialog/open";
const NATIVE_SAVE_URL    = "/am-vfx-tools/native-dialog/save";

// Setting controlling what happens when the user drops a media file onto
// the canvas. Default `media` reverses stock ComfyUI's behavior: instead
// of treating the drop as a workflow-recovery action (which only works
// for files with embedded `prompt`/`workflow` metadata), we save the
// drop to the input directory and create an AM Read node with the
// absolute path filled in. The `workflow` value restores the stock
// behavior, with the EXR/MKV gap covered by our own extractor (stock
// ComfyUI's frontend dispatcher has no EXR branch).
const DROP_MODE_SETTING  = "am-vfx-tools.dragdrop-mode";
const DROP_MODE_MEDIA    = "media";
const DROP_MODE_WORKFLOW = "workflow";

// Extensions our drop handler claims. Anything else falls through to the
// stock dispatcher (json / safetensors / latent / glb / ...).
const IMAGE_DROP_EXTS = new Set([
  "exr", "png", "jpg", "jpeg", "tif", "tiff", "dpx", "hdr", "webp",
  "tga", "bmp",
]);
const VIDEO_DROP_EXTS = new Set([
  "mov", "mp4", "mkv", "webm", "avi", "m4v", "mpg", "mpeg",
]);

// Read / Write nodes that get the 📁 Open in Explorer button on file_path.
// AM OCIO Colorspace doesn't (it has no file_path widget — it's a transform).
const TARGET_NODES = new Map([
  ["AMImageRead",  { kind: "open", label: "AM Read Image"           }],
  ["AMVideoRead",  { kind: "open", label: "AM Read Video"           }],
  ["AMImageWrite", { kind: "save", label: "AM Write Image"          }],
  ["AMVideoWrite", { kind: "save", label: "AM Write Video"          }],
]);

// All AM VFX Tools nodes spawn at the same fixed width on fresh creation —
// LiteGraph auto-sizes per-node based on widget content, so different
// AM nodes end up at different widths (200-280 px range), and even a
// min-width bump leaves them inconsistent because wider nodes stay at
// their auto-computed value. Forcing a single fixed width gives the
// canvas a uniform AM-Pipe look. 350 px is enough for `file_path`
// and the categorized colorspace dropdowns at default zoom without
// overwhelming the canvas — settled at 350 after iterating
// 300 → 400 → 600 → 700 → 350 with the user. Saved-workflow loads
// bypass this (ComfyUI restores the saved size AFTER `nodeCreated`
// fires; the resize only applies when a node is first dropped from
// the menu / drag-drop / template instantiation).
const WIDEN_NODES = new Set([
  "AMImageRead", "AMVideoRead",
  "AMImageWrite", "AMVideoWrite",
  "AMGrade", "AMGradeRGB",
  "AMOCIOColorspace", "AMOCIOLogConvert",
  "AMSeed",
]);
const TARGET_NODE_WIDTH = 350;

// Single source of truth for the width-forcing — called from
// `nodeCreated` (initial spawn) AND from any later widget surgery
// that might recompute size (e.g. Write nodes'
// `stripSeedControlWidget`, ComfyUI's seed-widget post-processor).
// No-op for non-AM-Pipe classes; idempotent.
//
// Two-pronged enforcement:
//   1. Patch `node.computeSize` so any `setSize(computeSize())` call
//      anywhere in the lifecycle (LiteGraph, ComfyUI, our strip
//      routine, ...) returns our target width while preserving the
//      auto-computed height. This is the load-bearing piece — it's
//      what catches the late `setSize(computeSize())` from ComfyUI's
//      seed-widget post-processor that was undoing our fix on Write
//      nodes after `nodeCreated` had already set the width.
//   2. Set `node.size[0]` directly to the target value, so the
//      current displayed size is correct without waiting for the
//      next computeSize() call.
//
// The `__am_vfx_tools_width_patched__` flag also serves as the "this is a
// fresh node we own" signal: saved-graph nodes go through
// `loadedGraphNode` (not `nodeCreated`), so the flag is never set on
// them, and `stripSeedControlWidget`'s reapply early-returns —
// preserving saved widths.
function applyAmPipeWidth(node) {
  try {
    const cls = node?.comfyClass ?? node?.type;
    if (!WIDEN_NODES.has(cls)) return;

    // (1) Patch computeSize once. Wraps the original so the height
    // stays naturally auto-computed; only the width is locked.
    if (!node.__am_vfx_tools_width_patched__) {
      const origCompute = node.computeSize;
      node.computeSize = function (...args) {
        let h = 100;
        try {
          const orig = origCompute ? origCompute.apply(this, args) : null;
          if (orig && typeof orig[1] === "number") h = orig[1];
        } catch (_e) { /* fall through with default height */ }
        return [TARGET_NODE_WIDTH, h];
      };
      node.__am_vfx_tools_width_patched__ = true;
    }

    // (2) Set current width.
    if (!Array.isArray(node.size) || node.size.length < 2) return;
    if (node.size[0] !== TARGET_NODE_WIDTH) {
      node.size[0] = TARGET_NODE_WIDTH;
      node.setDirtyCanvas?.(true, true);
    }
  } catch (e) {
    console.warn("[am-vfx-tools] could not resize node:", e);
  }
}

// Saved-graph-safe reapply: only force the width if `applyAmPipeWidth`
// was previously called on this node (= we're inside a fresh-creation
// lifecycle, not a saved-graph load). Used by `stripSeedControlWidget`
// to recover from its own `setSize(computeSize())` recalc — the
// computeSize patch alone catches it for fresh nodes, but this is a
// belt-and-suspenders direct re-set.
function reapplyAmPipeWidthIfFresh(node) {
  if (!node?.__am_vfx_tools_width_patched__) return;
  try {
    if (!Array.isArray(node.size) || node.size.length < 2) return;
    if (node.size[0] !== TARGET_NODE_WIDTH) {
      node.size[0] = TARGET_NODE_WIDTH;
      node.setDirtyCanvas?.(true, true);
    }
  } catch (_e) { /* ignore */ }
}

// Detect Range applies to BOTH read nodes — the backend picks the probe
// backend from the resolved file extension (sequence scandir for image
// patterns, PyAV header read for video containers).
const DETECT_RANGE_NODES = new Set(["AMImageRead", "AMVideoRead"]);

// Seed widgets (named "seed", INT) get an auto-generated
// "control_after_generate" widget appended by ComfyUI's frontend. On
// Write nodes the seed is metadata-only (default -1 → look up the AM
// Seed registry) and the per-prompt mutation knob is obsolete — AM
// Seed owns randomize/increment/decrement/fixed semantics server-side
// for the whole workflow. We splice the auto-generated widget out
// after node creation on these specific nodes.
const STRIP_SEED_CONTROL_NODES = new Set(["AMImageWrite", "AMVideoWrite"]);

// One-time migration shim: detect workflows saved while the
// `load_saved_from_disk` widget was briefly inserted MID-LIST (between
// file_path and ext) and shift it back to the tail to match the current
// append-only INPUT_TYPES shape. See `migrateLoadSavedFromDiskWidget`
// for the detection heuristic and rewrite logic.
const LOAD_SAVED_MIGRATION_NODES = new Set(["AMImageWrite", "AMVideoWrite"]);

// Per-node ext enums — the post-toggle widget at index 5 in the shape-
// FINAL widgets_values is the file extension. We use the enum to
// distinguish "shape FINAL" (string in enum) from "shape INTERMEDIATE"
// (BOOLEAN at index 5). Keep these in sync with the Python-side
// INPUT_TYPES `ext` choices on each Write node.
const _EXT_ENUMS = {
  "AMImageWrite": new Set(["exr", "png", "jpg", "tif", "dpx", "hdr", "webp"]),
  "AMVideoWrite": new Set(["mov", "mp4", "mkv", "webm"]),
};

function migrateLoadSavedFromDiskWidget(info, nodeName) {
  const wv = info && info.widgets_values;
  if (!Array.isArray(wv) || wv.length < 7) return;
  const extEnum = _EXT_ENUMS[nodeName];
  if (!extEnum) return;
  // shape INTERMEDIATE detection: index 5 is a boolean (the toggle's
  // mid-list position) AND index 6 is a value from the ext enum
  // (where ext should have been before the toggle was inserted).
  // shape FINAL: index 5 IS the ext string — boolean check fails,
  // function exits as a no-op. shape ORIGINAL: index 5 is also the
  // ext string — same no-op.
  const v5 = wv[5];
  const v6 = wv[6];
  const isV5Bool = (typeof v5 === "boolean");
  const isV6Ext  = (typeof v6 === "string" && extEnum.has(v6));
  if (!(isV5Bool && isV6Ext)) return;
  // Splice out index 5 and append at the end of the array.
  const toggleVal = wv.splice(5, 1)[0];
  wv.push(toggleVal);
  console.log(
    "[am-vfx-tools] migrated", nodeName,
    "load_saved_from_disk widget from mid-list (index 5) to tail —",
    "saved-workflow shape forwarded.",
  );
}

// Parse a JSON body even on non-OK responses (so we can surface server
// error fields like {"error":"outside-sandbox","path":"…","message":"…"}
// instead of a useless "HTTP 403").
async function parseMaybeJson(response) {
  try { return await response.json(); }
  catch (_e) { return null; }
}

async function callDetectRange(payload) {
  const r = await fetch(DETECT_RANGE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await parseMaybeJson(r);
  if (!r.ok) {
    const err = new Error(body?.message || `detect-range: HTTP ${r.status}`);
    err.body = body; err.status = r.status;
    throw err;
  }
  return body || {};
}

// Spawn the OS-native file manager at *path*. Server walks up to the
// deepest existing ancestor when *path* itself doesn't exist (Write
// outputs frequently don't exist yet on a fresh shot).
async function callOpenInExplorer(path) {
  const r = await fetch(OPEN_EXPLORER_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  const body = await parseMaybeJson(r);
  if (!r.ok) {
    const err = new Error(body?.message || `open-in-explorer: HTTP ${r.status}`);
    err.body = body; err.status = r.status;
    throw err;
  }
  return body || {};
}

// Upload a drop and probe its embedded workflow metadata in one round-trip.
// Reads response as text first so a parse failure can be logged with the
// raw body for diagnostics — `response.json()` would have consumed the
// stream and left us nothing to inspect on failure.
async function callDrop(file) {
  const fd = new FormData();
  fd.append("file", file, file.name);
  const r = await fetch(DROP_URL, { method: "POST", body: fd });
  const rawText = await r.text();
  let parsed = null;
  try { parsed = JSON.parse(rawText); }
  catch (e) {
    console.warn(
      "[am-vfx-tools] /drop response not valid JSON:", e,
      "head:", rawText.slice(0, 300),
    );
  }
  if (!r.ok) {
    const err = new Error(parsed?.message || `drop: HTTP ${r.status}`);
    err.body = parsed; err.status = r.status;
    throw err;
  }
  return parsed || {};
}

function dropExtension(file) {
  const m = (file?.name || "").match(/\.([^.]+)$/);
  return m ? m[1].toLowerCase() : "";
}

function dropKindFor(file) {
  const ext = dropExtension(file);
  if (IMAGE_DROP_EXTS.has(ext)) return "image";
  if (VIDEO_DROP_EXTS.has(ext)) return "video";
  return null;
}

// Read the active drag-drop mode from ComfyUI's settings store. We support
// both the modern (`extensionManager.setting.get`) and the legacy
// (`ui.settings.getSettingValue`) accessor — different frontend versions
// expose different surfaces, and the settings entry registers under both
// API surfaces transparently.
function dragDropMode() {
  try {
    const v = app?.extensionManager?.setting?.get?.(DROP_MODE_SETTING);
    if (typeof v === "string" && v) return v;
  } catch (_e) { /* ignore and fall through */ }
  try {
    const v = app?.ui?.settings?.getSettingValue?.(DROP_MODE_SETTING, DROP_MODE_MEDIA);
    if (typeof v === "string" && v) return v;
  } catch (_e) { /* ignore */ }
  return DROP_MODE_MEDIA;
}

// Place a freshly-created node near the canvas drop position so it lands
// where the user actually dropped (matching stock LoadImage behavior).
function dropPosition(canvas) {
  // ComfyUI sets `last_drop_position` on the LiteGraph canvas during the
  // drop event. Fall back to canvas-center math if absent.
  const dp = canvas?.last_drop_position;
  if (Array.isArray(dp) && dp.length >= 2) return [dp[0], dp[1]];
  const ds = canvas?.ds;
  if (ds?.offset && canvas?.canvas) {
    const cx = canvas.canvas.width  / 2 / (ds.scale || 1) - ds.offset[0];
    const cy = canvas.canvas.height / 2 / (ds.scale || 1) - ds.offset[1];
    return [cx, cy];
  }
  return [0, 0];
}

function setNodeWidget(node, name, value) {
  const w = node?.widgets?.find?.((x) => x.name === name);
  if (!w) return false;
  w.value = value;
  try { w.callback?.(value); } catch (_e) { /* ignore */ }
  return true;
}

async function createAmReadFromDrop(canvas, kind, info) {
  const nodeName = (kind === "video") ? "AMVideoRead" : "AMImageRead";
  const ctor = LiteGraph?.createNode || globalThis.LiteGraph?.createNode;
  if (typeof ctor !== "function") {
    throw new Error("LiteGraph.createNode not available");
  }
  const node = ctor.call(LiteGraph, nodeName);
  if (!node) {
    throw new Error(`could not create ${nodeName}`);
  }
  // Yield one tick so any async work the constructor / onNodeCreated chain
  // queued (e.g. the frontend's seed-control auto-append, dynamic widget
  // setup) settles before we touch widgets — mirrors stock ComfyUI's
  // createNode helper (`await new Promise(e => setTimeout(e, 0))`).
  // Without this yield setting `widget.value` immediately can race with
  // widget initialization and leave the value undefined.
  await new Promise((resolve) => setTimeout(resolve, 0));

  const [px, py] = dropPosition(canvas);
  const w = node.size?.[0] ?? 200;
  const h = node.size?.[1] ?? 100;
  node.pos = [px - w / 2, py - h / 2];

  const widgetPath = (typeof info?.path === "string" && info.path)
                     ? info.path
                     : (typeof info?.absolute_path === "string" && info.absolute_path
                        ? info.absolute_path : "");
  if (!widgetPath) {
    console.error(
      "[am-vfx-tools] /drop returned no path — file_path will be empty. " +
      "Server response:", info,
    );
  }

  // Set file_path BEFORE adding to graph so any configure pass triggered
  // by graph.add() picks up our value; set it again AFTER add() as
  // belt-and-suspenders against builds that don't configure on add.
  setNodeWidget(node, "file_path", widgetPath);
  app.graph.add(node);
  setNodeWidget(node, "file_path", widgetPath);

  node.setDirtyCanvas?.(true, true);
  return node;
}

// Wraps the original ComfyUI handleFile to intercept media drops. Files we
// don't claim (json / safetensors / glb / ...) pass through untouched, as
// do non-canvas-drop sources (e.g. file picker dialogs).
let _stockHandleFile = null;
async function amHandleFile(file, source, opts) {
  // Pass-through for unsupported types or non-drop sources.
  const kind = dropKindFor(file);
  if (!kind || source !== "file_drop") {
    return _stockHandleFile.call(this, file, source, opts);
  }

  const mode = dragDropMode();

  let info;
  try {
    info = await callDrop(file);
  } catch (e) {
    console.error("[am-vfx-tools] drop upload failed; falling through to stock handler:", e);
    return _stockHandleFile.call(this, file, source, opts);
  }

  // Workflow mode: try to load the embedded graph first; if absent, fall
  // back to creating an AM Read node (per the user's spec — "if no
  // embedded workflow, just fall back to loading the media instead").
  if (mode === DROP_MODE_WORKFLOW && info?.found && info?.workflow) {
    try {
      await app.loadGraphData(info.workflow);
      console.info(
        `[am-vfx-tools] drop -> loaded workflow from ${info.filename} ` +
        `(format=${info.format})`
      );
      return;
    } catch (e) {
      console.error("[am-vfx-tools] loadGraphData failed; falling back to AM Read:", e);
      // Fall through to AM Read creation.
    }
  }

  try {
    const node = await createAmReadFromDrop(app.canvas, kind, info);
    if (node) {
      const via = info?.resolved_via || "upload";
      console.info(
        `[am-vfx-tools] drop -> ${node.type} (via=${via}) at ${info.absolute_path || info.path}`
      );
    }
  } catch (e) {
    console.error("[am-vfx-tools] could not create AM Read for drop; falling through:", e);
    return _stockHandleFile.call(this, file, source, opts);
  }
}

function installDropHandler() {
  if (_stockHandleFile) return;  // already installed
  if (typeof app.handleFile !== "function") {
    // Frontend hasn't wired handleFile yet — retry on next tick.
    setTimeout(installDropHandler, 50);
    return;
  }
  _stockHandleFile = app.handleFile.bind(app);
  app.handleFile = amHandleFile;
  console.info("[am-vfx-tools] drag-drop handler installed");
}

function findWidget(node, name) {
  if (!node.widgets) return null;
  return node.widgets.find((w) => w.name === name);
}

function moveWidgetAbove(node, widget, anchorName) {
  // Move *widget* in node.widgets to sit directly above the widget
  // named *anchorName*. addWidget() always appends to the end; this
  // splices the entry into the right slot for visual ordering.
  const wIdx = node.widgets.indexOf(widget);
  if (wIdx === -1) return;
  node.widgets.splice(wIdx, 1);
  const anchorIdx = node.widgets.findIndex((w) => w.name === anchorName);
  if (anchorIdx === -1) {
    // anchor not found — restore at original position
    node.widgets.splice(wIdx, 0, widget);
    return;
  }
  node.widgets.splice(anchorIdx, 0, widget);
}

function addOpenInExplorerButton(node, nodeName) {
  const widget = findWidget(node, "file_path");
  if (!widget) return;

  if (node.__am_open_explorer_added__) return;
  node.__am_open_explorer_added__ = true;

  const button = node.addWidget("button", "📁 Open in Explorer", "open_in_explorer", async () => {
    try {
      const path = (widget && widget.value) || "";
      if (!path) {
        alert(
          "AM VFX Tools: file_path is empty — type or drop a file first."
        );
        return;
      }
      const j = await callOpenInExplorer(path);
      if (j?.opened) {
        console.info(`[am-vfx-tools] opened in file manager: ${j.opened}`);
      }
    } catch (e) {
      console.error("[am-vfx-tools] open-in-explorer failed:", e);
      if (e.body?.error === "no-existing-ancestor") {
        alert(
          "AM VFX Tools: nothing on this path exists yet — not even a parent " +
          "directory. Render at least one frame, or pick an existing " +
          "location first.\n\n" +
          `Path: ${e.body.path}`
        );
        return;
      }
      if (e.body?.error === "no-file-manager") {
        alert(
          "AM VFX Tools: no OS file manager available on this server.\n\n" +
          (e.body.message || "")
        );
        return;
      }
      alert("AM VFX Tools: open in explorer failed — " + e.message);
    }
  });

  // Sits directly above file_path.
  moveWidgetAbove(node, button, "file_path");
}


// ---------------------------------------------------------------------------
// Browse button + Copy File Path — port of the internal pack's
// `addBrowseButton` / `addCopyPathButton`. Browse calls the vendored
// filechooser routes (`/am-vfx-tools/{roots, native-dialog/open|save}`)
// for native OS dialog spawn; Copy is pure frontend (clipboard API +
// legacy textarea fallback).
// ---------------------------------------------------------------------------

let _rootsCache = null;
async function fetchRoots() {
  if (_rootsCache) return _rootsCache;
  try {
    const r = await fetch(ROOTS_URL);
    if (!r.ok) return [];
    const j = await r.json();
    _rootsCache = Array.isArray(j.roots) ? j.roots : [];
    return _rootsCache;
  } catch (_e) { return []; }
}

async function _parseMaybeJson(response) {
  try { return await response.json(); }
  catch (_e) { return null; }
}

async function callNativeOpen(default_dir, title) {
  const r = await fetch(NATIVE_OPEN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title: title || "AM Read — open",
      default_dir: default_dir || null,
    }),
  });
  const body = await _parseMaybeJson(r);
  if (!r.ok) {
    const err = new Error(body?.message || `native dialog open: HTTP ${r.status}`);
    err.body = body; err.status = r.status;
    throw err;
  }
  return body || {};
}

async function callNativeSave(default_dir, default_filename, title) {
  const r = await fetch(NATIVE_SAVE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title: title || "AM Write — choose folder",
      default_dir: default_dir || null,
      default_filename: default_filename || null,
    }),
  });
  const body = await _parseMaybeJson(r);
  if (!r.ok) {
    const err = new Error(body?.message || `native dialog save: HTTP ${r.status}`);
    err.body = body; err.status = r.status;
    throw err;
  }
  return body || {};
}

function _dirnameOf(p) {
  if (!p) return null;
  const idx = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  if (idx <= 0) return p;
  if (p[idx - 1] === ":") return p.slice(0, idx + 1);
  return p.slice(0, idx);
}

async function defaultDirFor(widget) {
  // Prefer the directory portion of the current widget value. Falls back
  // to the first configured sandbox root so the dialog opens somewhere
  // the server is willing to accept.
  const v = (widget && widget.value) || "";
  if (v) {
    const d = _dirnameOf(v);
    if (d) return d;
  }
  const roots = await fetchRoots();
  return roots[0] || null;
}

function defaultFilenameFor(nodeName) {
  switch (nodeName) {
    case "AMImageWrite": return "image.png";
    case "AMVideoWrite": return "video.mov";
    default:             return "";
  }
}

async function copyToClipboard(text) {
  if (navigator?.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (e) {
      console.info("[am-vfx-tools] navigator.clipboard.writeText failed; trying legacy fallback:", e);
    }
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "-1000px";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return Boolean(ok);
  } catch (e) {
    console.error("[am-vfx-tools] legacy clipboard fallback failed:", e);
    return false;
  }
}

function addBrowseButton(node, nodeName, spec /* { kind, label } */) {
  const widget = findWidget(node, "file_path");
  if (!widget) return;

  if (node.__am_browse_added__) return;
  node.__am_browse_added__ = true;

  const button = node.addWidget("button", "📂 Browse", "browse", async () => {
    try {
      const dir = await defaultDirFor(widget);
      const j = spec.kind === "open"
        ? await callNativeOpen(dir, spec.label)
        : await callNativeSave(dir, widget.value || defaultFilenameFor(nodeName), spec.label);
      if (j && j.path) {
        // Public pack stores the absolute path verbatim — no project-root
        // tokenisation (that's an internal-pipeline concern).
        widget.value = j.path;
        widget.callback?.(j.path);
        node.setDirtyCanvas(true, true);
      } else if (j && j.cancelled) {
        // user hit Cancel — quietly do nothing
      } else if (j && j.fallback) {
        alert(
          "AM VFX Tools: native file dialog unavailable on this server " +
          `(reason: ${j.fallback}). Type the path into file_path manually.`
        );
      }
    } catch (e) {
      console.error("[am-vfx-tools] browse failed:", e);
      // Sandbox rejection is the most likely 4xx — surface a friendly message.
      if (e.body?.message) {
        alert(`AM VFX Tools: ${e.body.message}`);
      } else {
        alert("AM VFX Tools: browse failed — " + e.message);
      }
    }
  });

  // Browse sits directly above file_path. Order at runtime is established
  // by the registration order in `beforeRegisterNodeDef`.
  moveWidgetAbove(node, button, "file_path");
}

function addCopyPathButton(node, nodeName) {
  const widget = findWidget(node, "file_path");
  if (!widget) return;

  if (node.__am_copy_path_added__) return;
  node.__am_copy_path_added__ = true;

  const button = node.addWidget("button", "📋 Copy File Path", "copy_path", async () => {
    try {
      const path = (widget && widget.value) || "";
      if (!path) {
        alert("AM VFX Tools: file_path is empty — nothing to copy.");
        return;
      }
      const ok = await copyToClipboard(path);
      if (ok) {
        console.info(`[am-vfx-tools] copied to clipboard: ${path}`);
      } else {
        alert(
          "AM VFX Tools: clipboard copy failed. Your browser may be blocking " +
          "clipboard access on non-HTTPS origins. Path:\n\n" + path
        );
      }
    } catch (e) {
      console.error("[am-vfx-tools] copy file path failed:", e);
      alert("AM VFX Tools: copy file path failed — " + e.message);
    }
  });

  moveWidgetAbove(node, button, "file_path");
}


function addDetectRangeButton(node) {
  const filePathWidget = findWidget(node, "file_path");
  if (!filePathWidget) return;

  if (node.__am_detect_range_added__) return;
  node.__am_detect_range_added__ = true;

  const button = node.addWidget("button", "🔍 Detect Range", "detect_range", async () => {
    const raw = (filePathWidget.value || "").trim();
    if (!raw) {
      alert(
        "AM VFX Tools: file_path is empty — type or drop one frame of " +
        "the sequence first."
      );
      return;
    }
    const payload = { path: raw };

    try {
      const j = await callDetectRange(payload);
      if (j && j.first != null && j.last != null) {
        const firstW = findWidget(node, "first_frame");
        const lastW  = findWidget(node, "last_frame");
        if (firstW) {
          firstW.value = j.first;
          firstW.callback?.(j.first);
        }
        if (lastW) {
          lastW.value = j.last;
          lastW.callback?.(j.last);
        }
        node.setDirtyCanvas(true, true);
        console.info(
          `[am-vfx-tools] detect-range -> ${j.pattern} | first=${j.first} last=${j.last} count=${j.count}`
        );
        if (j.count > 0 && (j.last - j.first + 1) !== j.count) {
          // Sequence has gaps — informational, not blocking.
          console.info(
            `[am-vfx-tools] sequence has gaps: ${j.count} frames between ${j.first} and ${j.last}.`
          );
        }
      } else {
        alert(
          "AM VFX Tools: no sequence detected at the resolved path.\n\n" +
          (j?.pattern ? `Resolved pattern: ${j.pattern}\n\n` : "") +
          "Make sure the path contains a frame token (#### or %05d) or " +
          "points to a frame in a numbered sequence (e.g. plate.0001.exr)."
        );
      }
    } catch (e) {
      console.error("[am-vfx-tools] detect-range failed:", e);
      alert("AM VFX Tools: detect-range failed — " + e.message);
    }
  });

  // Position: directly above the `frame_rate` widget if the node has
  // one (AM Read Image), else above `first_frame` (AM Read Video). The
  // canvas order on Read Image becomes:
  //   frame_mode → 🔍 Detect Range → frame_rate → first_frame
  // — Detect Range stays adjacent to `frame_mode` (the conceptually
  // related neighbor) without slipping below `frame_rate`.
  const hasFrameRate = (node.widgets || []).some((w) => w.name === "frame_rate");
  moveWidgetAbove(node, button, hasFrameRate ? "frame_rate" : "first_frame");
}

function stripSeedControlWidget(node) {
  // ComfyUI's frontend appends an auto-generated widget named
  // "control_after_generate" right after any "seed" / "noise_seed" INT
  // widget. On Write nodes that knob is obsolete — AM Seed owns
  // randomize/increment/decrement/fixed semantics server-side for the
  // whole workflow, and the Write seed is a metadata-only sentinel
  // (default -1 → look up the AM Seed registry). Splice the widget out
  // so the artist isn't presented with a dead knob.
  if (node.__am_seed_control_stripped__) return;
  const ctrl = findWidget(node, "control_after_generate");
  if (!ctrl) return;  // older frontend / no auto-gen — nothing to do
  try {
    const idx = node.widgets.indexOf(ctrl);
    if (idx >= 0) node.widgets.splice(idx, 1);
    node.__am_seed_control_stripped__ = true;
    // Recompute the node's height now that a widget is gone. For
    // fresh AM VFX Tools nodes, `node.computeSize` has been patched by
    // `applyAmPipeWidth` (called from `nodeCreated`) to return the
    // forced target width — so this `setSize(computeSize())` call
    // preserves our width while letting the height shrink to the
    // natural value. For saved-graph nodes (no patch), this falls
    // back to LiteGraph's auto-computed dimensions exactly as before.
    try { node.setSize?.(node.computeSize()); } catch (_e) { /* older frontend */ }
    reapplyAmPipeWidthIfFresh(node);
    node.setDirtyCanvas?.(true, true);
  } catch (e) {
    console.warn("[am-vfx-tools] could not strip seed control_after_generate widget:", e);
  }
}

app.registerExtension({
  name: "am-vfx-tools",
  // Settings panel entry — surfaces under "AM VFX Tools ▸ Drag-drop
  // mode" in the Settings dialog. Default `media` reverses stock
  // ComfyUI behavior so dropping an EXR / MOV / PNG creates an
  // AM Read node rather than trying (and failing, in the EXR case) to
  // load an embedded workflow. Switch to `workflow` to match stock
  // SaveImage's drag-drop-to-load-workflow flow — the AM-VFX-Tools
  // extractor covers EXR / MKV which stock ComfyUI's frontend
  // dispatcher misses.
  settings: [
    {
      id: DROP_MODE_SETTING,
      category: ["AM VFX Tools", "Drag-drop mode"],
      name: "Drag-drop mode",
      tooltip: (
        "What happens when you drop an image / video file onto the canvas:\n" +
        "  • Load media as AM Read node — uploads the file to ComfyUI's input/, " +
        "creates an AM Image/Video Read node with the absolute path filled in. " +
        "Default.\n" +
        "  • Load workflow from metadata — extracts the embedded workflow " +
        "(if any) and loads it onto the canvas. Falls back to media-load when " +
        "the file has no embedded workflow."
      ),
      type: "combo",
      options: [
        { value: DROP_MODE_MEDIA,    text: "Load media as AM Read node"     },
        { value: DROP_MODE_WORKFLOW, text: "Load workflow from metadata"   },
      ],
      defaultValue: DROP_MODE_MEDIA,
    },
  ],
  async setup() {
    // Install the drop interceptor once the app is wired. Doing it in
    // `setup()` (called after the frontend boots) avoids the race where
    // `app.handleFile` isn't yet defined when the extension loads.
    installDropHandler();
  },
  // Fires once per fresh node creation (menu drop / drag-drop / template
  // instantiation). NOT called for saved-graph loads — those go through
  // `loadedGraphNode` and get their saved size restored automatically,
  // so this resize only affects "new" nodes the artist drops onto an
  // empty canvas. Forces a fixed width on EVERY AM VFX Tools class (not a
  // min-bump) so they all spawn at the same size regardless of their
  // auto-computed widget width. Width is also re-applied from inside
  // `stripSeedControlWidget` (which fires on a setTimeout AFTER this
  // hook and would otherwise reset the size via setSize+computeSize).
  async nodeCreated(node) {
    applyAmPipeWidth(node);
  },
  async beforeRegisterNodeDef(nodeType, nodeData, _app) {
    const spec = TARGET_NODES.get(nodeData.name);
    const wantsDetectRange   = DETECT_RANGE_NODES.has(nodeData.name);
    const wantsStripSeedCtrl = STRIP_SEED_CONTROL_NODES.has(nodeData.name);
    const wantsLoadSavedMig  = LOAD_SAVED_MIGRATION_NODES.has(nodeData.name);
    if (!spec && !wantsDetectRange && !wantsStripSeedCtrl && !wantsLoadSavedMig) return;

    // Saved-workflow migration shim. Litegraph fires `onConfigure(info)`
    // during workflow deserialization with `info.widgets_values` mutable.
    // ComfyUI persists widget values positionally — adding a widget
    // anywhere except the end of INPUT_TYPES shifts every subsequent
    // saved value by one slot and silently corrupts every existing
    // workflow on disk.
    //
    // The current Write-node shape places `load_saved_from_disk` at
    // the END of INPUT_TYPES (append-only — see
    // `docs/media-io-sync-rule.md` invariant 33). Two earlier shapes
    // exist in the wild and need to be migrated forward:
    //
    //   shape ORIGINAL — no toggle. widgets_values is N entries.
    //   shape INTERMEDIATE — toggle inserted MID-LIST under file_path
    //                        (between file_path at 4 and ext at 5).
    //                        widgets_values has N+1 entries with a
    //                        BOOLEAN at index 5 and the original
    //                        ext-string shifted to index 6.
    //   shape FINAL — toggle appended at the END of required.
    //                 widgets_values has N+1 entries with the
    //                 ext-string back at its original index 5 and the
    //                 BOOLEAN somewhere near the tail.
    //
    // ORIGINAL → FINAL: ComfyUI defaults missing trailing entries —
    // no migration needed.
    //
    // INTERMEDIATE → FINAL: detect by checking widgets_values[5] is
    // BOOLEAN AND widgets_values[6] is in the ext enum, then splice
    // out index 5 and append it at the end of the array. Idempotent —
    // re-running on FINAL data is a no-op (widgets_values[5] would be
    // the ext string, not a boolean).
    if (wantsLoadSavedMig) {
      const origOnConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        try {
          migrateLoadSavedFromDiskWidget(info, nodeData.name);
        } catch (e) {
          console.error(
            "[am-vfx-tools] load_saved_from_disk migration failed for",
            nodeData.name, e,
          );
        }
        return origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
      };
    }

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated ? onCreated.apply(this, arguments) : undefined;
      // Registration order matters — moveWidgetAbove inserts each button
      // directly above file_path, so the FIRST registered button ends up
      // at the TOP. Final visual order top→bottom:
      //   Browse → Open in Explorer → Copy File Path → file_path.
      try {
        if (spec) addBrowseButton(this, nodeData.name, spec);
      } catch (e) {
        console.error("[am-vfx-tools] could not add Browse button:", e);
      }
      try {
        if (spec) addOpenInExplorerButton(this, nodeData.name);
      } catch (e) {
        console.error("[am-vfx-tools] could not add Open-in-Explorer button:", e);
      }
      try {
        if (spec) addCopyPathButton(this, nodeData.name);
      } catch (e) {
        console.error("[am-vfx-tools] could not add Copy File Path button:", e);
      }
      try {
        if (wantsDetectRange) addDetectRangeButton(this);
      } catch (e) {
        console.error("[am-vfx-tools] could not add Detect Range button:", e);
      }
      if (wantsStripSeedCtrl) {
        // The control_after_generate widget is appended *after* this
        // hook runs (during the frontend's seed-widget post-processing
        // pass), so retry on the next microtask + once after a tick.
        const apply = () => stripSeedControlWidget(this);
        try { apply(); } catch (_e) { /* ignore */ }
        try { Promise.resolve().then(apply); } catch (_e) { /* ignore */ }
        try { setTimeout(apply, 0); } catch (_e) { /* ignore */ }
      }
      return r;
    };
  },
});
