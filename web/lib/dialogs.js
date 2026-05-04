// Orchestrates the Save / Open / Recent flows.
//
// Save / Open: try the native OS dialog first (zenity/kdialog/yad on Linux,
// PowerShell on Windows). If unavailable / headless / errored, fall back to
// the in-browser file browser. The "preferNative" user setting (default true)
// lets the user force the in-browser browser.

import { app } from "../../../scripts/app.js";
import { state, persist } from "./state.js";
import {
  loadWorkflow, saveWorkflow, getRecent, clearRecent,
  nativeOpen, nativeSave, getNativeAvailable,
} from "./api.js";
import { pickViaBrowser } from "./browser.js";

// Lazy, memoized native-availability check. Resolves the race between
// setup()'s async fetch and a user clicking Open / Save As right away —
// without this, state.nativeAvailable would still be its `null` default
// at click time and we'd silently fall back to the in-browser browser.
let _nativeCheckPromise = null;
async function ensureNativeChecked() {
  if (state.nativeAvailable !== null) return;
  if (_nativeCheckPromise) return _nativeCheckPromise;
  _nativeCheckPromise = (async () => {
    try {
      const nat = await getNativeAvailable();
      state.nativeAvailable = !!nat.available;
      state.nativeTool = nat.tool || null;
    } catch (err) {
      console.warn("[AM VFX Tools] native availability check failed:", err);
      state.nativeAvailable = false;
    }
  })();
  return _nativeCheckPromise;
}

function basename(p) { return p.split(/[\\/]/).pop(); }

function dirname(p) {
  const idx = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  if (idx < 0) return p;
  if (idx === 0) return p.slice(0, 1);
  if (p[idx - 1] === ":") return p.slice(0, idx + 1);
  return p.slice(0, idx);
}

function relTime(ts) {
  const s = Math.floor(Date.now() / 1000 - ts);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function toast(severity, summary, detail, life = 3000) {
  app.extensionManager?.toast?.add({ severity, summary, detail, life });
}

function defaultSaveFilename() {
  if (state.lastSaved) return basename(state.lastSaved);
  return "workflow.json";
}

function defaultDir() {
  if (state.lastDir) return state.lastDir;
  if (state.lastSaved) return dirname(state.lastSaved);
  return state.roots[0] || null;
}

// ---------------------------------------------------------------------------
// Save / Open orchestration
// ---------------------------------------------------------------------------

async function pickSavePath() {
  await ensureNativeChecked();
  console.debug(
    `[AM VFX Tools] pickSavePath preferNative=${state.preferNative} nativeAvailable=${state.nativeAvailable} tool=${state.nativeTool}`
  );
  if (state.preferNative && state.nativeAvailable) {
    try {
      const r = await nativeSave({
        defaultDir: defaultDir(),
        defaultFilename: defaultSaveFilename(),
        title: "AM VFX Tools — Save Workflow As",
      });
      if (r.path) return { path: r.path, overwrite: !!r.overwrite };
      if (r.cancelled) return null;
      console.warn("[AM VFX Tools] native save fell back, opening in-browser browser:", r);
    } catch (err) {
      console.warn("[AM VFX Tools] native save error, opening in-browser browser:", err);
    }
  }
  return pickViaBrowser({
    mode: "save",
    title: "AM VFX Tools — Save Workflow As",
    defaultDir: defaultDir(),
    defaultFilename: defaultSaveFilename(),
  });
}

async function pickOpenPath() {
  await ensureNativeChecked();
  console.debug(
    `[AM VFX Tools] pickOpenPath preferNative=${state.preferNative} nativeAvailable=${state.nativeAvailable} tool=${state.nativeTool}`
  );
  if (state.preferNative && state.nativeAvailable) {
    try {
      const r = await nativeOpen({
        defaultDir: defaultDir(),
        title: "AM VFX Tools — Open Workflow",
      });
      if (r.path) return { path: r.path, overwrite: false };
      if (r.cancelled) return null;
      console.warn("[AM VFX Tools] native open fell back, opening in-browser browser:", r);
    } catch (err) {
      console.warn("[AM VFX Tools] native open error, opening in-browser browser:", err);
    }
  }
  return pickViaBrowser({
    mode: "open",
    title: "AM VFX Tools — Open Workflow",
    defaultDir: defaultDir(),
  });
}

// ---------------------------------------------------------------------------
// Public actions
// ---------------------------------------------------------------------------

export async function openSaveDialog() {
  const pick = await pickSavePath();
  if (!pick) return;
  await trySave(pick.path, pick.overwrite);
}

export async function openOpenDialog() {
  const pick = await pickOpenPath();
  if (!pick) return;
  await tryLoad(pick.path);
}

// Exported so extension.js incrementalSave can reuse the full save flow
// (stampGraphMetadata + serialize + save + retagActiveWorkflow + toast).
export async function trySave(path, overwrite) {
  try {
    // Stamp metadata into app.graph.extra BEFORE serialize so it gets
    // saved into the JSON file and round-trips on later loads.
    // stampGraphMetadata is async (it tokenises absPath via a backend
    // round-trip); awaiting before serialize() guarantees the
    // ${PROJECT_ROOT}/... form makes it into the saved JSON.
    await stampGraphMetadata(path);
    const workflow = app.graph.serialize();
    const res = await saveWorkflow(path, workflow, overwrite);
    state.lastSaved = res.path;
    state.lastDir = dirname(res.path);
    persist();
    // After saving, retag the active workflow so the tab title reflects the
    // new filename (covers both fresh saves and Save-As renames).
    retagActiveWorkflow(res.path, workflow);
    toast("success", "AM VFX Tools", `Saved ${basename(res.path)}`);
  } catch (err) {
    if (err.status === 409) {
      const ok = confirm(
        `${basename(path)} already exists.\n\nOverwrite?`
      );
      if (ok) return trySave(path, true);
      return;
    }
    toast("error", "AM VFX Tools — Save failed", String(err.message || err), 6000);
  }
}

async function tryLoad(path) {
  try {
    const wf = await loadWorkflow(path);
    await loadAsNamedWorkflow(path, wf);
    state.lastSaved = path;
    state.lastDir = dirname(path);
    persist();
    toast("success", "AM VFX Tools", `Opened ${basename(path)}`);
  } catch (err) {
    toast("error", "AM VFX Tools — Open failed", String(err.message || err), 6000);
  }
}

/**
 * Load a workflow into ComfyUI such that the topbar tab shows the actual
 * filename instead of "Unsaved Workflow N", and the workflow is treated
 * as persisted/clean (no modified-dot until the user actually edits).
 *
 * Goes via app.extensionManager.workflow + workflowService when available
 * (current ComfyUI frontend); falls back to plain app.loadGraphData on
 * older versions where those APIs aren't exposed. Stashes the absolute
 * path on the workflow object as `amPipePath` so future custom nodes can
 * read it at execution time.
 */
async function loadAsNamedWorkflow(absPath, workflowJson) {
  // Both `openWorkflow(wf)` AND `loadGraphData(json, true, true, wf)` end up
  // calling `wf.load()` (verified in stack traces from the user's console).
  // That 404s for our external paths because load() fetches via ComfyUI's
  // user-files API. Worse, leaving a half-attached temp workflow in the
  // store crashes the next load with "Cannot read properties of null
  // (reading 'store')" at beforeLoadNewGraph. So we DON'T call
  // createTemporary at all. Plain loadGraphData lets ComfyUI create its
  // own "Unsaved Workflow N" temp; we then patch that temp's filename so
  // the tab reads the way we want.
  await app.loadGraphData(workflowJson);
  await stampGraphMetadata(absPath);
  // Defer retag past Vue's reactive flush — loadGraphData triggers
  // changeTracker mutations that would otherwise stomp our _isModified=false
  // reset. Two rAFs is the standard "after current frame's effects" idiom.
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  retagActiveWorkflow(absPath, workflowJson);
  // NOTE: a delayed reactive cascade re-flags the workflow as modified
  // hundreds of ms after our retag, so the "Save changes?" prompt still
  // appears on close even for untouched workflows. Multiple suppression
  // attempts (modified-guard interval, every plausible changeTracker
  // method, size mutation) failed. See runbook §8.1 for the full log
  // and untried avenues. User can hold Shift while closing to bypass.
  console.info(`[AM VFX Tools] loaded: ${basename(absPath)}`);
}

/**
 * After a successful save / load, patch the active workflow so the topbar
 * tab reflects our filename and the workflow looks "clean" (no save prompt
 * on close). **Critical**: we DO NOT touch `path`. The workflow store keys
 * workflows by path; changing it broke tab-close, +new-workflow, and the
 * second open in earlier rounds with "null (reading 'store')" at
 * beforeLoadNewGraph.
 *
 * What we touch + why:
 *   - filename → the tab-label binding (independent field on this build)
 *   - size > -1 → flips isPersisted=true so the "Unsaved" badge clears
 *   - changeTracker.{reset,checkpoint,commit,markClean,markSaved} →
 *     clear-modified-state methods. We probe-and-call all plausible names
 *     since the change-tracker API isn't documented for extensions.
 *   - _isModified / isModified → both the public reactive field and any
 *     private backing field, in case one is a getter.
 *   - amPipePath → out-of-band slot for future custom write-nodes
 *
 * Readback at the end logs the actual state so any field that didn't take
 * effect is visible in DevTools (the close-save prompt is hard to debug
 * without seeing the workflow's post-retag fields).
 */
function retagActiveWorkflow(absPath, workflowJson) {
  const ws = app.extensionManager?.workflow;
  const active = ws?.activeWorkflow;
  if (!active) {
    console.warn("[AM VFX Tools] retag: no activeWorkflow");
    return;
  }
  const filename = basename(absPath);
  const bytes = JSON.stringify(workflowJson).length;

  try { active.amPipePath = absPath; } catch {}
  try { active.filename = filename; } catch (e) { console.warn("[AM VFX Tools] set filename threw:", e); }
  try { active.size = bytes; } catch {}

  // Try every plausible "clear modified" entry point on changeTracker
  const ct = active.changeTracker;
  if (ct) {
    for (const m of ["reset", "checkpoint", "commit", "markClean", "markSaved", "save"]) {
      try { if (typeof ct[m] === "function") ct[m](); } catch {}
    }
  }
  try { active._isModified = false; } catch {}
  try { active.isModified = false; } catch {}

  // Diagnostic readback — surfaces any field that didn't take, so the
  // close-save-prompt issue is debuggable without another diagnostic round.
  console.info("[AM VFX Tools] retag readback:", {
    filename: active.filename,
    path: active.path,
    size: active.size,
    isTemporary: active.isTemporary,
    isPersisted: active.isPersisted,
    _isModified: active._isModified,
    isModified: active.isModified,
    inModifiedList: ws?.modifiedWorkflows?.length != null
      ? ws.modifiedWorkflows.some?.((w) => w === active)
      : "n/a",
    trackerKeys: ct ? Object.keys(ct).slice(0, 20) : null,
  });
}

const _VERSION_RE = /(?<![A-Za-z0-9])([vV])(\d+)(?![A-Za-z0-9])/;

/**
 * Persist a small workflow-origin metadata block into `app.graph.extra.am_vfx_tools`
 * so it round-trips through the workflow JSON. Public pack uses this purely
 * for the in-app version-tracking + last-saved-path memory; no Python nodes
 * read it back. Stays under a clearly-namespaced key so it doesn't collide
 * with any other custom-node pack's `extra` writes.
 */
async function stampGraphMetadata(absPath) {
  const g = app.graph;
  if (!g || !absPath) return;
  if (!g.extra || typeof g.extra !== "object") g.extra = {};
  const filename = basename(absPath);
  const stem = filename.replace(/\.json$/i, "");
  const m = stem.match(_VERSION_RE);
  g.extra.am_vfx_tools = {
    path: absPath,
    filename,
    stem,
    version_label: m ? `${m[1]}${m[2]}` : null,  // e.g. "v001" or null
    version_num: m ? parseInt(m[2], 10) : null,  // e.g. 1
    version_width: m ? m[2].length : null,        // e.g. 3 (for zero-pad preservation)
    stamped_at: Date.now(),
    extension: "comfyui-am-vfx-tools",
    extension_version: "0.2.0",
  };
}

// ---------------------------------------------------------------------------
// Recent dialog
// ---------------------------------------------------------------------------

function buildModalShell(title) {
  const overlay = document.createElement("div");
  overlay.className = "am-vfx-tools-overlay";
  const modal = document.createElement("div");
  modal.className = "am-vfx-tools-modal";
  overlay.appendChild(modal);
  const header = document.createElement("div");
  header.className = "am-vfx-tools-header";
  header.textContent = title;
  modal.appendChild(header);
  const body = document.createElement("div");
  body.className = "am-vfx-tools-body";
  modal.appendChild(body);
  const footer = document.createElement("div");
  footer.className = "am-vfx-tools-footer";
  modal.appendChild(footer);

  let resolveClose;
  const closed = new Promise((r) => { resolveClose = r; });
  function close(value) {
    overlay.remove();
    document.removeEventListener("keydown", onKey);
    resolveClose(value);
  }
  function onKey(e) { if (e.key === "Escape") { e.preventDefault(); close(null); } }
  document.addEventListener("keydown", onKey);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(null); });
  document.body.appendChild(overlay);
  return { overlay, modal, header, body, footer, close, closed };
}

export async function openRecentDialog() {
  const { body, footer, close } = buildModalShell("AM VFX Tools — Recent Files");

  const status = document.createElement("div");
  status.className = "am-vfx-tools-status";
  status.textContent = "Loading…";
  body.appendChild(status);

  const list = document.createElement("div");
  list.className = "am-vfx-tools-recent-list";
  body.appendChild(list);

  const clear = document.createElement("button");
  clear.className = "am-vfx-tools-btn";
  clear.textContent = "Clear list";
  const cancel = document.createElement("button");
  cancel.className = "am-vfx-tools-btn";
  cancel.textContent = "Close";
  footer.appendChild(clear);
  footer.appendChild(cancel);

  cancel.addEventListener("click", () => close(null));
  clear.addEventListener("click", async () => {
    try {
      await clearRecent();
      list.innerHTML = "";
      status.textContent = "List cleared.";
    } catch (err) {
      status.textContent = `Clear failed: ${err.message}`;
    }
  });

  try {
    const { entries } = await getRecent();
    status.textContent = entries.length ? "" : "No recent files.";
    for (const e of entries) {
      const row = document.createElement("div");
      row.className = "am-vfx-tools-recent-item";
      const action = document.createElement("span");
      action.className = "am-vfx-tools-recent-action";
      action.textContent = e.action === "save" ? "↑" : "↓";
      action.title = e.action;
      const pathEl = document.createElement("span");
      pathEl.className = "am-vfx-tools-recent-path";
      pathEl.textContent = e.path;
      const time = document.createElement("span");
      time.className = "am-vfx-tools-recent-time";
      time.textContent = relTime(e.ts);
      row.appendChild(action);
      row.appendChild(pathEl);
      row.appendChild(time);
      row.addEventListener("click", async () => {
        close(null);
        await tryLoad(e.path);
      });
      list.appendChild(row);
    }
  } catch (err) {
    status.textContent = `Failed to load recent: ${err.message}`;
  }
}
