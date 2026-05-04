import { app } from "../../scripts/app.js";
import { state, persist } from "./lib/state.js";
import { openSaveDialog, openOpenDialog, openRecentDialog, trySave } from "./lib/dialogs.js";
import { getRoots, getNativeAvailable, nextVersion, revealFolder } from "./lib/api.js";
import { injectStyles } from "./lib/styles.js";

const EXT_NAME = "AMVFXTools.WorkfileIO";
const MENU_LABEL = "AM VFX Tools";

function basename(p) { return p.split(/[\\/]/).pop(); }

function dirname(p) {
  const idx = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  if (idx < 0) return p;
  if (idx === 0) return p.slice(0, 1);
  if (p[idx - 1] === ":") return p.slice(0, idx + 1);
  return p.slice(0, idx);
}

function toast(severity, summary, detail, life = 3000) {
  app.extensionManager?.toast?.add({ severity, summary, detail, life });
}

async function plainSave() {
  // Standard Save semantics: write to the current file path, no prompt.
  // Falls through to Save As when no current path is known (first save).
  if (!state.lastSaved) return openSaveDialog();
  await trySave(state.lastSaved, /*overwrite=*/true);
}

async function copyCurrentPath() {
  // Paths-as-strings workaround: zenity's GTK file chooser doesn't expose
  // the current directory as copyable text (Ctrl+L shows an empty entry,
  // not the current path), so we provide a direct command that puts the
  // active workflow's absolute path on the clipboard.
  if (!state.lastSaved) {
    toast("warn", "AM VFX Tools", "No AM VFX Tools workflow currently loaded — open or save one first.", 4000);
    return;
  }
  try {
    await navigator.clipboard.writeText(state.lastSaved);
    toast("success", "AM VFX Tools", `Path copied: ${state.lastSaved}`, 4000);
  } catch (err) {
    // Clipboard API requires a secure context; localhost qualifies but we
    // surface failures explicitly so the user can paste from the toast.
    console.error("[AM VFX Tools] clipboard write failed:", err);
    toast("error", "AM VFX Tools — Copy failed",
      `${err.message}\n\nPath: ${state.lastSaved}`, 12000);
  }
}

async function openCurrentFolder() {
  // Spawn the OS file manager (xdg-open on Linux, explorer on Windows) at
  // the parent directory of the active workflow. Backend handles the OS
  // dispatch and sandbox check.
  if (!state.lastSaved) {
    toast("warn", "AM VFX Tools", "No AM VFX Tools workflow currently loaded.", 4000);
    return;
  }
  try {
    const r = await revealFolder(state.lastSaved);
    toast("success", "AM VFX Tools", `Opened ${r.opened}`, 3000);
  } catch (err) {
    toast("error", "AM VFX Tools — Open Folder failed", String(err.message || err), 6000);
  }
}

async function incrementalSave() {
  if (!state.lastSaved) return openSaveDialog();
  let suggestion;
  try {
    const r = await nextVersion(state.lastSaved);
    suggestion = r.suggestion;
  } catch (err) {
    if (err.status === 422 && err.body?.error === "no-version") {
      // No version pattern detected — let user choose to fall back to Save As.
      const wantSaveAs = window.confirm(
        `${err.body.message}\n\n` +
        `Click OK to open Save As (so you can give the file a versioned name).\n` +
        `Click Cancel to abort and rename the file manually.`
      );
      if (wantSaveAs) openSaveDialog();
      return;
    }
    toast("error", "AM VFX Tools — Save Incremental failed", String(err.message || err), 6000);
    return;
  }
  // Reuse the standard save flow so stampGraphMetadata + retagActiveWorkflow
  // both fire — keeps the workflow tab title and graph.extra in sync after
  // an incremental save just like after Save As.
  await trySave(suggestion, false);
}

// ---------------------------------------------------------------------------
// Move our top-level menu to position 2 (between the first built-in menu and
// the rest). The "menubar" is actually a single PrimeVue TieredMenu popup
// behind the Comfy logo, driven by useMenuItemStore().menuItems — there is
// no .p-menubar in the DOM. The store has no public insert-at-index API, so
// we splice the reactive menuItems array directly. Mutating the Pinia ref
// re-renders the popup correctly and survives every popup open/close.
// ---------------------------------------------------------------------------

function getMenuItemStore() {
  // The menu store isn't exposed as a property on app.extensionManager
  // (confirmed by the diagnostic dump — em has 27 keys, none of them menu*).
  // app.extensionManager IS a Pinia store (the workspace store), so its
  // private `_p` field is the Pinia instance with `_s` (the stores Map).
  // We dig for a store that has a reactive `menuItems` array. This is
  // private-API territory but the only path available; if a future ComfyUI
  // changes Pinia internals we'll detect it via the null return and the
  // menu just stays at its default position.
  const em = app?.extensionManager;
  if (!em) return null;
  // Direct property (in case a future build promotes it to em.*)
  if (em.menuItem?.menuItems) return em.menuItem;
  // Pinia internals
  const stores = em._p?._s;
  if (stores?.get) {
    // Try the documented store id first
    const direct = stores.get("menuItem") || stores.get("menu") || stores.get("menuItems");
    if (direct?.menuItems) return direct;
    // Walk all stores and find one with a menuItems array
    for (const [, store] of stores.entries()) {
      try {
        if (Array.isArray(store?.menuItems)) {
          return store;
        }
      } catch { /* getter may throw on some stores */ }
    }
  }
  return null;
}

// Marker we tag on our injected separator so customizeMenu is idempotent —
// running again leaves the menu unchanged when already correctly placed.
const SEPARATOR_MARKER = "_amPipeSeparator";
const HIDDEN_TOP_LABELS = ["File"]; // ComfyUI's File menu — replaced by AM VFX Tools

function customizeMenu() {
  const store = getMenuItemStore();
  const items = store?.menuItems;
  if (!Array.isArray(items)) return false;

  const ourIdx = items.findIndex((i) => i?.label === MENU_LABEL);
  if (ourIdx < 0) return false;

  const hasHiddenTop = items.some((i) => HIDDEN_TOP_LABELS.includes(i?.label));
  const inPlace = ourIdx === 0 && items[1]?.separator && items[1]?.[SEPARATOR_MARKER];
  if (inPlace && !hasHiddenTop) return true;

  // Lift our menu out, strip any pre-existing AM VFX Tools separator (idempotency
  // across re-runs), and remove the suppressed top-level entries.
  const [ours] = items.splice(ourIdx, 1);
  const staleSepIdx = items.findIndex((i) => i?.[SEPARATOR_MARKER]);
  if (staleSepIdx >= 0) items.splice(staleSepIdx, 1);
  for (const lbl of HIDDEN_TOP_LABELS) {
    const idx = items.findIndex((i) => i?.label === lbl);
    if (idx >= 0) items.splice(idx, 1);
  }

  // Prepend AM VFX Tools + separator. items[0]=AM VFX Tools, items[1]=separator,
  // items[2..]=remaining menu sequence (New, Edit, View, …) without File.
  items.unshift(ours, { separator: true, [SEPARATOR_MARKER]: true });
  console.info(`[${EXT_NAME}] menu customized: AM VFX Tools at top + separator, hidden=${HIDDEN_TOP_LABELS.join(",")}`);
  return true;
}

// ---------------------------------------------------------------------------
// Diagnostic — dumps exactly which APIs ComfyUI exposes on this build so we
// can target the right paths for menu reorder and workflow tab title shim
// instead of guessing. Visible in DevTools console under [AM VFX Tools DIAG].
// ---------------------------------------------------------------------------
function diagnosticDump() {
  try {
    const em = app.extensionManager;
    console.group("[AM VFX Tools DIAG]");
    console.info("app.extensionManager:", em);
    console.info("app.extensionManager keys:", em ? Object.keys(em).sort() : "(null)");
    if (em) {
      for (const k of Object.keys(em).sort()) {
        const v = em[k];
        if (v && typeof v === "object") {
          let keys;
          try { keys = Object.keys(v).sort(); } catch { keys = "(unenumerable)"; }
          console.info(`em.${k} (${typeof v}):`, Array.isArray(keys) ? keys.slice(0, 30) : keys);
        } else {
          console.info(`em.${k} (${typeof v}):`, v);
        }
      }
    }
    // Probe the specific paths we care about
    const probes = [
      "extensionManager.workflow",
      "extensionManager.workflow.createTemporary",
      "extensionManager.workflow.activeWorkflow",
      "extensionManager.workflowService",
      "extensionManager.workflowService.openWorkflow",
      "extensionManager.menuItem",
      "extensionManager.menuItem.menuItems",
      "extensionManager.menubar",
      "extensionManager.menu",
    ];
    for (const path of probes) {
      const parts = path.split(".");
      let v = app;
      for (const p of parts) {
        v = v?.[p];
        if (v == null) break;
      }
      console.info(`probe app.${path}:`, v == null ? "MISSING" : `present (${typeof v})`);
    }
    // Enumerate all Pinia store IDs reachable from em._p._s — this is where
    // menuItemStore lives when it exists. If "menuItem" doesn't appear in
    // the list, our reorder logic will give up gracefully.
    const stores = em?._p?._s;
    if (stores?.keys) {
      const ids = [];
      for (const k of stores.keys()) ids.push(k);
      console.info("Pinia store IDs:", ids.sort());
      // Identify any store that has a menuItems array — that's our target
      const menuish = [];
      for (const [id, store] of stores.entries()) {
        try {
          if (Array.isArray(store?.menuItems)) menuish.push({ id, length: store.menuItems.length });
        } catch { /* skip */ }
      }
      console.info("stores with .menuItems array:", menuish);
    }
    if (window.__PINIA__) {
      console.info("window.__PINIA__:", Object.keys(window.__PINIA__));
    }
    if (app.workflowManager) {
      console.info("app.workflowManager keys:", Object.keys(app.workflowManager));
    }
    console.groupEnd();
  } catch (err) {
    console.error("[AM VFX Tools DIAG] dump failed:", err);
  }
}

function startMenuReorder() {
  // Extension menu items are appended after extensionService runs us, which
  // happens after app.registerExtension returns. Retry a few times to cover
  // the gap.
  let n = 0;
  const tick = () => {
    if (customizeMenu() || ++n >= 40) return;
    setTimeout(tick, 100);
  };
  setTimeout(tick, 50);
}

// Keybinding-store surgery to override ComfyUI's defaults (Ctrl+S, Ctrl+O,
// Ctrl+Shift+S) was attempted and abandoned: probe-based unbind found and
// removed the entries, but our subsequent registration didn't take effect
// (menu showed no shortcuts, ComfyUI's defaults still fired). We reverted
// to non-conflicting combos. See runbook §8.1 for the full dead-end log.

app.registerExtension({
  name: EXT_NAME,

  settings: [
    {
      // ID hierarchy renders as Settings → AM VFX Tools → Workfile IO → <name>
      id: "AM VFX Tools.Workfile IO.preferNative",
      name: "Prefer native OS file dialogs (Save / Open)",
      tooltip: "When enabled and a native dialog tool (zenity / kdialog / yad / PowerShell) is available, AM VFX Tools uses the OS file picker. Otherwise it falls back to its own in-browser file browser.",
      type: "boolean",
      defaultValue: true,
      onChange: (newVal) => {
        state.preferNative = !!newVal;
        persist();
      },
    },
  ],

  async setup() {
    injectStyles();
    try {
      const info = await getRoots();
      state.os = info.os;
      state.roots = info.roots || [];
    } catch (err) {
      console.error(`[${EXT_NAME}] failed to fetch roots:`, err);
    }
    try {
      const nat = await getNativeAvailable();
      state.nativeAvailable = !!nat.available;
      state.nativeTool = nat.tool || null;
      console.info(
        `[${EXT_NAME}] os=${state.os} roots=`, state.roots,
        `native=${state.nativeAvailable ? state.nativeTool : "none"} (display=${nat.display_ok})`,
      );
    } catch (err) {
      console.error(`[${EXT_NAME}] failed to query native dialog availability:`, err);
    }
    if (!state.roots.length) {
      toast("warn", "AM VFX Tools — no sandbox roots configured",
        "Set AM_PIPE_WORK_FILE_ROOTS or ensure default roots exist.", 8000);
    }
    const pref = app.extensionManager?.setting?.get("AM VFX Tools.Workfile IO.preferNative");
    if (typeof pref === "boolean") state.preferNative = pref;
    console.info(`[${EXT_NAME}] preferNative=${state.preferNative}`);

    // Diagnostic dump — prints exactly which APIs are exposed on this build
    // so we can target the right paths instead of guessing. Stops once the
    // workflow / menu fixes are confirmed working in production.
    diagnosticDump();

    startMenuReorder();
  },

  commands: [
    { id: "am-vfx-tools.workfile.open",       label: "Open",                function: openOpenDialog },
    { id: "am-vfx-tools.workfile.recent",     label: "Open Recent",         function: openRecentDialog },
    { id: "am-vfx-tools.workfile.save",       label: "Save",                function: plainSave },
    { id: "am-vfx-tools.workfile.saveAs",     label: "Save As",             function: openSaveDialog },
    { id: "am-vfx-tools.workfile.incSave",    label: "Save Incremental",    function: incrementalSave },
    { id: "am-vfx-tools.workfile.openFolder", label: "Open Current Folder", function: openCurrentFolder },
    { id: "am-vfx-tools.workfile.copyPath",   label: "Copy Current Path",   function: copyCurrentPath },
  ],

  menuCommands: [
    {
      path: [MENU_LABEL],
      commands: [
        "am-vfx-tools.workfile.open",
        "am-vfx-tools.workfile.recent",
        "am-vfx-tools.workfile.save",
        "am-vfx-tools.workfile.saveAs",
        "am-vfx-tools.workfile.incSave",
        "am-vfx-tools.workfile.openFolder",
        "am-vfx-tools.workfile.copyPath",
      ],
    },
  ],

  keybindings: [
    // Combos chosen to NOT conflict with ComfyUI's defaults (Ctrl+S,
    // Ctrl+O, Ctrl+Shift+S). Conflict-suppression via Pinia-store
    // surgery was attempted and reverted — see runbook §8.1 for the
    // dead-end log so we don't repeat it.
    { combo: { key: "O", ctrl: true, shift: true },           commandId: "am-vfx-tools.workfile.open" },
    { combo: { key: "O", ctrl: true, alt: true },             commandId: "am-vfx-tools.workfile.recent" },
    { combo: { key: "S", ctrl: true, shift: true },           commandId: "am-vfx-tools.workfile.save" },
    { combo: { key: "S", ctrl: true, alt: true },             commandId: "am-vfx-tools.workfile.saveAs" },
    { combo: { key: "S", ctrl: true, shift: true, alt: true }, commandId: "am-vfx-tools.workfile.incSave" },
  ],
});
