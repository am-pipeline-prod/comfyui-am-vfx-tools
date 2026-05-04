// Shared singleton state for AM VFX Tools Workfile I/O.
// Persisted (subset) to localStorage so per-user prefs survive reloads.

const STORAGE_KEY = "am_vfx_tools.workfile-io.state";

function _loadPersisted() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* ignore — corrupt entry, treat as empty */ }
  return {};
}

const _persisted = _loadPersisted();

export const state = {
  // session-only (filled by extension setup)
  os: null,                  // "linux" | "windows"
  roots: [],                 // string[] of resolved root paths
  nativeAvailable: null,     // null = unknown, true/false = checked. Lazily
                             // resolved via ensureNativeChecked() below so a
                             // user click before setup()'s async fetch finishes
                             // doesn't fall back to the in-browser browser.
  nativeTool: null,          // "zenity"|"kdialog"|"yad"|"powershell"|null

  // persisted
  lastSaved: _persisted.lastSaved || null,    // string|null — most recent save/load
  lastDir: _persisted.lastDir || null,        // string|null — directory pre-fill
  preferNative: _persisted.preferNative !== false,  // default true
};

export function persist() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      lastSaved: state.lastSaved,
      lastDir: state.lastDir,
      preferNative: state.preferNative,
    }));
  } catch { /* ignore — quota / disabled */ }
}
