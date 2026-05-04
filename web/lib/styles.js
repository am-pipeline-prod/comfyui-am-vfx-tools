const CSS = `
.am-vfx-tools-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.5);
  display: flex; align-items: center; justify-content: center;
  z-index: 10000;
  font-family: var(--font-family, sans-serif);
}
.am-vfx-tools-modal {
  background: var(--comfy-menu-bg, #202020);
  color: var(--input-text, #ddd);
  border: 1px solid var(--border-color, #444);
  border-radius: 6px;
  min-width: 540px; max-width: 80vw; max-height: 80vh;
  display: flex; flex-direction: column;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.am-vfx-tools-modal-browser {
  width: 800px; max-width: 90vw; height: 600px; max-height: 85vh;
}
.am-vfx-tools-header {
  padding: 12px 16px; border-bottom: 1px solid var(--border-color, #444);
  font-weight: 600; font-size: 14px;
}
.am-vfx-tools-body {
  padding: 16px; flex: 1; overflow-y: auto;
  display: flex; flex-direction: column; gap: 10px;
}
.am-vfx-tools-footer {
  padding: 10px 16px; border-top: 1px solid var(--border-color, #444);
  display: flex; gap: 8px; justify-content: flex-end;
  align-items: center;
}
.am-vfx-tools-footer-form {
  padding: 10px 16px; border-top: 1px solid var(--border-color, #444);
  display: flex; flex-direction: column; gap: 6px;
}
.am-vfx-tools-form-row {
  display: flex; align-items: center; gap: 8px;
}
.am-vfx-tools-form-label { font-size: 12px; min-width: 70px; opacity: 0.85; }
.am-vfx-tools-filename-input { flex: 1; }

.am-vfx-tools-toolbar {
  display: flex; align-items: center; gap: 6px;
  padding: 8px 12px; border-bottom: 1px solid var(--border-color, #333);
  background: rgba(0,0,0,0.15);
}
.am-vfx-tools-roots-select {
  background: var(--comfy-input-bg, #181818);
  color: var(--input-text, #ddd);
  border: 1px solid var(--border-color, #555); border-radius: 4px;
  padding: 4px 6px; font-size: 12px;
  font-family: ui-monospace, monospace;
}
.am-vfx-tools-breadcrumb {
  flex: 1; display: flex; flex-wrap: wrap; align-items: center; gap: 2px;
  padding: 4px 8px; min-height: 28px;
  background: var(--comfy-input-bg, #181818);
  border: 1px solid var(--border-color, #555); border-radius: 4px;
  cursor: text; font-family: ui-monospace, monospace; font-size: 12px;
}
.am-vfx-tools-breadcrumb-seg {
  padding: 1px 5px; border-radius: 3px; cursor: pointer;
}
.am-vfx-tools-breadcrumb-seg:hover {
  background: var(--p-primary-color, #4a9eff); color: white;
}
.am-vfx-tools-breadcrumb-sep { opacity: 0.4; }
.am-vfx-tools-breadcrumb-input {
  width: 100%; border: none; background: transparent; padding: 0;
  font-family: inherit; font-size: inherit; color: inherit;
}
.am-vfx-tools-breadcrumb-input:focus { outline: none; }

.am-vfx-tools-list-wrap {
  flex: 1; display: flex; flex-direction: column; min-height: 0;
  border-bottom: 1px solid var(--border-color, #333);
}
.am-vfx-tools-list-header {
  display: grid;
  grid-template-columns: 28px 1fr 120px 80px;
  gap: 6px;
  padding: 6px 12px;
  background: rgba(0,0,0,0.25);
  font-size: 11px; opacity: 0.7;
  border-bottom: 1px solid var(--border-color, #333);
}
.am-vfx-tools-list-header > div:first-child { grid-column: 1 / 3; }
.am-vfx-tools-col-header { cursor: pointer; user-select: none; }
.am-vfx-tools-col-header:hover { color: var(--p-primary-color, #4a9eff); }
.am-vfx-tools-list {
  flex: 1; overflow-y: auto; outline: none;
}
.am-vfx-tools-list:focus { outline: 1px solid var(--p-primary-color, #4a9eff); outline-offset: -1px; }
.am-vfx-tools-list-row {
  display: grid;
  grid-template-columns: 28px 1fr 120px 80px;
  gap: 6px;
  padding: 4px 12px;
  cursor: default; user-select: none;
  font-size: 12px;
  border-bottom: 1px solid rgba(255,255,255,0.03);
}
.am-vfx-tools-list-row:hover { background: rgba(255,255,255,0.04); }
.am-vfx-tools-list-row-selected,
.am-vfx-tools-list-row-selected:hover {
  background: var(--p-primary-color, #4a9eff); color: white;
}
.am-vfx-tools-list-icon { text-align: center; }
.am-vfx-tools-list-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: ui-monospace, monospace; }
.am-vfx-tools-list-mtime { font-size: 11px; opacity: 0.75; }
.am-vfx-tools-list-size { font-size: 11px; opacity: 0.75; text-align: right; }

.am-vfx-tools-hint { font-size: 11px; opacity: 0.6; }
.am-vfx-tools-input {
  box-sizing: border-box;
  padding: 5px 8px;
  background: var(--comfy-input-bg, #181818);
  color: var(--input-text, #ddd);
  border: 1px solid var(--border-color, #555); border-radius: 4px;
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 12px;
}
.am-vfx-tools-input:focus { outline: 1px solid var(--p-primary-color, #4a9eff); }
.am-vfx-tools-checkbox-row {
  display: flex; align-items: center; gap: 6px;
  font-size: 12px; cursor: pointer;
}
.am-vfx-tools-status { font-size: 12px; min-height: 1.2em; opacity: 0.85; }

.am-vfx-tools-btn {
  padding: 5px 12px; background: transparent;
  color: var(--input-text, #ddd);
  border: 1px solid var(--border-color, #555); border-radius: 4px;
  cursor: pointer; font-size: 12px;
  white-space: nowrap;
}
.am-vfx-tools-btn:hover { background: rgba(255,255,255,0.05); }
.am-vfx-tools-btn-icon {
  padding: 4px 8px; min-width: 28px;
}
.am-vfx-tools-btn-primary {
  background: var(--p-primary-color, #4a9eff); color: white;
  border-color: var(--p-primary-color, #4a9eff);
}
.am-vfx-tools-btn-primary:hover { filter: brightness(1.1); }
.am-vfx-tools-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.am-vfx-tools-recent-list {
  display: flex; flex-direction: column;
  border: 1px solid var(--border-color, #444); border-radius: 4px;
  max-height: 50vh; overflow-y: auto;
}
.am-vfx-tools-recent-item {
  display: grid; grid-template-columns: 24px 1fr auto; gap: 8px;
  padding: 6px 10px; cursor: pointer;
  border-bottom: 1px solid var(--border-color, #333);
  font-size: 12px;
}
.am-vfx-tools-recent-item:last-child { border-bottom: none; }
.am-vfx-tools-recent-item:hover { background: rgba(255,255,255,0.05); }
.am-vfx-tools-recent-action { opacity: 0.6; }
.am-vfx-tools-recent-path { font-family: ui-monospace, monospace; overflow-wrap: anywhere; }
.am-vfx-tools-recent-time { opacity: 0.5; white-space: nowrap; }
`;

export function injectStyles() {
  if (document.getElementById("am-vfx-tools-work-file-io-styles")) return;
  const style = document.createElement("style");
  style.id = "am-vfx-tools-work-file-io-styles";
  style.textContent = CSS;
  document.head.appendChild(style);
}
