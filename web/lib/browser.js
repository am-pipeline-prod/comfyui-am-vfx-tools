// In-browser file browser dialog — fallback when native OS dialog isn't
// available. Designed to feel close to a desktop Save/Open dialog: clickable
// breadcrumbs, sortable columns, keyboard navigation, type-ahead jumping,
// New Folder support.
//
// Public API:
//   pickViaBrowser({ mode: "open"|"save", title, defaultDir, defaultFilename })
//     → Promise<{ path, overwrite } | null>
//
// All path manipulation is OS-aware (separator) so it works from Windows or
// Linux frontends pointed at either a Windows or a Linux server.

import { state } from "./state.js";
import { listDir, mkdir } from "./api.js";

const sep = () => (state.os === "windows" ? "\\" : "/");

function joinPath(dir, name) {
  if (!dir) return name;
  const s = sep();
  return dir.endsWith(s) ? dir + name : dir + s + name;
}

function dirname(p) {
  const idx = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  if (idx < 0) return p;
  if (idx === 0) return p.slice(0, 1);            // "/foo" → "/"
  if (p[idx - 1] === ":") return p.slice(0, idx + 1); // "Z:\foo" → "Z:\"
  return p.slice(0, idx);
}

function basename(p) { return p.split(/[\\/]/).pop(); }

function findRootFor(p) {
  // Find the longest configured root that is a prefix of p.
  return state.roots
    .filter((r) => p === r || p.startsWith(r.endsWith(sep()) ? r : r + sep()))
    .sort((a, b) => b.length - a.length)[0] || null;
}

function splitSegments(p) {
  // Anchor breadcrumbs at the configured root that contains p, so the user
  // never gets a clickable segment that would navigate outside the sandbox.
  // Falls back to filesystem-root anchoring if no root matches.
  if (!p) return [];
  const s = sep();
  const root = findRootFor(p);

  let prefixPath, prefixLabel, rest;
  if (root) {
    prefixPath = root;
    prefixLabel = root;
    rest = p.slice(root.length).replace(/^[\\/]+/, "");
  } else {
    // Fallback: filesystem root
    if (state.os === "windows" && /^[A-Z]:[\\/]?/i.test(p)) {
      prefixPath = p.slice(0, 2) + s;
      prefixLabel = p.slice(0, 2);
    } else {
      prefixPath = "/";
      prefixLabel = "/";
    }
    rest = p.slice(prefixPath.length).replace(/^[\\/]+/, "");
  }

  const out = [{ label: prefixLabel, fullPath: prefixPath }];
  if (rest) {
    let acc = prefixPath;
    for (const seg of rest.split(/[\\/]/)) {
      if (!seg) continue;
      acc = acc.endsWith(s) ? acc + seg : acc + s + seg;
      out.push({ label: seg, fullPath: acc });
    }
  }
  return out;
}

function fmtSize(bytes) {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}

function fmtTime(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  const now = Date.now();
  const ageDays = (now - d.getTime()) / 86400000;
  if (ageDays < 1) {
    return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  }
  if (ageDays < 365) {
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

// ---------------------------------------------------------------------------
// Main dialog
// ---------------------------------------------------------------------------

export async function pickViaBrowser({ mode, title, defaultDir, defaultFilename }) {
  const isSave = mode === "save";

  // ---- root container -------------------------------------------------
  const overlay = document.createElement("div");
  overlay.className = "am-vfx-tools-overlay";
  const modal = document.createElement("div");
  modal.className = "am-vfx-tools-modal am-vfx-tools-modal-browser";
  overlay.appendChild(modal);

  // header
  const header = document.createElement("div");
  header.className = "am-vfx-tools-header";
  header.textContent = title;
  modal.appendChild(header);

  // toolbar
  const toolbar = document.createElement("div");
  toolbar.className = "am-vfx-tools-toolbar";
  modal.appendChild(toolbar);

  // body — table-like list
  const listWrap = document.createElement("div");
  listWrap.className = "am-vfx-tools-list-wrap";
  modal.appendChild(listWrap);

  const colHeader = document.createElement("div");
  colHeader.className = "am-vfx-tools-list-header";
  listWrap.appendChild(colHeader);

  const list = document.createElement("div");
  list.className = "am-vfx-tools-list";
  list.tabIndex = 0;
  listWrap.appendChild(list);

  // footer — filename input + actions
  const footerForm = document.createElement("div");
  footerForm.className = "am-vfx-tools-footer-form";
  modal.appendChild(footerForm);

  const footer = document.createElement("div");
  footer.className = "am-vfx-tools-footer";
  modal.appendChild(footer);

  document.body.appendChild(overlay);

  // ---- state ----------------------------------------------------------
  const initialDir = (() => {
    if (defaultDir) return defaultDir;
    if (state.lastDir) return state.lastDir;
    if (state.roots[0]) return state.roots[0];
    return state.os === "windows" ? "C:\\" : "/";
  })();

  let currentDir = initialDir;
  let entries = [];
  let selectedIdx = -1;
  let sortBy = "name";
  let sortDir = "asc";
  let pathEditing = false;

  // type-ahead buffer
  let typeAheadBuf = "";
  let typeAheadTimer = null;

  // resolution promise
  let resolveDialog;
  const closed = new Promise((r) => { resolveDialog = r; });

  function close(value) {
    overlay.remove();
    document.removeEventListener("keydown", onGlobalKey);
    resolveDialog(value);
  }

  function onGlobalKey(e) {
    if (e.key === "Escape") { e.preventDefault(); close(null); }
  }
  document.addEventListener("keydown", onGlobalKey);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(null); });

  // ---- toolbar --------------------------------------------------------
  const rootsSelect = document.createElement("select");
  rootsSelect.className = "am-vfx-tools-roots-select";
  rootsSelect.title = "Jump to root";
  for (const r of state.roots) {
    const opt = document.createElement("option");
    opt.value = r;
    opt.textContent = r;
    rootsSelect.appendChild(opt);
  }
  rootsSelect.value = state.roots.find((r) => currentDir.startsWith(r)) || state.roots[0] || "";
  rootsSelect.addEventListener("change", () => navigate(rootsSelect.value));
  toolbar.appendChild(rootsSelect);

  const upBtn = document.createElement("button");
  upBtn.className = "am-vfx-tools-btn am-vfx-tools-btn-icon";
  upBtn.textContent = "↑";
  upBtn.title = "Parent folder (Backspace)";
  upBtn.addEventListener("click", goUp);
  toolbar.appendChild(upBtn);

  const breadcrumb = document.createElement("div");
  breadcrumb.className = "am-vfx-tools-breadcrumb";
  breadcrumb.title = "Click to navigate · double-click to edit as text";
  breadcrumb.addEventListener("dblclick", () => togglePathEdit(true));
  toolbar.appendChild(breadcrumb);

  const editPathBtn = document.createElement("button");
  editPathBtn.className = "am-vfx-tools-btn am-vfx-tools-btn-icon";
  editPathBtn.textContent = "✎";
  editPathBtn.title = "Edit path as text";
  editPathBtn.addEventListener("click", () => togglePathEdit(!pathEditing));
  toolbar.appendChild(editPathBtn);

  if (isSave) {
    const newFolderBtn = document.createElement("button");
    newFolderBtn.className = "am-vfx-tools-btn";
    newFolderBtn.textContent = "+ New Folder";
    newFolderBtn.addEventListener("click", promptNewFolder);
    toolbar.appendChild(newFolderBtn);
  }

  // ---- column header --------------------------------------------------
  for (const [key, label, cls] of [
    ["name", "Name", "am-vfx-tools-col-name"],
    ["mtime", "Modified", "am-vfx-tools-col-mtime"],
    ["size", "Size", "am-vfx-tools-col-size"],
  ]) {
    const c = document.createElement("div");
    c.className = `am-vfx-tools-col-header ${cls}`;
    c.textContent = label;
    c.addEventListener("click", () => {
      if (sortBy === key) sortDir = sortDir === "asc" ? "desc" : "asc";
      else { sortBy = key; sortDir = "asc"; }
      renderList();
    });
    colHeader.appendChild(c);
  }

  // ---- footer form ----------------------------------------------------
  let filenameInput, overwriteBox;
  if (isSave) {
    const row = document.createElement("div");
    row.className = "am-vfx-tools-form-row";
    const lbl = document.createElement("label");
    lbl.textContent = "Filename:";
    lbl.className = "am-vfx-tools-form-label";
    filenameInput = document.createElement("input");
    filenameInput.className = "am-vfx-tools-input am-vfx-tools-filename-input";
    filenameInput.value = defaultFilename || "workflow.json";
    filenameInput.spellcheck = false;
    filenameInput.autocomplete = "off";
    filenameInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); doConfirm(); }
    });
    row.appendChild(lbl);
    row.appendChild(filenameInput);
    footerForm.appendChild(row);

    const ovRow = document.createElement("label");
    ovRow.className = "am-vfx-tools-checkbox-row";
    overwriteBox = document.createElement("input");
    overwriteBox.type = "checkbox";
    ovRow.appendChild(overwriteBox);
    ovRow.appendChild(document.createTextNode(" Overwrite if exists"));
    footerForm.appendChild(ovRow);
  }

  const status = document.createElement("div");
  status.className = "am-vfx-tools-status";
  footerForm.appendChild(status);

  // ---- footer buttons -------------------------------------------------
  const cancelBtn = document.createElement("button");
  cancelBtn.className = "am-vfx-tools-btn";
  cancelBtn.textContent = "Cancel";
  cancelBtn.addEventListener("click", () => close(null));

  const confirmBtn = document.createElement("button");
  confirmBtn.className = "am-vfx-tools-btn am-vfx-tools-btn-primary";
  confirmBtn.textContent = isSave ? "Save" : "Open";
  confirmBtn.addEventListener("click", doConfirm);

  footer.appendChild(cancelBtn);
  footer.appendChild(confirmBtn);

  // ---- behavior -------------------------------------------------------
  function setStatus(msg) { status.textContent = msg; }

  async function navigate(toDir) {
    setStatus("");
    let res;
    try {
      res = await listDir(toDir);
    } catch (err) {
      setStatus(`Could not open ${toDir}: ${err.message}`);
      return;
    }
    currentDir = res.dir;
    entries = res.entries;
    selectedIdx = -1;
    rootsSelect.value = state.roots.find((r) => currentDir.startsWith(r)) || rootsSelect.value;
    renderBreadcrumb();
    renderList();
    list.focus();
  }

  function goUp() {
    // Don't navigate above the root that anchors the current breadcrumbs.
    const root = findRootFor(currentDir);
    if (root && currentDir === root) return;
    const parent = dirname(currentDir);
    if (parent && parent !== currentDir) navigate(parent);
  }

  function renderBreadcrumb() {
    breadcrumb.innerHTML = "";
    if (pathEditing) {
      const input = document.createElement("input");
      input.className = "am-vfx-tools-input am-vfx-tools-breadcrumb-input";
      input.value = currentDir;
      input.spellcheck = false;
      input.autocomplete = "off";
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          pathEditing = false;
          navigate(input.value);
        } else if (e.key === "Escape") {
          e.preventDefault();
          pathEditing = false;
          renderBreadcrumb();
        }
      });
      input.addEventListener("blur", () => {
        pathEditing = false;
        renderBreadcrumb();
      });
      breadcrumb.appendChild(input);
      setTimeout(() => { input.focus(); input.select(); }, 0);
      return;
    }
    const segs = splitSegments(currentDir);
    segs.forEach((s, i) => {
      if (i > 0) {
        const sepEl = document.createElement("span");
        sepEl.className = "am-vfx-tools-breadcrumb-sep";
        sepEl.textContent = sep();
        breadcrumb.appendChild(sepEl);
      }
      const segEl = document.createElement("span");
      segEl.className = "am-vfx-tools-breadcrumb-seg";
      segEl.textContent = s.label;
      segEl.addEventListener("click", () => navigate(s.fullPath));
      breadcrumb.appendChild(segEl);
    });
  }

  function togglePathEdit(on) {
    pathEditing = !!on;
    renderBreadcrumb();
  }

  function sortedEntries() {
    const arr = [...entries];
    arr.sort((a, b) => {
      // Always group dirs first regardless of sort
      if (a.type !== b.type) return a.type === "dir" ? -1 : 1;
      let av, bv;
      if (sortBy === "name") { av = a.name.toLowerCase(); bv = b.name.toLowerCase(); }
      else if (sortBy === "mtime") { av = a.mtime || 0; bv = b.mtime || 0; }
      else if (sortBy === "size") { av = a.size || 0; bv = b.size || 0; }
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return sortDir === "asc" ? cmp : -cmp;
    });
    return arr;
  }

  function renderList() {
    list.innerHTML = "";
    const arr = sortedEntries();
    arr.forEach((e, i) => {
      const row = document.createElement("div");
      row.className = "am-vfx-tools-list-row";
      if (i === selectedIdx) row.classList.add("am-vfx-tools-list-row-selected");
      const icon = document.createElement("span");
      icon.className = "am-vfx-tools-list-icon";
      icon.textContent = e.type === "dir" ? "📁" : "📄";
      const name = document.createElement("span");
      name.className = "am-vfx-tools-list-name";
      name.textContent = e.name;
      const time = document.createElement("span");
      time.className = "am-vfx-tools-list-mtime";
      time.textContent = fmtTime(e.mtime);
      const size = document.createElement("span");
      size.className = "am-vfx-tools-list-size";
      size.textContent = e.type === "dir" ? "—" : fmtSize(e.size);
      row.appendChild(icon);
      row.appendChild(name);
      row.appendChild(time);
      row.appendChild(size);
      row.addEventListener("click", () => {
        selectedIdx = i;
        renderList();
        if (isSave && e.type === "file") {
          filenameInput.value = e.name;
        }
      });
      row.addEventListener("dblclick", () => {
        if (e.type === "dir") navigate(joinPath(currentDir, e.name));
        else if (isSave) { filenameInput.value = e.name; doConfirm(); }
        else doConfirmWithEntry(e);
      });
      list.appendChild(row);
    });
    if (selectedIdx >= 0 && list.children[selectedIdx]) {
      list.children[selectedIdx].scrollIntoView({ block: "nearest" });
    }
    list._sortedEntries = arr; // stash for keyboard handlers
  }

  function moveSelection(delta) {
    const arr = list._sortedEntries || [];
    if (!arr.length) return;
    if (selectedIdx < 0) selectedIdx = delta > 0 ? 0 : arr.length - 1;
    else selectedIdx = Math.max(0, Math.min(arr.length - 1, selectedIdx + delta));
    renderList();
  }

  function activateSelection() {
    const arr = list._sortedEntries || [];
    const e = arr[selectedIdx];
    if (!e) return;
    if (e.type === "dir") navigate(joinPath(currentDir, e.name));
    else if (isSave) { filenameInput.value = e.name; }
    else doConfirmWithEntry(e);
  }

  function typeAhead(ch) {
    typeAheadBuf += ch.toLowerCase();
    clearTimeout(typeAheadTimer);
    typeAheadTimer = setTimeout(() => { typeAheadBuf = ""; }, 800);
    const arr = list._sortedEntries || [];
    const idx = arr.findIndex((e) => e.name.toLowerCase().startsWith(typeAheadBuf));
    if (idx >= 0) { selectedIdx = idx; renderList(); }
  }

  list.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); moveSelection(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); moveSelection(-1); }
    else if (e.key === "Enter") { e.preventDefault(); activateSelection(); }
    else if (e.key === "Backspace") { e.preventDefault(); goUp(); }
    else if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      typeAhead(e.key);
    }
  });

  async function promptNewFolder() {
    const name = prompt("New folder name:");
    if (!name) return;
    try {
      await mkdir(joinPath(currentDir, name));
      await navigate(joinPath(currentDir, name));
    } catch (err) {
      setStatus(`mkdir failed: ${err.message}`);
    }
  }

  async function doConfirm() {
    if (isSave) {
      const fn = (filenameInput.value || "").trim();
      if (!fn) { setStatus("Filename is required."); return; }
      if (!fn.toLowerCase().endsWith(".json")) {
        setStatus("Filename must end in .json");
        return;
      }
      const target = joinPath(currentDir, fn);
      close({ path: target, overwrite: !!overwriteBox.checked });
    } else {
      const arr = list._sortedEntries || [];
      const e = arr[selectedIdx];
      if (!e) { setStatus("Select a file."); return; }
      doConfirmWithEntry(e);
    }
  }

  function doConfirmWithEntry(e) {
    if (e.type !== "file") { setStatus("Select a file."); return; }
    close({ path: joinPath(currentDir, e.name), overwrite: false });
  }

  // ---- initial load --------------------------------------------------
  navigate(currentDir);

  return closed;
}
