// Thin fetch wrappers for the /am-vfx-tools/workfile-io/* backend.
const BASE = "/am-vfx-tools/workfile-io";

async function _request(url, init = {}) {
  const res = await fetch(url, init);
  if (!res.ok) {
    let msg;
    try { msg = (await res.text()) || res.statusText; }
    catch { msg = res.statusText; }
    const err = new Error(`${res.status} ${msg}`);
    err.status = res.status;
    err.body = msg;
    throw err;
  }
  if (res.status === 204) return null;
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res.text();
}

export async function getRoots() {
  return _request(`${BASE}/roots`);
}

export async function getNativeAvailable() {
  return _request(`${BASE}/native-dialog/available`);
}

export async function nativeOpen({ defaultDir, title } = {}) {
  return _request(`${BASE}/native-dialog/open`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ default_dir: defaultDir, title }),
  });
}

export async function nativeSave({ defaultDir, defaultFilename, title } = {}) {
  return _request(`${BASE}/native-dialog/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      default_dir: defaultDir,
      default_filename: defaultFilename,
      title,
    }),
  });
}

export async function listDir(dir) {
  return _request(`${BASE}/list?dir=${encodeURIComponent(dir)}`);
}

export async function mkdir(path) {
  return _request(`${BASE}/mkdir`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
}

export async function revealFolder(path) {
  return _request(`${BASE}/reveal`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
}

export async function loadWorkflow(path) {
  return _request(`${BASE}/load?path=${encodeURIComponent(path)}`);
}

export async function saveWorkflow(path, workflow, overwrite = false) {
  const res = await fetch(`${BASE}/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, workflow, overwrite }),
  });
  if (res.status === 409) {
    const body = await res.json().catch(() => ({}));
    const err = new Error(`file exists: ${body.path || path}`);
    err.status = 409;
    err.existing = body;
    throw err;
  }
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    const err = new Error(`${res.status} ${body}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export async function nextVersion(path) {
  const res = await fetch(`${BASE}/next-version?path=${encodeURIComponent(path)}`);
  if (res.status === 422) {
    const body = await res.json().catch(() => ({}));
    const err = new Error(body.message || "no version pattern in filename");
    err.status = 422;
    err.body = body;
    throw err;
  }
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    const err = new Error(`${res.status} ${body}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export async function getRecent() {
  return _request(`${BASE}/recent`);
}

export async function clearRecent() {
  return _request(`${BASE}/recent/clear`, { method: "POST" });
}

// Path portability — the internal pipeline pack tokenises absolute paths
// into ${PROJECT_ROOT}/<rel> form for cross-OS workflow round-trips. The
// public pack stores absolute paths verbatim; if you need cross-OS
// portability for a particular workflow, keep your sandbox roots at the
// same logical path on both OSes (e.g. `~/Documents`) so the absolute
// paths match.
