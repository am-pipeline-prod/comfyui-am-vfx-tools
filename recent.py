"""Recent-files JSON store (newest first, capped at config.RECENT_MAX)."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from . import config, sandbox

log = logging.getLogger("am_vfx_tools.workfile-io")


def _atomic_write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}.{int(time.time() * 1000)}")
    tmp.write_bytes(data)
    os.replace(tmp, p)


def _load_raw() -> list[dict]:
    p = config.RECENT_STORE
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("[am-vfx-tools/workfile-io] recent store unreadable, treating as empty: %s", e)
        return []
    if not isinstance(data, list):
        log.warning("[am-vfx-tools/workfile-io] recent store is not a list, resetting")
        return []
    return [e for e in data if isinstance(e, dict) and isinstance(e.get("path"), str)]


def _save_raw(entries: list[dict]) -> None:
    try:
        body = json.dumps(entries, indent=2).encode("utf-8")
        _atomic_write(config.RECENT_STORE, body)
    except OSError as e:
        log.warning("[am-vfx-tools/workfile-io] could not write recent store: %s", e)


def add(path: str, action: str) -> None:
    """Add an entry; dedup by path; keep newest ``RECENT_MAX``."""
    entries = _load_raw()
    entries = [e for e in entries if e.get("path") != path]
    entries.insert(0, {"path": path, "ts": time.time(), "action": action})
    if len(entries) > config.RECENT_MAX:
        entries = entries[: config.RECENT_MAX]
    _save_raw(entries)


def read() -> list[dict]:
    """Return current entries, dropping any whose path no longer exists or
    no longer validates against the sandbox (lazy cleanup, not persisted)."""
    out = []
    for e in _load_raw():
        p = e.get("path")
        if not isinstance(p, str):
            continue
        try:
            sandbox.validate(p, must_be_json=True)
        except Exception:
            continue
        if not Path(p).is_file():
            continue
        out.append(e)
    return out


def clear() -> None:
    _save_raw([])
