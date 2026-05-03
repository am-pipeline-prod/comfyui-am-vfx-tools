"""AM Pipe — process-global seed registry.

The AM Seed node publishes its resolved seed here at **IS_CHANGED**
time (which ComfyUI calls for every node before any node executes —
see ``execution.py`` ``set_prompt → add_keys → is_changed_cache.get``
chain). AM Image Write / AM Video Write read it back at execute()
time — when their own ``seed`` parameter is the sentinel ``-1`` —
via :func:`find_seed_for_prompt`, which scans the active prompt dict
for ``AMSeed`` class entries and resolves their published values.

Why "publish from IS_CHANGED" rather than "publish from execute()":
    * Output nodes (AM Seed and Write nodes are both
      ``OUTPUT_NODE=True``) execute in a topological order that's not
      controllable from a custom-node author's perspective when the
      two aren't wired. If publish happens in ``execute()`` and Write
      runs before AM Seed, Write's lookup misses or — worse — finds
      the previous prompt's stale value (because ``id(prompt)``
      keying isn't unique across prompts in CPython: memory
      addresses get reused). The classic symptom is "file metadata
      shows a different seed than the UI / console log".
    * IS_CHANGED is invoked for every node during the pre-execution
      cache-signature build (``IsChangedCache.get`` from
      ``set_prompt``). By the time any node's ``execute()`` runs, AM
      Seed's IS_CHANGED has already published. Write's lookup is
      therefore guaranteed-consistent regardless of execute() order.

Why we key by ``node_id`` (the AM Seed's UNIQUE_ID) rather than by
prompt object identity:
    * ``PROMPT`` hidden input is empty (``{}``) inside IS_CHANGED —
      ComfyUI calls ``get_input_data`` with ``dynprompt=None`` from
      the cache layer, so the PROMPT slot is not populated.
    * ``node_id`` is stable inside IS_CHANGED (UNIQUE_ID is wired in
      whether dynprompt is None or not).
    * Write-side disambiguation: Write's ``execute()`` *does* receive
      the full PROMPT dict, so it can scan ``prompt`` for ``AMSeed``
      class entries and resolve the matching node_ids — see
      :func:`find_seed_for_prompt`.

Cross-prompt staleness: if prompt N publishes seed_N under
``node_id=42`` and prompt N+1 also has an AM Seed with
``node_id=42``, IS_CHANGED in N+1 overwrites with seed_(N+1) BEFORE
any execute() runs. So lookups inside prompt N+1 always see
seed_(N+1). If prompt N+1 has NO AM Seed,
:func:`find_seed_for_prompt` finds no AMSeed entries in the prompt
dict and returns ``None`` — the stale ``42 → seed_N`` entry is
ignored, and the write node correctly omits the seed metadata.

State is process-local. ComfyUI worker processes don't share state;
each worker handles one prompt at a time. Bounded by LRU eviction
on every publish.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Mapping, Optional

_MAX_ENTRIES = 64

_lock = threading.Lock()
_registry: "OrderedDict[str, int]" = OrderedDict()


def publish(node_id: Any, seed: int) -> None:
    """Record *seed* under *node_id* (the AM Seed's UNIQUE_ID).

    Idempotent — calling twice with the same args is a no-op for
    consumers. Calling with a different seed for the same node_id
    overwrites (the typical pattern when AM Seed runs again with a
    fresh randomize / increment / decrement value).
    """
    if node_id is None:
        return
    nid = str(node_id)
    with _lock:
        _registry[nid] = int(seed)
        _registry.move_to_end(nid)
        while len(_registry) > _MAX_ENTRIES:
            _registry.popitem(last=False)


def lookup(node_id: Any) -> Optional[int]:
    """Return the seed published under *node_id*, or ``None`` if absent."""
    if node_id is None:
        return None
    with _lock:
        return _registry.get(str(node_id))


def find_seed_for_prompt(prompt: Mapping[str, Any]) -> Optional[int]:
    """Find the seed published by an AM Seed node in *prompt*.

    Scans *prompt* for entries whose ``class_type`` is ``AMSeed`` and
    looks each up in the registry. Returns the seed of the first
    match in **node-id-sorted order** (lowest id first → typically
    the AM Seed the artist added first; deterministic tiebreaker for
    multi-AMSeed workflows, where the user can override by wiring
    the desired AM Seed directly into the write node).

    Returns ``None`` when:
      * *prompt* isn't a usable dict (defensive — IS_CHANGED is the
        publish point, not execute(), so the registry should be
        populated before this is called),
      * the prompt contains no ``AMSeed`` nodes,
      * the AM Seed nodes in the prompt haven't been published to
        the registry (shouldn't happen in practice — every AM Seed
        publishes during its own IS_CHANGED).

    Skipping unpublished AMSeed nodes (rather than returning a stale
    value from a different node_id) is intentional: the registry is
    a shared global, and a Write node should only consume a seed
    that demonstrably came from THIS prompt.
    """
    if not isinstance(prompt, Mapping):
        return None
    am_seed_ids = []
    for nid, node_def in prompt.items():
        if not isinstance(node_def, Mapping):
            continue
        if node_def.get("class_type") == "AMSeed":
            am_seed_ids.append(str(nid))
    if not am_seed_ids:
        return None

    # Sort numerically when the ids are numeric (the common case in
    # ComfyUI workflows), else fall back to string sort. Either way
    # deterministic so multiple Write nodes in the same prompt agree
    # on which AMSeed they pick.
    def _sort_key(s: str):
        try:
            return (0, int(s))
        except (TypeError, ValueError):
            return (1, s)

    am_seed_ids.sort(key=_sort_key)
    for nid in am_seed_ids:
        v = lookup(nid)
        if v is not None:
            return v
    return None


def evict(node_id: Any) -> None:
    """Drop the entry for *node_id* (mostly useful for tests)."""
    if node_id is None:
        return
    with _lock:
        _registry.pop(str(node_id), None)


def clear() -> None:
    """Drop everything (test helper)."""
    with _lock:
        _registry.clear()
