"""AM Seed — reproducible, render-farm-safe seed generator.

Solves three problems with the stock ComfyUI seed widget:

* **Render-farm reproducibility.** Stock ComfyUI computes the
  ``control_after_generate`` mutation in the **frontend JS** before
  the prompt JSON is submitted; on a headless server, the seed
  arrives baked-in and never re-randomizes between submissions
  (https://github.com/Comfy-Org/ComfyUI/issues/11905). AMSeed performs
  all mode logic server-side. Resolution happens in :meth:`IS_CHANGED`
  (see below) so the seed is freshly drawn on every prompt and
  available *before any node executes*. The resolved seed itself
  doubles as the cache-invalidation key — for randomize / increment
  / decrement it differs every call so ComfyUI's input-cache always
  treats the node as "changed"; for fixed it stays stable so the
  cache cleanly hits.

* **Global-variable wiring.** AMSeed is an ``OUTPUT_NODE=True`` node
  so it's always part of the execution plan. AM Image Write / AM
  Video Write read the seed from ``_core.seed_registry`` when their
  own seed parameter is the sentinel ``-1``, removing the need to
  fan-out a single seed across many writers via explicit wires.

  **Why publish from IS_CHANGED, not execute().** The two output
  nodes (AM Seed and Write) execute in a topological order that's
  not controllable from a custom-node perspective when they aren't
  wired — Write often runs *before* AM Seed. A previous design
  published in execute() and broke for that reason: Write either
  saw an empty registry (seed metadata omitted entirely) or a stale
  value left over from a previous prompt (file metadata mismatched
  the UI). IS_CHANGED is invoked for every node *before* any node
  executes (during ComfyUI's pre-execution cache-signature build —
  see ``execution.py`` ``set_prompt → add_keys → IsChangedCache.get``).
  Resolving + publishing there guarantees the registry is populated
  before Write's execute() runs, regardless of execute order.

* **API-node compatibility.** Default ``max`` is ``2_147_483_647``
  (int32 signed max). Many third-party API nodes (Replicate /
  RunPod / DALL-E wrappers) reject seeds above that cap; staying
  inside int32 keeps results portable across the ecosystem.

Modes:
    randomize   – uniform pick in ``[min, max]`` via SystemRandom
                  (default; freshly drawn each prompt execution).
    fixed       – emit ``value`` clamped to ``[min, max]``.
    increment   – ``value + step * n``, clamped at ``max``; ``n``
                  advances by one per execution while widget params
                  are unchanged.
    decrement   – ``value - step * n``, clamped at ``min``.

The increment/decrement counter is process-local and resets when any
of (value, step, min, max) change, or on server restart. In the
interactive case the frontend ticks ``value`` after each run via the
ws push, so the counter resets to 1 each call and the effective
sequence becomes ``value + step`` per run; in headless the counter
drives the same sequence directly. Either way the observable seed
chain is identical (e.g. 10 → 11 → 12 → 13).

To stash the resolved seed between IS_CHANGED and execute() we use
a process-global dict keyed by node_id. IS_CHANGED writes; execute()
reads. The stash is overwritten on every IS_CHANGED call, so it
can't go stale within a prompt — and the next prompt's IS_CHANGED
overwrites again before any execute() runs.
"""
from __future__ import annotations

import builtins
import logging
import random
import threading
from typing import Any, Dict, Optional, Tuple

from ._core import seed_registry

log = logging.getLogger("am_vfx_tools.media-io.seed")


# `PromptServer.instance.send_sync(event, data)` is how custom nodes push
# websocket events to the frontend. Imported defensively so the module
# still loads under the dcc-core test runner / headless render farm where
# the comfy server isn't on the import path. When unavailable we silently
# skip the push — the seed itself is already resolved + published to the
# registry, the file metadata still lands correctly.
try:  # pragma: no cover — import-time only
    from server import PromptServer as _PromptServer
except Exception:
    _PromptServer = None


MODE_RANDOMIZE = "randomize"
MODE_FIXED     = "fixed"
MODE_INCREMENT = "increment"
MODE_DECREMENT = "decrement"
_MODES = [MODE_RANDOMIZE, MODE_FIXED, MODE_INCREMENT, MODE_DECREMENT]

# int32 signed max — practical cap for downstream API node compatibility.
_DEFAULT_MAX = 2_147_483_647

_rng = random.SystemRandom()

# Per-node execution counter for increment / decrement modes. Key
# includes the widget params so any param edit resets the chain.
# Bounded so pathological workflows can't grow the dict unboundedly.
_counters: Dict[Tuple[Any, ...], int] = {}
_counters_lock = threading.Lock()
_COUNTERS_MAX = 1024


# Process-global stash: node_id → most-recently-resolved seed (during
# this process's IS_CHANGED). execute() reads from here so the value
# the UI receives via ws push matches the value Write embedded into
# the file metadata (both came from the same IS_CHANGED resolution
# inside the same prompt — Write reads the registry, execute reads
# the stash, both populated by the same IS_CHANGED call).
_resolved_stash: Dict[str, int] = {}
_stash_lock = threading.Lock()


def _advance_counter(key: Tuple[Any, ...]) -> int:
    with _counters_lock:
        n = _counters.get(key, 0) + 1
        _counters[key] = n
        if len(_counters) > _COUNTERS_MAX:
            for k in list(_counters)[: _COUNTERS_MAX // 2]:
                _counters.pop(k, None)
        return n


def _resolve_seed(
    *,
    mode: str,
    value: int,
    step: int,
    seed_min: int,
    seed_max: int,
    node_id: Any,
) -> int:
    if seed_min > seed_max:
        seed_min, seed_max = seed_max, seed_min  # tolerate user typo

    clamp = lambda x: builtins.max(seed_min, builtins.min(seed_max, int(x)))

    if mode == MODE_FIXED:
        return clamp(value)

    if mode == MODE_RANDOMIZE:
        return _rng.randint(int(seed_min), int(seed_max))

    if mode in (MODE_INCREMENT, MODE_DECREMENT):
        key = (node_id, int(value), int(step), int(seed_min), int(seed_max))
        n = _advance_counter(key)
        delta = int(step) * n
        raw = int(value) + delta if mode == MODE_INCREMENT else int(value) - delta
        return clamp(raw)

    log.warning("AMSeed: unknown mode %r — falling back to fixed", mode)
    return clamp(value)


def _resolve_publish_and_stash(
    *,
    mode: str,
    value: int,
    step: int,
    seed_min: int,
    seed_max: int,
    unique_id: Any,
) -> int:
    """Eagerly resolve the seed AND populate both the stash + registry.

    Called from :meth:`AMSeed.IS_CHANGED` so the registry is populated
    before any node's execute() runs in the current prompt. execute()
    later reads from the stash to make sure the value it pushes to the
    UI matches the value Write embedded into file metadata.
    """
    seed = _resolve_seed(
        mode=mode,
        value=value,
        step=step,
        seed_min=seed_min,
        seed_max=seed_max,
        node_id=unique_id,
    )
    nid = str(unique_id or "")
    if nid:
        with _stash_lock:
            _resolved_stash[nid] = int(seed)
        try:
            seed_registry.publish(node_id=nid, seed=int(seed))
        except Exception:  # pragma: no cover — best-effort
            log.exception("AMSeed: failed to publish to registry")
    return int(seed)


def _read_stash(unique_id: Any) -> Optional[int]:
    nid = str(unique_id or "")
    if not nid:
        return None
    with _stash_lock:
        return _resolved_stash.get(nid)


class AMSeed:
    """ComfyUI node — reproducible, render-farm-safe seed generator."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (_MODES, {
                    "default": MODE_RANDOMIZE,
                    "tooltip": (
                        "Seed selection mode. "
                        "randomize = fresh draw from [min,max] each prompt (default). "
                        "fixed = emit `value` clamped to [min,max]. "
                        "increment / decrement = `value ± step·n` per execution, "
                        "clamped at the bounds. Mode logic runs server-side so it "
                        "behaves identically on headless render farms."
                    ),
                }),
                # Named ``value`` (not ``seed``) on purpose: ComfyUI's
                # frontend auto-appends a ``control_after_generate``
                # widget to any INT named ``seed`` / ``noise_seed``, and
                # that widget's frontend mutation logic would compete
                # with the server-side mode logic implemented here.
                # ``value`` is the artist-set base for fixed / increment
                # / decrement modes — mode-specific applicability is
                # documented per widget; we deliberately surface every
                # widget at all times rather than hiding "non-applicable"
                # ones, because the previous show/hide JS hack didn't
                # mesh cleanly with ComfyUI's widget-lifecycle.
                "value": ("INT", {
                    "default": 0,
                    "min":     0,
                    "max":     _DEFAULT_MAX,
                    "tooltip": (
                        "fixed: emitted directly (clamped to [min,max]).\n"
                        "increment / decrement: starting base; advances by `step` per execution.\n"
                        "randomize: ignored — a fresh value is drawn from [min,max]."
                    ),
                }),
                "step":  ("INT", {
                    "default": 1,
                    "min":     1,
                    "max":     1_000_000,
                    "tooltip": "increment / decrement step. Ignored in fixed / randomize modes.",
                }),
                "min":   ("INT", {
                    "default": 0,
                    "min":     0,
                    "max":     _DEFAULT_MAX,
                    "tooltip": "Lower bound used by randomize and as a clamp for the others.",
                }),
                "max":   ("INT", {
                    "default": _DEFAULT_MAX,
                    "min":     0,
                    "max":     _DEFAULT_MAX,
                    "tooltip": (
                        "Upper bound used by randomize and as a clamp for the others. "
                        "Default 2,147,483,647 (int32 max) for compatibility with API "
                        "nodes that reject larger seeds."
                    ),
                }),
            },
            "optional": {},
            "hidden": {
                # `unique_id` keys both the stash + the registry. We do
                # NOT request `prompt`: it'd be empty (`{}`) inside
                # IS_CHANGED anyway because ComfyUI's IsChangedCache
                # invokes `get_input_data` with `dynprompt=None`. The
                # write-side path uses its own `prompt` hidden input to
                # find the AMSeed node-ids in the active prompt and
                # look them up in the registry.
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    OUTPUT_TOOLTIPS = (
        "Resolved seed value. Wire into samplers / API nodes that take a seed input. "
        "AM Image Write / AM Video Write also pick up this seed automatically via "
        "the process-global registry (their `seed` widget at -1 = look up).",
    )
    FUNCTION     = "execute"
    CATEGORY     = "AM VFX Tools/Util"
    OUTPUT_NODE  = True  # always execute, even when the seed isn't wired

    @classmethod
    def IS_CHANGED(cls, mode, value, step, min, max, unique_id=None, **_kw):
        # Resolve the seed eagerly + publish to the registry + stash for
        # execute() to read. This is the load-bearing step: it makes
        # sure the registry is populated BEFORE any node's execute()
        # runs in the current prompt (IsChangedCache.get is called for
        # every node during the pre-execution cache-signature build).
        #
        # The resolved seed itself doubles as the cache-invalidation
        # key:
        #   * randomize / increment / decrement — a fresh value every
        #     call, so ComfyUI's cache sees "different from last time"
        #     and forces a re-execute (the ws push fires + UI ticks).
        #   * fixed — same value every call (when widgets unchanged),
        #     so ComfyUI's cache hits and execute() is skipped. That's
        #     fine: the registry was populated *here* in IS_CHANGED, so
        #     Write still finds the value. The UI doesn't tick because
        #     fixed mode has no value to tick to.
        seed = _resolve_publish_and_stash(
            mode=mode,
            value=value,
            step=step,
            seed_min=min,
            seed_max=max,
            unique_id=unique_id,
        )
        return int(seed)

    def execute(
        self,
        mode: str,
        value: int,
        step: int,
        min: int,
        max: int,
        unique_id: Optional[Any] = None,
    ):
        # Read from the stash (populated by IS_CHANGED in the same
        # prompt). Falls back to a fresh resolve only if the stash is
        # missing — that path shouldn't fire in practice because
        # ComfyUI always calls IS_CHANGED for every node before any
        # execute(), but the fallback keeps the node usable in any
        # adversarial harness that bypasses IS_CHANGED.
        stashed = _read_stash(unique_id)
        if stashed is not None:
            seed = stashed
        else:
            log.warning(
                "AMSeed: no stashed seed for node_id=%r — resolving fresh "
                "in execute(). This shouldn't happen under normal ComfyUI "
                "execution; the registry/stash should have been populated "
                "by IS_CHANGED.", unique_id,
            )
            seed = _resolve_seed(
                mode=mode,
                value=value,
                step=step,
                seed_min=min,
                seed_max=max,
                node_id=unique_id,
            )
            try:
                seed_registry.publish(node_id=str(unique_id or ""), seed=int(seed))
            except Exception:  # pragma: no cover
                log.exception("AMSeed: registry publish in execute() failed")

        log.info(
            "AMSeed: mode=%s resolved=%s (value=%s step=%s min=%s max=%s)",
            mode, seed, value, step, min, max,
        )

        # Push the resolved seed to the frontend so the `value` widget
        # ticks visually after each execution — matches the behavior of
        # stock ComfyUI's `control_after_generate` but with the mutation
        # happening server-side AFTER resolution (preserving
        # render-farm reproducibility — a headless worker without a
        # connected client just skips the push). The frontend extension
        # subscribes to `am_pipe.am_seed.resolved` and writes the value
        # into the matching node's `value` widget.
        if _PromptServer is not None:
            try:
                _PromptServer.instance.send_sync(
                    "am_pipe.am_seed.resolved",
                    {
                        "node_id": str(unique_id or ""),
                        "seed":    int(seed),
                        "mode":    str(mode),
                    },
                )
            except Exception:  # pragma: no cover — push is best-effort
                log.debug("AMSeed: send_sync push failed (no client?)",
                          exc_info=True)

        # `ui.text` echoes the resolved seed onto the node's output
        # panel — secondary feedback in case the websocket push failed
        # (e.g. disconnected client). The `result` tuple is unchanged.
        return {
            "ui": {"text": [f"resolved seed: {int(seed)} ({mode})"]},
            "result": (int(seed),),
        }
