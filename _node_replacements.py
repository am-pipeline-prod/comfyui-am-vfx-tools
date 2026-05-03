"""Node replacement registrations — auto-migrate old saved workflows
when this pack's nodes evolve in shape-breaking ways.

See [`docs/custom-node-widget-evolution.md`](../../docs/custom-node-widget-evolution.md)
for the full policy. Boiled down:

* Adding a widget to an existing node → just append it to INPUT_TYPES.
  No version bump, no migration, no entry in this file.
* Renaming, removing, reordering, or type-changing a widget → bump the
  node's class to `_v<N+1>` and add ONE NodeReplace entry below mapping
  the old node_id to the new one.

Imports are guarded so older ComfyUI without `comfy_api` still loads
the pack normally — saved workflows just won't auto-migrate, the artist
has to recreate the node manually. ComfyUI ≥ 0.3.48 (Aug 2025) ships
`comfy_api.v0_0_2` in core. The pack's `pyproject.toml` declares the
pin; this defensive try/except is the runtime belt to that suspenders.

Pin `comfy_api.v0_0_2` rather than `comfy_api.latest` — the docs
explicitly warn `latest` "will be changed without warning until
`v0_0_3` is cut." Pinning insulates us; expect one or two cheap rename
sweeps when v0_0_3 lands.
"""
from __future__ import annotations

import logging

log = logging.getLogger("am_pipe.media-io.replacements")

try:
    from comfy_api.v0_0_2 import ComfyExtension, io  # type: ignore[import-not-found]
    _V3_API_AVAILABLE = True
except ImportError:
    _V3_API_AVAILABLE = False
    ComfyExtension = object  # type: ignore[misc,assignment]
    io = None  # type: ignore[assignment]


class AMPipeMediaIOExtension(ComfyExtension):  # type: ignore[misc]
    """ComfyExtension hook — registers all node-replacement migrations
    for the `am-pipe-media-io` pack.

    Add ONE `io.NodeReplace(...)` entry per breaking change. Each entry
    is independent — chained migrations (v1→v2→v3 on the same node)
    each get their own entry. ComfyUI walks them in registration order
    when migrating an old saved workflow.
    """

    async def get_node_list(self):
        """V3-schema node contributions from this extension.

        We use the V1-schema (the traditional `NODE_CLASS_MAPPINGS`
        dict path) for our actual nodes — they're registered in
        `__init__.py`. This extension class exists only to host the
        `node_replacement` registrations, so the V3 node list is empty.

        Required by `ComfyExtension`'s abstract base class — without
        this stub the class can't instantiate and `on_load()` never
        runs (silent asyncio TypeError in the boot logs).
        """
        return []

    async def on_load(self):
        if not _V3_API_AVAILABLE:
            return

        # ------------------------------------------------------------------
        # Migrations registered to date: NONE.
        # ------------------------------------------------------------------
        #
        # When you make the first breaking change to a node in this pack,
        # add the registration here. Template:
        #
        #   await self.api.node_replacement.register(
        #       io.NodeReplace(
        #           old_node_id="AMImageWrite",
        #           new_node_id="AMImageWriteV2",
        #           old_widget_ids=[
        #               # Full positional list of widget names from the OLD
        #               # node, in their saved-workflow order. Get this by
        #               # reading `INPUT_TYPES["required"]` keys top-to-bottom
        #               # then `INPUT_TYPES["optional"]` keys.
        #               "mode", "output_class", "custom", "file_path",
        #               # ... etc
        #           ],
        #           input_mapping=[
        #               # Carry old value forward, same name:
        #               {"new_id": "mode",        "old_id": "mode"},
        #               # Rename:
        #               {"new_id": "destination", "old_id": "output_class"},
        #               # Drop: simply omit the entry.
        #               # New widget added in v2 with constant default:
        #               {"new_id": "new_widget",  "set_value": "default"},
        #           ],
        #           # output_mapping only needed if RETURN_TYPES order changed.
        #           # Default: output sockets carry over by index.
        #       )
        #   )
        #
        # See [`docs/custom-node-widget-evolution.md`](../../docs/custom-node-widget-evolution.md)
        # §"How to do a _v<N+1> bump" for the full recipe and the
        # `input_mapping` field semantics table.

        log.debug("am-pipe-media-io: no node replacements registered (clean state)")


async def register_replacements():
    """Entry point called from the pack's `__init__.py` at module-load
    time. Wraps `AMPipeMediaIOExtension.on_load()` with the V3-API
    availability guard so older ComfyUI doesn't break.
    """
    if not _V3_API_AVAILABLE:
        log.info(
            "am-pipe-media-io: comfy_api.v0_0_2 not available "
            "(ComfyUI < 0.3.48?) — saved-workflow auto-migration disabled. "
            "Nodes themselves load normally."
        )
        return
    await AMPipeMediaIOExtension().on_load()
