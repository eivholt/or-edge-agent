"""Reconcile a detector event against a synthetic surgical pathway.

Pure logic — no LLM, no network.  Compares what the detector sees
(visible counts) with what the pathway says is required (item → count)
and returns a list of deficit dicts.
"""

from __future__ import annotations

import logfire


@logfire.instrument("reconcile case_id={case[case_id]}")
def reconcile(event: dict, case: dict) -> list[dict]:
    """Return deficit info for the gap between *event* and *case*.

    Supports quantity-aware comparison:
      - ``required_items`` may be ``dict[str, int]`` (item → count)
        or ``list[str]`` (each entry counts as 1).
      - ``visible_items`` may be ``dict[str, int]`` (item → count)
        or ``list[str]`` (each entry counts as 1).

    Each returned dict has shape: ``{"item": str, "have": int, "need": int}``.
    """
    # Normalise required_items to dict[str, int]
    raw_required = case.get("required_items", [])
    if isinstance(raw_required, dict):
        required = dict(raw_required)
    else:
        required: dict[str, int] = {}
        for item in raw_required:
            required[item] = required.get(item, 0) + 1

    # Normalise visible_items to dict[str, int]
    raw_visible = event.get("visible_items", [])
    if isinstance(raw_visible, dict):
        visible = dict(raw_visible)
    else:
        visible: dict[str, int] = {}
        for item in raw_visible:
            visible[item] = visible.get(item, 0) + 1

    deficits: list[dict] = []

    for item, need in sorted(required.items()):
        have = visible.get(item, 0)
        if have < need:
            deficits.append({"item": item, "have": have, "need": need})

    return deficits
