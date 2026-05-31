"""Reconcile a detector event against a synthetic surgical pathway.

Pure logic — no LLM, no network.  Compares what the detector sees
(visible counts / missing_or_uncertain) with what the pathway says is
required (item → count) and returns a list of proposed tool-call dicts
the agent should consider.
"""

from __future__ import annotations

import logfire


@logfire.instrument("reconcile case_id={case[case_id]}")
def reconcile(event: dict, case: dict) -> list[dict]:
    """Return proposed tool_calls for the gap between *event* and *case*.

    Supports quantity-aware comparison:
      - ``required_items`` may be ``dict[str, int]`` (item → count)
        or ``list[str]`` (each entry counts as 1).
      - ``visible_items`` may be ``dict[str, int]`` (item → count)
        or ``list[str]`` (each entry counts as 1).

    Each returned dict has the shape expected by ``validate_decision``:
    ``{"name": "...", "arguments": {...}}``.
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

    missing = set(event.get("missing_or_uncertain", []))

    event_type = event.get("event_type", "")
    flagged_task_type = "human_review" if event_type == "sterile_zone_ambiguity" else "missing_supply"

    calls: list[dict] = []

    for item, need in sorted(required.items()):
        have = visible.get(item, 0)
        flagged = item in missing

        if flagged:
            # Detector explicitly flagged this item as uncertain.
            deficit = max(0, need - have)
            if deficit > 0:
                calls.append({
                    "name": "create_or_task",
                    "arguments": {
                        "case_id": case["case_id"],
                        "task_type": flagged_task_type,
                        "priority": case.get("priority", "normal"),
                        "summary": (
                            f"{item} deficit: need {need}, have {have} "
                            f"in {event.get('zone', 'unknown zone')}"
                        ),
                        "reason": (
                            f"Detector flagged {item} as uncertain. "
                            f"Required {need}, visible {have}."
                        ),
                    },
                })
            # else: flagged but no deficit (have >= need) — no action needed
        elif have < need:
            # Not flagged, but count is short.
            calls.append({
                "name": "create_or_task",
                "arguments": {
                    "case_id": case["case_id"],
                    "task_type": "missing_supply",
                    "priority": case.get("priority", "normal"),
                    "summary": (
                        f"{item} deficit: need {need}, have {have} "
                        f"in {event.get('zone', 'unknown zone')}"
                    ),
                    "reason": (
                        f"Pathway requires {need} {item} but detector "
                        f"sees only {have}."
                    ),
                },
            })

    # Handle pathway-change events — table may look ready but procedure changed.
    # Per protocol: ALL THREE actions are required.
    if event.get("event_type") == "visually_ready_but_pathway_changed":
        calls.append({
            "name": "create_or_task",
            "arguments": {
                "case_id": case["case_id"],
                "task_type": "procedure_change_review",
                "priority": "high",
                "summary": f"Procedure changed — review setup for {case.get('procedure', 'unknown')}",
                "reason": "Surgical pathway changed after initial setup. Requires review.",
            },
        })
        calls.append({
            "name": "create_or_task",
            "arguments": {
                "case_id": case["case_id"],
                "task_type": "porter_hold",
                "priority": "high",
                "summary": "Hold porter — procedure changed, setup under review",
                "reason": "Porter must wait until procedure change review is complete.",
            },
        })
        calls.append({
            "name": "set_or_prep_light",
            "arguments": {
                "room_id": event.get("room_id", "OR-?"),
                "color": "yellow",
            },
        })

    return calls
