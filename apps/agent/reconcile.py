"""Reconcile a detector event against a synthetic surgical pathway.

Pure logic — no LLM, no network.  Compares what the detector sees
(visible / missing_or_uncertain) with what the pathway says is required
and returns a list of proposed tool-call dicts the agent should consider.
"""

from __future__ import annotations


def reconcile(event: dict, case: dict) -> list[dict]:
    """Return proposed tool_calls for the gap between *event* and *case*.

    Each returned dict has the shape expected by ``validate_decision``:
    ``{"name": "...", "arguments": {...}}``.
    """
    required = set(case.get("required_items", []))
    visible = set(event.get("visible_items", []))
    missing = set(event.get("missing_or_uncertain", []))

    # Items the detector flagged as missing *and* the pathway actually needs.
    actionable_missing = missing & required

    # Items the pathway needs that are neither visible nor flagged.
    unaccounted = required - visible - missing

    calls: list[dict] = []

    for item in sorted(actionable_missing):
        calls.append({
            "name": "create_synthetic_or_task",
            "arguments": {
                "case_id": case["case_id"],
                "task_type": "missing_supply",
                "priority": case.get("priority", "normal"),
                "summary": f"{item} missing from {event.get('zone', 'unknown zone')}",
                "reason": f"Detector flagged {item} as missing/uncertain and pathway requires it.",
            },
        })

    for item in sorted(unaccounted):
        calls.append({
            "name": "create_synthetic_or_task",
            "arguments": {
                "case_id": case["case_id"],
                "task_type": "missing_supply",
                "priority": case.get("priority", "normal"),
                "summary": f"{item} not yet seen in {event.get('zone', 'unknown zone')}",
                "reason": f"Pathway requires {item} but detector has not reported it.",
            },
        })

    # Handle pathway-change events — table may look ready but procedure changed.
    if event.get("event_type") == "visually_ready_but_pathway_changed":
        calls.append({
            "name": "create_synthetic_or_task",
            "arguments": {
                "case_id": case["case_id"],
                "task_type": "procedure_change_review",
                "priority": "high",
                "summary": f"Procedure changed — review setup for {case.get('procedure', 'unknown')}",
                "reason": "Surgical pathway changed after initial setup. Requires review.",
            },
        })

    return calls
