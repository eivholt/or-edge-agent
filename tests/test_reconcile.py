"""Test 1: same physical event, different pathway → different outcome.

Now quantity-aware: required_items and visible_items are dicts of item → count.
"""

from apps.agent.reconcile import reconcile

# Shared detector event — identical for both cases
EVENT = {
    "event_id": "evt-9001",
    "room_id": "OR-2",
    "case_id": "PLACEHOLDER",
    "event_type": "or_setup_state_change",
    "visible_items": {
        "scalpel": 2,
        "scissors": 2,
        "sponge": 4,
        "tweezers": 3,
    },
    "missing_or_uncertain": [],
    "zone": "back_table",
    "confidence": 0.88,
    "timestamp": "2026-05-15T08:00:00+02:00",
}

# Case A: requires only 1 of each — all present
CASE_A = {
    "case_id": "CASE-A",
    "procedure": "synthetic minor excision",
    "phase": "pre_op_setup",
    "priority": "normal",
    "required_items": {"scalpel": 1, "scissors": 1, "sponge": 2},
}

# Case B: requires more tweezers than visible (3 visible but needs 4)
CASE_B = {
    "case_id": "CASE-B",
    "procedure": "synthetic laparoscopic biopsy",
    "phase": "pre_op_setup",
    "priority": "normal",
    "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 4},
}


def test_no_escalation_when_counts_sufficient():
    """Case A: all required counts met → no tool calls."""
    calls = reconcile(EVENT, CASE_A)
    assert calls == [], f"Expected no escalation, got {calls}"


def test_missing_supply_when_count_deficit():
    """Case B: need 4 tweezers but only 3 visible → deficit."""
    deficits = reconcile(EVENT, CASE_B)
    assert len(deficits) == 1
    d = deficits[0]
    assert d["item"] == "tweezers"
    assert d["have"] == 3
    assert d["need"] == 4


# ── Quantity deficit tests ───────────────────────────────────────────


def test_multiple_deficits():
    """Multiple items below required count → one deficit per item."""
    event = {
        "event_id": "evt-qd",
        "room_id": "OR-2",
        "case_id": "CASE-QD",
        "event_type": "or_setup_state_change",
        "visible_items": {"scalpel": 1, "scissors": 0},
        "missing_or_uncertain": [],
        "zone": "back_table",
        "confidence": 0.90,
        "timestamp": "2026-05-15T08:00:00+02:00",
    }
    case = {
        "case_id": "CASE-QD",
        "procedure": "synthetic procedure",
        "priority": "normal",
        "required_items": {"scalpel": 2, "scissors": 2},
    }
    deficits = reconcile(event, case)
    assert len(deficits) == 2
    items = [d["item"] for d in deficits]
    assert "scalpel" in items
    assert "scissors" in items


def test_flagged_uncertain_with_deficit():
    """Item flagged uncertain AND count is short → deficit."""
    event = {
        "event_id": "evt-fu",
        "room_id": "OR-2",
        "case_id": "CASE-FU",
        "event_type": "or_setup_state_change",
        "visible_items": {"scalpel": 2, "scissors": 1},
        "missing_or_uncertain": ["scissors"],
        "zone": "back_table",
        "confidence": 0.85,
        "timestamp": "2026-05-15T08:00:00+02:00",
    }
    case = {
        "case_id": "CASE-FU",
        "procedure": "synthetic procedure",
        "priority": "normal",
        "required_items": {"scalpel": 2, "scissors": 2},
    }
    deficits = reconcile(event, case)
    assert len(deficits) == 1
    assert deficits[0]["item"] == "scissors"


def test_flagged_uncertain_but_count_sufficient():
    """Item flagged uncertain but visible count meets requirement → no task."""
    event = {
        "event_id": "evt-fs",
        "room_id": "OR-2",
        "case_id": "CASE-FS",
        "event_type": "or_setup_state_change",
        "visible_items": {"scalpel": 2, "scissors": 2},
        "missing_or_uncertain": ["scissors"],
        "zone": "back_table",
        "confidence": 0.85,
        "timestamp": "2026-05-15T08:00:00+02:00",
    }
    case = {
        "case_id": "CASE-FS",
        "procedure": "synthetic procedure",
        "priority": "normal",
        "required_items": {"scalpel": 2, "scissors": 2},
    }
    calls = reconcile(event, case)
    assert calls == [], f"Count is sufficient even though flagged: {calls}"


def test_item_not_required_no_task():
    """Item visible but not required → no task even if count is low."""
    event = {
        "event_id": "evt-nr",
        "room_id": "OR-2",
        "case_id": "CASE-NR",
        "event_type": "or_setup_state_change",
        "visible_items": {"scalpel": 2, "scissors": 1, "sponge": 1},
        "missing_or_uncertain": [],
        "zone": "back_table",
        "confidence": 0.90,
        "timestamp": "2026-05-15T08:00:00+02:00",
    }
    case = {
        "case_id": "CASE-NR",
        "procedure": "synthetic procedure",
        "priority": "normal",
        "required_items": {"scalpel": 2},
    }
    calls = reconcile(event, case)
    assert calls == [], f"sponge and scissors not required: {calls}"


# ── Procedure-change event ───────────────────────────────────────────


def test_procedure_change_produces_review_call():
    """visually_ready_but_pathway_changed — reconcile only returns deficits.
    No deficits here since all counts meet requirements."""
    event = {
        "event_id": "evt-pc",
        "room_id": "OR-2",
        "case_id": "CASE-PC",
        "event_type": "visually_ready_but_pathway_changed",
        "visible_items": {"scalpel": 2, "scissors": 2, "tweezers": 2},
        "missing_or_uncertain": [],
        "zone": "back_table",
        "confidence": 0.91,
        "timestamp": "2026-05-15T08:00:00+02:00",
    }
    case = {
        "case_id": "CASE-PC",
        "procedure": "synthetic cholecystectomy",
        "priority": "high",
        "required_items": {"scalpel": 2, "scissors": 2, "tweezers": 2},
    }
    deficits = reconcile(event, case)
    assert deficits == [], f"No deficits expected when counts match: {deficits}"


def test_procedure_change_with_deficit():
    """Pathway changed AND items below count → deficits returned."""
    event = {
        "event_id": "evt-pc2",
        "room_id": "OR-2",
        "case_id": "CASE-PC2",
        "event_type": "visually_ready_but_pathway_changed",
        "visible_items": {"scalpel": 1},
        "missing_or_uncertain": ["scissors"],
        "zone": "back_table",
        "confidence": 0.91,
        "timestamp": "2026-05-15T08:00:00+02:00",
    }
    case = {
        "case_id": "CASE-PC2",
        "procedure": "synthetic cholecystectomy",
        "priority": "high",
        "required_items": {"scalpel": 2, "scissors": 2, "tweezers": 2},
    }
    deficits = reconcile(event, case)
    items = [d["item"] for d in deficits]
    assert "scalpel" in items, "scalpel deficit (need 2, have 1)"
    assert "scissors" in items, "scissors deficit (need 2, have 0)"
    assert "tweezers" in items, "tweezers deficit (need 2, have 0)"
