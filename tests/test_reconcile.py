"""Test 1: same physical event, different pathway → different outcome.

Now quantity-aware: required_items and visible_items are dicts of item → count.
"""

from apps.agent.reconcile import reconcile

# Shared detector event — identical for both cases
EVENT = {
    "case_id": "PLACEHOLDER",
    "room_id": "OR-2",
    "image_path": "frames/frame_all_present.png",
    "visible_items": {
        "scalpel": 2,
        "scissors": 2,
        "sponge": 4,
        "tweezers": 3,
    },
}

# Case A: requires only 1 of each — all present
CASE_A = {
    "case_id": "CASE-A",
    "procedure": "synthetic minor excision",
    "priority": "normal",
    "required_items": {"scalpel": 1, "scissors": 1, "sponge": 2},
}

# Case B: requires more tweezers than visible (3 visible but needs 4)
CASE_B = {
    "case_id": "CASE-B",
    "procedure": "synthetic laparoscopic biopsy",
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
        "case_id": "CASE-QD",
        "room_id": "OR-2",
        "image_path": "frames/frame_all_present.png",
        "visible_items": {"scalpel": 1, "scissors": 0},
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
    """Item with count short → deficit."""
    event = {
        "case_id": "CASE-FU",
        "room_id": "OR-2",
        "image_path": "frames/frame_all_present.png",
        "visible_items": {"scalpel": 2, "scissors": 1},
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


def test_count_sufficient_no_deficit():
    """Visible count meets requirement → no deficit."""
    event = {
        "case_id": "CASE-FS",
        "room_id": "OR-2",
        "image_path": "frames/frame_all_present.png",
        "visible_items": {"scalpel": 2, "scissors": 2},
    }
    case = {
        "case_id": "CASE-FS",
        "procedure": "synthetic procedure",
        "priority": "normal",
        "required_items": {"scalpel": 2, "scissors": 2},
    }
    calls = reconcile(event, case)
    assert calls == [], f"Count is sufficient: {calls}"


def test_item_not_required_no_task():
    """Item visible but not required → no task even if count is low."""
    event = {
        "case_id": "CASE-NR",
        "room_id": "OR-2",
        "image_path": "frames/frame_all_present.png",
        "visible_items": {"scalpel": 2, "scissors": 1, "sponge": 1},
    }
    case = {
        "case_id": "CASE-NR",
        "procedure": "synthetic procedure",
        "priority": "normal",
        "required_items": {"scalpel": 2},
    }
    calls = reconcile(event, case)
    assert calls == [], f"sponge and scissors not required: {calls}"


# ── No procedure-change concept — just count-based tests ─────────────


def test_no_deficit_when_all_counts_match():
    """All counts meet requirements → no deficits."""
    event = {
        "case_id": "CASE-PC",
        "room_id": "OR-2",
        "image_path": "frames/frame_all_present.png",
        "visible_items": {"scalpel": 2, "scissors": 2, "tweezers": 2},
    }
    case = {
        "case_id": "CASE-PC",
        "procedure": "synthetic cholecystectomy",
        "priority": "high",
        "required_items": {"scalpel": 2, "scissors": 2, "tweezers": 2},
    }
    deficits = reconcile(event, case)
    assert deficits == [], f"No deficits expected when counts match: {deficits}"


def test_deficit_when_counts_short():
    """Items below count → deficits returned."""
    event = {
        "case_id": "CASE-PC2",
        "room_id": "OR-2",
        "image_path": "frames/frame_all_present.png",
        "visible_items": {"scalpel": 1},
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
