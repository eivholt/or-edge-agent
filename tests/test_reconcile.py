"""Test 1: same physical event, different pathway → different outcome."""

from apps.agent.reconcile import reconcile

# Shared detector event — identical for both cases
EVENT = {
    "event_id": "evt-9001",
    "room_id": "OR-2",
    "case_id": "PLACEHOLDER",
    "event_type": "or_setup_state_change",
    "visible_items": ["scalpel", "forceps", "trocar", "specimen_cup"],
    "missing_or_uncertain": ["suction_tip"],
    "zone": "back_table",
    "confidence": 0.88,
    "timestamp": "2026-05-15T08:00:00+02:00",
}

# Case A: procedure does NOT require suction_tip
CASE_A = {
    "case_id": "CASE-A",
    "procedure": "synthetic minor excision",
    "phase": "pre_op_setup",
    "priority": "normal",
    "required_items": ["scalpel", "forceps", "trocar", "specimen_cup"],
}

# Case B: procedure DOES require suction_tip
CASE_B = {
    "case_id": "CASE-B",
    "procedure": "synthetic laparoscopic biopsy",
    "phase": "pre_op_setup",
    "priority": "normal",
    "required_items": ["scalpel", "forceps", "trocar", "specimen_cup", "suction_tip"],
}


def test_no_escalation_when_pathway_does_not_need_item():
    """Case A: suction_tip missing but not required → no tool calls."""
    calls = reconcile(EVENT, CASE_A)
    assert calls == [], f"Expected no escalation, got {calls}"


def test_missing_supply_when_pathway_requires_item():
    """Case B: suction_tip missing and required → missing_supply task."""
    calls = reconcile(EVENT, CASE_B)
    assert len(calls) == 1
    call = calls[0]
    assert call["name"] == "create_synthetic_or_task"
    assert call["arguments"]["task_type"] == "missing_supply"
    assert "suction_tip" in call["arguments"]["summary"]


# ── Procedure-change event ───────────────────────────────────────────


def test_procedure_change_produces_review_call():
    """visually_ready_but_pathway_changed → procedure_change_review task."""
    event = {
        "event_id": "evt-pc",
        "room_id": "OR-2",
        "case_id": "CASE-PC",
        "event_type": "visually_ready_but_pathway_changed",
        "visible_items": ["scalpel", "forceps", "trocar"],
        "missing_or_uncertain": [],
        "zone": "back_table",
        "confidence": 0.91,
        "timestamp": "2026-05-15T08:00:00+02:00",
    }
    case = {
        "case_id": "CASE-PC",
        "procedure": "synthetic cholecystectomy",
        "priority": "high",
        "required_items": ["scalpel", "forceps", "trocar"],
    }
    calls = reconcile(event, case)
    assert len(calls) == 1
    assert calls[0]["arguments"]["task_type"] == "procedure_change_review"
    assert calls[0]["arguments"]["priority"] == "high"


def test_procedure_change_with_missing_items():
    """Pathway changed AND items missing → both gaps + procedure_change_review."""
    event = {
        "event_id": "evt-pc2",
        "room_id": "OR-2",
        "case_id": "CASE-PC2",
        "event_type": "visually_ready_but_pathway_changed",
        "visible_items": ["scalpel"],
        "missing_or_uncertain": ["forceps"],
        "zone": "back_table",
        "confidence": 0.91,
        "timestamp": "2026-05-15T08:00:00+02:00",
    }
    case = {
        "case_id": "CASE-PC2",
        "procedure": "synthetic cholecystectomy",
        "priority": "high",
        "required_items": ["scalpel", "forceps", "trocar"],
    }
    calls = reconcile(event, case)
    types = [c["arguments"]["task_type"] for c in calls]
    assert "missing_supply" in types, "Should flag forceps as missing_supply"
    assert "procedure_change_review" in types, "Should include procedure_change_review"
