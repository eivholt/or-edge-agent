"""Test 1: same physical event, different pathway — end-to-end via LLM.

Requires the VLM running at $VLM_BASE_URL (default http://localhost:8001/v1).
Mark with ``pytest -m llm`` or run the whole suite.
"""

import pytest

from apps.agent.run_fixture import ask_agent, get_resources
from apps.agent.validation import validate_decision

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

RESOURCES = get_resources("OR-2")

# Case A: procedure does NOT require suction_tip
CASE_A = {
    "case_id": "CASE-A",
    "procedure": "synthetic minor excision",
    "phase": "pre_op_setup",
    "priority": "normal",
    "required_items": ["scalpel", "forceps", "trocar", "specimen_cup"],
    "open_items": [],
    "porter_release_allowed": False,
}

# Case B: procedure DOES require suction_tip
CASE_B = {
    "case_id": "CASE-B",
    "procedure": "synthetic laparoscopic biopsy",
    "phase": "pre_op_setup",
    "priority": "normal",
    "required_items": ["scalpel", "forceps", "trocar", "specimen_cup", "suction_tip"],
    "open_items": [],
    "porter_release_allowed": False,
}


@pytest.mark.llm
def test_no_escalation_when_pathway_does_not_need_item():
    """Case A: suction_tip missing but not required → no missing_supply task."""
    decision = ask_agent(EVENT, CASE_A, RESOURCES)
    errors = validate_decision(decision, EVENT)
    assert not errors, f"Validation errors: {errors}"

    task_types = [
        c["arguments"].get("task_type")
        for c in decision["tool_calls"]
        if c["name"] == "create_synthetic_or_task"
    ]
    assert "missing_supply" not in task_types, (
        f"LLM should not create a missing_supply task when the pathway "
        f"does not require suction_tip, got: {decision}"
    )


@pytest.mark.llm
def test_missing_supply_when_pathway_requires_item():
    """Case B: suction_tip missing and required → missing_supply task."""
    decision = ask_agent(EVENT, CASE_B, RESOURCES)
    errors = validate_decision(decision, EVENT)
    assert not errors, f"Validation errors: {errors}"

    supply_calls = [
        c for c in decision["tool_calls"]
        if c["name"] == "create_synthetic_or_task"
        and c["arguments"].get("task_type") == "missing_supply"
    ]
    assert len(supply_calls) >= 1, (
        f"LLM should create a missing_supply task for suction_tip, got: {decision}"
    )
    summaries = " ".join(c["arguments"].get("summary", "") for c in supply_calls)
    normalized = summaries.lower().replace("_", " ")
    assert "suction tip" in normalized, (
        f"missing_supply task should mention suction_tip, got: {summaries}"
    )
