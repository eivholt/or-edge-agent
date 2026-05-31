"""Test 1: same physical event, different pathway — end-to-end via LLM.

Requires:
  - Synthetic EMR API running: uvicorn synthetic_emr.api:app --port 9000
  - vLLM running at $VLM_BASE_URL (default http://localhost:8081/v1)

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
    "visible_items": {
        "scalpel": 2,
        "scissors": 2,
        "sponge": 4,
        "tweezers": 3,
    },
    "missing_or_uncertain": ["tweezers"],
    "zone": "back_table",
    "confidence": 0.88,
    "timestamp": "2026-05-15T08:00:00+02:00",
    "image_path": "frames/frame_all_present.png",
}

RESOURCES = get_resources("OR-2")


@pytest.mark.llm
def test_no_escalation_when_pathway_does_not_need_item():
    """Case A: tweezers flagged uncertain but not required → no task or resupply."""
    event_a = dict(EVENT, case_id="CASE-A")
    decision = ask_agent(event_a, RESOURCES)
    errors = validate_decision(decision, event_a)
    assert not errors, f"Validation errors: {errors}"

    resupply_calls = [
        c for c in decision["tool_calls"]
        if c["name"] in ("request_resupply", "request_spd_resupply")
    ]
    assert not resupply_calls, (
        f"LLM should not request resupply when tweezers is "
        f"not required by the pathway, got: {decision}"
    )


@pytest.mark.llm
def test_resupply_when_pathway_requires_item():
    """Case B: scissors deficit → agent should call request_resupply."""
    # Override event to have scissors deficit
    event_with_deficit = dict(EVENT, case_id="CASE-B")
    event_with_deficit["visible_items"] = {
        "scalpel": 2,
        "scissors": 1,
        "sponge": 4,
        "tweezers": 1,
    }
    event_with_deficit["missing_or_uncertain"] = ["scissors"]

    decision = ask_agent(event_with_deficit, RESOURCES)
    errors = validate_decision(decision, event_with_deficit)
    assert not errors, f"Validation errors: {errors}"

    resupply_calls = [
        c for c in decision["tool_calls"]
        if c["name"] in ("request_resupply", "request_spd_resupply")
    ]
    assert len(resupply_calls) >= 1, (
        f"LLM should request resupply for scissors deficit, got: {decision}"
    )
    all_args = " ".join(
        str(c["arguments"]) for c in resupply_calls
    ).lower()
    assert "scissors" in all_args, (
        f"Resupply call should mention scissors, got: {all_args}"
    )
