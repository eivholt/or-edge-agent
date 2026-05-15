"""Tests that exercise the VLM pipeline:
  - Local VLM: Ministral-3-3B via vLLM (inspect_scene_local)
  - Remote VLM: Azure gpt-4o (inspect_scene_remote / ask_vlm)

Requires:
  - Synthetic EMR API running: uvicorn synthetic_emr.api:app --port 9000
  - vLLM running Ministral-3-3B at $VLM_BASE_URL
  - Azure OpenAI VLM configured in .env

Run with: pytest -m llm tests/test_vlm_inspect.py -v
"""

import pytest

from apps.vlm.ask_vlm import ask_vlm
from apps.agent.run_fixture import ask_agent, get_case, get_resources
from apps.agent.validation import validate_decision

IMAGE_PATH = "data/surgery_tools_1024.jpg"
RESOURCES = get_resources("OR-2")


# ── Direct VLM test ─────────────────────────────────────────────────


@pytest.mark.llm
def test_vlm_identifies_instruments():
    """Azure gpt-4o should identify surgical instruments in the image."""
    result = ask_vlm(IMAGE_PATH, "List the surgical instruments visible. Be brief.")
    lower = result.lower()
    # The image shows scalpels, scissors, forceps — at least some should be identified
    found = sum(1 for kw in ["scalpel", "scissor", "forceps"] if kw in lower)
    assert found >= 2, f"VLM should identify at least 2 instrument types, got: {result}"


# ── Agent calls VLM via inspect_scene ────────────────────────────────


# Event with low confidence triggers the agent to want visual verification
EVENT_UNCERTAIN = {
    "event_id": "evt-9050",
    "room_id": "OR-2",
    "case_id": "CASE-VLM",
    "event_type": "sterile_zone_ambiguity",
    "visible_items": ["scalpel", "forceps"],
    "missing_or_uncertain": ["scissors"],
    "zone": "back_table",
    "confidence": 0.82,
    "timestamp": "2026-05-15T09:00:00+02:00",
    "image_path": IMAGE_PATH,
}


@pytest.mark.llm
def test_agent_handles_uncertain_event_with_missing_required():
    """Agent should create a task for scissors (missing + required).

    The agent receives a low-confidence event with scissors uncertain.
    Scissors IS required by the pathway, so reconciliation flags it.
    The agent may optionally call inspect_scene to verify, then should
    create a missing_supply or human_review task.
    """
    case = get_case("CASE-VLM")
    decision = ask_agent(EVENT_UNCERTAIN, case, RESOURCES)
    errors = validate_decision(decision, EVENT_UNCERTAIN)
    assert not errors, f"Validation errors: {errors}"

    # Agent should have created at least one task for scissors
    task_calls = [
        c for c in decision["tool_calls"]
        if c["name"] == "create_synthetic_or_task"
    ]
    assert len(task_calls) >= 1, (
        f"Agent should create a task for missing scissors, got: {decision}"
    )

    # Check the task references scissors
    all_summaries = " ".join(
        c["arguments"].get("summary", "") + " " + c["arguments"].get("reason", "")
        for c in task_calls
    ).lower()
    assert "scissor" in all_summaries, (
        f"Task should reference scissors, got: {all_summaries}"
    )


@pytest.mark.llm
def test_vlm_inspect_scene_direct():
    """Call inspect_scene tool function directly (bypassing agent) to verify
    the Azure VLM integration end-to-end with a specific question."""
    result = ask_vlm(
        IMAGE_PATH,
        "Is there a suction tip visible on this table? Answer yes or no and explain briefly."
    )
    lower = result.lower()
    # The image doesn't show a suction tip
    assert "no" in lower or "not" in lower or "don't" in lower or "absent" in lower, (
        f"VLM should report no suction tip visible, got: {result}"
    )
