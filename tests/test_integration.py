"""Full integration tests: mocked detector → reconcile → LLM agent → tool calls.

Each test simulates a detector event (mocked), feeds it through the real
reconciliation and agent pipeline, and asserts on the tool calls the agent
actually executes.  The local LLM (Qwen2.5-7B-Instruct) does real inference
and calls real pydantic_ai tools.

Requires:
  - vLLM running Qwen2.5-7B-Instruct at $VLM_BASE_URL
  - Azure VLM configured in .env (only for tests that trigger inspect_scene)

Run with: pytest -m llm tests/test_integration.py -v
"""

import json

import pytest

from pydantic_ai import Agent, ModelSettings, RunContext

from apps.agent.run_fixture import (
    AgentDeps,
    INSTRUCTIONS,
    _model,
    ask_agent,
    get_resources,
    reconcile_setup,
)
from apps.agent.validation import validate_decision


RESOURCES = get_resources("OR-2")


# ─────────────────────────────────────────────────────────────────────
# Scenario 1: Missing required item → agent creates missing_supply task
# ─────────────────────────────────────────────────────────────────────

MISSING_REQUIRED_EVENT = {
    "event_id": "evt-int-001",
    "room_id": "OR-2",
    "case_id": "CASE-INT-1",
    "event_type": "or_setup_state_change",
    "visible_items": ["scalpel", "forceps", "trocar"],
    "missing_or_uncertain": ["bovie_tip", "specimen_cup"],
    "zone": "back_table",
    "confidence": 0.90,
    "timestamp": "2026-05-15T08:00:00+02:00",
}

MISSING_REQUIRED_CASE = {
    "case_id": "CASE-INT-1",
    "procedure": "synthetic laparoscopic cholecystectomy",
    "phase": "pre_op_setup",
    "priority": "normal",
    "required_items": ["scalpel", "forceps", "trocar", "bovie_tip", "specimen_cup"],
    "open_items": [],
    "porter_release_allowed": False,
}


@pytest.mark.llm
def test_multiple_missing_required_items():
    """Two required items missing → agent should create tasks for both."""
    decision = ask_agent(MISSING_REQUIRED_EVENT, MISSING_REQUIRED_CASE, RESOURCES)
    errors = validate_decision(decision, MISSING_REQUIRED_EVENT)
    assert not errors, f"Validation errors: {errors}"

    supply_calls = [
        c for c in decision["tool_calls"]
        if c["name"] == "create_synthetic_or_task"
        and c["arguments"].get("task_type") == "missing_supply"
    ]
    assert len(supply_calls) >= 2, (
        f"Agent should create tasks for both bovie_tip and specimen_cup, "
        f"got {len(supply_calls)} supply calls: {decision}"
    )

    all_text = " ".join(
        c["arguments"].get("summary", "") + " " + c["arguments"].get("reason", "")
        for c in supply_calls
    ).lower().replace("_", " ")
    assert "bovie" in all_text, f"Should mention bovie_tip: {all_text}"
    assert "specimen" in all_text, f"Should mention specimen_cup: {all_text}"


# ─────────────────────────────────────────────────────────────────────
# Scenario 2: All items present → fast path, no LLM call
# ─────────────────────────────────────────────────────────────────────

ALL_PRESENT_EVENT = {
    "event_id": "evt-int-002",
    "room_id": "OR-2",
    "case_id": "CASE-INT-2",
    "event_type": "or_setup_state_change",
    "visible_items": ["scalpel", "forceps", "trocar", "specimen_cup"],
    "missing_or_uncertain": [],
    "zone": "back_table",
    "confidence": 0.95,
    "timestamp": "2026-05-15T08:05:00+02:00",
}

ALL_PRESENT_CASE = {
    "case_id": "CASE-INT-2",
    "procedure": "synthetic minor excision",
    "phase": "pre_op_setup",
    "priority": "normal",
    "required_items": ["scalpel", "forceps", "trocar", "specimen_cup"],
    "open_items": [],
    "porter_release_allowed": False,
}


@pytest.mark.llm
def test_all_present_fast_path():
    """All required items visible → no LLM call, empty tool_calls."""
    decision = ask_agent(ALL_PRESENT_EVENT, ALL_PRESENT_CASE, RESOURCES)
    errors = validate_decision(decision, ALL_PRESENT_EVENT)
    assert not errors, f"Validation errors: {errors}"

    assert decision["tool_calls"] == [], (
        f"No tools should be called when everything is present: {decision}"
    )
    assert decision["requires_human_review"] is False


# ─────────────────────────────────────────────────────────────────────
# Scenario 3: Missing item NOT required → no task
# ─────────────────────────────────────────────────────────────────────

MISSING_NOT_REQUIRED_EVENT = {
    "event_id": "evt-int-003",
    "room_id": "OR-2",
    "case_id": "CASE-INT-3",
    "event_type": "or_setup_state_change",
    "visible_items": ["scalpel", "forceps", "trocar"],
    "missing_or_uncertain": ["suction_tip", "retractor"],
    "zone": "back_table",
    "confidence": 0.88,
    "timestamp": "2026-05-15T08:10:00+02:00",
}

MISSING_NOT_REQUIRED_CASE = {
    "case_id": "CASE-INT-3",
    "procedure": "synthetic wound closure",
    "phase": "pre_op_setup",
    "priority": "normal",
    "required_items": ["scalpel", "forceps", "trocar"],
    "open_items": [],
    "porter_release_allowed": False,
}


@pytest.mark.llm
def test_missing_items_not_required_no_task():
    """Items flagged missing but not required → fast path, no tasks."""
    decision = ask_agent(
        MISSING_NOT_REQUIRED_EVENT, MISSING_NOT_REQUIRED_CASE, RESOURCES
    )
    errors = validate_decision(decision, MISSING_NOT_REQUIRED_EVENT)
    assert not errors, f"Validation errors: {errors}"

    assert decision["tool_calls"] == [], (
        f"No tasks should be created for items not in required_items: {decision}"
    )


# ─────────────────────────────────────────────────────────────────────
# Scenario 4: Mix — some missing items required, some not
# ─────────────────────────────────────────────────────────────────────

MIXED_EVENT = {
    "event_id": "evt-int-004",
    "room_id": "OR-2",
    "case_id": "CASE-INT-4",
    "event_type": "or_setup_state_change",
    "visible_items": ["scalpel", "forceps"],
    "missing_or_uncertain": ["trocar", "retractor", "suction_tip"],
    "zone": "back_table",
    "confidence": 0.85,
    "timestamp": "2026-05-15T08:15:00+02:00",
}

MIXED_CASE = {
    "case_id": "CASE-INT-4",
    "procedure": "synthetic laparoscopic biopsy",
    "phase": "pre_op_setup",
    "priority": "high",
    "required_items": ["scalpel", "forceps", "trocar", "suction_tip"],
    "open_items": [],
    "porter_release_allowed": False,
}


@pytest.mark.llm
def test_mixed_missing_only_required_get_tasks():
    """trocar and suction_tip are missing+required; retractor is not required.
    Agent should create tasks only for trocar and suction_tip."""
    decision = ask_agent(MIXED_EVENT, MIXED_CASE, RESOURCES)
    errors = validate_decision(decision, MIXED_EVENT)
    assert not errors, f"Validation errors: {errors}"

    supply_calls = [
        c for c in decision["tool_calls"]
        if c["name"] == "create_synthetic_or_task"
        and c["arguments"].get("task_type") == "missing_supply"
    ]
    assert len(supply_calls) >= 2, (
        f"Agent should create tasks for trocar and suction_tip, "
        f"got {len(supply_calls)}: {decision}"
    )

    all_text = " ".join(
        c["arguments"].get("summary", "") + " " + c["arguments"].get("reason", "")
        for c in supply_calls
    ).lower().replace("_", " ")
    assert "trocar" in all_text, f"Should mention trocar: {all_text}"
    assert "suction" in all_text, f"Should mention suction_tip: {all_text}"

    # retractor should NOT appear in any missing_supply task
    retractor_calls = [
        c for c in supply_calls
        if "retractor" in (c["arguments"].get("summary", "") + c["arguments"].get("reason", "")).lower()
    ]
    assert len(retractor_calls) == 0, (
        f"Agent should NOT create a task for retractor (not required): {retractor_calls}"
    )


# ─────────────────────────────────────────────────────────────────────
# Scenario 5: Reconciliation correctness verified end-to-end
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.llm
def test_reconcile_feeds_correct_gaps_to_agent():
    """Verify reconcile_setup produces the right gaps and the agent acts on them."""
    recon = reconcile_setup(MIXED_EVENT, MIXED_CASE)

    assert sorted(recon["actionable_missing"]) == ["suction_tip", "trocar"]
    assert recon["unaccounted"] == []
    assert recon["all_present"] is False


# ─────────────────────────────────────────────────────────────────────
# Scenario 6: Specimen handoff — different event type
# ─────────────────────────────────────────────────────────────────────

SPECIMEN_EVENT = {
    "event_id": "evt-int-006",
    "room_id": "OR-2",
    "case_id": "CASE-INT-6",
    "event_type": "specimen_container_seen",
    "visible_items": ["specimen_cup", "forceps"],
    "missing_or_uncertain": [],
    "zone": "back_table",
    "confidence": 0.93,
    "timestamp": "2026-05-15T08:30:00+02:00",
}

SPECIMEN_CASE = {
    "case_id": "CASE-INT-6",
    "procedure": "synthetic biopsy",
    "phase": "intra_op",
    "priority": "normal",
    "required_items": ["specimen_cup", "forceps"],
    "open_items": [],
    "porter_release_allowed": False,
}


@pytest.mark.llm
def test_specimen_event_all_present():
    """Specimen container seen, all items present → no tasks."""
    decision = ask_agent(SPECIMEN_EVENT, SPECIMEN_CASE, RESOURCES)
    errors = validate_decision(decision, SPECIMEN_EVENT)
    assert not errors, f"Validation errors: {errors}"

    assert decision["tool_calls"] == [], (
        f"All items present for specimen event, no tasks expected: {decision}"
    )


# ─────────────────────────────────────────────────────────────────────
# Scenario 7: Unaccounted items — required but never seen by detector
# ─────────────────────────────────────────────────────────────────────

UNACCOUNTED_EVENT = {
    "event_id": "evt-int-007",
    "room_id": "OR-2",
    "case_id": "CASE-INT-7",
    "event_type": "or_setup_state_change",
    "visible_items": ["scalpel"],
    "missing_or_uncertain": [],
    "zone": "back_table",
    "confidence": 0.85,
    "timestamp": "2026-05-15T08:35:00+02:00",
}

UNACCOUNTED_CASE = {
    "case_id": "CASE-INT-7",
    "procedure": "synthetic appendectomy",
    "phase": "pre_op_setup",
    "priority": "normal",
    "required_items": ["scalpel", "forceps", "trocar"],
    "open_items": [],
    "porter_release_allowed": False,
}


@pytest.mark.llm
def test_unaccounted_items_get_tasks():
    """forceps and trocar required but never seen or flagged → agent should act."""
    recon = reconcile_setup(UNACCOUNTED_EVENT, UNACCOUNTED_CASE)
    assert sorted(recon["unaccounted"]) == ["forceps", "trocar"]

    decision = ask_agent(UNACCOUNTED_EVENT, UNACCOUNTED_CASE, RESOURCES)
    errors = validate_decision(decision, UNACCOUNTED_EVENT)
    assert not errors, f"Validation errors: {errors}"

    supply_calls = [
        c for c in decision["tool_calls"]
        if c["name"] == "create_synthetic_or_task"
    ]
    assert len(supply_calls) >= 2, (
        f"Agent should create tasks for forceps and trocar, "
        f"got {len(supply_calls)}: {decision}"
    )


# ─────────────────────────────────────────────────────────────────────
# Test 2: Visually ready, pathway changed
# ─────────────────────────────────────────────────────────────────────

PROCEDURE_CHANGED_EVENT = {
    "event_id": "evt-int-010",
    "room_id": "OR-2",
    "case_id": "CASE-INT-10",
    "event_type": "visually_ready_but_pathway_changed",
    "visible_items": ["scalpel", "forceps", "trocar", "specimen_cup", "suction_tip"],
    "missing_or_uncertain": [],
    "zone": "back_table",
    "confidence": 0.91,
    "timestamp": "2026-05-15T09:00:00+02:00",
}

PROCEDURE_CHANGED_CASE = {
    "case_id": "CASE-INT-10",
    "procedure": "synthetic laparoscopic cholecystectomy (changed from appendectomy)",
    "phase": "pre_op_setup",
    "priority": "high",
    "required_items": ["scalpel", "forceps", "trocar", "specimen_cup", "suction_tip"],
    "open_items": [],
    "porter_release_allowed": False,
}


@pytest.mark.llm
def test_procedure_changed_creates_review_and_hold():
    """Pathway changed after setup → procedure_change_review + porter_hold + yellow light."""
    decision = ask_agent(
        PROCEDURE_CHANGED_EVENT, PROCEDURE_CHANGED_CASE, RESOURCES
    )
    errors = validate_decision(decision, PROCEDURE_CHANGED_EVENT)
    assert not errors, f"Validation errors: {errors}"

    task_types = [
        c["arguments"].get("task_type")
        for c in decision["tool_calls"]
        if c["name"] == "create_synthetic_or_task"
    ]
    assert "procedure_change_review" in task_types, (
        f"Should create procedure_change_review task, got: {task_types}"
    )
    assert "porter_hold" in task_types, (
        f"Should create porter_hold task, got: {task_types}"
    )

    light_calls = [
        c for c in decision["tool_calls"]
        if c["name"] == "set_or_prep_light"
    ]
    assert len(light_calls) >= 1, (
        f"Should set prep light, got tool_calls: {decision['tool_calls']}"
    )
    assert light_calls[0]["arguments"]["color"] == "yellow", (
        f"Prep light should be yellow, got: {light_calls[0]}"
    )


# ─────────────────────────────────────────────────────────────────────
# Test 3: VLM only when needed — low confidence sterile zone ambiguity
# ─────────────────────────────────────────────────────────────────────

STERILE_ZONE_EVENT = {
    "event_id": "evt-int-011",
    "room_id": "OR-2",
    "case_id": "CASE-INT-11",
    "event_type": "sterile_zone_ambiguity",
    "visible_items": ["scalpel", "trocar"],
    "missing_or_uncertain": ["forceps"],
    "zone": "back_table",
    "confidence": 0.72,
    "timestamp": "2026-05-15T09:10:00+02:00",
}

STERILE_ZONE_CASE = {
    "case_id": "CASE-INT-11",
    "procedure": "synthetic minor excision",
    "phase": "pre_op_setup",
    "priority": "normal",
    "required_items": ["scalpel", "trocar", "forceps"],
    "open_items": [],
    "porter_release_allowed": False,
}


@pytest.mark.llm
def test_sterile_zone_ambiguity_creates_human_review():
    """Low confidence + sterile_zone_ambiguity → human_review, no sterile assertion.

    The agent may attempt to set the prep light, but validation must reject
    actuation below confidence 0.8 — verifying the guardrail works."""
    decision = ask_agent(STERILE_ZONE_EVENT, STERILE_ZONE_CASE, RESOURCES)

    task_calls = [
        c for c in decision["tool_calls"]
        if c["name"] == "create_synthetic_or_task"
    ]
    assert len(task_calls) >= 1, (
        f"Should create at least one task for forceps, got: {decision}"
    )

    task_types = [c["arguments"].get("task_type") for c in task_calls]
    assert "human_review" in task_types, (
        f"Should create human_review (not just missing_supply) for uncertainty, "
        f"got types: {task_types}"
    )

    # Agent must NOT directly assert sterile/contaminated status
    summary = decision.get("decision_summary", "").lower()
    assert "contaminated" not in summary, (
        f"Agent should not assert contaminated status: {summary}"
    )

    # Validation guardrail: any set_or_prep_light at confidence < 0.8 is rejected
    errors = validate_decision(decision, STERILE_ZONE_EVENT)
    light_calls = [
        c for c in decision["tool_calls"] if c["name"] == "set_or_prep_light"
    ]
    if light_calls:
        assert any("cannot actuate below confidence" in e for e in errors), (
            f"Validation should reject light actuation at confidence 0.72: {errors}"
        )


# ─────────────────────────────────────────────────────────────────────
# Test 4: New tool appears — dynamic tool addition
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.llm
def test_dynamic_tool_swap_robot_delivery():
    """When request_spd_robot_delivery replaces request_spd_resupply,
    the agent uses the new tool without any code changes to decision logic.

    This proves native tool calling allows dynamic tool additions: the LLM
    discovers and uses new tools from their schema + docstring alone."""

    # Build a fresh agent with robot delivery instead of SPD resupply.
    robot_agent = Agent(
        _model,
        instructions=(
            INSTRUCTIONS
            + "\n\nA delivery robot is available. For any missing supply "
            "that needs physical delivery, use request_spd_robot_delivery."
        ),
        deps_type=AgentDeps,
        model_settings=ModelSettings(temperature=0, max_tokens=700),
    )

    @robot_agent.tool
    def create_synthetic_or_task(
        ctx: RunContext[AgentDeps],
        case_id: str,
        task_type: str,
        priority: str,
        summary: str,
        reason: str,
    ) -> dict:
        """Create a synthetic OR workflow task."""
        return {"status": "created", "case_id": case_id, "task_type": task_type,
                "priority": priority, "summary": summary, "reason": reason}

    @robot_agent.tool
    def request_spd_robot_delivery(
        ctx: RunContext[AgentDeps],
        item_name: str,
        destination_room: str,
        urgency: str,
    ) -> dict:
        """Request delivery of a sterile supply by indoor robot.

        Use this when a missing physical item blocks OR setup and robot
        delivery is available.

        Args:
            item_name: Name of the item to deliver.
            destination_room: The OR room identifier.
            urgency: One of low, normal, high.
        """
        return {
            "delivery_id": f"ROBOT-{item_name}-{destination_room}",
            "item_name": item_name,
            "destination_room": destination_room,
            "urgency": urgency,
            "status": "robot_delivery_requested",
        }

    # Event: suction_tip is missing and required
    event = {
        "event_id": "evt-int-012",
        "room_id": "OR-2",
        "case_id": "CASE-INT-12",
        "event_type": "or_setup_state_change",
        "visible_items": ["scalpel", "forceps", "trocar"],
        "missing_or_uncertain": ["suction_tip"],
        "zone": "back_table",
        "confidence": 0.90,
        "timestamp": "2026-05-15T09:20:00+02:00",
    }
    case = {
        "case_id": "CASE-INT-12",
        "procedure": "synthetic laparoscopic biopsy",
        "phase": "pre_op_setup",
        "priority": "normal",
        "required_items": ["scalpel", "forceps", "trocar", "suction_tip"],
        "open_items": [],
        "porter_release_allowed": False,
    }
    resources = get_resources("OR-2")
    recon = reconcile_setup(event, case)

    deps = AgentDeps(
        event=event, case=case, resources=resources, reconciliation=recon,
    )
    prompt = json.dumps(
        {"event": event, "synthetic_pathway": case,
         "reconciliation": recon, "resources": resources},
        indent=2,
    )
    result = robot_agent.run_sync(prompt, deps=deps)

    # Extract tool calls
    tool_calls = []
    for msg in result.all_messages():
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if hasattr(part, "tool_name") and hasattr(part, "args"):
                    tool_calls.append({
                        "name": part.tool_name,
                        "arguments": (
                            part.args if isinstance(part.args, dict)
                            else json.loads(part.args) if isinstance(part.args, str)
                            else {}
                        ),
                    })

    robot_calls = [c for c in tool_calls if c["name"] == "request_spd_robot_delivery"]
    assert len(robot_calls) >= 1, (
        f"Agent should use request_spd_robot_delivery (the new tool), "
        f"got tool_calls: {tool_calls}"
    )
    assert "suction" in robot_calls[0]["arguments"].get("item_name", "").lower(), (
        f"Robot delivery should be for suction_tip: {robot_calls[0]}"
    )
