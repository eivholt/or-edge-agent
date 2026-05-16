"""Orchestrator benchmark — progressive stress tests for Ministral 3B tool calling.

Levels:
  1. Basic:       Single clear signal → single correct tool call
  2. Standard:    Multiple gaps → correct multi-tool response
  3. Nuanced:     Confidence thresholds, task_type discrimination, priority mapping
  4. Complex:     Multi-step combos (procedure change = 3 tools), conditional logic
  5. Adversarial: Conflicting signals, near-boundary confidence, distractors

Requirements:
  - Synthetic EMR API running on :9000
  - vLLM running Ministral 3B on :8081

Run:  pytest tests/test_benchmark_orchestrator.py -v --tb=short
Slow: pytest tests/test_benchmark_orchestrator.py -v --tb=short -x  (stop on first failure)
"""

import time
from dataclasses import dataclass, field

import pytest

from apps.agent.run_fixture import ask_agent, get_case, get_resources, reconcile_setup
from apps.agent.validation import validate_decision

pytestmark = pytest.mark.llm

RESOURCES = get_resources("OR-BENCH")


# ── Helpers ──────────────────────────────────────────────────────────


@dataclass
class BenchResult:
    name: str
    level: int
    passed: bool
    latency_ms: float
    tool_calls: list
    errors: list
    notes: str = ""


RESULTS: list[BenchResult] = []


def _run(name: str, level: int, event: dict, case_id: str, checks: callable):
    """Run agent, validate, apply checks, record result."""
    case = get_case(case_id)
    t0 = time.perf_counter()
    decision = ask_agent(event, case, RESOURCES)
    latency = (time.perf_counter() - t0) * 1000

    val_errors = validate_decision(decision, event)
    check_errors = checks(decision)
    all_errors = val_errors + check_errors
    passed = len(all_errors) == 0

    result = BenchResult(
        name=name,
        level=level,
        passed=passed,
        latency_ms=latency,
        tool_calls=decision.get("tool_calls", []),
        errors=all_errors,
    )
    RESULTS.append(result)
    assert not all_errors, (
        f"[L{level}] {name} — {len(all_errors)} error(s):\n"
        + "\n".join(f"  • {e}" for e in all_errors)
        + f"\n  Tool calls: {decision.get('tool_calls', [])}"
    )


def _tool_names(decision):
    return [tc["name"] for tc in decision.get("tool_calls", [])]


def _task_types(decision):
    return [
        tc["arguments"].get("task_type")
        for tc in decision.get("tool_calls", [])
        if tc["name"] == "create_synthetic_or_task"
    ]


def _light_colors(decision):
    return [
        tc["arguments"].get("color")
        for tc in decision.get("tool_calls", [])
        if tc["name"] == "set_or_prep_light"
    ]


# ═════════════════════════════════════════════════════════════════════
# LEVEL 1 — BASIC: single signal → single correct tool
# ═════════════════════════════════════════════════════════════════════


class TestLevel1Basic:
    """Single, unambiguous signal → exactly the right tool call."""

    def test_L1_01_all_present_green_light(self):
        """Everything present, high confidence → green light only."""
        event = {
            "event_id": "bench-L1-01", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-1",
            "event_type": "or_setup_state_change",
            "visible_items": {"scalpel": 1, "scissors": 1, "sponge": 2},
            "missing_or_uncertain": [],
            "zone": "back_table", "confidence": 0.95,
            "timestamp": "2026-05-16T08:00:00Z",
        }

        def checks(d):
            errs = []
            lights = _light_colors(d)
            if "green" not in lights:
                errs.append("Expected green light when all present")
            task_calls = [tc for tc in d["tool_calls"] if tc["name"] == "create_synthetic_or_task"]
            if task_calls:
                errs.append(f"No tasks expected when all present, got {len(task_calls)}")
            return errs

        _run("all_present→green", 1, event, "CASE-BENCH-1", checks)

    def test_L1_02_single_missing_item_task(self):
        """One item flagged missing with deficit → missing_supply task."""
        event = {
            "event_id": "bench-L1-02", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-1",
            "event_type": "or_setup_state_change",
            "visible_items": {"scalpel": 1, "sponge": 2},
            "missing_or_uncertain": ["scissors"],
            "zone": "back_table", "confidence": 0.90,
            "timestamp": "2026-05-16T08:01:00Z",
        }

        def checks(d):
            errs = []
            types = _task_types(d)
            if "missing_supply" not in types:
                errs.append("Expected missing_supply task for scissors")
            text = " ".join(
                tc["arguments"].get("summary", "") + tc["arguments"].get("reason", "")
                for tc in d["tool_calls"] if tc["name"] == "create_synthetic_or_task"
            ).lower()
            if "scissors" not in text:
                errs.append("Task should mention 'scissors'")
            return errs

        _run("single_missing→task", 1, event, "CASE-BENCH-1", checks)

    def test_L1_03_no_action_surplus(self):
        """Visible counts exceed required — no tasks needed."""
        event = {
            "event_id": "bench-L1-03", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-1",
            "event_type": "or_setup_state_change",
            "visible_items": {"scalpel": 3, "scissors": 2, "sponge": 5},
            "missing_or_uncertain": [],
            "zone": "back_table", "confidence": 0.93,
            "timestamp": "2026-05-16T08:02:00Z",
        }

        def checks(d):
            errs = []
            task_calls = [tc for tc in d["tool_calls"] if tc["name"] == "create_synthetic_or_task"]
            if task_calls:
                errs.append(f"No tasks expected with surplus, got {len(task_calls)}")
            return errs

        _run("surplus→no_action", 1, event, "CASE-BENCH-1", checks)


# ═════════════════════════════════════════════════════════════════════
# LEVEL 2 — STANDARD: multiple signals → correct multi-tool response
# ═════════════════════════════════════════════════════════════════════


class TestLevel2Standard:
    """Multiple gaps or signals → agent must issue multiple correct tools."""

    def test_L2_01_two_missing_items(self):
        """Two items flagged missing with deficit → two missing_supply tasks."""
        event = {
            "event_id": "bench-L2-01", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-2",
            "event_type": "or_setup_state_change",
            "visible_items": {"scalpel": 2, "scissors": 1, "sponge": 4, "tweezers": 1},
            "missing_or_uncertain": ["scissors", "tweezers"],
            "zone": "back_table", "confidence": 0.88,
            "timestamp": "2026-05-16T08:10:00Z",
        }

        def checks(d):
            errs = []
            supply = [tc for tc in d["tool_calls"]
                      if tc["name"] == "create_synthetic_or_task"
                      and tc["arguments"].get("task_type") == "missing_supply"]
            if len(supply) < 2:
                errs.append(f"Expected ≥2 missing_supply tasks, got {len(supply)}")
            text = " ".join(
                tc["arguments"].get("summary", "") + tc["arguments"].get("reason", "")
                for tc in supply
            ).lower()
            if "scissors" not in text:
                errs.append("Should mention scissors")
            if "tweezers" not in text:
                errs.append("Should mention tweezers")
            return errs

        _run("two_missing→two_tasks", 2, event, "CASE-BENCH-2", checks)

    def test_L2_02_three_items_missing(self):
        """Three items with deficits → three tasks."""
        event = {
            "event_id": "bench-L2-02", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-2",
            "event_type": "or_setup_state_change",
            "visible_items": {"scalpel": 2, "scissors": 0, "sponge": 2, "tweezers": 1},
            "missing_or_uncertain": ["scissors", "sponge", "tweezers"],
            "zone": "back_table", "confidence": 0.85,
            "timestamp": "2026-05-16T08:11:00Z",
        }

        def checks(d):
            errs = []
            supply = [tc for tc in d["tool_calls"]
                      if tc["name"] == "create_synthetic_or_task"]
            if len(supply) < 3:
                errs.append(f"Expected ≥3 tasks for 3 deficits, got {len(supply)}")
            return errs

        _run("three_missing→three_tasks", 2, event, "CASE-BENCH-2", checks)

    def test_L2_03_unaccounted_not_flagged(self):
        """Items not flagged by detector but with count deficit → tasks."""
        event = {
            "event_id": "bench-L2-03", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-2",
            "event_type": "or_setup_state_change",
            "visible_items": {"scalpel": 2},
            "missing_or_uncertain": [],
            "zone": "back_table", "confidence": 0.90,
            "timestamp": "2026-05-16T08:12:00Z",
        }

        def checks(d):
            errs = []
            recon = reconcile_setup(event, get_case("CASE-BENCH-2"))
            expected_unaccounted = len(recon["unaccounted"])
            tasks = [tc for tc in d["tool_calls"]
                     if tc["name"] == "create_synthetic_or_task"]
            if len(tasks) < expected_unaccounted:
                errs.append(
                    f"Expected ≥{expected_unaccounted} tasks for unaccounted items "
                    f"({recon['unaccounted']}), got {len(tasks)}"
                )
            return errs

        _run("unaccounted→tasks", 2, event, "CASE-BENCH-2", checks)


# ═════════════════════════════════════════════════════════════════════
# LEVEL 3 — NUANCED: confidence thresholds, task_type discrimination
# ═════════════════════════════════════════════════════════════════════


class TestLevel3Nuanced:
    """Tests that require understanding confidence thresholds and
    choosing the correct task_type or light color."""

    def test_L3_01_low_confidence_human_review(self):
        """Confidence < 0.80 + missing item → human_review (not missing_supply)."""
        event = {
            "event_id": "bench-L3-01", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-1",
            "event_type": "or_setup_state_change",
            "visible_items": {"scalpel": 1, "sponge": 2},
            "missing_or_uncertain": ["scissors"],
            "zone": "back_table", "confidence": 0.72,
            "timestamp": "2026-05-16T08:20:00Z",
        }

        def checks(d):
            errs = []
            types = _task_types(d)
            if "human_review" not in types:
                errs.append("Expected human_review at low confidence, got: " + str(types))
            # Light is OK if VLM was invoked first (per instructions rule 4)
            vlm_called = any(tc["name"] in ("inspect_scene_local", "inspect_scene_remote")
                             for tc in d.get("tool_calls", []))
            lights = _light_colors(d)
            if lights and not vlm_called:
                errs.append(f"Should not set light below confidence 0.8 without VLM, got: {lights}")
            return errs

        _run("low_conf→human_review", 3, event, "CASE-BENCH-1", checks)

    def test_L3_02_sterile_zone_ambiguity_human_review(self):
        """sterile_zone_ambiguity event → human_review regardless of confidence."""
        event = {
            "event_id": "bench-L3-02", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-1",
            "event_type": "sterile_zone_ambiguity",
            "visible_items": {"scalpel": 1, "scissors": 1, "sponge": 1},
            "missing_or_uncertain": ["sponge"],
            "zone": "back_table", "confidence": 0.85,
            "timestamp": "2026-05-16T08:21:00Z",
        }

        def checks(d):
            errs = []
            types = _task_types(d)
            if "human_review" not in types:
                errs.append(
                    "sterile_zone_ambiguity should trigger human_review, got: " + str(types)
                )
            return errs

        _run("sterile_zone→human_review", 3, event, "CASE-BENCH-1", checks)

    def test_L3_03_high_confidence_missing_supply_not_review(self):
        """High confidence + missing item → missing_supply (not human_review)."""
        event = {
            "event_id": "bench-L3-03", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-2",
            "event_type": "or_setup_state_change",
            "visible_items": {"scalpel": 2, "scissors": 1, "sponge": 4, "tweezers": 2},
            "missing_or_uncertain": ["scissors"],
            "zone": "back_table", "confidence": 0.92,
            "timestamp": "2026-05-16T08:22:00Z",
        }

        def checks(d):
            errs = []
            types = _task_types(d)
            if "missing_supply" not in types:
                errs.append(f"Expected missing_supply at high conf, got: {types}")
            if "human_review" in types:
                errs.append("Should not trigger human_review at confidence 0.92")
            return errs

        _run("high_conf→missing_supply", 3, event, "CASE-BENCH-2", checks)

    def test_L3_04_flagged_but_no_deficit(self):
        """Item flagged uncertain but count meets requirement → no task for it."""
        event = {
            "event_id": "bench-L3-04", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-1",
            "event_type": "or_setup_state_change",
            "visible_items": {"scalpel": 2, "scissors": 1, "sponge": 3},
            "missing_or_uncertain": ["scalpel"],
            "zone": "back_table", "confidence": 0.88,
            "timestamp": "2026-05-16T08:23:00Z",
        }

        def checks(d):
            errs = []
            # scalpel: need 1, have 2 — no deficit despite being flagged
            task_calls = [tc for tc in d["tool_calls"]
                          if tc["name"] == "create_synthetic_or_task"]
            for tc in task_calls:
                text = (tc["arguments"].get("summary", "") +
                        tc["arguments"].get("reason", "")).lower()
                if "scalpel" in text:
                    errs.append("Should NOT create task for scalpel (no deficit: need 1, have 2)")
            return errs

        _run("flagged_no_deficit→no_task", 3, event, "CASE-BENCH-1", checks)


# ═════════════════════════════════════════════════════════════════════
# LEVEL 4 — COMPLEX: multi-step combos, conditional tool sequences
# ═════════════════════════════════════════════════════════════════════


class TestLevel4Complex:
    """Multi-tool combinations that require following compound rules."""

    def test_L4_01_procedure_changed_triple(self):
        """Pathway changed → must call all three:
        procedure_change_review + porter_hold + yellow light."""
        event = {
            "event_id": "bench-L4-01", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-4",
            "event_type": "visually_ready_but_pathway_changed",
            "visible_items": {"scalpel": 3, "scissors": 2, "sponge": 6, "tweezers": 2},
            "missing_or_uncertain": [],
            "zone": "back_table", "confidence": 0.91,
            "timestamp": "2026-05-16T08:30:00Z",
        }

        def checks(d):
            errs = []
            types = _task_types(d)
            if "procedure_change_review" not in types:
                errs.append(f"Missing procedure_change_review, got: {types}")
            if "porter_hold" not in types:
                errs.append(f"Missing porter_hold, got: {types}")
            lights = _light_colors(d)
            if "yellow" not in lights:
                errs.append(f"Missing yellow light, got: {lights}")
            return errs

        _run("procedure_changed→triple", 4, event, "CASE-BENCH-4", checks)

    def test_L4_02_procedure_changed_with_deficits(self):
        """Pathway changed AND items are short → triple + missing_supply tasks."""
        event = {
            "event_id": "bench-L4-02", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-4",
            "event_type": "visually_ready_but_pathway_changed",
            "visible_items": {"scalpel": 2, "scissors": 1, "sponge": 4, "tweezers": 2},
            "missing_or_uncertain": ["scissors", "sponge"],
            "zone": "back_table", "confidence": 0.88,
            "timestamp": "2026-05-16T08:31:00Z",
        }

        def checks(d):
            errs = []
            types = _task_types(d)
            # Procedure change triple
            if "procedure_change_review" not in types:
                errs.append("Missing procedure_change_review")
            if "porter_hold" not in types:
                errs.append("Missing porter_hold")
            lights = _light_colors(d)
            if "yellow" not in lights:
                errs.append("Missing yellow light")
            # Plus supply tasks for deficits (scalpel need 3 have 2, scissors need 2 have 1, sponge need 6 have 4)
            supply = [tc for tc in d["tool_calls"]
                      if tc["name"] == "create_synthetic_or_task"
                      and tc["arguments"].get("task_type") in ("missing_supply", "human_review")]
            if len(supply) < 2:
                errs.append(f"Expected ≥2 supply/review tasks for deficits, got {len(supply)}")
            return errs

        _run("procedure_changed+deficits", 4, event, "CASE-BENCH-4", checks)

    def test_L4_03_high_priority_propagation(self):
        """High-priority case → tasks should inherit high priority."""
        event = {
            "event_id": "bench-L4-03", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-5",
            "event_type": "or_setup_state_change",
            "visible_items": {"scalpel": 1, "scissors": 1, "sponge": 2, "tweezers": 1},
            "missing_or_uncertain": ["scalpel", "scissors", "tweezers"],
            "zone": "back_table", "confidence": 0.87,
            "timestamp": "2026-05-16T08:32:00Z",
        }

        def checks(d):
            errs = []
            tasks = [tc for tc in d["tool_calls"]
                     if tc["name"] == "create_synthetic_or_task"]
            if not tasks:
                errs.append("Expected tasks for missing items")
                return errs
            for tc in tasks:
                prio = tc["arguments"].get("priority")
                if prio != "high":
                    errs.append(
                        f"High-priority case should produce high-priority tasks, "
                        f"got '{prio}' for {tc['arguments'].get('task_type')}"
                    )
            return errs

        _run("high_priority_propagation", 4, event, "CASE-BENCH-5", checks)


# ═════════════════════════════════════════════════════════════════════
# LEVEL 5 — ADVERSARIAL: conflicting signals, boundaries, distractors
# ═════════════════════════════════════════════════════════════════════


class TestLevel5Adversarial:
    """Edge cases and tricky scenarios that probe model understanding."""

    def test_L5_01_exact_boundary_confidence(self):
        """Confidence exactly 0.80 — should use missing_supply (not human_review)."""
        event = {
            "event_id": "bench-L5-01", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-1",
            "event_type": "or_setup_state_change",
            "visible_items": {"scalpel": 1, "sponge": 2},
            "missing_or_uncertain": ["scissors"],
            "zone": "back_table", "confidence": 0.80,
            "timestamp": "2026-05-16T08:40:00Z",
        }

        def checks(d):
            errs = []
            types = _task_types(d)
            if "missing_supply" not in types:
                errs.append(f"At exactly 0.80, should use missing_supply, got: {types}")
            return errs

        _run("boundary_conf_0.80→supply", 5, event, "CASE-BENCH-1", checks)

    def test_L5_02_all_items_flagged_but_surplus(self):
        """Every item flagged uncertain, but all counts exceed requirements.
        Agent should NOT create tasks (counts are fine)."""
        event = {
            "event_id": "bench-L5-02", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-1",
            "event_type": "or_setup_state_change",
            "visible_items": {"scalpel": 3, "scissors": 2, "sponge": 4},
            "missing_or_uncertain": ["scalpel", "scissors", "sponge"],
            "zone": "back_table", "confidence": 0.85,
            "timestamp": "2026-05-16T08:41:00Z",
        }

        def checks(d):
            errs = []
            tasks = [tc for tc in d["tool_calls"]
                     if tc["name"] == "create_synthetic_or_task"]
            if tasks:
                errs.append(
                    f"All counts exceed requirements — no tasks expected, got {len(tasks)}: "
                    + str([tc["arguments"].get("summary") for tc in tasks])
                )
            return errs

        _run("all_flagged_but_surplus→no_tasks", 5, event, "CASE-BENCH-1", checks)

    def test_L5_03_mixed_flagged_and_unaccounted(self):
        """Mix: scissors flagged+deficit, tweezers unaccounted+deficit,
        sponge flagged but no deficit. Only scissors+tweezers get tasks."""
        event = {
            "event_id": "bench-L5-03", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-2",
            "event_type": "or_setup_state_change",
            "visible_items": {"scalpel": 2, "scissors": 1, "sponge": 4, "tweezers": 0},
            "missing_or_uncertain": ["scissors", "sponge"],
            "zone": "back_table", "confidence": 0.90,
            "timestamp": "2026-05-16T08:42:00Z",
        }

        def checks(d):
            errs = []
            tasks = [tc for tc in d["tool_calls"]
                     if tc["name"] == "create_synthetic_or_task"]
            text = " ".join(
                tc["arguments"].get("summary", "") + " " + tc["arguments"].get("reason", "")
                for tc in tasks
            ).lower()
            if "scissors" not in text:
                errs.append("Should create task for scissors (flagged + deficit)")
            if "tweezers" not in text:
                errs.append("Should create task for tweezers (unaccounted, need 2 have 0)")
            if "sponge" in text:
                errs.append("Should NOT create task for sponge (flagged but no deficit)")
            return errs

        _run("mixed_flagged_unaccounted", 5, event, "CASE-BENCH-2", checks)

    def test_L5_04_zero_visible_items(self):
        """Nothing visible at all — all 4 required items unaccounted."""
        event = {
            "event_id": "bench-L5-04", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-2",
            "event_type": "or_setup_state_change",
            "visible_items": {},
            "missing_or_uncertain": [],
            "zone": "back_table", "confidence": 0.82,
            "timestamp": "2026-05-16T08:43:00Z",
        }

        def checks(d):
            errs = []
            tasks = [tc for tc in d["tool_calls"]
                     if tc["name"] == "create_synthetic_or_task"]
            if len(tasks) < 4:
                errs.append(
                    f"Empty table with 4 required items → ≥4 tasks, got {len(tasks)}"
                )
            return errs

        _run("empty_table→all_tasks", 5, event, "CASE-BENCH-2", checks)

    def test_L5_05_no_spd_unless_explicit(self):
        """Agent should NOT call request_spd_resupply spontaneously —
        instructions say 'only when explicitly told to order sterile resupply'."""
        event = {
            "event_id": "bench-L5-05", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-2",
            "event_type": "or_setup_state_change",
            "visible_items": {"scalpel": 1, "scissors": 0, "sponge": 2, "tweezers": 1},
            "missing_or_uncertain": ["scissors", "sponge", "tweezers"],
            "zone": "back_table", "confidence": 0.88,
            "timestamp": "2026-05-16T08:44:00Z",
        }

        def checks(d):
            errs = []
            spd = [tc for tc in d["tool_calls"]
                   if tc["name"] in ("request_spd_resupply", "request_spd_robot_delivery")]
            if spd:
                errs.append(
                    f"Agent should not spontaneously call SPD resupply, got {len(spd)} call(s)"
                )
            return errs

        _run("no_spontaneous_spd", 5, event, "CASE-BENCH-2", checks)

    def test_L5_06_procedure_change_no_green(self):
        """Procedure changed → yellow light (never green)."""
        event = {
            "event_id": "bench-L5-06", "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-4",
            "event_type": "visually_ready_but_pathway_changed",
            "visible_items": {"scalpel": 3, "scissors": 2, "sponge": 6, "tweezers": 2},
            "missing_or_uncertain": [],
            "zone": "back_table", "confidence": 0.93,
            "timestamp": "2026-05-16T08:45:00Z",
        }

        def checks(d):
            errs = []
            lights = _light_colors(d)
            if "green" in lights:
                errs.append("Should NOT set green for procedure_changed (must be yellow)")
            if "yellow" not in lights:
                errs.append(f"Expected yellow light, got: {lights}")
            return errs

        _run("procedure_change→no_green", 5, event, "CASE-BENCH-4", checks)


# ═════════════════════════════════════════════════════════════════════
# REPORT — printed at end of session
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True, scope="session")
def print_benchmark_report():
    """Print a summary table after all tests complete."""
    yield
    if not RESULTS:
        return
    print("\n")
    print("=" * 72)
    print("  ORCHESTRATOR BENCHMARK REPORT — Ministral 3B")
    print("=" * 72)
    print(f"  {'Test':<40} {'L':>2} {'Result':>8} {'Latency':>10}")
    print("-" * 72)

    by_level = {}
    for r in RESULTS:
        by_level.setdefault(r.level, []).append(r)
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"  {r.name:<40} {r.level:>2} {status:>8} {r.latency_ms:>8.0f}ms")
        if not r.passed:
            for e in r.errors[:3]:
                print(f"    → {e}")

    print("-" * 72)
    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r.passed)
    avg_latency = sum(r.latency_ms for r in RESULTS) / total if total else 0
    print(f"  Total: {passed}/{total} passed  |  Avg latency: {avg_latency:.0f}ms")
    print()

    for level in sorted(by_level):
        lvl_results = by_level[level]
        lvl_pass = sum(1 for r in lvl_results if r.passed)
        print(f"  Level {level}: {lvl_pass}/{len(lvl_results)} passed")

    print("=" * 72)
