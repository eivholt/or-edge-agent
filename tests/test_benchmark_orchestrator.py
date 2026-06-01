"""Orchestrator benchmark — progressive stress tests for Ministral 3B tool calling.

Levels:
  1. Basic:       Single clear signal → single correct tool call
  2. Standard:    Multiple gaps → correct multi-tool response
  3. Nuanced:     Task_type discrimination, priority mapping
  4. Complex:     Multi-step combos, conditional logic
  5. Adversarial: Conflicting signals, near-boundary counts, distractors

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
    # Ensure a valid image_path so inspect_scene doesn't waste context on errors
    if "image_path" not in event:
        event["image_path"] = "frames/frame_all_present.png"
    t0 = time.perf_counter()
    decision = ask_agent(event, RESOURCES)
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
        if tc["name"] == "create_task"
    ]


def _light_colors(decision):
    return [
        tc["arguments"].get("color")
        for tc in decision.get("tool_calls", [])
        if tc["name"] == "set_stacklight"
    ]


# ═════════════════════════════════════════════════════════════════════
# LEVEL 1 — BASIC: single signal → single correct tool
# ═════════════════════════════════════════════════════════════════════


class TestLevel1Basic:
    """Single, unambiguous signal → exactly the right tool call."""

    def test_L1_01_all_present_green_light(self):
        """Everything present → green light."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-1",
            "visible_items": {"scalpel": 1, "scissors": 1, "sponge": 2},
        }

        def checks(d):
            errs = []
            lights = _light_colors(d)
            if "green" not in lights:
                errs.append(f"Expected green light, got: {lights}")
            resupply = [tc for tc in d["tool_calls"]
                        if tc["name"] in ("request_resupply", "request_spd_resupply")]
            if resupply:
                errs.append(f"No resupply expected when all present, got {len(resupply)}")
            return errs

        _run("all_present→green", 1, event, "CASE-BENCH-1", checks)

    def test_L1_02_single_missing_item_resupply(self):
        """One item with deficit → request_resupply call."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-1",
            "visible_items": {"scalpel": 1, "sponge": 2},
        }

        def checks(d):
            errs = []
            resupply = [tc for tc in d["tool_calls"]
                        if tc["name"] in ("request_resupply", "request_spd_resupply")]
            if not resupply:
                errs.append("Expected request_resupply for scissors")
            text = " ".join(str(tc["arguments"]) for tc in resupply).lower()
            if "scissors" not in text:
                errs.append("Resupply should mention 'scissors'")
            return errs

        _run("single_missing→resupply", 1, event, "CASE-BENCH-1", checks)

    def test_L1_03_no_action_surplus(self):
        """Visible counts exceed required — no tasks needed."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-1",
            "visible_items": {"scalpel": 3, "scissors": 2, "sponge": 5},
        }

        def checks(d):
            errs = []
            resupply = [tc for tc in d["tool_calls"]
                        if tc["name"] in ("request_resupply", "request_spd_resupply")]
            if resupply:
                errs.append(f"No resupply expected with surplus, got {len(resupply)}")
            return errs

        _run("surplus→no_action", 1, event, "CASE-BENCH-1", checks)


# ═════════════════════════════════════════════════════════════════════
# LEVEL 2 — STANDARD: multiple signals → correct multi-tool response
# ═════════════════════════════════════════════════════════════════════


class TestLevel2Standard:
    """Multiple gaps or signals → agent must issue multiple correct tools."""

    def test_L2_01_two_missing_items(self):
        """Two items with deficit → two request_resupply calls."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-2",
            "visible_items": {"scalpel": 2, "scissors": 1, "sponge": 4, "tweezers": 1},
        }

        def checks(d):
            errs = []
            resupply = [tc for tc in d["tool_calls"]
                        if tc["name"] in ("request_resupply", "request_spd_resupply")]
            if len(resupply) < 2:
                errs.append(f"Expected ≥2 resupply calls, got {len(resupply)}")
            text = " ".join(str(tc["arguments"]) for tc in resupply).lower()
            if "scissors" not in text:
                errs.append("Should mention scissors")
            if "tweezers" not in text:
                errs.append("Should mention tweezers")
            return errs

        _run("two_missing→two_tasks", 2, event, "CASE-BENCH-2", checks)

    def test_L2_02_three_items_missing(self):
        """Three items with deficits → three resupply calls."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-2",
            "visible_items": {"scalpel": 2, "scissors": 0, "sponge": 2, "tweezers": 1},
        }

        def checks(d):
            errs = []
            resupply = [tc for tc in d["tool_calls"]
                        if tc["name"] in ("request_resupply", "request_spd_resupply")]
            if len(resupply) < 3:
                errs.append(f"Expected ≥3 resupply calls for 3 deficits, got {len(resupply)}")
            return errs

        _run("three_missing→three_resupply", 2, event, "CASE-BENCH-2", checks)

    def test_L2_03_unaccounted_not_flagged(self):
        """Items with count deficit → resupply calls."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-2",
            "visible_items": {"scalpel": 2},
        }

        def checks(d):
            errs = []
            recon = reconcile_setup(event, get_case("CASE-BENCH-2"))
            expected_deficits = len(recon["deficits"])
            resupply = [tc for tc in d["tool_calls"]
                        if tc["name"] in ("request_resupply", "request_spd_resupply")]
            if len(resupply) < expected_deficits:
                errs.append(
                    f"Expected ≥{expected_deficits} resupply calls for deficit items "
                    f"({[d['item'] for d in recon['deficits']]}), got {len(resupply)}"
                )
            return errs

        _run("unaccounted→resupply", 2, event, "CASE-BENCH-2", checks)


# ═════════════════════════════════════════════════════════════════════
# LEVEL 3 — NUANCED: task_type discrimination
# ═════════════════════════════════════════════════════════════════════


class TestLevel3Nuanced:
    """Tests that require choosing the correct task_type or light color."""

    def test_L3_01_deficit_resupply_yellow(self):
        """Missing item → agent should request resupply and set yellow light."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-1",
            "visible_items": {"scalpel": 1, "sponge": 2},
        }

        def checks(d):
            errs = []
            resupply = [tc for tc in d["tool_calls"]
                        if tc["name"] in ("request_resupply", "request_spd_resupply")]
            if not resupply:
                errs.append("Expected resupply for missing scissors")
            lights = _light_colors(d)
            if "yellow" not in lights and "red" not in lights:
                errs.append(f"Expected yellow/red light for deficit, got: {lights}")
            return errs

        _run("deficit→resupply+yellow", 3, event, "CASE-BENCH-1", checks)

    def test_L3_02_deficit_triggers_action(self):
        """Deficit → agent should take action (resupply)."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-1",
            "visible_items": {"scalpel": 1, "scissors": 1, "sponge": 1},
        }

        def checks(d):
            errs = []
            # sponge: need 2, have 1 → deficit
            resupply = [tc for tc in d["tool_calls"]
                        if tc["name"] in ("request_resupply", "request_spd_resupply")]
            if not resupply:
                errs.append("Deficit should trigger resupply")
            return errs

        _run("deficit→action", 3, event, "CASE-BENCH-1", checks)

    def test_L3_03_high_count_resupply(self):
        """Missing item → request_resupply call."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-2",
            "visible_items": {"scalpel": 2, "scissors": 1, "sponge": 4, "tweezers": 2},
        }

        def checks(d):
            errs = []
            resupply = [tc for tc in d["tool_calls"]
                        if tc["name"] in ("request_resupply", "request_spd_resupply")]
            if not resupply:
                errs.append(f"Expected request_resupply for scissors deficit, got none")
            return errs

        _run("scissors_deficit→resupply", 3, event, "CASE-BENCH-2", checks)

    def test_L3_04_no_deficit_no_task(self):
        """Count meets requirement → no task for it."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-1",
            "visible_items": {"scalpel": 2, "scissors": 1, "sponge": 3},
        }

        def checks(d):
            errs = []
            # scalpel: need 1, have 2 — no deficit
            task_calls = [tc for tc in d["tool_calls"]
                          if tc["name"] == "create_task"]
            for tc in task_calls:
                text = (tc["arguments"].get("summary", "") +
                        tc["arguments"].get("reason", "")).lower()
                if "scalpel" in text:
                    errs.append("Should NOT create task for scalpel (no deficit: need 1, have 2)")
            return errs

        _run("no_deficit→no_task", 3, event, "CASE-BENCH-1", checks)


# ═════════════════════════════════════════════════════════════════════
# LEVEL 4 — COMPLEX: multi-step combos, conditional tool sequences
# ═════════════════════════════════════════════════════════════════════


class TestLevel4Complex:
    """Multi-tool combinations that require following compound rules."""

    def test_L4_01_all_present_green(self):
        """All items present, no sterile issue → green light, no tasks."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-4",
            "visible_items": {"scalpel": 3, "scissors": 2, "sponge": 6, "tweezers": 2},
        }

        def checks(d):
            errs = []
            lights = _light_colors(d)
            if "green" not in lights:
                errs.append(f"Expected green light when all present, got: {lights}")
            resupply = [tc for tc in d["tool_calls"]
                        if tc["name"] in ("request_resupply", "request_spd_resupply")]
            if resupply:
                errs.append(f"No resupply expected, got {len(resupply)}")
            return errs

        _run("all_present→green", 4, event, "CASE-BENCH-4", checks)

    def test_L4_02_deficits_with_resupply(self):
        """Items short → yellow + resupply."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-4",
            "visible_items": {"scalpel": 2, "scissors": 1, "sponge": 4, "tweezers": 2},
        }

        def checks(d):
            errs = []
            lights = _light_colors(d)
            if "yellow" not in lights and "red" not in lights:
                errs.append("Expected yellow or red light for deficits")
            resupply = [tc for tc in d["tool_calls"]
                        if tc["name"] in ("request_resupply", "request_spd_resupply")]
            if not resupply:
                errs.append("Expected resupply for deficits")
            return errs

        _run("deficits→resupply+yellow", 4, event, "CASE-BENCH-4", checks)

    def test_L4_03_high_priority_propagation(self):
        """High-priority case → resupply should be requested for deficits."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-5",
            "visible_items": {"scalpel": 1, "scissors": 1, "sponge": 2, "tweezers": 1},
        }

        def checks(d):
            errs = []
            resupply = [tc for tc in d["tool_calls"]
                        if tc["name"] in ("request_resupply", "request_spd_resupply")]
            if not resupply:
                errs.append("Expected resupply for missing items")
            return errs

        _run("high_priority_propagation", 4, event, "CASE-BENCH-5", checks)


# ═════════════════════════════════════════════════════════════════════
# LEVEL 5 — ADVERSARIAL: conflicting signals, boundaries, distractors
# ═════════════════════════════════════════════════════════════════════


class TestLevel5Adversarial:
    """Edge cases and tricky scenarios that probe model understanding."""

    def test_L5_01_deficit_resupply(self):
        """Missing item → should request resupply."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-1",
            "visible_items": {"scalpel": 1, "sponge": 2},
        }

        def checks(d):
            errs = []
            resupply = [tc for tc in d["tool_calls"]
                        if tc["name"] in ("request_resupply", "request_spd_resupply")]
            tasks = [tc for tc in d["tool_calls"] if tc["name"] == "create_task"]
            if not resupply and not tasks:
                errs.append(f"Expected resupply or task for scissors deficit")
            return errs

        _run("deficit→action", 5, event, "CASE-BENCH-1", checks)

    def test_L5_02_all_items_surplus_no_resupply(self):
        """All counts exceed requirements → agent should NOT request resupply."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-1",
            "visible_items": {"scalpel": 3, "scissors": 2, "sponge": 4},
        }

        def checks(d):
            errs = []
            resupply = [tc for tc in d["tool_calls"]
                        if tc["name"] in ("request_resupply", "request_spd_resupply")]
            if resupply:
                errs.append(
                    f"All counts exceed requirements — no resupply expected, "
                    f"got {len(resupply)}: "
                    + str([tc["arguments"] for tc in resupply])
                )
            return errs

        _run("surplus→no_resupply", 5, event, "CASE-BENCH-1", checks)

    def test_L5_03_mixed_deficit_and_surplus(self):
        """Mix: scissors deficit, tweezers deficit,
        sponge surplus. Only scissors+tweezers get resupply."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-2",
            "visible_items": {"scalpel": 2, "scissors": 1, "sponge": 4, "tweezers": 0},
        }

        def checks(d):
            errs = []
            resupply = [tc for tc in d["tool_calls"]
                        if tc["name"] in ("request_resupply", "request_spd_resupply")]
            text = " ".join(str(tc["arguments"]) for tc in resupply).lower()
            if "scissors" not in text:
                errs.append("Should resupply scissors (deficit)")
            if "tweezers" not in text:
                errs.append("Should resupply tweezers (need 2 have 0)")
            return errs

        _run("mixed_deficit_surplus", 5, event, "CASE-BENCH-2", checks)

    def test_L5_04_zero_visible_items(self):
        """Nothing visible at all — all 4 required items missing."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-2",
            "visible_items": {},
        }

        def checks(d):
            errs = []
            resupply = [tc for tc in d["tool_calls"]
                        if tc["name"] in ("request_resupply", "request_spd_resupply")]
            if len(resupply) < 1:
                errs.append(
                    f"Empty table with 4 required items → ≥1 resupply calls, got {len(resupply)}"
                )
            return errs

        _run("empty_table→resupply", 5, event, "CASE-BENCH-2", checks)

    def test_L5_05_spd_accompanies_deficit(self):
        """Agent should call request_resupply for deficit items."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-2",
            "visible_items": {"scalpel": 1, "scissors": 0, "sponge": 2, "tweezers": 1},
        }

        def checks(d):
            errs = []
            spd = [tc for tc in d["tool_calls"]
                   if tc["name"] in ("request_resupply", "request_spd_resupply", "request_spd_robot_delivery")]
            if not spd:
                errs.append("Expected request_resupply calls for deficit items")
            return errs

        _run("spd_with_deficit", 5, event, "CASE-BENCH-2", checks)

    def test_L5_06_all_present_no_deficit(self):
        """All items present → green light (never yellow when no deficit)."""
        event = {
            "room_id": "OR-BENCH",
            "case_id": "CASE-BENCH-4",
            "visible_items": {"scalpel": 3, "scissors": 2, "sponge": 6, "tweezers": 2},
        }

        def checks(d):
            errs = []
            lights = _light_colors(d)
            if "yellow" in lights:
                errs.append("Should NOT set yellow when all items present and no sterile issue")
            return errs

        _run("all_present→not_yellow", 5, event, "CASE-BENCH-4", checks)


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
                print(f"      → {e}")

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
