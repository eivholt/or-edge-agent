"""Integration tests — full pipeline: EI inference → agent → validation.

Runs actual Edge Impulse model inference on real frame images, feeds
detected object counts into the agent, and validates tool call behavior.

Requires:
  - vLLM running Ministral 3B on :8081
  - Synthetic EMR API running on :9000

Run:  pytest tests/test_integration.py -v --tb=short
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from apps.agent.run_fixture import ask_agent
from apps.agent.reconcile import reconcile
from apps.agent.validation import validate_decision
from apps.detector.inference import detect

pytestmark = pytest.mark.llm

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Expected outcomes per scenario ───────────────────────────────────
# EI detections are live from the model.
#
# CASE-1042 required: scalpel:1, scissors:3, sponge:6, tweezers:2
# CASE-1044 required: scalpel:1, scissors:2, sponge:4, tweezers:1

EXPECTED = {
    "all_present": {
        # CASE-1044: EI sees scalpel:1, scissors:2, sponge:4, tweezers:1 → no deficits
        # VLM verdict=false → green light (no sterile issue, no deficits)
        "light": "green",
        "has_deficits": False,
        "min_resupply": 0,
    },
    "missing_scissors": {
        # CASE-1042: EI sees scalpel:2, sponge:3, tweezers:2, no scissors
        # VLM verdict=false → yellow light from deficits
        "light": "yellow",
        "has_deficits": True,
        "min_resupply": 1,
    },
    "missing_something": {
        # CASE-1042: EI sees scissors:2, scalpel:1, sponge:4, tweezers:1
        # VLM verdict=false → yellow light from deficits
        "light": "yellow",
        "has_deficits": True,
        "min_resupply": 1,
    },
    "instrument_out_of_zone": {
        # CASE-1045: required scalpel:2, scissors:1, sponge:3, tweezers:2
        # EI sees scalpel:2, scissors:1, sponge:3, tweezers:2 → no deficits
        # VLM verdict=true → red light + human_review task
        "light": "red",
        "has_deficits": False,
        "min_resupply": 0,
        "min_any_tasks": 1,
    },
    "sterile_zone_ambiguity": {
        # CASE-1042: EI sees scalpel:1, scissors:2, sponge:3 → multiple deficits
        # VLM verdict=false → yellow light from deficits
        "light": "yellow",
        "has_deficits": True,
        "min_resupply": 0,
    },
}

RUNS_PER_SCENARIO = 3


# ── Helpers ──────────────────────────────────────────────────────────


def _load_scenario(name: str) -> dict:
    return json.loads((SCENARIOS_DIR / f"{name}.json").read_text())


def _run_ei(event: dict) -> dict:
    """Run Edge Impulse inference and populate visible_items from detections."""
    image_path = event.get("image_path")
    if not image_path:
        return event
    frame = Path(image_path)
    if not frame.is_absolute():
        frame = DATA_DIR / image_path
    if frame.exists():
        det_result = detect(frame)
        labels = [d.label for d in det_result.detections]
        event["visible_items"] = dict(Counter(labels))
    return event


def _extract(decision: dict):
    """Extract key fields from an agent decision."""
    tool_calls = decision.get("tool_calls", [])
    tool_names = [tc["name"] for tc in tool_calls]

    lights = [tc for tc in tool_calls if tc["name"] == "set_stacklight"]
    light_color = lights[-1]["arguments"]["color"] if lights else None

    supply_tasks = []  # no longer used
    review_tasks = [
        tc for tc in tool_calls
        if tc["name"] == "create_task"
        and tc["arguments"].get("task_type") == "human_review"
    ]
    resupply_count = sum(1 for tc in tool_calls if tc["name"] == "request_resupply")
    has_inspect = "inspect_scene" in tool_names

    return {
        "tool_names": tool_names,
        "light": light_color,
        "supply_tasks": supply_tasks,
        "review_tasks": review_tasks,
        "resupply_count": resupply_count,
        "has_inspect": has_inspect,
    }


# ── Parametrized integration tests (3x each) ────────────────────────


@pytest.fixture(params=list(EXPECTED.keys()))
def scenario_name(request):
    return request.param


class TestFullPipeline:
    """EI inference → agent reasoning → tool calls → validation."""

    def test_scenario_reliable(self, scenario_name):
        """Each scenario must pass all N runs."""
        expect = EXPECTED[scenario_name]
        failures = []

        for run in range(RUNS_PER_SCENARIO):
            event = _load_scenario(scenario_name)
            event = _run_ei(event)

            decision = ask_agent(event, {"room_id": "OR-2", "cloud_connected": False})

            val_errors = validate_decision(decision, event)
            if val_errors:
                failures.append(f"run {run+1}: validation errors {val_errors}")
                continue

            ext = _extract(decision)
            errs = []

            # Light color (accept single value or list of acceptable values)
            expected_lights = expect["light"]
            if isinstance(expected_lights, str):
                expected_lights = [expected_lights]
            if ext["light"] not in expected_lights:
                errs.append(f"light={ext['light']}, expected one of {expected_lights}")

            # Resupply checks
            if expect["has_deficits"]:
                if ext["resupply_count"] < expect["min_resupply"]:
                    errs.append(
                        f"resupply={ext['resupply_count']}, "
                        f"expected >={expect['min_resupply']}"
                    )
            else:
                if ext["resupply_count"] > 0:
                    errs.append(f"unexpected resupply: {ext['resupply_count']}")

            # Task checks (apply regardless of deficit status)
            min_any = expect.get("min_any_tasks", 0)
            if min_any:
                total_tasks = len(ext["review_tasks"])
                if total_tasks < min_any:
                    errs.append(
                        f"review_tasks={total_tasks}, expected >={min_any}"
                    )

            # inspect_scene should always be called when objects are detected
            if event.get("visible_items") and not ext["has_inspect"]:
                errs.append("inspect_scene not called")

            if errs:
                failures.append(f"run {run+1}: {errs}")

        assert not failures, (
            f"{scenario_name}: {len(failures)}/{RUNS_PER_SCENARIO} failed:\n"
            + "\n".join(failures)
        )


# ── EI detection tests ──────────────────────────────────────────────


class TestDetector:
    """Verify EI model outputs on real frames."""

    KNOWN_CLASSES = {"scalpel", "scissors", "sponge", "tweezers"}

    @pytest.mark.parametrize("name", list(EXPECTED.keys()))
    def test_only_known_classes(self, name):
        event = _load_scenario(name)
        event = _run_ei(event)
        detected = set(event.get("visible_items", {}).keys())
        assert detected.issubset(self.KNOWN_CLASSES), (
            f"Unknown classes: {detected - self.KNOWN_CLASSES}"
        )

    @pytest.mark.parametrize("name", list(EXPECTED.keys()))
    def test_detects_at_least_one(self, name):
        event = _load_scenario(name)
        event = _run_ei(event)
        total = sum(event.get("visible_items", {}).values())
        assert total > 0, "EI detected nothing"


# ── Reconciliation tests ────────────────────────────────────────────


class TestReconciliation:
    """Verify reconcile logic with live EI detections vs EMR cases."""

    @pytest.mark.parametrize("name", list(EXPECTED.keys()))
    def test_deficits_match_expectations(self, name):
        import httpx

        event = _load_scenario(name)
        event = _run_ei(event)

        case = httpx.get(
            f"http://localhost:9000/cases/{event['case_id']}", timeout=10
        ).json()
        deficits = reconcile(event, case)
        expect = EXPECTED[name]

        if expect["has_deficits"]:
            assert len(deficits) > 0, (
                f"Expected deficits but got none. "
                f"visible={event.get('visible_items')}, "
                f"required={case['required_items']}"
            )
        else:
            assert len(deficits) == 0, (
                f"Expected no deficits but got {deficits}. "
                f"visible={event.get('visible_items')}, "
                f"required={case['required_items']}"
            )
