import json
from pathlib import Path

import pytest

from apps.detector.models import ORSceneEvent

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def scenario_files():
    return sorted(SCENARIOS_DIR.glob("*.json"))


@pytest.fixture(params=scenario_files(), ids=lambda p: p.stem)
def scenario(request):
    return json.loads(request.param.read_text())


def test_scenario_validates(scenario):
    event = ORSceneEvent.model_validate(scenario)
    assert event.event_id == scenario["event_id"]
    assert event.case_id == scenario["case_id"]
    assert 0.0 <= event.confidence <= 1.0


def test_missing_scissors():
    data = json.loads((SCENARIOS_DIR / "missing_scissors.json").read_text())
    event = ORSceneEvent.model_validate(data)
    assert event.event_type == "or_setup_state_change"
    assert "scissors" in event.missing_or_uncertain
    assert "tweezers" in event.missing_or_uncertain
    # visible_items has reduced counts
    assert event.visible_items["scissors"] == 1
    assert event.visible_items["tweezers"] == 1


def test_rejects_invalid_confidence():
    data = json.loads((SCENARIOS_DIR / "missing_scissors.json").read_text())
    data["confidence"] = 1.5
    with pytest.raises(Exception):
        ORSceneEvent.model_validate(data)


def test_rejects_unknown_event_type():
    data = json.loads((SCENARIOS_DIR / "missing_scissors.json").read_text())
    data["event_type"] = "unknown_type"
    with pytest.raises(Exception):
        ORSceneEvent.model_validate(data)


def test_rejects_bad_event_id():
    data = json.loads((SCENARIOS_DIR / "missing_scissors.json").read_text())
    data["event_id"] = "bad-id"
    with pytest.raises(Exception):
        ORSceneEvent.model_validate(data)
