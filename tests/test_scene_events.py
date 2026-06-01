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
    assert event.case_id == scenario["case_id"]
    assert event.room_id == scenario["room_id"]
    assert event.image_path == scenario["image_path"]


def test_missing_scissors():
    data = json.loads((SCENARIOS_DIR / "missing_scissors.json").read_text())
    event = ORSceneEvent.model_validate(data)
    # visible_items populated at runtime by EI inference, not in JSON
    assert event.visible_items == {}


def test_visible_items_populated_at_runtime():
    """Visible items can be added after loading the scenario."""
    data = json.loads((SCENARIOS_DIR / "missing_scissors.json").read_text())
    data["visible_items"] = {"scalpel": 1, "scissors": 0}
    event = ORSceneEvent.model_validate(data)
    assert event.visible_items == {"scalpel": 1, "scissors": 0}
