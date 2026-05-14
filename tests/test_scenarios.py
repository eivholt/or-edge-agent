import json
from pathlib import Path


REQUIRED_EVENT_KEYS = {
    "event_id",
    "room_id",
    "case_id",
    "event_type",
    "visible_items",
    "missing_or_uncertain",
    "confidence",
    "timestamp",
}


def test_all_scenarios_are_valid_json():
    for path in Path("scenarios").glob("*.json"):
        data = json.loads(path.read_text())
        assert REQUIRED_EVENT_KEYS.issubset(data.keys()), path
        assert isinstance(data["visible_items"], list)
        assert isinstance(data["missing_or_uncertain"], list)
        assert 0 <= data["confidence"] <= 1