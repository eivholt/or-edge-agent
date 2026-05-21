import json
from pathlib import Path


REQUIRED_EVENT_KEYS = {
    "event_id",
    "room_id",
    "case_id",
    "event_type",
    "confidence",
    "timestamp",
}


def test_all_scenarios_are_valid_json():
    for path in Path("scenarios").glob("*.json"):
        data = json.loads(path.read_text())
        assert REQUIRED_EVENT_KEYS.issubset(data.keys()), path
        if "visible_items" in data:
            assert isinstance(data["visible_items"], dict)
            assert all(isinstance(v, int) for v in data["visible_items"].values())
        if "missing_or_uncertain" in data:
            assert isinstance(data["missing_or_uncertain"], list)
        assert 0 <= data["confidence"] <= 1