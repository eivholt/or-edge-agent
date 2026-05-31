import json
from pathlib import Path


REQUIRED_EVENT_KEYS = {
    "event_id",
    "room_id",
    "case_id",
    "event_type",
    "timestamp",
}


def test_all_scenarios_are_valid_json():
    for path in Path("scenarios").glob("*.json"):
        data = json.loads(path.read_text())
        assert REQUIRED_EVENT_KEYS.issubset(data.keys()), (
            f"{path.name} missing keys: {REQUIRED_EVENT_KEYS - data.keys()}"
        )
        if "visible_items" in data:
            assert isinstance(data["visible_items"], dict)
            assert all(isinstance(v, int) for v in data["visible_items"].values())