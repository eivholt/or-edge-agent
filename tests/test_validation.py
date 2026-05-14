from apps.agent.validation import validate_decision


def test_rejects_unknown_tool():
    decision = {
        "tool_calls": [
            {"name": "delete_patient_record", "arguments": {}}
        ]
    }
    event = {"confidence": 0.9}
    errors = validate_decision(decision, event)
    assert errors


def test_rejects_actuation_with_low_confidence():
    decision = {
        "tool_calls": [
            {
                "name": "set_or_prep_light",
                "arguments": {
                    "room_id": "OR-2",
                    "color": "yellow",
                    "duration_seconds": 5
                }
            }
        ]
    }
    event = {"confidence": 0.5}
    errors = validate_decision(decision, event)
    assert errors