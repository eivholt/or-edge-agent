from apps.agent.validation import validate_decision, should_call_vlm


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
                    "color": "yellow"
                }
            }
        ]
    }
    event = {"confidence": 0.5}
    errors = validate_decision(decision, event)
    assert errors


def test_should_call_vlm_low_confidence():
    event = {
        "confidence": 0.72,
        "event_type": "or_setup_state_change",
        "missing_or_uncertain": [],
    }
    assert should_call_vlm(event) is True


def test_should_call_vlm_sterile_zone_ambiguity():
    event = {
        "confidence": 0.90,
        "event_type": "sterile_zone_ambiguity",
        "missing_or_uncertain": [],
    }
    assert should_call_vlm(event) is True


def test_should_not_call_vlm_high_confidence_normal():
    event = {
        "confidence": 0.95,
        "event_type": "or_setup_state_change",
        "missing_or_uncertain": [],
    }
    assert should_call_vlm(event) is False