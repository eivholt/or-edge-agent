from apps.agent.validation import validate_decision


def test_rejects_unknown_tool():
    decision = {
        "tool_calls": [
            {"name": "delete_patient_record", "arguments": {}}
        ]
    }
    event = {}
    errors = validate_decision(decision, event)
    assert errors


def test_accepts_allowed_tools():
    decision = {
        "tool_calls": [
            {"name": "get_case", "arguments": {"case_id": "CASE-1042"}},
            {"name": "check_supplies", "arguments": {}},
            {"name": "set_stacklight", "arguments": {"color": "green"}},
        ]
    }
    event = {}
    errors = validate_decision(decision, event)
    assert not errors


def test_rejects_invalid_task_type():
    decision = {
        "tool_calls": [
            {"name": "create_task", "arguments": {"task_type": "invalid_type"}}
        ]
    }
    event = {}
    errors = validate_decision(decision, event)
    assert errors


def test_rejects_invalid_light_color():
    decision = {
        "tool_calls": [
            {"name": "set_stacklight", "arguments": {"color": "blue"}}
        ]
    }
    event = {}
    errors = validate_decision(decision, event)
    assert errors


def test_rejects_resupply_without_item():
    decision = {
        "tool_calls": [
            {"name": "request_resupply", "arguments": {}}
        ]
    }
    event = {}
    errors = validate_decision(decision, event)
    assert errors