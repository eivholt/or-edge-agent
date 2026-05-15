ALLOWED_TOOLS = {
    "create_synthetic_or_task",
    "request_spd_resupply",
    "request_spd_robot_delivery",
    "set_or_prep_light",
    "inspect_scene"
}

ALLOWED_TASK_TYPES = {
    "missing_supply",
    "human_review",
    "porter_hold",
    "porter_release",
    "specimen_handoff",
    "wrong_case_cart",
    "procedure_change_review"
}

ALLOWED_PRIORITIES = {"low", "normal", "high"}
ALLOWED_LIGHTS = {"green", "yellow", "red"}

VLM_TRIGGER_EVENT_TYPES = {
    "instrument_out_of_zone",
    "sterile_zone_ambiguity",
    "wrong_case_cart_candidate",
}


def should_call_vlm(event: dict) -> bool:
    if event["confidence"] < 0.80:
        return True
    if event["event_type"] in VLM_TRIGGER_EVENT_TYPES:
        return True
    if event["missing_or_uncertain"]:
        return True
    return False


def validate_decision(decision: dict, event: dict) -> list[str]:
    errors: list[str] = []

    if "tool_calls" not in decision or not isinstance(decision["tool_calls"], list):
        return ["decision.tool_calls must be a list"]

    for i, call in enumerate(decision["tool_calls"]):
        name = call.get("name")
        args = call.get("arguments", {})

        if name not in ALLOWED_TOOLS:
            errors.append(f"tool_calls[{i}].name not allowed: {name}")
            continue

        if name == "create_synthetic_or_task":
            if args.get("task_type") not in ALLOWED_TASK_TYPES:
                errors.append(f"tool_calls[{i}] invalid task_type")
            if args.get("priority") not in ALLOWED_PRIORITIES:
                errors.append(f"tool_calls[{i}] invalid priority")

        if name == "request_spd_resupply":
            if args.get("urgency") not in ALLOWED_PRIORITIES:
                errors.append(f"tool_calls[{i}] invalid urgency")
            if not args.get("item_name"):
                errors.append(f"tool_calls[{i}] missing item_name")

        if name == "request_spd_robot_delivery":
            if args.get("urgency") not in ALLOWED_PRIORITIES:
                errors.append(f"tool_calls[{i}] invalid urgency")
            if not args.get("item_name"):
                errors.append(f"tool_calls[{i}] missing item_name")

        if name == "set_or_prep_light":
            if args.get("color") not in ALLOWED_LIGHTS:
                errors.append(f"tool_calls[{i}] invalid light color")
            duration = args.get("duration_seconds")
            if not isinstance(duration, int) or not (1 <= duration <= 10):
                errors.append(f"tool_calls[{i}] invalid duration_seconds")
            if event.get("confidence", 0) < 0.8:
                errors.append(f"tool_calls[{i}] cannot actuate below confidence 0.8")

    return errors