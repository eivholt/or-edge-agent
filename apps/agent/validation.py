import logfire

ALLOWED_TOOLS = {
    "get_surgical_pathway",
    "reconcile_instruments",
    "create_or_task",
    "request_spd_resupply",
    "request_spd_robot_delivery",
    "set_or_prep_light",
    "inspect_scene",
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

ALLOWED_LIGHTS = {"green", "yellow", "red"}

VLM_TRIGGER_EVENT_TYPES = {
    "instrument_out_of_zone",
    "sterile_zone_ambiguity",
    "wrong_case_cart_candidate",
    "specimen_ready_check",
    "room_turnover_check",
    "ppe_compliance_check",
}


@logfire.instrument("should_call_vlm event_type={event[event_type]}")
def should_call_vlm(event: dict) -> bool:
    if event["event_type"] in VLM_TRIGGER_EVENT_TYPES:
        return True
    if event.get("missing_or_uncertain"):
        return True
    return False


@logfire.instrument("validate_decision")
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

        if name == "create_or_task":
            if args.get("task_type") not in ALLOWED_TASK_TYPES:
                errors.append(f"tool_calls[{i}] invalid task_type")

        if name == "request_spd_resupply":
            if not args.get("item_name"):
                errors.append(f"tool_calls[{i}] missing item_name")

        if name == "request_spd_robot_delivery":
            if not args.get("item_name"):
                errors.append(f"tool_calls[{i}] missing item_name")

        if name == "set_or_prep_light":
            if args.get("color") not in ALLOWED_LIGHTS:
                errors.append(f"tool_calls[{i}] invalid light color")

    return errors