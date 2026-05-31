import logfire

ALLOWED_TOOLS = {
    "get_case",
    "check_supplies",
    "create_task",
    "request_resupply",
    "set_stacklight",
    "inspect_scene",
}

ALLOWED_TASK_TYPES = {
    "human_review",
}

ALLOWED_LIGHTS = {"green", "yellow", "red"}


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

        if name == "create_task":
            if args.get("task_type") not in ALLOWED_TASK_TYPES:
                errors.append(f"tool_calls[{i}] invalid task_type")

        if name == "request_resupply":
            if not args.get("item_name"):
                errors.append(f"tool_calls[{i}] missing item_name")

        if name == "set_stacklight":
            if args.get("color") not in ALLOWED_LIGHTS:
                errors.append(f"tool_calls[{i}] invalid light color")

    return errors