from mcp.server.fastmcp import FastMCP

mcp = FastMCP("or-logistics")


@mcp.tool()
def get_available_or_resources(room_id: str) -> dict:
    """
    Return current synthetic OR logistics resources for a room.

    Use this before delegating work to a porter, sterile processing runner,
    indoor robot, remote VLM, local VLM, or human review queue.
    """
    return {
        "room_id": room_id,
        "sterile_processing_robot": {
            "available": True,
            "eta_seconds": 180
        },
        "human_runner": {
            "available": True,
            "eta_seconds": 420
        },
        "porter": {
            "available": True,
            "eta_seconds": 300
        },
        "local_vlm": {
            "available": True,
            "estimated_latency_seconds": 5
        },
        "pc_gpu_vlm": {
            "available": True,
            "estimated_latency_seconds": 3
        }
    }


@mcp.tool()
def request_spd_resupply(item_name: str, room_id: str, urgency: str) -> dict:
    """
    Request synthetic sterile processing delivery for a missing item.

    Use this when an OR setup task requires a missing physical item and
    synthetic policy allows delivery.

    Do not use for items requiring direct human sign-off.
    Urgency must be one of: low, normal, high.
    """
    if urgency not in {"low", "normal", "high"}:
        return {"error": "urgency must be low, normal, or high"}

    return {
        "request_id": f"SPD-{item_name}-{room_id}",
        "item_name": item_name,
        "room_id": room_id,
        "urgency": urgency,
        "status": "requested"
    }


@mcp.tool()
def set_or_prep_light(room_id: str, color: str, duration_seconds: int) -> dict:
    """
    Set the demo OR prep status light.

    Use green for simulated logistics-ready state.
    Use yellow for review-needed state.
    Use red only for high-confidence simulated safety or custody exceptions.
    Never use this as a real clinical alarm.
    Duration must be between 1 and 10 seconds.
    """
    if color not in {"green", "yellow", "red"}:
        return {"error": "color must be green, yellow, or red"}
    if duration_seconds < 1 or duration_seconds > 10:
        return {"error": "duration_seconds must be between 1 and 10"}

    return {
        "room_id": room_id,
        "color": color,
        "duration_seconds": duration_seconds,
        "status": "set"
    }


if __name__ == "__main__":
    mcp.run()