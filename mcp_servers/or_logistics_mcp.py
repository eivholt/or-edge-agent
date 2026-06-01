import logfire
from mcp.server.fastmcp import FastMCP

logfire.configure(service_name="mcp-or-logistics")

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
def request_resupply(item_name: str, room_id: str, urgency: str) -> dict:
    """
    Request sterile processing delivery for a missing item.

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
def set_stacklight(room_id: str, color: str) -> dict:
    """
    Set the OR prep status stacklight.

    Green = logistics-ready, Yellow = review-needed.
    """
    if color not in {"green", "yellow"}:
        return {"error": "color must be green or yellow"}

    return {
        "room_id": room_id,
        "color": color,
        "status": "set"
    }


@mcp.tool()
def reconcile_setup(
    required_items: dict[str, int],
    visible_items: dict[str, int],
) -> dict:
    """
    Compare detected item counts against the surgical pathway's required counts.

    Returns which required items have a deficit.
    Use this BEFORE deciding whether to create tasks or request resupply.

    - deficits: items required but with insufficient visible count.
    - all_present: true if every required item meets its count.
    """
    deficits = []

    for item, need in sorted(required_items.items()):
        have = visible_items.get(item, 0)
        if have < need:
            deficits.append(item)

    return {
        "deficits": deficits,
        "all_present": len(deficits) == 0,
    }


@mcp.tool()
def request_spd_robot_delivery(item_name: str, destination_room: str, urgency: str) -> dict:
    """
    Request delivery of a sterile supply or instrument set by indoor robot.

    Use this when a missing physical item blocks OR setup and robot delivery
    is available for the current synthetic pathway.

    Do not use for items requiring direct human sign-off.
    Urgency must be one of: low, normal, high.
    """
    if urgency not in {"low", "normal", "high"}:
        return {"error": "urgency must be low, normal, or high"}

    return {
        "delivery_id": f"ROBOT-{item_name}-{destination_room}",
        "item_name": item_name,
        "destination_room": destination_room,
        "urgency": urgency,
        "eta_seconds": 180,
        "status": "robot_delivery_requested"
    }

if __name__ == "__main__":
    mcp.run()