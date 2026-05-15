from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ORSceneEvent(BaseModel):
    event_id: str = Field(pattern=r"^evt-\d+$")
    room_id: str
    case_id: str
    event_type: Literal[
        "or_setup_state_change",
        "visually_ready_but_pathway_changed",
        "specimen_container_seen",
        "sterile_zone_ambiguity",
        "instrument_out_of_zone",
        "wrong_case_cart_candidate",
    ]
    visible_items: list[str]
    missing_or_uncertain: list[str]
    zone: str
    confidence: float = Field(ge=0.0, le=1.0)
    image_path: str | None = None
    timestamp: datetime
