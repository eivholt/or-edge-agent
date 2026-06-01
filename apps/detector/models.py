from pydantic import BaseModel, Field


class ORSceneEvent(BaseModel):
    case_id: str
    room_id: str
    image_path: str
    visible_items: dict[str, int] = Field(default_factory=dict)
