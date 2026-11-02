from typing import Optional
from pydantic import BaseModel


class WebcamStartRequest(BaseModel):
    camera_index: int = 0
    frame_skip: int = 5   # run detection + recognition every N frames


class WebcamStatusResponse(BaseModel):
    is_running: bool
    message: str
    camera_index: Optional[int] = None
