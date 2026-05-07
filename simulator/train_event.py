from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid

class BlockAhead(BaseModel):
    block_id: str
    occupying_train: str
    estimated_clearance_sec: float

class TrainEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    train_id: str
    block_id: str
    block_idx: int
    subdivision: str = "Harrisburg_Sub"
    current_speed_mph: float
    max_authorized_speed_mph: float = 79.0
    gross_tonnage: int = 18400
    signal_aspect_ahead: str  # "clear", "approach", "stop"
    blocks_ahead_occupied: List[BlockAhead] = []
    schedule_adherence_sec: float  # negative = late
    track_grade_pct: float = 0.0
    
    def to_kafka_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")