import time
import random
from simulator.train_event import TrainEvent, BlockAhead

BLOCKS = [f"B_{i:03d}" for i in range(1, 21)]
TRAIN_IDS = ["T_001", "T_002", "T_003", "T_004", "T_005"]

class TrainState:
    def __init__(self, train_id: str, block_idx: int):
        self.train_id = train_id
        self.block_idx = block_idx
        self.speed_mph = random.uniform(45, 70)
        self.schedule_adherence_sec = 0.0

def compute_signal(block_idx: int, all_states: list[TrainState]) -> str:
    occupied_indices = {s.block_idx for s in all_states}
    if (block_idx + 1) in occupied_indices:
        return "stop" if block_idx in occupied_indices else "approach"
    return "clear"

def simulate_step(states: list[TrainState]) -> list[TrainEvent]:
    occupied = {s.block_idx: s.train_id for s in states}
    events = []

    for s in states:
        signal = compute_signal(s.block_idx, states)

        # Physics
        if signal == "stop":
            s.speed_mph = max(0, s.speed_mph - random.uniform(8, 15))
        elif signal == "approach":
            s.speed_mph = max(20, s.speed_mph - random.uniform(2, 6))
        else:
            s.speed_mph = min(79, s.speed_mph + random.uniform(1, 4))
            if random.random() > 0.5:
                s.block_idx = min(s.block_idx + 1, len(BLOCKS) - 1)

        s.schedule_adherence_sec += random.uniform(-8, 3)

        # Blocks ahead
        blocks_ahead = []
        for lookahead in range(1, 3):
            ahead_idx = s.block_idx + lookahead
            if ahead_idx in occupied and occupied[ahead_idx] != s.train_id:
                blocks_ahead.append(BlockAhead(
                    block_id=BLOCKS[ahead_idx],
                    occupying_train=occupied[ahead_idx],
                    estimated_clearance_sec=random.uniform(60, 300)
                ))

        event = TrainEvent(
            train_id=s.train_id,
            block_id=BLOCKS[s.block_idx],
            block_idx=s.block_idx,
            current_speed_mph=round(s.speed_mph, 1),
            signal_aspect_ahead=signal,
            blocks_ahead_occupied=blocks_ahead,
            schedule_adherence_sec=round(s.schedule_adherence_sec, 1),
            track_grade_pct=round(random.uniform(-1.5, 1.5), 2)
        )
        events.append(event)

    return events

if __name__ == "__main__":
    states = [
        TrainState(tid, i * 3)
        for i, tid in enumerate(TRAIN_IDS)
    ]
    print("Generator running. Ctrl+C to stop.\n")
    while True:
        events = simulate_step(states)
        for e in events:
            print(e.model_dump_json(indent=2))
        time.sleep(2)