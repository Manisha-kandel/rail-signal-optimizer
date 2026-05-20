"""
Day 15 — Custom Gymnasium environment for rail speed advisory MDP.

Single-train agent navigates the 20-block Harrisburg Subdivision corridor.
Four background trains are updated each step using the rule-based physics
from train_generator.py. The agent controls train T_001.

Observation (8 floats, Box[-1, 1]):
    0  speed_norm         speed / 79.0
    1  signal_stop        1.0 if stop else 0.0
    2  signal_approach    1.0 if approach else 0.0
    3  signal_clear       1.0 if clear else 0.0
    4  blocks_ahead_norm  occupied blocks ahead (0-2) / 2.0
    5  adherence_norm     clip(adherence_sec / 300, -1, 1)
    6  grade_norm         clip(grade_pct / 3, -1, 1)
    7  progress_norm      block_idx / 19.0

Action (Discrete 5):
    0 → 0 mph   1 → 20 mph   2 → 40 mph   3 → 60 mph   4 → 79 mph

Reward per step:
    primary  : -abs(schedule_adherence_sec) / 300
    secondary: -0.3 * deceleration / 79   (braking energy cost)
    bonus    : +0.1 * speed / 79          (efficiency on clear signal)
    clipped to [-2.0, 1.0]
"""
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

N_BLOCKS = 20
MAX_SPEED = 79.0
SPEED_TARGETS = [0.0, 20.0, 40.0, 60.0, 79.0]
MAX_STEPS = 200
BRAKING_ALPHA = 0.3


class _TrainState:
    def __init__(self, train_id: str, block_idx: int, speed: float = 50.0) -> None:
        self.train_id = train_id
        self.block_idx = block_idx
        self.speed_mph = speed
        self.schedule_adherence_sec = 0.0


def _compute_signal(block_idx: int, occupied: set[int]) -> str:
    if (block_idx + 1) % N_BLOCKS in occupied:
        return "stop"
    if (block_idx + 2) % N_BLOCKS in occupied:
        return "approach"
    return "clear"


class RailCorridorEnv(gym.Env):
    """Harrisburg Sub 20-block single-train speed advisory environment."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.full(8, -1.0, dtype=np.float32),
            high=np.full(8, 1.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(SPEED_TARGETS))

        self._agent: _TrainState = _TrainState("T_001", 0)
        self._bg: list[_TrainState] = []
        self._grade: float = 0.0
        self._step: int = 0

    # ── Gymnasium API ──────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._agent = _TrainState("T_001", 0, speed=random.uniform(45, 70))
        self._bg = [
            _TrainState(f"T_00{i + 2}", i * 4, speed=random.uniform(45, 70))
            for i in range(1, 5)
        ]
        self._grade = round(random.uniform(-1.5, 1.5), 2)
        self._step = 0
        return self._obs(), {}

    def step(self, action: int):
        a = self._agent
        target = SPEED_TARGETS[int(action)]
        prev_speed = a.speed_mph

        # All occupied indices (agent + background)
        occupied = {bg.block_idx for bg in self._bg} | {a.block_idx}
        signal = _compute_signal(a.block_idx, occupied)

        # --- Agent physics: move toward target with inertia ---
        if target < a.speed_mph:
            a.speed_mph = max(target, a.speed_mph - random.uniform(6, 14))
        else:
            a.speed_mph = min(target, a.speed_mph + random.uniform(1, 4))

        # Signal safety override (environment enforces, agent can't ignore)
        if signal == "stop":
            a.speed_mph = max(0.0, a.speed_mph - random.uniform(8, 20))
        elif signal == "approach":
            a.speed_mph = min(a.speed_mph, 40.0)

        a.speed_mph = float(np.clip(a.speed_mph, 0.0, MAX_SPEED))

        # --- Block advancement ---
        if signal != "stop" and a.speed_mph > 5.0:
            next_blk = (a.block_idx + 1) % N_BLOCKS
            if next_blk not in occupied and random.random() > 0.5:
                a.block_idx = next_blk

        # --- Schedule adherence ---
        grade_drag = -self._grade * 2.0
        if signal == "clear" and a.speed_mph > 60:
            a.schedule_adherence_sec += random.uniform(-2.0, 4.0)
        else:
            a.schedule_adherence_sec += random.uniform(-8.0, 1.0) + grade_drag

        # Occasional grade change
        if random.random() < 0.1:
            self._grade = round(random.uniform(-1.5, 1.5), 2)

        # --- Background trains (rule-based) ---
        bg_occupied = {bg.block_idx for bg in self._bg} | {a.block_idx}
        for bg in self._bg:
            bg_sig = _compute_signal(bg.block_idx, bg_occupied)
            if bg_sig == "stop":
                bg.speed_mph = max(0.0, bg.speed_mph - random.uniform(8, 15))
            elif bg_sig == "approach":
                bg.speed_mph = max(20.0, bg.speed_mph - random.uniform(2, 6))
            else:
                bg.speed_mph = min(MAX_SPEED, bg.speed_mph + random.uniform(1, 4))
                nxt = (bg.block_idx + 1) % N_BLOCKS
                if nxt not in bg_occupied and random.random() > 0.5:
                    bg.block_idx = nxt
            bg.schedule_adherence_sec += random.uniform(-8, 3)

        # --- Reward ---
        adherence_r = -abs(a.schedule_adherence_sec) / 300.0
        braking_r = -BRAKING_ALPHA * max(0.0, prev_speed - a.speed_mph) / MAX_SPEED
        speed_bonus = 0.1 * (a.speed_mph / MAX_SPEED) if signal == "clear" else 0.0
        reward = float(np.clip(adherence_r + braking_r + speed_bonus, -2.0, 1.0))

        self._step += 1
        truncated = self._step >= MAX_STEPS
        info = {
            "step": self._step,
            "speed_mph": round(a.speed_mph, 1),
            "signal": signal,
            "block_idx": a.block_idx,
            "schedule_adherence_sec": round(a.schedule_adherence_sec, 1),
            "grade_pct": self._grade,
        }
        return self._obs(), reward, False, truncated, info

    def render(self) -> None:
        a = self._agent
        occupied = {bg.block_idx for bg in self._bg} | {a.block_idx}
        sig = _compute_signal(a.block_idx, occupied)
        print(
            f"step={self._step:3d}  blk={a.block_idx:2d}  {sig:8s}"
            f"  {a.speed_mph:5.1f} mph  adherence={a.schedule_adherence_sec:+.0f}s"
        )

    # ── Internal ───────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        a = self._agent
        occupied = {bg.block_idx for bg in self._bg} | {a.block_idx}
        sig = _compute_signal(a.block_idx, occupied)

        blocks_ahead = sum(
            1 for i in range(1, 3)
            if (a.block_idx + i) % N_BLOCKS in occupied
        )
        return np.array(
            [
                a.speed_mph / MAX_SPEED,
                1.0 if sig == "stop" else 0.0,
                1.0 if sig == "approach" else 0.0,
                1.0 if sig == "clear" else 0.0,
                blocks_ahead / 2.0,
                float(np.clip(a.schedule_adherence_sec / 300.0, -1.0, 1.0)),
                float(np.clip(self._grade / 3.0, -1.0, 1.0)),
                a.block_idx / (N_BLOCKS - 1),
            ],
            dtype=np.float32,
        )
