"""
Day 17 — PPO agent training on the RailCorridorEnv.

Local run:
    python -m models.train_ppo

Saves:
    models/artifacts/ppo_rail_policy.zip   (stable-baselines3 format)
    models/artifacts/ppo_rail_policy.onnx  (exported on Day 18)

Colab equivalent — run cells 1-4 of results/experiment_log.ipynb after
uploading the repository; no other changes needed.
"""
import os
import sys
import time
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from models.rl_environment import RailCorridorEnv

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "ppo_rail_policy")


class ProgressCallback(BaseCallback):
    """Prints a one-line progress update every N steps."""

    def __init__(self, log_every: int = 10_000) -> None:
        super().__init__(verbose=0)
        self._log_every = log_every
        self._last_log = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log >= self._log_every:
            ep_info = self.locals.get("infos", [{}])[0]
            adh = ep_info.get("schedule_adherence_sec", 0.0)
            spd = ep_info.get("speed_mph", 0.0)
            print(
                f"  step={self.num_timesteps:>7,}  "
                f"adherence={adh:+.0f}s  speed={spd:.1f} mph"
            )
            self._last_log = self.num_timesteps
        return True


def train(total_timesteps: int = 200_000, n_envs: int = 4) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Vectorized environment for faster training
    env = make_vec_env(RailCorridorEnv, n_envs=n_envs)
    eval_env = Monitor(RailCorridorEnv())

    # PPO with default MlpPolicy (64x64 ReLU — appropriate for 8-dim tabular state)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # entropy bonus: encourages exploration
        verbose=0,
        seed=42,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=ARTIFACTS_DIR,
        log_path=ARTIFACTS_DIR,
        eval_freq=max(5_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )

    print(f"Training PPO for {total_timesteps:,} steps on {n_envs} envs...")
    t0 = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[ProgressCallback(log_every=20_000), eval_cb],
        progress_bar=False,
    )

    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.0f}s")

    model.save(MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}.zip")


def run() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on RailCorridorEnv")
    parser.add_argument("--steps", type=int, default=200_000,
                        help="Total training timesteps (default 200k, ~3 min local)")
    parser.add_argument("--envs", type=int, default=4,
                        help="Number of parallel envs (default 4)")
    args = parser.parse_args()

    print("=" * 50)
    print("  Rail Signal Optimizer - PPO Training")
    print("=" * 50)
    train(total_timesteps=args.steps, n_envs=args.envs)


if __name__ == "__main__":
    run()
