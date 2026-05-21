"""
Day 16 / Day 20 — Rule-based baseline and PPO evaluation harness.

Rule-based policy:
    stop     → action 0 (0 mph)
    approach → action 1 (20 mph)
    clear    → action 4 (79 mph)

The same harness is reused on Day 20 to compare rule-based vs PPO.
Run standalone to benchmark the rule-based baseline.
"""
import argparse
import numpy as np
from models.rl_environment import RailCorridorEnv

# Observation indices (matches rl_environment.py)
OBS_SIGNAL_STOP = 1
OBS_SIGNAL_APPROACH = 2


def rule_based_action(obs: np.ndarray) -> int:
    """Deterministic 3-aspect signal policy: stop→0 mph, approach→20 mph, clear→79 mph."""
    if obs[OBS_SIGNAL_STOP] > 0.5:
        return 0   # 0 mph
    if obs[OBS_SIGNAL_APPROACH] > 0.5:
        return 1   # 20 mph
    return 4       # 79 mph


def run_episodes(
    policy,
    n_episodes: int = 100,
    seed: int = 0,
    label: str = "policy",
) -> dict:
    env = RailCorridorEnv()
    ep_rewards: list[float] = []
    ep_adherences: list[float] = []
    ep_avg_speeds: list[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        speeds: list[float] = []
        done = False

        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            speeds.append(info["speed_mph"])
            done = terminated or truncated

        ep_rewards.append(ep_reward)
        ep_adherences.append(info["schedule_adherence_sec"])
        ep_avg_speeds.append(float(np.mean(speeds)))

    return {
        "label": label,
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
        "mean_adherence_sec": float(np.mean(ep_adherences)),
        "mean_avg_speed_mph": float(np.mean(ep_avg_speeds)),
    }


def print_results(r: dict) -> None:
    print(f"\n-- {r['label']} ({r['n_episodes']} episodes) --")
    print(f"  Mean reward          : {r['mean_reward']:+.4f}  +/- {r['std_reward']:.4f}")
    print(f"  Mean final adherence : {r['mean_adherence_sec']:+.1f} s")
    print(f"  Mean avg speed       : {r['mean_avg_speed_mph']:.1f} mph")


def run() -> None:
    parser = argparse.ArgumentParser(description="Rail policy evaluator")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained PPO zip (optional)")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    print("=" * 50)
    print("  Rail Signal Optimizer - Policy Evaluation")
    print("=" * 50)

    baseline = run_episodes(rule_based_action, args.episodes, label="Rule-based baseline")
    print_results(baseline)

    if args.model:
        try:
            from stable_baselines3 import PPO as _PPO
            model = _PPO.load(args.model)

            def ppo_action(obs: np.ndarray) -> int:
                action, _ = model.predict(obs, deterministic=True)
                return int(action)

            ppo_res = run_episodes(ppo_action, args.episodes, label="PPO agent")
            print_results(ppo_res)

            delta_r = ppo_res["mean_reward"] - baseline["mean_reward"]
            delta_a = ppo_res["mean_adherence_sec"] - baseline["mean_adherence_sec"]
            print(f"\n  PPO vs baseline: reward {delta_r:+.4f}  adherence {delta_a:+.1f}s")
        except ImportError:
            print("\n[WARN] stable-baselines3 not installed — skipping PPO eval")


if __name__ == "__main__":
    run()
