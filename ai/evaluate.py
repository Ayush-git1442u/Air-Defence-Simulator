"""
ai/evaluate.py
─────────────────────────────────────────────────────────────────
Evaluates trained RL agent vs hardcoded router.
Runs N scenarios with each and compares success rates.

Usage:
    python ai/evaluate.py                       # 100 test episodes
    python ai/evaluate.py --episodes 500        # more episodes
    python ai/evaluate.py --difficulty hard     # hard only
─────────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from ai.rl_environment import AirDefenseEnv
from entities.ad_layer import create_indian_ad_layers
from core.simulation import SimulationEngine
from core.router import ThreatRouter
from scenarios.wave_generator import generate_training_episode, build_threat_objects


# ─────────────────────────────────────────
# Evaluate Hardcoded Router
# ─────────────────────────────────────────

def run_hardcoded(scenario: dict) -> dict:
    """Run one scenario with the original hardcoded router."""
    threats = build_threat_objects(scenario)
    layers  = create_indian_ad_layers()
    engine  = SimulationEngine(
        threats       = threats,
        layers        = layers,
        scenario_name = scenario["name"],
        silent        = True,
    )
    engine.run()
    return engine.get_metrics()


# ─────────────────────────────────────────
# Evaluate RL Agent
# ─────────────────────────────────────────

def run_rl_agent(model, scenario: dict) -> dict:
    """Run one scenario with the trained RL agent."""
    env = AirDefenseEnv()
    obs, _ = env.reset()

    # Override with the specific scenario
    env.threats = build_threat_objects(scenario)
    env.layers  = create_indian_ad_layers()
    env.engine  = SimulationEngine(
        threats       = env.threats,
        layers        = env.layers,
        scenario_name = scenario["name"],
        silent        = True,
    )
    obs = env._get_observation()

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return env.engine.get_metrics()


# ─────────────────────────────────────────
# Compare
# ─────────────────────────────────────────

def evaluate(n_episodes: int = 100, difficulty: str = "random",
             model_path: str = "ai/models/best_model.zip"):

    print(f"\n{'═'*60}")
    print(f"  EVALUATION — RL Agent vs Hardcoded Router")
    print(f"  Episodes   : {n_episodes}")
    print(f"  Difficulty : {difficulty}")
    print(f"  Model      : {model_path}")
    print(f"{'═'*60}\n")

    # Load trained model
    if not os.path.exists(model_path):
        print(f"  ❌ Model not found: {model_path}")
        print(f"  Run 'python ai/train.py' first.\n")
        return

    model = PPO.load(model_path)
    print(f"  ✅ Model loaded\n")

    hardcoded_results = []
    rl_results        = []

    for ep in range(n_episodes):
        scenario = generate_training_episode(difficulty)

        # Both use same scenario for fair comparison
        hc = run_hardcoded(scenario)
        rl = run_rl_agent(model, scenario)

        hardcoded_results.append(hc)
        rl_results.append(rl)

        if (ep + 1) % 10 == 0:
            hc_avg = np.mean([r["intercepted"]/r["total"]*100
                              for r in hardcoded_results[-10:]])
            rl_avg = np.mean([r["intercepted"]/r["total"]*100
                              for r in rl_results[-10:]])
            print(f"  Episode {ep+1:>4} | "
                  f"Hardcoded: {hc_avg:>5.1f}% | "
                  f"RL Agent: {rl_avg:>5.1f}%")

    # ── Final summary ─────────────────────────────────────────────────
    def avg_success(results):
        return np.mean([r["intercepted"] / r["total"] * 100
                        for r in results if r["total"] > 0])

    def avg_breached(results):
        return np.mean([r["breached"] for r in results])

    hc_success  = avg_success(hardcoded_results)
    rl_success  = avg_success(rl_results)
    hc_breached = avg_breached(hardcoded_results)
    rl_breached = avg_breached(rl_results)
    improvement = rl_success - hc_success

    print(f"\n{'─'*60}")
    print(f"  📊 FINAL RESULTS ({n_episodes} episodes)")
    print(f"{'─'*60}")
    print(f"  {'Metric':<25} {'Hardcoded':>12} {'RL Agent':>12}")
    print(f"  {'─'*49}")
    print(f"  {'Success Rate':<25} {hc_success:>11.1f}% {rl_success:>11.1f}%")
    print(f"  {'Avg Breaches/Episode':<25} {hc_breached:>12.1f} {rl_breached:>12.1f}")
    print(f"  {'─'*49}")
    print(f"  {'Improvement':<25} {improvement:>+11.1f}%")

    if improvement > 0:
        print(f"\n  ✅ RL Agent is {improvement:.1f}% better than hardcoded router!")
    else:
        print(f"\n  ⚠️  RL Agent needs more training.")

    print()


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",   type=int, default=100)
    parser.add_argument("--difficulty", type=str, default="random",
                        choices=["easy", "medium", "hard", "random"])
    parser.add_argument("--model",      type=str,
                        default="ai/models/best_model.zip")
    args = parser.parse_args()

    evaluate(
        n_episodes  = args.episodes,
        difficulty  = args.difficulty,
        model_path  = args.model,
    )
