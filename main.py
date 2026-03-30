"""
Air Defense Simulation — main.py
─────────────────────────────────────────────────────────────────
Entry point for all modes.

Usage:
    python main.py simulate                      # Phase 2 — hardcoded router
    python main.py simulate --scenario drone     # specific scenario
    python main.py train                         # Phase 3 — train RL agent
    python main.py train --steps 2000000         # train longer
    python main.py train --difficulty hard       # train on hard scenarios
    python main.py evaluate                      # compare RL vs hardcoded
    python main.py manual                        # manual test with trained AI
    python main.py manual --hardcoded            # manual test with hardcoded
─────────────────────────────────────────────────────────────────
"""

import sys
import os
import argparse
import numpy as np

from entities.threat import Threat, RCS
from entities.ad_layer import create_indian_ad_layers
from core.simulation import SimulationEngine
from scenarios.generate_drone import generate_drone_only
from scenarios.generate_ballistic import generate_ballistic_only
from scenarios.generate_saturation import generate_saturation
from scenarios.wave_generator import build_threat_objects


# ─────────────────────────────────────────
# Shared: build threats from scenario dict
# ─────────────────────────────────────────

RCS_MAP = {
    "LARGE" : RCS.LARGE,
    "MEDIUM": RCS.MEDIUM,
    "SMALL" : RCS.SMALL,
    "MICRO" : RCS.MICRO,
}

def build_threats(scenario: dict) -> list:
    threats = []
    for t in scenario["threats"]:
        pos = t["position"]
        threat = Threat(
            id              = t["id"],
            position        = pos,
            velocity        = t["velocity"],
            rcs             = RCS_MAP[t["rcs"]],
            origin_distance = float(np.linalg.norm(pos[:2])),
        )
        threats.append(threat)
    return threats


def run_scenario(scenario: dict):
    print(f"\n  📂 Scenario : {scenario['name']}")
    print(f"  📝 {scenario['description']}")
    print(f"  🎯 Threats  : {len(scenario['threats'])}")
    threats = build_threats(scenario)
    layers  = create_indian_ad_layers()
    engine  = SimulationEngine(threats=threats, layers=layers,
                               scenario_name=scenario["name"])
    engine.run()


# ─────────────────────────────────────────
# Mode: simulate
# ─────────────────────────────────────────

SCENARIO_MAP = {
    "generate_drone"      : ("scenarios.generate_drone",      "generate_drone_only"),
    "generate_ballistic"  : ("scenarios.generate_ballistic",  "generate_ballistic_only"),
    "generate_saturation" : ("scenarios.generate_saturation", "generate_saturation"),
}

def mode_simulate(args):
    if args.scenario:
        fn = SCENARIO_MAP.get(args.scenario)
        if not fn:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {list(SCENARIO_MAP.keys())}")
            sys.exit(1)
        run_scenario(fn())
    else:
        print("\n🚀 Running all scenarios...\n")
        for name, fn in SCENARIO_MAP.items():
            run_scenario(fn())
            print("\n" + "─"*60 + "\n")
        print("✅ All scenarios complete. Check output/ folder.")


# ─────────────────────────────────────────
# Mode: train
# ─────────────────────────────────────────

def mode_train(args):
    from ai.train import train
    train(
        total_steps = args.steps,
        difficulty  = args.difficulty,
        n_envs      = args.envs,
    )


# ─────────────────────────────────────────
# Mode: evaluate
# ─────────────────────────────────────────

def mode_evaluate(args):
    from ai.evaluate import evaluate
    evaluate(
        n_episodes  = args.episodes,
        difficulty  = args.difficulty,
        model_path  = args.model,
    )


# ─────────────────────────────────────────
# Mode: manual
# ─────────────────────────────────────────

def mode_manual(args):
    from tests.manual_scenario import my_scenario, run_with_rl, run_with_hardcoded
    threats = my_scenario(
        n_ballistic = args.ballistic,
        n_cruise    = args.cruise,
        n_rocket    = args.rocket,
        n_drone     = args.drone,
    )
    total = len(threats)
    print(f"Manual Scenario: {total} threats")
    print(f"     Ballistic:{args.ballistic}  Cruise:{args.cruise}  Rocket:{args.rocket}  Drone:{args.drone}")
    if args.hardcoded:
        run_with_hardcoded(threats)
    else:
        run_with_rl(threats, args.model)


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Air Defense Simulation")
    sub    = parser.add_subparsers(dest="mode")

    # simulate
    p_sim = sub.add_parser("simulate")
    p_sim.add_argument("--scenario", type=str, default=None)

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--steps",      type=int, default=1_000_000)
    p_train.add_argument("--difficulty", type=str, default="random",
                         choices=["easy","medium","hard","random"])
    p_train.add_argument("--envs",       type=int, default=4)

    # evaluate
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--episodes",   type=int, default=100)
    p_eval.add_argument("--difficulty", type=str, default="random")
    p_eval.add_argument("--model",      type=str,
                        default="ai/models/best_model.zip")

    # manual
    p_man = sub.add_parser("manual")
    p_man.add_argument("--model",     type=str, default="ai/models/best_model.zip")
    p_man.add_argument("--hardcoded", action="store_true")
    p_man.add_argument("--ballistic", type=int, default=3)
    p_man.add_argument("--cruise",    type=int, default=5)
    p_man.add_argument("--rocket",    type=int, default=5)
    p_man.add_argument("--drone",     type=int, default=8)

    args = parser.parse_args()

    if   args.mode == "simulate" : mode_simulate(args)
    elif args.mode == "train"    : mode_train(args)
    elif args.mode == "evaluate" : mode_evaluate(args)
    elif args.mode == "manual"   : mode_manual(args)
    else:
        parser.print_help()