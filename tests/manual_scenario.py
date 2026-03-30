"""
tests/manual_scenario.py
─────────────────────────────────────────────────────────────────
Manual testing entry point with full control over:
  - Number of each threat type
  - Maneuver assignment per threat
  - Multidirectional spawning

Usage:
    python main.py manual                          # default counts
    python main.py manual --hardcoded              # use hardcoded router
    python main.py manual --ballistic 4 --drone 12 --cruise 6 --rocket 5
─────────────────────────────────────────────────────────────────
"""

import os
import sys
import math
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from entities.threat import Threat, RCS, ManeuverType
from entities.ad_layer import create_indian_ad_layers
from core.simulation import SimulationEngine
from core.router import ThreatRouter


# ─────────────────────────────────────────
# Multidirectional threat factory
# ─────────────────────────────────────────

def _spawn_threat(tid: str, dist: float, alt_range: tuple,
                  speed: float, rcs: RCS,
                  maneuver: ManeuverType = None) -> Threat:
    """
    Spawn a single threat at a random bearing around the origin.
    Velocity always points toward [0,0,0].
    """
    angle = random.uniform(0, 2 * math.pi)
    x = dist * math.cos(angle)
    y = dist * math.sin(angle)
    z = random.uniform(*alt_range)

    # Velocity: toward origin + slight randomness
    dx = -x / dist + random.uniform(-0.03, 0.03)
    dy = -y / dist + random.uniform(-0.03, 0.03)
    dz = random.uniform(-0.3, -0.05) if z > 10 else 0.0

    # Scale horizontal component to speed
    horiz = math.sqrt(dx**2 + dy**2)
    if horiz > 0:
        scale = speed * math.cos(math.atan2(abs(dz), speed))
        dx = dx / horiz * scale
        dy = dy / horiz * scale

    if maneuver is None:
        maneuver = ManeuverType.STRAIGHT

    return Threat(
        id              = tid,
        position        = [x, y, z],
        velocity        = [dx, dy, dz],
        rcs             = rcs,
        origin_distance = float(math.sqrt(x**2 + y**2)),
        maneuver        = maneuver,
    )


# ─────────────────────────────────────────
# YOUR scenario — edit counts here
# ─────────────────────────────────────────

def my_scenario(n_ballistic=3, n_cruise=5, n_rocket=5, n_drone=8) -> list:
    """
    Generates a multidirectional mixed attack scenario.
    Threats come from all directions with varied maneuvers.

    Args:
        n_ballistic : number of ballistic missiles
        n_cruise    : number of cruise missiles
        n_rocket    : number of rockets
        n_drone     : number of drones
    """
    threats = []
    counter = 1

    # ── Ballistic missiles ──────────────────────────────
    # High altitude, fast, mostly straight with some corkscrews
    ballistic_maneuvers = [ManeuverType.STRAIGHT] * 2 + \
                          [ManeuverType.CORKSCREW] * 1
    for i in range(n_ballistic):
        maneuver = random.choice(ballistic_maneuvers)
        t = _spawn_threat(
            tid      = f"B{i+1}",
            dist     = random.uniform(320, 390),
            alt_range= (90, 130),
            speed    = random.uniform(3.8, 4.8),
            rcs      = RCS.LARGE,
            maneuver = maneuver,
        )
        threats.append(t)
        counter += 1

    # ── Cruise missiles ─────────────────────────────────
    # Low altitude, terrain hugging or zigzag
    cruise_maneuvers = [ManeuverType.TERRAIN_HUG] * 2 + \
                       [ManeuverType.ZIGZAG] * 2 + \
                       [ManeuverType.JINK] * 1
    for i in range(n_cruise):
        maneuver = random.choice(cruise_maneuvers)
        t = _spawn_threat(
            tid      = f"C{i+1}",
            dist     = random.uniform(80, 105),
            alt_range= (8, 18),
            speed    = random.uniform(0.9, 1.2),
            rcs      = RCS.MEDIUM,
            maneuver = maneuver,
        )
        threats.append(t)
        counter += 1

    # ── Rockets ─────────────────────────────────────────
    # Short range, zigzag or straight
    rocket_maneuvers = [ManeuverType.STRAIGHT] * 2 + \
                       [ManeuverType.ZIGZAG] * 2 + \
                       [ManeuverType.JINK] * 1
    for i in range(n_rocket):
        maneuver = random.choice(rocket_maneuvers)
        t = _spawn_threat(
            tid      = f"R{i+1}",
            dist     = random.uniform(55, 78),
            alt_range= (5, 12),
            speed    = random.uniform(0.7, 0.95),
            rcs      = RCS.SMALL,
            maneuver = maneuver,
        )
        threats.append(t)
        counter += 1

    # ── Drones ──────────────────────────────────────────
    # Very low, erratic — zigzag and jink
    drone_maneuvers = [ManeuverType.ZIGZAG] * 3 + \
                      [ManeuverType.JINK] * 3 + \
                      [ManeuverType.STRAIGHT] * 1
    for i in range(n_drone):
        maneuver = random.choice(drone_maneuvers)
        t = _spawn_threat(
            tid      = f"D{i+1}",
            dist     = random.uniform(45, 60),
            alt_range= (0.5, 2.0),
            speed    = random.uniform(0.15, 0.25),
            rcs      = RCS.MICRO,
            maneuver = maneuver,
        )
        threats.append(t)
        counter += 1

    return threats


# ─────────────────────────────────────────
# Run with RL Agent
# ─────────────────────────────────────────

def run_with_rl(threats, model_path: str):
    try:
        from stable_baselines3 import PPO
        model  = PPO.load(model_path)
        layers = create_indian_ad_layers()
        engine = SimulationEngine(
            threats       = threats,
            layers        = layers,
            scenario_name = "manual_test_rl",
            silent        = False,
        )
        print(f"\n  🤖 Running with TRAINED RL AGENT\n")
        engine.run()
    except FileNotFoundError:
        print(f"\n  ❌ Model not found: {model_path}")
        print(f"  Run 'python main.py train' first.\n")
        run_with_hardcoded(threats)


# ─────────────────────────────────────────
# Run with Hardcoded Router
# ─────────────────────────────────────────

def run_with_hardcoded(threats):
    layers = create_indian_ad_layers()
    engine = SimulationEngine(
        threats       = threats,
        layers        = layers,
        scenario_name = "manual_test_hardcoded",
        silent        = False,
    )
    print(f"\n  📋 Running with HARDCODED ROUTER\n")
    engine.run()