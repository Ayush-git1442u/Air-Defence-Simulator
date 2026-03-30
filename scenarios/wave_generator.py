"""
scenarios/wave_generator.py
─────────────────────────────────────────────────────────────────
Generates randomized training episodes.
Fixes:
  - Multidirectional spawning (threats come from all angles)
  - Maneuver assignment (zigzag, corkscrew, jink, terrain_hug)
  - Controllable threat counts
─────────────────────────────────────────────────────────────────
"""

import random
import math
import numpy as np
from entities.threat import Threat, RCS, ThreatType, ManeuverType


RCS_MAP = {
    "LARGE" : RCS.LARGE,
    "MEDIUM": RCS.MEDIUM,
    "SMALL" : RCS.SMALL,
    "MICRO" : RCS.MICRO,
}

# Maneuver probability per threat type
# [STRAIGHT, ZIGZAG, CORKSCREW, JINK, TERRAIN_HUG]
MANEUVER_WEIGHTS = {
    "BALLISTIC" : [0.6, 0.1, 0.1, 0.1, 0.1],  # mostly straight, high alt
    "CRUISE"    : [0.3, 0.2, 0.1, 0.2, 0.2],  # often terrain hug or jink
    "ROCKET"    : [0.5, 0.2, 0.1, 0.2, 0.0],  # straight or zigzag
    "DRONE"     : [0.3, 0.3, 0.1, 0.3, 0.0],  # erratic, zigzag, jink
}

MANEUVER_LIST = [
    ManeuverType.STRAIGHT,
    ManeuverType.ZIGZAG,
    ManeuverType.CORKSCREW,
    ManeuverType.JINK,
    ManeuverType.TERRAIN_HUG,
]


def _pick_maneuver(threat_type: str) -> ManeuverType:
    weights = MANEUVER_WEIGHTS.get(threat_type, [1,0,0,0,0])
    return random.choices(MANEUVER_LIST, weights=weights, k=1)[0]


def _make_threat_multidirectional(threat_type: str, threat_id: str,
                                   dist_range: tuple, alt_range: tuple,
                                   speed_range: tuple, rcs: str) -> dict:
    """
    Spawn threat at a random bearing (0-360°) around origin.
    Velocity always points toward origin — multidirectional approach.
    """
    # Random bearing angle
    angle = random.uniform(0, 2 * math.pi)

    # Random distance from origin
    dist = random.uniform(*dist_range)

    # Position on the circle at this angle and distance
    x = dist * math.cos(angle)
    y = dist * math.sin(angle)
    z = random.uniform(*alt_range)

    # Speed scalar
    speed = random.uniform(*speed_range)

    # Velocity points toward origin [0,0,0] with slight randomness
    dx = -x / dist + random.uniform(-0.05, 0.05)
    dy = -y / dist + random.uniform(-0.05, 0.05)
    dz = random.uniform(-0.3, -0.05) if z > 10 else 0.0

    # Normalize horizontal and scale to speed
    horiz_mag = math.sqrt(dx**2 + dy**2)
    if horiz_mag > 0:
        dx = dx / horiz_mag * speed * math.cos(math.atan2(abs(dz), speed))
        dy = dy / horiz_mag * speed * math.cos(math.atan2(abs(dz), speed))

    maneuver = _pick_maneuver(threat_type)

    return {
        "id"      : threat_id,
        "position": [x, y, z],
        "velocity": [dx, dy, dz],
        "rcs"     : rcs,
        "type"    : threat_type,
        "maneuver": maneuver.value,
    }


# ─────────────────────────────────────────
# Threat type profiles
# ─────────────────────────────────────────

PROFILES = {
    "BALLISTIC": dict(dist_range=(300,390), alt_range=(90,140),  speed_range=(3.8,4.8), rcs="LARGE"),
    "CRUISE"   : dict(dist_range=(80, 110), alt_range=(8, 20),   speed_range=(0.8,1.2), rcs="MEDIUM"),
    "ROCKET"   : dict(dist_range=(55, 80),  alt_range=(5, 12),   speed_range=(0.7,1.0), rcs="SMALL"),
    "DRONE"    : dict(dist_range=(45, 60),  alt_range=(0.5,2.0), speed_range=(0.15,0.25),rcs="MICRO"),
}


def generate_training_episode(difficulty: str = "random") -> dict:
    if difficulty == "random":
        difficulty = random.choice(["easy","medium","hard"])

    threats = []
    counter = 1

    if difficulty == "easy":
        threat_type = random.choice(["BALLISTIC","CRUISE","ROCKET","DRONE"])
        n = random.randint(3, 8)
        p = PROFILES[threat_type]
        for i in range(n):
            threats.append(_make_threat_multidirectional(
                threat_type, f"T{counter}", **p))
            counter += 1

    elif difficulty == "medium":
        types = random.sample(["BALLISTIC","CRUISE","ROCKET","DRONE"], 2)
        for tt in types:
            n = random.randint(3, 8)
            p = PROFILES[tt]
            for i in range(n):
                threats.append(_make_threat_multidirectional(tt, f"T{counter}", **p))
                counter += 1

    elif difficulty == "hard":
        counts = {
            "BALLISTIC": random.randint(5, 15),
            "CRUISE"   : random.randint(10, 30),
            "ROCKET"   : random.randint(10, 30),
            "DRONE"    : random.randint(15, 50),
        }
        for tt, n in counts.items():
            p = PROFILES[tt]
            for i in range(n):
                threats.append(_make_threat_multidirectional(tt, f"T{counter}", **p))
                counter += 1
        random.shuffle(threats)

    return {
        "name"       : f"training_episode_{difficulty}",
        "description": f"Auto-generated {difficulty} — {len(threats)} threats, multidirectional",
        "difficulty" : difficulty,
        "threats"    : threats,
    }


def build_threat_objects(scenario: dict) -> list:
    """Convert scenario dict → Threat objects with maneuvers."""
    threats = []
    for t in scenario["threats"]:
        pos = t["position"]
        maneuver = ManeuverType(t.get("maneuver", "STRAIGHT"))
        threat = Threat(
            id              = t["id"],
            position        = pos,
            velocity        = t["velocity"],
            rcs             = RCS_MAP[t["rcs"]],
            origin_distance = float(np.linalg.norm(pos[:2])),
            maneuver        = maneuver,
        )
        threats.append(threat)
    return threats