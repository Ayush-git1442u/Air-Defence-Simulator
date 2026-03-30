import numpy as np
import math
import random
from enum import Enum


# ─────────────────────────────────────────
# Enums
# ─────────────────────────────────────────

class ThreatType(Enum):
    BALLISTIC = "BALLISTIC"
    CRUISE    = "CRUISE"
    ROCKET    = "ROCKET"
    DRONE     = "DRONE"
    UNKNOWN   = "UNKNOWN"


class RCS(Enum):
    LARGE  = "LARGE"
    MEDIUM = "MEDIUM"
    SMALL  = "SMALL"
    MICRO  = "MICRO"


class ThreatStatus(Enum):
    INBOUND     = "INBOUND"
    INTERCEPTED = "INTERCEPTED"
    BREACHED    = "BREACHED"


class ManeuverType(Enum):
    STRAIGHT    = "STRAIGHT"     # constant velocity, no deviation
    ZIGZAG      = "ZIGZAG"       # lateral oscillation left/right
    CORKSCREW   = "CORKSCREW"    # spiral approach
    JINK        = "JINK"         # random sudden direction changes
    TERRAIN_HUG = "TERRAIN_HUG"  # altitude drops as it approaches


# ─────────────────────────────────────────
# Threat Entity
# ─────────────────────────────────────────

class Threat:
    """
    Represents a single incoming threat.
    Supports multidirectional spawning and 4 maneuver types.
    """

    def __init__(self, id: str, position: list, velocity: list,
                 rcs: RCS, origin_distance: float = 0.0,
                 maneuver: ManeuverType = ManeuverType.STRAIGHT):

        self.id              = id
        self.position        = np.array(position, dtype=float)
        self.velocity        = np.array(velocity, dtype=float)
        self.rcs             = rcs
        self.origin_distance = origin_distance
        self.maneuver        = maneuver

        # Derived
        self.speed           = float(np.linalg.norm(self.velocity))
        self.altitude        = float(self.position[2])

        # State
        self.type            = ThreatType.UNKNOWN
        self.status          = ThreatStatus.INBOUND
        self.classified      = False
        self.assigned        = False
        self.detected_time   = None   # sim time when first detected

        # Maneuver state
        self._tick           = 0
        self._jink_timer     = 0
        self._jink_delta     = np.zeros(3)

        # Trajectory log
        self.trajectory_log  = [self.position.tolist()]

    # ─────────────────────────────────────────
    # Movement
    # ─────────────────────────────────────────

    def update_position(self, dt: float):
        """Move threat forward applying maneuver if assigned."""
        self._tick += 1
        maneuver_vel = self._compute_maneuver()
        self.position += (self.velocity + maneuver_vel) * dt
        self.altitude  = float(self.position[2])
        self.trajectory_log.append(self.position.tolist())

    def _compute_maneuver(self) -> np.ndarray:
        """
        Returns an additional velocity vector based on maneuver type.
        Always perpendicular to main approach so it doesn't change speed.
        """
        t = self._tick

        if self.maneuver == ManeuverType.STRAIGHT:
            return np.zeros(3)

        elif self.maneuver == ManeuverType.ZIGZAG:
            # Oscillate laterally — perpendicular to approach direction
            perp = self._lateral_unit()
            amplitude = self.speed * 0.3
            return perp * amplitude * math.sin(t * 0.25)

        elif self.maneuver == ManeuverType.CORKSCREW:
            # Spiral — combine lateral + vertical oscillation
            perp = self._lateral_unit()
            amplitude = self.speed * 0.2
            lat  = perp * amplitude * math.sin(t * 0.3)
            vert = np.array([0, 0, amplitude * 0.5 * math.cos(t * 0.3)])
            return lat + vert

        elif self.maneuver == ManeuverType.JINK:
            # Random sudden direction changes every 8-15 ticks
            self._jink_timer -= 1
            if self._jink_timer <= 0:
                self._jink_timer = random.randint(8, 15)
                perp = self._lateral_unit()
                direction = random.choice([-1, 1])
                self._jink_delta = perp * self.speed * 0.4 * direction
            return self._jink_delta

        elif self.maneuver == ManeuverType.TERRAIN_HUG:
            # Altitude drops as threat approaches — hug the ground
            dist = self.distance_to_origin()
            if dist > 10:
                # Gradually descend toward low altitude
                target_alt = max(0.5, dist * 0.02)
                diff = target_alt - self.altitude
                return np.array([0, 0, diff * 0.1])
            return np.zeros(3)

        return np.zeros(3)

    def _lateral_unit(self) -> np.ndarray:
        """Unit vector perpendicular to velocity in the horizontal plane."""
        v = self.velocity.copy()
        v[2] = 0  # horizontal only
        mag = np.linalg.norm(v)
        if mag < 1e-6:
            return np.array([0, 1, 0])
        v_norm = v / mag
        # Perpendicular in horizontal plane
        return np.array([-v_norm[1], v_norm[0], 0])

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def distance_to_origin(self) -> float:
        """2D horizontal distance from origin."""
        return float(np.linalg.norm(self.position[:2]))

    def to_dict(self) -> dict:
        return {
            "id"              : self.id,
            "type"            : self.type.value,
            "rcs"             : self.rcs.value,
            "speed"           : round(self.speed, 3),
            "maneuver"        : self.maneuver.value,
            "origin_distance" : self.origin_distance,
            "status"          : self.status.value,
            "trajectory"      : self.trajectory_log,
        }

    def __repr__(self):
        return (f"Threat(id={self.id}, type={self.type.value}, "
                f"speed={self.speed:.2f}km/s, alt={self.altitude:.1f}km, "
                f"maneuver={self.maneuver.value}, status={self.status.value})")