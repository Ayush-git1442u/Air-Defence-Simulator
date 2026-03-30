import numpy as np
from .threat import ThreatType, ThreatStatus


class ADLayer:
    """
    Represents a single Air Defence layer (S-400, Barak-8, or Akash).
    Manages engagement capacity, ammo, and firing logic.
    """

    def __init__(self, name: str, max_range: float, min_range: float,
                 max_altitude: float, min_altitude: float,
                 reaction_time: float, ammo: int,
                 max_simultaneous: int, reload_time: float,
                 interceptor_speed: float,
                 effective_against: list, kill_probability: dict):
        """
        Args:
            name                : "S-400" | "Barak-8" | "Akash"
            max_range           : Maximum engagement range (km)
            min_range           : Minimum engagement range (km)
            max_altitude        : Maximum intercept altitude (km)
            min_altitude        : Minimum intercept altitude (km)
            reaction_time       : Seconds from detect to fire
            ammo                : Total interceptors available
            max_simultaneous    : Max threats engaged at once
            reload_time         : Seconds between shots
            interceptor_speed   : Speed of fired interceptor (km/s)
            effective_against   : List of ThreatType this layer handles
            kill_probability    : Dict {ThreatType: 0.0–1.0}
        """
        self.name              = name
        self.max_range         = max_range
        self.min_range         = min_range
        self.max_altitude      = max_altitude
        self.min_altitude      = min_altitude
        self.reaction_time     = reaction_time
        self.ammo              = ammo
        self.max_simultaneous  = max_simultaneous
        self.reload_time       = reload_time
        self.interceptor_speed = interceptor_speed
        self.effective_against = effective_against
        self.kill_probability  = kill_probability

        # Runtime state
        self.currently_engaging = []   # list of threat IDs being engaged
        self.total_fired         = 0   # total interceptors fired this sim

    def can_engage(self, threat) -> bool:
        """
        Check if this layer is physically capable of engaging a threat.
        Checks: threat type, range, altitude.
        """
        if threat.type not in self.effective_against:
            return False

        distance = threat.distance_to_origin()

        if not (self.min_range <= distance <= self.max_range):
            return False

        if not (self.min_altitude <= threat.altitude <= self.max_altitude):
            return False

        return True

    def is_available(self) -> bool:
        """Has ammo and not at max simultaneous engagement capacity."""
        return (
            self.ammo > 0 and
            len(self.currently_engaging) < self.max_simultaneous
        )

    def fire(self, threat):
        """
        Fire at a threat — deduct ammo, register engagement.
        """
        self.ammo -= 1
        self.total_fired += 1
        self.currently_engaging.append(threat.id)

    def release(self, threat_id: str):
        """
        Release a threat from engagement tracking
        (called after intercept or miss).
        """
        if threat_id in self.currently_engaging:
            self.currently_engaging.remove(threat_id)

    def get_kill_probability(self, threat) -> float:
        """Return kill probability for a given threat type."""
        return self.kill_probability.get(threat.type, 0.0)

    def to_dict(self) -> dict:
        """Serialize current state for JSON export."""
        return {
            "name"         : self.name,
            "ammo_remaining": self.ammo,
            "total_fired"  : self.total_fired,
            "engaging"     : self.currently_engaging.copy(),
        }

    def __repr__(self):
        return (f"ADLayer(name={self.name}, ammo={self.ammo}, "
                f"engaging={len(self.currently_engaging)}/{self.max_simultaneous})")


# ─────────────────────────────────────────
# Factory — create the 3 Indian AD layers
# ─────────────────────────────────────────

def create_indian_ad_layers() -> list:
    """
    Returns the three layers of India's multilayer air defence:
    S-400, Barak-8, Akash — with real approximate specifications.
    """

    s400 = ADLayer(
        name               = "S-400",
        max_range          = 400,
        min_range          = 40,
        max_altitude       = 185,
        min_altitude       = 10,
        reaction_time      = 9,
        ammo               = 16,
        max_simultaneous   = 36,
        reload_time        = 15,
        interceptor_speed  = 4.8,        # km/s
        effective_against  = [ThreatType.BALLISTIC, ThreatType.CRUISE],
        kill_probability   = {
            ThreatType.BALLISTIC : 0.92,
            ThreatType.CRUISE    : 0.75,
        }
    )

    barak8 = ADLayer(
        name               = "Barak-8",
        max_range          = 100,
        min_range          = 0.5,
        max_altitude       = 16,
        min_altitude       = 0.02,
        reaction_time      = 5,
        ammo               = 32,
        max_simultaneous   = 12,
        reload_time        = 8,
        interceptor_speed  = 2.0,        # km/s
        effective_against  = [ThreatType.CRUISE, ThreatType.ROCKET,
                               ThreatType.DRONE],
        kill_probability   = {
            ThreatType.CRUISE  : 0.85,
            ThreatType.ROCKET  : 0.80,
            ThreatType.DRONE   : 0.70,
        }
    )

    akash = ADLayer(
        name               = "Akash",
        max_range          = 60,
        min_range          = 0.5,
        max_altitude       = 18,
        min_altitude       = 0.02,
        reaction_time      = 8,
        ammo               = 24,
        max_simultaneous   = 8,
        reload_time        = 6,
        interceptor_speed  = 0.7,        # km/s
        effective_against  = [ThreatType.DRONE, ThreatType.ROCKET,
                               ThreatType.CRUISE],
        kill_probability   = {
            ThreatType.DRONE   : 0.88,
            ThreatType.ROCKET  : 0.75,
            ThreatType.CRUISE  : 0.60,
        }
    )

    return [s400, barak8, akash]