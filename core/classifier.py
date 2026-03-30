from entities.threat import ThreatType, RCS


class ThreatClassifier:
    """
    Classifies an incoming threat based on its physical properties.
    Uses hardcoded rules in Phase 2.
    In Phase 3 this will be replaced by a trained ML model (scikit-learn).
    """

    def classify(self, threat) -> ThreatType:
        """
        Reads threat speed, altitude, and RCS → assigns a ThreatType.
        Rules are evaluated in priority order (most specific first).

        Args:
            threat : Threat object with speed, altitude, rcs set

        Returns:
            ThreatType enum value
        """
        s = threat.speed      # km/s
        a = threat.altitude   # km

        # ── BALLISTIC ───────────────────────────────────────────────
        # Very fast, very high altitude, parabolic arc
        # Examples: Agni, Shaheen ballistic missiles
        if s > 3.0 and a > 80:
            return ThreatType.BALLISTIC

        # ── CRUISE ──────────────────────────────────────────────────
        # Fast, low-to-mid altitude, terrain hugging or flat path
        # Examples: BrahMos, Tomahawk cruise missiles
        elif s > 0.8 and a < 50:
            return ThreatType.CRUISE

        # ── ROCKET ──────────────────────────────────────────────────
        # Medium speed, low altitude, short range
        # Examples: Grad rockets, artillery rockets
        elif 0.4 < s <= 0.8 and a < 20:
            return ThreatType.ROCKET

        # ── DRONE ───────────────────────────────────────────────────
        # Very slow, very low altitude, tiny radar cross section
        # Examples: Loitering munitions, UAV swarms
        elif s <= 0.3 and threat.rcs == RCS.MICRO:
            return ThreatType.DRONE

        # ── UNKNOWN ─────────────────────────────────────────────────
        # Does not match any known profile
        else:
            return ThreatType.UNKNOWN

    def classify_all(self, threats: list) -> list:
        """
        Classify a list of threats in one call.
        Returns list of (threat, assigned_type) tuples.
        """
        results = []
        for threat in threats:
            assigned = self.classify(threat)
            threat.type       = assigned
            threat.classified = True
            results.append((threat, assigned))
        return results
