from entities.threat import ThreatType


class ThreatRouter:
    """
    Decides which AD layer engages which threat.

    Uses the Minimum Sufficiency Principle:
        Use the least powerful layer that still has a high kill probability.
        Don't waste an S-400 interceptor on a drone.
        Don't assign Akash to a ballistic it can't reach.

    In Phase 3 this will be replaced by a trained RL agent (PPO).
    """

    def assign(self, threat, layers: list) -> tuple:
        """
        Find the best available layer for a given threat.

        Args:
            threat  : Classified Threat object
            layers  : List of ADLayer objects

        Returns:
            (best_layer, reason_string)
            best_layer is None if no layer can engage (BREACH)
        """

        # ── Step 1: Find all capable + available layers ──────────────
        candidates = [
            layer for layer in layers
            if layer.can_engage(threat) and layer.is_available()
        ]

        # ── Step 2: Nobody can handle it → BREACH ────────────────────
        if not candidates:
            reason = self._breach_reason(threat, layers)
            return None, reason

        # ── Step 3: Minimum sufficiency selection ─────────────────────
        # Sort by:
        #   1. Higher kill probability for this threat type (preferred)
        #   2. Shorter max range (prefer less powerful / cheaper system)
        best = min(
            candidates,
            key=lambda l: (
                -l.kill_probability.get(threat.type, 0.0),  # negate = higher is better
                l.max_range                                   # lower range = less overkill
            )
        )

        best.fire(threat)
        threat.assigned = True
        return best, "ASSIGNED"

    def _breach_reason(self, threat, layers: list) -> str:
        """
        Diagnose WHY no layer could engage — useful for event log.
        """
        capable_but_busy = [
            l for l in layers
            if l.can_engage(threat) and not l.is_available()
        ]
        capable_but_empty = [
            l for l in layers
            if threat.type in l.effective_against and l.ammo == 0
        ]
        out_of_range = [
            l for l in layers
            if threat.type in l.effective_against
            and not l.can_engage(threat)
        ]

        if capable_but_busy:
            names = [l.name for l in capable_but_busy]
            return f"SATURATED — {names} at max capacity"

        if capable_but_empty:
            names = [l.name for l in capable_but_empty]
            return f"NO_AMMO — {names} depleted"

        if out_of_range:
            names = [l.name for l in out_of_range]
            return f"OUT_OF_RANGE — {names} cannot reach"

        return "NO_CAPABLE_LAYER — threat type not covered"
