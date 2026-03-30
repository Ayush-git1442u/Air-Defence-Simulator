import json
import os
import numpy as np
from entities.threat import ThreatStatus, ThreatType
from entities.interceptor import Interceptor, InterceptorStatus
from core.classifier import ThreatClassifier
from core.router import ThreatRouter


class SimulationEngine:
    """
    Main simulation loop.

    Fixes applied:
    - Re-engagement queue (threats retried every tick)
    - Staggered launches (reaction_time per layer respected)
    - silent mode for RL training
    - router injection for RL agent
    """

    ORIGIN        = [0.0, 0.0, 0.0]
    BREACH_RADIUS = 1.0
    MAX_TIME      = 600

    def __init__(self, threats: list, layers: list,
                 scenario_name: str = "simulation",
                 silent: bool = False,
                 router=None):
        self.threats          = threats
        self.layers           = layers
        self.scenario_name    = scenario_name
        self.silent           = silent
        self.interceptors     = []
        self.classifier       = ThreatClassifier()
        self.router           = router if router else ThreatRouter()
        self.time             = 0
        self.tick             = 1
        self.event_log        = []
        self._interceptor_ctr = 0
        self._pending_queue   = []

        # Fix 5: track when each threat was detected
        # so reaction_time delay can be applied before firing
        self._detection_times = {}   # threat_id → sim time detected

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _log(self, event: str, **kwargs):
        entry = {"time": self.time, "event": event}
        entry.update(kwargs)
        self.event_log.append(entry)
        if not self.silent:
            print(f"  [t={self.time:>4}s]  {event:<22} {kwargs}")

    def _active_threats(self):
        return [t for t in self.threats if t.status == ThreatStatus.INBOUND]

    def _get_threat(self, threat_id: str):
        return next((t for t in self.threats if t.id == threat_id), None)

    def _get_layer(self, layer_name: str):
        return next((l for l in self.layers if l.name == layer_name), None)

    def _new_interceptor_id(self) -> str:
        self._interceptor_ctr += 1
        return f"I{self._interceptor_ctr}"

    def _launch_interceptor(self, threat, layer):
        interceptor = Interceptor(
            id              = self._new_interceptor_id(),
            parent_layer    = layer.name,
            target_id       = threat.id,
            launch_position = self.ORIGIN,
            speed           = layer.interceptor_speed,
            launch_time     = self.time,
        )
        self.interceptors.append(interceptor)
        self._log("ASSIGNED",
                  threat=threat.id,
                  layer=layer.name,
                  interceptor=interceptor.id)

    # ─────────────────────────────────────────
    # Main Loop
    # ─────────────────────────────────────────

    def run(self):
        if not self.silent:
            print(f"\n{'═'*60}")
            print(f"  SIMULATION START — Scenario: {self.scenario_name}")
            print(f"{'═'*60}\n")

        while self._active_threats() and self.time < self.MAX_TIME:
            self.time += self.tick
            self._update_threats()
            self._retry_pending()
            self._update_interceptors()

        if not self.silent:
            print(f"\n{'═'*60}")
            print(f"  SIMULATION END — Duration: {self.time}s")
            print(f"{'═'*60}")
            self._print_summary()

        self.export_results()

    # ─────────────────────────────────────────
    # Threat Update
    # ─────────────────────────────────────────

    def _update_threats(self):
        for threat in self._active_threats():

            # Move forward with maneuver
            threat.update_position(self.tick)

            # Classify on first tick
            if not threat.classified:
                threat.type             = self.classifier.classify(threat)
                threat.classified       = True
                threat.detected_time    = self.time
                self._detection_times[threat.id] = self.time
                self._log("DETECTED",
                          threat=threat.id,
                          type=threat.type.value,
                          speed=round(threat.speed, 2),
                          alt=round(threat.altitude, 1),
                          maneuver=threat.maneuver.value)

            # ── Fix 5: Staggered launch — respect reaction_time ──
            # Don't assign immediately. Wait until:
            #   current_time >= detection_time + layer.reaction_time
            if not threat.assigned:
                detection_time = self._detection_times.get(threat.id, 0)

                # Find candidates that have completed reaction time
                ready_layers = [
                    l for l in self.layers
                    if self.time >= detection_time + l.reaction_time
                ]

                if not ready_layers:
                    # Still waiting for reaction time — skip this tick
                    continue

                layer, reason = self.router.assign(threat, ready_layers)
                if layer:
                    threat.assigned = True
                    self._launch_interceptor(threat, layer)
                else:
                    if threat not in self._pending_queue:
                        self._pending_queue.append(threat)
                        self._log("QUEUED", threat=threat.id, reason=reason)

            # Breach check — 2D horizontal
            if np.linalg.norm(threat.position[:2]) < self.BREACH_RADIUS:
                threat.status = ThreatStatus.BREACHED
                if threat in self._pending_queue:
                    self._pending_queue.remove(threat)
                self._log("BREACH", threat=threat.id, type=threat.type.value)

    # ─────────────────────────────────────────
    # Re-engagement queue
    # ─────────────────────────────────────────

    def _retry_pending(self):
        still_pending = []
        for threat in self._pending_queue:
            if threat.status != ThreatStatus.INBOUND:
                continue
            if threat.assigned:
                continue   # already assigned this tick via _update_threats
            detection_time = self._detection_times.get(threat.id, 0)
            ready_layers   = [
                l for l in self.layers
                if self.time >= detection_time + l.reaction_time
            ]
            layer, reason = self.router.assign(threat, ready_layers)
            if layer:
                threat.assigned = True
                self._launch_interceptor(threat, layer)
                self._log("RE_ENGAGED", threat=threat.id, layer=layer.name)
            else:
                still_pending.append(threat)
        self._pending_queue = still_pending

    # ─────────────────────────────────────────
    # Interceptor Update
    # ─────────────────────────────────────────

    def _update_interceptors(self):
        for interceptor in self.interceptors:
            if interceptor.status != InterceptorStatus.PURSUING:
                continue

            target = self._get_threat(interceptor.target_id)
            if target is None or target.status != ThreatStatus.INBOUND:
                interceptor.status = InterceptorStatus.MISSED
                continue

            layer     = self._get_layer(interceptor.parent_layer)
            kill_prob = layer.get_kill_probability(target) if layer else 0.5

            interceptor.update(self.tick, target.position)
            result = interceptor.check_intercept(target, kill_prob)

            if result == "DETONATED":
                target.status = ThreatStatus.INTERCEPTED
                if target in self._pending_queue:
                    self._pending_queue.remove(target)
                if layer:
                    layer.release(target.id)
                self._log("INTERCEPTED",
                          threat=target.id,
                          interceptor=interceptor.id,
                          layer=interceptor.parent_layer)

            elif result == "MISSED":
                if layer:
                    layer.release(target.id)
                if target.status == ThreatStatus.INBOUND and target not in self._pending_queue:
                    target.assigned = False
                    self._pending_queue.append(target)
                self._log("MISSED",
                          threat=target.id,
                          interceptor=interceptor.id,
                          layer=interceptor.parent_layer)

    # ─────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────

    def _print_summary(self):
        intercepted = sum(1 for t in self.threats if t.status == ThreatStatus.INTERCEPTED)
        breached    = sum(1 for t in self.threats if t.status == ThreatStatus.BREACHED)
        total       = len(self.threats)
        print(f"\n  📊 RESULTS")
        print(f"  Total threats  : {total}")
        print(f"  Intercepted    : {intercepted}  ✅")
        print(f"  Breached       : {breached}  ❌")
        print(f"  Success rate   : {intercepted/total*100:.1f}%")
        print(f"\n  💥 AMMO REMAINING")
        for layer in self.layers:
            print(f"  {layer.name:<12} : {layer.ammo} rounds left (fired {layer.total_fired})")

    # ─────────────────────────────────────────
    # Metrics for RL
    # ─────────────────────────────────────────

    def get_metrics(self) -> dict:
        intercepted = sum(1 for t in self.threats if t.status == ThreatStatus.INTERCEPTED)
        breached    = sum(1 for t in self.threats if t.status == ThreatStatus.BREACHED)
        inbound     = sum(1 for t in self.threats if t.status == ThreatStatus.INBOUND)
        return {
            "intercepted"    : intercepted,
            "breached"       : breached,
            "inbound"        : inbound,
            "total"          : len(self.threats),
            "ammo_remaining" : {l.name: l.ammo for l in self.layers},
            "pending"        : len(self._pending_queue),
            "time"           : self.time,
        }

    # ─────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────

    def export_results(self):
        intercepted = sum(1 for t in self.threats if t.status == ThreatStatus.INTERCEPTED)
        breached    = sum(1 for t in self.threats if t.status == ThreatStatus.BREACHED)
        results = {
            "scenario"         : self.scenario_name,
            "duration_seconds" : self.time,
            "summary": {
                "total_threats"  : len(self.threats),
                "intercepted"    : intercepted,
                "breached"       : breached,
                "success_rate"   : round(intercepted / len(self.threats) * 100, 1),
                "ammo_remaining" : {l.name: l.ammo for l in self.layers},
                "total_fired"    : {l.name: l.total_fired for l in self.layers},
            },
            "threats"      : [t.to_dict() for t in self.threats],
            "interceptors" : [i.to_dict() for i in self.interceptors],
            "events"       : self.event_log,
        }
        os.makedirs("output", exist_ok=True)
        path = f"output/{self.scenario_name}_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        if not self.silent:
            print(f"\n  ✅ Exported → {path}\n")
        return path