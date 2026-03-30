"""
ai/rl_environment.py
─────────────────────────────────────────────────────────────────
Gymnasium wrapper around SimulationEngine.
This turns your simulation into a "game" that the RL agent plays.

The agent plays one EPISODE = one full scenario from start to finish.
Each STEP = one tick of the simulation (1 second).

STATE  — what the agent observes each tick
ACTION — what assignment decision the agent makes
REWARD — signal telling agent if it did well or badly
─────────────────────────────────────────────────────────────────
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from entities.ad_layer import create_indian_ad_layers
from core.simulation import SimulationEngine
from core.classifier import ThreatClassifier
from scenarios.wave_generator import generate_training_episode, build_threat_objects
from entities.threat import ThreatStatus, ThreatType


# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────

MAX_THREATS    = 60     # max threats per episode (pad state to this size)
N_LAYERS       = 3      # S-400, Barak-8, Akash
N_ACTIONS      = MAX_THREATS * N_LAYERS + 1   # assign threat i to layer j, or HOLD

THREAT_TYPE_MAP = {
    ThreatType.BALLISTIC : 0,
    ThreatType.CRUISE    : 1,
    ThreatType.ROCKET    : 2,
    ThreatType.DRONE     : 3,
    ThreatType.UNKNOWN   : 4,
}


class AirDefenseEnv(gym.Env):
    """
    Gymnasium environment for multilayer air defense simulation.

    Observation space:
        For each threat slot (MAX_THREATS):
            [active, x, y, z, vx, vy, vz, speed, altitude, type_enc, distance]
        For each layer (N_LAYERS):
            [ammo_ratio, load_ratio]
        = MAX_THREATS * 11 + N_LAYERS * 2 features

    Action space:
        Discrete — agent picks one of:
            0               : HOLD (do nothing this tick)
            1..MAX_THREATS  : assign threat slot i to best available layer
                              (layer is auto-selected by minimum sufficiency)
        This keeps action space manageable — agent decides WHICH threat
        to prioritize, router handles WHICH layer.

    Reward:
        +100  per intercept
        -500  per breach
        -10   per wasted interceptor (overkill)
        -200  per layer fully depleted early
        +1    per tick with zero breaches (survival bonus)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, difficulty: str = "random"):
        super().__init__()
        self.difficulty = difficulty
        self.classifier = ThreatClassifier()

        # ── Observation space ────────────────────────────────────────
        # Flat vector: threat features + layer features
        n_obs = MAX_THREATS * 11 + N_LAYERS * 2
        self.observation_space = spaces.Box(
            low   = -1.0,
            high  =  1.0,
            shape = (n_obs,),
            dtype = np.float32,
        )

        # ── Action space ─────────────────────────────────────────────
        # Discrete: 0=HOLD, 1..MAX_THREATS = prioritize threat i
        self.action_space = spaces.Discrete(MAX_THREATS + 1)

        # Runtime state — reset each episode
        self.engine       = None
        self.threats      = None
        self.layers       = None
        self._prev_intercepted = 0
        self._prev_breached    = 0

    # ─────────────────────────────────────────
    # Reset — start a new episode
    # ─────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate a fresh random scenario
        scenario     = generate_training_episode(self.difficulty)
        self.threats = build_threat_objects(scenario)
        self.layers  = create_indian_ad_layers()

        # Create simulation in silent mode — no prints during training
        self.engine = SimulationEngine(
            threats       = self.threats,
            layers        = self.layers,
            scenario_name = scenario["name"],
            silent        = True,
        )

        self._prev_intercepted = 0
        self._prev_breached    = 0

        # Classify all threats immediately (radar sees them on spawn)
        for threat in self.threats:
            if not threat.classified:
                threat.type       = self.classifier.classify(threat)
                threat.classified = True

        obs  = self._get_observation()
        info = {"scenario": scenario["name"], "n_threats": len(self.threats)}
        return obs, info

    # ─────────────────────────────────────────
    # Step — one tick of simulation
    # ─────────────────────────────────────────

    def step(self, action: int):
        """
        Execute one simulation tick with the agent's chosen action.

        Action 0        = HOLD — let simulation run without extra assignment
        Action 1..N     = prioritize threat at slot (action-1) for immediate assignment
        """

        # ── Apply action ─────────────────────────────────────────────
        if action > 0:
            threat_idx = action - 1
            active_threats = [t for t in self.threats
                              if t.status == ThreatStatus.INBOUND
                              and not t.assigned]
            if threat_idx < len(active_threats):
                target_threat = active_threats[threat_idx]
                layer, reason = self.engine.router.assign(
                    target_threat, self.layers
                )
                if layer:
                    target_threat.assigned = True
                    self.engine._launch_interceptor(target_threat, layer)

        # ── Advance simulation one tick ───────────────────────────────
        self.engine.time += self.engine.tick
        self.engine._update_threats()
        self.engine._update_interceptors()

        # ── Compute reward ────────────────────────────────────────────
        metrics = self.engine.get_metrics()
        reward  = self._compute_reward(metrics)

        # ── Check if episode is done ──────────────────────────────────
        terminated = (
            metrics["inbound"] == 0 or           # all threats resolved
            self.engine.time >= self.engine.MAX_TIME
        )
        truncated = self.engine.time >= self.engine.MAX_TIME

        obs  = self._get_observation()
        info = metrics

        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────
    # Reward Function
    # ─────────────────────────────────────────

    def _compute_reward(self, metrics: dict) -> float:
        reward = 0.0

        # Delta since last tick
        new_intercepted = metrics["intercepted"] - self._prev_intercepted
        new_breached    = metrics["breached"]    - self._prev_breached

        # ── Core rewards ─────────────────────────────────────────────
        reward += new_intercepted * 100.0    # +100 per intercept
        reward -= new_breached    * 500.0    # -500 per breach

        # ── Survival bonus ────────────────────────────────────────────
        if new_breached == 0:
            reward += 1.0                    # +1 per clean tick

        # ── Ammo conservation penalty ─────────────────────────────────
        # Penalize if any layer is fully depleted
        for layer in self.layers:
            if layer.ammo == 0 and metrics["inbound"] > 0:
                reward -= 5.0                # -5 per tick with empty layer + threats still coming

        # Update previous state
        self._prev_intercepted = metrics["intercepted"]
        self._prev_breached    = metrics["breached"]

        return reward

    # ─────────────────────────────────────────
    # Observation Builder
    # ─────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """
        Build a flat normalized observation vector.
        Padding: if fewer than MAX_THREATS threats exist, pad with zeros.
        """
        obs = np.zeros(MAX_THREATS * 11 + N_LAYERS * 2, dtype=np.float32)

        active = [t for t in self.threats if t.status == ThreatStatus.INBOUND]

        # ── Threat features ───────────────────────────────────────────
        for i, threat in enumerate(active[:MAX_THREATS]):
            base = i * 11
            obs[base + 0]  = 1.0                                         # active flag
            obs[base + 1]  = np.clip(threat.position[0] / 400.0, -1, 1)  # x (norm by max range)
            obs[base + 2]  = np.clip(threat.position[1] / 100.0, -1, 1)  # y
            obs[base + 3]  = np.clip(threat.position[2] / 200.0, -1, 1)  # z (altitude)
            obs[base + 4]  = np.clip(threat.velocity[0] / 5.0,  -1, 1)   # vx
            obs[base + 5]  = np.clip(threat.velocity[1] / 1.0,  -1, 1)   # vy
            obs[base + 6]  = np.clip(threat.velocity[2] / 2.0,  -1, 1)   # vz
            obs[base + 7]  = np.clip(threat.speed      / 5.0,   -1, 1)   # speed
            obs[base + 8]  = np.clip(threat.altitude   / 200.0,  0, 1)   # altitude
            obs[base + 9]  = THREAT_TYPE_MAP.get(threat.type, 4) / 4.0   # type encoded
            obs[base + 10] = np.clip(
                threat.distance_to_origin() / 400.0, 0, 1
            )                                                              # distance

        # ── Layer features ────────────────────────────────────────────
        layer_specs = [
            ("S-400",   16, 36),
            ("Barak-8", 32, 12),
            ("Akash",   24,  8),
        ]
        base = MAX_THREATS * 11
        for i, (name, max_ammo, max_sim) in enumerate(layer_specs):
            layer = self.engine._get_layer(name)
            if layer:
                obs[base + i*2 + 0] = layer.ammo / max_ammo                         # ammo ratio
                obs[base + i*2 + 1] = len(layer.currently_engaging) / max_sim        # load ratio

        return obs

    # ─────────────────────────────────────────
    # Render — human readable tick summary
    # ─────────────────────────────────────────

    def render(self):
        metrics = self.engine.get_metrics()
        print(f"  t={self.engine.time:>4}s | "
              f"inbound={metrics['inbound']:>3} | "
              f"intercepted={metrics['intercepted']:>3} | "
              f"breached={metrics['breached']:>3} | "
              f"ammo={metrics['ammo_remaining']}")
