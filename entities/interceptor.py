import numpy as np
from enum import Enum


class InterceptorStatus(Enum):
    PURSUING  = "PURSUING"
    DETONATED = "DETONATED"
    MISSED    = "MISSED"


class Interceptor:
    """
    An interceptor missile fired by an AD layer to destroy a threat.
    Uses proportional navigation in 3D to chase its target.
    """

    KILL_RADIUS = 0.5    # km — detonate if within 500m of target

    def __init__(self, id: str, parent_layer: str, target_id: str,
                 launch_position: list, speed: float, launch_time: int):
        """
        Args:
            id              : Unique ID e.g. "I1"
            parent_layer    : Name of the AD layer that fired this
            target_id       : ID of the threat being chased
            launch_position : [x, y, z] in km — where it was launched from
            speed           : Interceptor speed in km/s
            launch_time     : Simulation time (seconds) when fired
        """
        self.id              = id
        self.parent_layer    = parent_layer
        self.target_id       = target_id
        self.position        = np.array(launch_position, dtype=float)
        self.speed           = speed
        self.launch_time     = launch_time
        self.velocity        = np.zeros(3, dtype=float)
        self.status          = InterceptorStatus.PURSUING

        # Store every position tick for JSON export + animation
        self.trajectory_log  = [self.position.tolist()]

    def update(self, dt: float, target_position: np.ndarray):
        """
        Move interceptor toward target using proportional navigation.
        Steers directly toward current target position each tick.

        Args:
            dt              : Time step in seconds
            target_position : Current [x, y, z] of the target threat
        """
        if self.status != InterceptorStatus.PURSUING:
            return

        direction = target_position - self.position
        distance  = float(np.linalg.norm(direction))

        # Already within kill radius — handled by check_intercept
        if distance < self.KILL_RADIUS:
            return

        # Normalize and scale to interceptor speed
        self.velocity  = (direction / distance) * self.speed
        self.position += self.velocity * dt
        self.trajectory_log.append(self.position.tolist())

    def check_intercept(self, target, kill_probability: float) -> str:
        """
        Check if this interceptor has reached its target.
        Applies probabilistic kill roll — not every intercept destroys.

        Returns:
            "DETONATED" | "MISSED" | "PURSUING"
        """
        import random

        distance = float(np.linalg.norm(target.position - self.position))

        if distance < self.KILL_RADIUS:
            if random.random() < kill_probability:
                self.status = InterceptorStatus.DETONATED
                return "DETONATED"
            else:
                # Close but failed to destroy
                self.status = InterceptorStatus.MISSED
                return "MISSED"

        return "PURSUING"

    def to_dict(self) -> dict:
        """Serialize for JSON export."""
        return {
            "id"           : self.id,
            "parent_layer" : self.parent_layer,
            "target"       : self.target_id,
            "status"       : self.status.value,
            "launch_time"  : self.launch_time,
            "trajectory"   : self.trajectory_log,
        }

    def __repr__(self):
        return (f"Interceptor(id={self.id}, layer={self.parent_layer}, "
                f"target={self.target_id}, status={self.status.value})")