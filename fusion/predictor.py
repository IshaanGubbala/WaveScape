"""
Predictive threat extrapolation.
While Gemma inference runs (2-5s), uses optical flow velocity to extrapolate
threat position at 30fps. Haptics update continuously, not just on Gemma response.

This is the key innovation: sub-inference-cycle haptic updates.
"""
import time
import math
import threading
from dataclasses import dataclass
from typing import Optional, Callable
from .stream_parser import ThreatAlert


@dataclass
class MotionVector:
    direction_deg: float
    speed_mps: float
    label: str
    confidence: float
    origin_distance_m: float
    captured_at: float


class ThreatPredictor:
    """
    Extrapolates threat position between Gemma inference calls.
    Updates at `update_hz` using last known velocity vector.
    Calls `haptic_cb` with extrapolated direction + urgency at each update.
    """

    def __init__(
        self,
        haptic_cb: Callable[[float, str, float], None],
        update_hz: float = 10.0,
    ):
        self.haptic_cb = haptic_cb  # (direction_deg, urgency, confidence)
        self.update_hz = update_hz
        self._lock = threading.Lock()
        self._active_threat: Optional[ThreatAlert] = None
        self._motion: Optional[MotionVector] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def update_threat(self, alert: ThreatAlert, motion: Optional[MotionVector] = None) -> None:
        with self._lock:
            self._active_threat = alert
            if motion:
                self._motion = motion

    def update_motion(self, motion: MotionVector) -> None:
        with self._lock:
            self._motion = motion

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        interval = 1.0 / self.update_hz
        while self._running:
            t_start = time.monotonic()
            with self._lock:
                alert = self._active_threat
                motion = self._motion

            if alert and alert.is_actionable():
                direction = self._extrapolate_direction(alert, motion)
                confidence = self._decay_confidence(alert)
                if confidence > 0.15:
                    self.haptic_cb(direction, alert.urgency, confidence)

            elapsed = time.monotonic() - t_start
            time.sleep(max(0, interval - elapsed))

    def _extrapolate_direction(
        self, alert: ThreatAlert, motion: Optional[MotionVector]
    ) -> float:
        if motion is None:
            return alert.direction_degrees

        elapsed = time.time() - alert.timestamp
        if elapsed > 3.0 or motion.speed_mps < 0.5:
            return alert.direction_degrees

        # Threat moving toward user: shift direction toward user heading
        # Simple model: angular velocity based on lateral motion
        # distance decreases at speed_mps * cos(bearing_diff)
        bearing_diff = (motion.direction_deg - alert.direction_degrees) % 360
        if bearing_diff > 180:
            bearing_diff -= 360

        # If threat moving perpendicular to user, direction shifts
        lateral_shift = motion.speed_mps * math.sin(math.radians(bearing_diff)) * elapsed
        distance = max(alert.distance_meters - motion.speed_mps * elapsed * 0.5, 1.0)
        angular_shift = math.degrees(math.atan2(lateral_shift, distance))

        return (alert.direction_degrees + angular_shift) % 360

    @staticmethod
    def _decay_confidence(alert: ThreatAlert) -> float:
        elapsed = time.time() - alert.timestamp
        half_life = 3.0
        return alert.confidence * math.exp(-elapsed * math.log(2) / half_life)
