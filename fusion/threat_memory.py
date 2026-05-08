"""
Exponential-decay threat memory.
Keeps threat confidence alive between slow Gemma inference cycles.
Prevents alert gaps when inference takes 2-5s.
"""
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from .stream_parser import ThreatAlert


DECAY_HALF_LIFE = 2.0  # seconds — threat confidence halves every 2s
DUPLICATE_WINDOW = 1.5  # seconds — suppress repeat alerts of same type+direction
HISTORY_LEN = 5


@dataclass
class MemoryEntry:
    alert: ThreatAlert
    original_confidence: float
    inserted_at: float = field(default_factory=time.time)

    def current_confidence(self) -> float:
        elapsed = time.time() - self.inserted_at
        return self.original_confidence * math.exp(
            -elapsed * math.log(2) / DECAY_HALF_LIFE
        )

    def is_alive(self) -> bool:
        return self.current_confidence() > 0.15


class ThreatMemory:
    """
    Rolling window of recent threats with exponential confidence decay.
    Answers: "Is there still an active threat even if Gemma hasn't responded yet?"
    """

    def __init__(self):
        self._entries: deque[MemoryEntry] = deque(maxlen=HISTORY_LEN)

    def insert(self, alert: ThreatAlert) -> None:
        if alert.threat_type == "none" or alert.confidence < 0.3:
            return
        self._entries.append(
            MemoryEntry(alert=alert, original_confidence=alert.confidence)
        )

    def active_threat(self) -> Optional[ThreatAlert]:
        """Returns highest-confidence live threat, confidence adjusted for decay."""
        best: Optional[MemoryEntry] = None
        for entry in self._entries:
            if not entry.is_alive():
                continue
            if best is None or entry.current_confidence() > best.current_confidence():
                best = entry
        if best is None:
            return None
        # Return alert with decayed confidence
        alert = best.alert
        alert.confidence = best.current_confidence()
        return alert

    def is_duplicate(self, alert: ThreatAlert) -> bool:
        """True if same threat type + similar direction seen within duplicate window."""
        now = time.time()
        for entry in self._entries:
            if now - entry.inserted_at > DUPLICATE_WINDOW:
                continue
            if entry.alert.threat_type != alert.threat_type:
                continue
            angle_diff = abs(entry.alert.direction_degrees - alert.direction_degrees)
            angle_diff = min(angle_diff, 360 - angle_diff)
            if angle_diff < 30:
                return True
        return False

    def summary_tokens(self) -> str:
        """Compact history string for Gemma prompt context. ~30 tokens max."""
        alive = [e for e in self._entries if e.is_alive()]
        if not alive:
            return "history:none"
        parts = []
        for e in alive[-3:]:
            c = e.current_confidence()
            parts.append(f"{e.alert.threat_type}@{e.alert.direction_degrees:.0f}deg:{c:.2f}")
        return "history:" + "|".join(parts)
