"""
Cascade router: decides which inference tier to invoke.
Tier 0: No action (threat below threshold)
Tier 1: Haptic direct from LiteRT (fast, high-confidence clear detections)
Tier 2: Gemma inference (ambiguous or complex multi-threat)

Routing score = f(detection_confidence, audio_confidence, motion_magnitude, threat_memory)
"""
import time
from dataclasses import dataclass
from typing import Optional
from .threat_memory import ThreatMemory


@dataclass
class SensorSnapshot:
    detections: list[dict]
    audio_labels: list[tuple[str, float]]
    heading_deg: Optional[float]
    pitch_deg: Optional[float]
    flow_vectors: Optional[list[dict]]
    captured_at: float = 0.0

    def __post_init__(self):
        if self.captured_at == 0.0:
            self.captured_at = time.time()

    def max_detection_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return max(d["confidence"] for d in self.detections)

    def max_audio_confidence(self) -> float:
        if not self.audio_labels:
            return 0.0
        return max(c for _, c in self.audio_labels)

    def max_motion_magnitude(self) -> float:
        if not self.flow_vectors:
            return 0.0
        return max(f["magnitude"] for f in self.flow_vectors)

    def is_stairs_likely(self) -> bool:
        return (
            self.pitch_deg is not None and abs(self.pitch_deg) > 10
            and self.max_detection_confidence() < 0.5
        )


TIER1_THRESHOLD = 0.62   # Direct haptic, no LLM (car@0.85 conf + <3m prox → 0.625)
TIER2_THRESHOLD = 0.18   # Invoke Gemma
GEMMA_COOLDOWN  = 0.8    # Min seconds between Gemma calls


class CascadeRouter:
    """
    Scores each sensor snapshot and routes to appropriate action tier.
    Prevents Gemma spam via cooldown. Uses memory to stay alert between calls.
    """

    def __init__(self, memory: ThreatMemory):
        self.memory = memory
        self._last_gemma_call: float = 0.0
        self._gemma_call_count: int = 0
        self.last_score: float = 0.0

    def route(self, snap: SensorSnapshot) -> int:
        score = self._score(snap)
        self.last_score = score

        if score >= TIER1_THRESHOLD:
            return 1

        now = time.time()
        cooldown_ok = (now - self._last_gemma_call) >= GEMMA_COOLDOWN

        if score >= TIER2_THRESHOLD and cooldown_ok:
            self._last_gemma_call = now
            self._gemma_call_count += 1
            return 2

        # Check memory: even if live score low, memory may sustain alert
        active = self.memory.active_threat()
        if active and active.confidence > 0.35:
            return 1  # Sustain from memory, no new Gemma call

        return 0

    def _score(self, snap: SensorSnapshot) -> float:
        vis  = snap.max_detection_confidence() * 0.50
        aud  = snap.max_audio_confidence()     * 0.25
        mot  = snap.max_motion_magnitude()     * 0.15
        pit  = 0.10 if snap.is_stairs_likely() else 0.0

        # Proximity boost: close objects are more urgent regardless of YOLO confidence.
        # <3m → +0.20, <6m → +0.10, <12m → +0.05
        min_dist = min(
            (d.get("distance_m") or 99.0 for d in snap.detections),
            default=99.0,
        )
        prox = 0.20 if min_dist < 3 else (0.10 if min_dist < 6 else (0.05 if min_dist < 12 else 0.0))

        return min(vis + aud + mot + pit + prox, 1.0)

    def stats(self) -> dict:
        return {
            "gemma_calls": self._gemma_call_count,
            "last_call_ago": round(time.time() - self._last_gemma_call, 1),
        }
