"""
Live Sensor Logit Steering (LSLS) — practical Idea 4 implementation.

True LSAI (mid-residual activation injection) needs llama.cpp C source patches
that are infeasible in a single session. This achieves the same end-effect via
output-layer logit_bias:

  Sensor state → boost token IDs for active concepts → bias generation
  without spending any prompt tokens to describe the sensor situation.

Pipeline:
  1. At startup: tokenize each concept's vocabulary via /tokenize endpoint
  2. Per Tier 2 call: sensor_to_logit_bias(snap) → dict[token_id, bias]
  3. Pass to /v1/chat/completions as logit_bias param

Stacks with Idea 1 (KV cache), Idea 2 (json_schema grammar). Grammar already
constrains enums (urgency, haptic_pattern); LSLS biases the unconstrained
fields (direction_degrees, threat_type) toward sensor-aligned tokens.
"""
from __future__ import annotations
import requests
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .router import SensorSnapshot


# Concept → vocabulary words. Words tokenized at startup.
# Multiple variants/casings since Gemma may pick any. Set-deduped after tokenize.
_CONCEPT_VOCAB: dict[str, list[str]] = {
    # Direction phrases (boost direction_degrees prose / natural_language)
    "direction_left":   ["left", " left", "Left", " Left", "270"],
    "direction_right":  ["right", " right", "Right", " Right", "90"],
    "direction_ahead":  ["ahead", " ahead", "Ahead", " Ahead", " front", "0"],
    "direction_behind": ["behind", " behind", "180"],

    # Urgency keywords (json_schema enum already constrains, bias picks correct one)
    "urgency_critical": ["critical", " critical", "Critical"],
    "urgency_high":     ["high", " high", "High"],
    "urgency_medium":   ["medium", " medium", "Medium"],
    "urgency_low":      ["low", " low", "Low"],

    # Haptic patterns (json_schema enum, bias picks correct one)
    "haptic_left":      ["rapid_left", "_left"],
    "haptic_right":     ["rapid_right", "_right"],
    "haptic_center":    ["rapid_center", "_center"],
    "haptic_pulse":     ["slow_pulse", "pulse"],

    # Object type tokens (boost threat_type field)
    "obj_vehicle":      ["vehicle", " vehicle", "car", " car", "Car", "truck", " truck"],
    "obj_person":       ["person", " person", "Person", "pedestrian"],
    "obj_animal":       ["animal", " animal", "dog", " dog"],

    # Audio context tokens (natural_language / threat_type bias)
    "audio_horn":       ["horn", " horn", "honking"],
    "audio_siren":      ["siren", " siren"],
}


class LogitSteerer:
    """
    Tokenizes concept vocab once at construction, computes per-call logit_bias
    from live SensorSnapshot.
    """

    def __init__(self, server_url: str = "http://localhost:8081", weight_scale: float = 3.0):
        self.server_url = server_url
        self.weight_scale = weight_scale
        self._token_map: dict[str, list[int]] = {}
        self._calibrate()

    def _calibrate(self) -> None:
        """Tokenize each concept's words via /tokenize, dedupe IDs."""
        for concept, words in _CONCEPT_VOCAB.items():
            ids: set[int] = set()
            for w in words:
                try:
                    r = requests.post(
                        f"{self.server_url}/tokenize",
                        json={"content": w}, timeout=5,
                    )
                    if r.ok:
                        ids.update(r.json().get("tokens", []))
                except Exception:
                    pass
            self._token_map[concept] = sorted(ids)

    def calibration_summary(self) -> str:
        return ", ".join(f"{k}:{len(v)}t" for k, v in self._token_map.items())

    def bias_for(self, snap: "SensorSnapshot") -> dict[int, float]:
        """Compute logit_bias dict from sensor snapshot. Returns {token_id: bias}."""
        bias: dict[int, float] = {}

        # ---- Direction (from detection bbox direction_deg) ----
        if snap.detections:
            # Weight by detection confidence — focus on most prominent object
            total_w = sum(d.get("confidence", 0.0) for d in snap.detections) or 1.0
            avg_dir = sum(d.get("direction_deg", 180) * d.get("confidence", 0.0)
                          for d in snap.detections) / total_w
            if avg_dir < 67.5 or avg_dir >= 292.5:
                self._add(bias, "direction_ahead", 1.0)
            elif avg_dir < 112.5:
                self._add(bias, "direction_right", 1.5)
                self._add(bias, "haptic_right", 1.5)
            elif avg_dir < 247.5:
                self._add(bias, "direction_behind", 1.0)
            else:
                self._add(bias, "direction_left", 1.5)
                self._add(bias, "haptic_left", 1.5)

        # ---- Urgency (from confidence + motion + distance) ----
        max_conf = snap.max_detection_confidence()
        any_closing = any(
            d.get("motion") in ("closing", "closing-fast") for d in snap.detections
        )
        nearest = min(
            (d.get("distance_m", 999.0) for d in snap.detections if d.get("distance_m") is not None),
            default=999.0,
        )

        if (max_conf > 0.85 and any_closing and nearest < 5.0):
            self._add(bias, "urgency_critical", 2.0)
            self._add(bias, "haptic_center", 0.5)   # gentle nudge if no L/R signal
        elif max_conf > 0.65 or any_closing:
            self._add(bias, "urgency_high", 1.5)
        elif max_conf > 0.4:
            self._add(bias, "urgency_medium", 1.0)
        else:
            self._add(bias, "urgency_low", 0.8)

        # ---- Object type (from YOLO labels) ----
        labels = {d.get("label", "") for d in snap.detections}
        if labels & {"car", "truck", "bus", "motorcycle", "bicycle"}:
            self._add(bias, "obj_vehicle", 2.0)
        if "person" in labels:
            self._add(bias, "obj_person", 2.0)
        if labels & {"dog", "cat", "horse", "cow", "sheep"}:
            self._add(bias, "obj_animal", 1.5)

        # ---- Audio context ----
        audio_lower = " ".join(l.lower() for l, _ in (snap.audio_labels or []))
        if "horn" in audio_lower or "honk" in audio_lower:
            self._add(bias, "audio_horn", 2.0)
            # Horn implies vehicle even if YOLO missed it
            self._add(bias, "obj_vehicle", 1.0)
        if "siren" in audio_lower:
            self._add(bias, "audio_siren", 2.0)
            self._add(bias, "urgency_critical", 1.5)

        return bias

    def _add(self, bias: dict[int, float], concept: str, weight: float) -> None:
        scaled = weight * self.weight_scale
        for tid in self._token_map.get(concept, []):
            bias[tid] = bias.get(tid, 0.0) + scaled
