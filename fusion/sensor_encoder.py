"""
Compact sensor state encoder.
Converts raw sensor data → dense prompt tokens.
5x shorter than verbose descriptions → faster prefill → lower latency.

Format: spatial grid + audio labels + IMU heading
Example output: "V:left:car:0.91:8m right:person:0.72:3m | A:horn:0.96,engine:0.88 | H:ahead | Mv:fast-left"
"""
import math
from typing import Optional


def degrees_to_plain(deg: float) -> str:
    """Convert degrees to plain direction word. 0=ahead, 90=right, 180=behind, 270=left."""
    deg = deg % 360
    if deg < 22.5 or deg >= 337.5:
        return "ahead"
    elif deg < 67.5:
        return "right-ahead"
    elif deg < 112.5:
        return "right"
    elif deg < 157.5:
        return "right-behind"
    elif deg < 202.5:
        return "behind"
    elif deg < 247.5:
        return "left-behind"
    elif deg < 292.5:
        return "left"
    else:
        return "left-ahead"


def encode_detections(detections: list[dict], max_objects: int = 4) -> str:
    """
    detections: [{"label": "car", "confidence": 0.91, "direction_deg": 315,
                  "distance_m": 8.0, "velocity_mps": 3.2, "eta_s": 2.5, "motion": "closing-fast"}]
    Encodes trajectory data when available: closing-fast@3.2m/s:ETA2s
    """
    if not detections:
        return "V:clear"
    # Sort by threat priority: closing-fast > closing > static > receding
    priority = {"closing-fast": 0, "closing": 1, "static": 2, "receding": 3}
    top = sorted(
        detections,
        key=lambda d: (priority.get(d.get("motion", "static"), 2), -d["confidence"]),
    )[:max_objects]
    parts = []
    for d in top:
        direction = degrees_to_plain(d["direction_deg"])
        dist = f"{d['distance_m']:.0f}m" if d.get("distance_m") else "?"
        token = f"{direction}:{d['label']}:{d['confidence']:.2f}:{dist}"
        # Append trajectory if tracked for ≥2 frames
        if d.get("motion") and d.get("track_age", 0) >= 2:
            motion = d["motion"]
            token += f":{motion}"
            if d.get("velocity_mps") is not None:
                token += f"@{d['velocity_mps']:.1f}m/s"
            if d.get("eta_s") is not None:
                token += f":ETA{d['eta_s']:.0f}s"
        parts.append(token)
    return "V:" + " ".join(parts)


def encode_audio(audio_labels: list[tuple[str, float]], top_k: int = 3) -> str:
    """
    audio_labels: [("horn", 0.96), ("engine_noise", 0.88), ...]
    """
    if not audio_labels:
        return "A:quiet"
    top = sorted(audio_labels, key=lambda x: x[1], reverse=True)[:top_k]
    parts = [f"{label}:{conf:.2f}" for label, conf in top]
    return "A:" + ",".join(parts)


def encode_imu(
    heading_deg: Optional[float],
    velocity_mps: Optional[float] = None,
    pitch_deg: Optional[float] = None,
) -> str:
    """IMU orientation + motion state."""
    parts = []
    if heading_deg is not None:
        parts.append(f"H:{degrees_to_plain(heading_deg)}")
    if pitch_deg is not None:
        if pitch_deg < -15:
            parts.append("Pitch:down")
        elif pitch_deg > 15:
            parts.append("Pitch:up")
    if velocity_mps is not None:
        if velocity_mps < 0.3:
            parts.append("Mv:still")
        elif velocity_mps < 1.2:
            parts.append("Mv:walking")
        else:
            parts.append("Mv:running")
    return " ".join(parts) if parts else "IMU:?"


def encode_motion_vectors(flow_vectors: list[dict]) -> str:
    """
    flow_vectors: [{"direction_deg": 90, "magnitude": 0.87, "label": "car"}]
    Encodes dominant motion as compact token.
    """
    if not flow_vectors:
        return ""
    dominant = max(flow_vectors, key=lambda f: f["magnitude"])
    direction = degrees_to_plain(dominant["direction_deg"])
    speed = "fast" if dominant["magnitude"] > 0.6 else "slow"
    return f"Mv:{speed}-{direction}"


def build_prompt(
    detections: list[dict],
    audio_labels: list[tuple[str, float]],
    heading_deg: Optional[float] = None,
    pitch_deg: Optional[float] = None,
    flow_vectors: Optional[list[dict]] = None,
    history_summary: str = "history:none",
) -> str:
    """Builds complete compact sensor prompt for Gemma."""
    parts = [
        encode_detections(detections),
        encode_audio(audio_labels),
        encode_imu(heading_deg, pitch_deg=pitch_deg),
    ]
    if flow_vectors:
        mv = encode_motion_vectors(flow_vectors)
        if mv:
            parts.append(mv)
    parts.append(history_summary)
    return " | ".join(parts)


class DeltaEncoder:
    """
    Delta Prompt Encoding (Idea 1, part 2).

    Tracks sensor state across Gemma calls. Changed fields are marked with *
    so Gemma's attention concentrates on what's new. Works synergistically
    with StreamParser's KV-cache history: the previous full state sits in the
    cached turn, so Gemma always has full context even when current prompt is
    abbreviated.

    Usage:
        enc = DeltaEncoder()
        prompt = enc.build_prompt(detections, audio_labels, ...)
    """

    def __init__(self) -> None:
        self._last: dict[str, str] = {}

    def build_prompt(
        self,
        detections: list[dict],
        audio_labels: list[tuple[str, float]],
        heading_deg: Optional[float] = None,
        pitch_deg: Optional[float] = None,
        flow_vectors: Optional[list[dict]] = None,
        history_summary: str = "history:none",
    ) -> str:
        cur: dict[str, str] = {
            "V": encode_detections(detections),
            "A": encode_audio(audio_labels),
            "H": encode_imu(heading_deg, pitch_deg=pitch_deg),
        }
        if flow_vectors:
            mv = encode_motion_vectors(flow_vectors)
            if mv:
                cur["Mv"] = mv
        cur["hist"] = history_summary

        if not self._last:
            # First call — full state, no annotations
            self._last = dict(cur)
            return " | ".join(cur.values())

        # Annotate changed fields with * so Gemma focuses on them
        parts = []
        for k, v in cur.items():
            if self._last.get(k) != v:
                parts.append(f"{v}*")   # * = changed since last call
            else:
                parts.append(v)

        self._last = dict(cur)
        return " | ".join(parts)
