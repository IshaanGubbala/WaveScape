"""
Lightweight multi-object tracker using IoU matching across frames.
Computes velocity, acceleration, and collision ETA per tracked object.
No external dependencies — pure numpy.
"""
import time
import math
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


MAX_AGE = 5          # frames before track is dropped
IOU_THRESHOLD = 0.25 # min IoU to match detection to existing track
MAX_TRACKS = 6


@dataclass
class TrackState:
    track_id: int
    label: str
    confidence: float
    direction_deg: float
    distance_m: float
    bbox: list              # [x1, y1, x2, y2]
    history: deque = field(default_factory=lambda: deque(maxlen=8))
    age: int = 0
    missed: int = 0
    first_seen: float = field(default_factory=time.time)

    def update(self, detection: dict) -> None:
        self.history.append({
            "distance_m": detection["distance_m"],
            "direction_deg": detection["direction_deg"],
            "t": time.time(),
        })
        self.label = detection["label"]
        self.confidence = detection["confidence"]
        self.direction_deg = detection["direction_deg"]
        self.distance_m = detection["distance_m"]
        self.bbox = detection.get("bbox", self.bbox)
        self.missed = 0
        self.age += 1

    def velocity_mps(self) -> float:
        """Positive = approaching (distance decreasing), negative = receding."""
        if len(self.history) < 2:
            return 0.0
        pts = list(self.history)
        dt = pts[-1]["t"] - pts[0]["t"]
        if dt < 0.05:
            return 0.0
        dd = pts[0]["distance_m"] - pts[-1]["distance_m"]  # positive = approaching
        return dd / dt

    def acceleration_mps2(self) -> float:
        if len(self.history) < 3:
            return 0.0
        pts = list(self.history)
        n = len(pts)
        mid = n // 2
        dt1 = pts[mid]["t"] - pts[0]["t"]
        dt2 = pts[-1]["t"] - pts[mid]["t"]
        if dt1 < 0.05 or dt2 < 0.05:
            return 0.0
        v1 = (pts[0]["distance_m"] - pts[mid]["distance_m"]) / dt1
        v2 = (pts[mid]["distance_m"] - pts[-1]["distance_m"]) / dt2
        return (v2 - v1) / ((dt1 + dt2) / 2)

    def eta_seconds(self) -> Optional[float]:
        """Seconds until object reaches 0m (collision). None if receding or static."""
        v = self.velocity_mps()
        if v <= 0.1:
            return None
        d = self.distance_m
        a = self.acceleration_mps2()
        if abs(a) < 0.1:
            return d / v
        # quadratic: d = v*t - 0.5*a*t^2 (a positive = accelerating toward)
        discriminant = v**2 + 2 * a * d
        if discriminant < 0:
            return None
        return (-v + math.sqrt(max(0, discriminant))) / (-a) if a != 0 else d / v

    def motion_label(self) -> str:
        v = self.velocity_mps()
        if v > 3.0:
            return "closing-fast"
        elif v > 0.5:
            return "closing"
        elif v < -0.5:
            return "receding"
        return "static"


def _iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class ObjectTracker:
    """
    Frame-to-frame IoU tracker. Call update() each frame with YOLO detections.
    Returns enriched detections with velocity, acceleration, ETA.
    """

    def __init__(self):
        self._tracks: list[TrackState] = []
        self._next_id = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """Match detections to existing tracks, return enriched detections."""
        # Match by IoU
        matched_det = set()
        matched_trk = set()

        for ti, trk in enumerate(self._tracks):
            best_iou, best_di = 0.0, -1
            for di, det in enumerate(detections):
                if di in matched_det:
                    continue
                if det["label"] != trk.label:
                    continue
                if not det.get("bbox") or not trk.bbox:
                    # fallback: match by direction proximity
                    angle_diff = abs(det["direction_deg"] - trk.direction_deg) % 360
                    angle_diff = min(angle_diff, 360 - angle_diff)
                    if angle_diff < 20:
                        iou = 0.3
                    else:
                        iou = 0.0
                else:
                    iou = _iou(det["bbox"], trk.bbox)
                if iou > best_iou:
                    best_iou, best_di = iou, di
            if best_iou >= IOU_THRESHOLD and best_di >= 0:
                matched_det.add(best_di)
                matched_trk.add(ti)
                self._tracks[ti].update(detections[best_di])

        # Unmatched tracks — increment missed
        for ti, trk in enumerate(self._tracks):
            if ti not in matched_trk:
                trk.missed += 1

        # New detections → new tracks
        for di, det in enumerate(detections):
            if di not in matched_det:
                trk = TrackState(
                    track_id=self._next_id,
                    label=det["label"],
                    confidence=det["confidence"],
                    direction_deg=det["direction_deg"],
                    distance_m=det["distance_m"],
                    bbox=det.get("bbox", []),
                )
                trk.update(det)
                self._tracks.append(trk)
                self._next_id += 1

        # Prune dead tracks
        self._tracks = [t for t in self._tracks if t.missed < MAX_AGE]
        self._tracks = sorted(self._tracks, key=lambda t: t.confidence, reverse=True)[:MAX_TRACKS]

        # Build enriched output
        return self._enrich(detections)

    def _enrich(self, detections: list[dict]) -> list[dict]:
        """Add velocity/eta/motion_label to each detection from matched track."""
        enriched = []
        for det in detections:
            best_trk = None
            best_score = -1
            for trk in self._tracks:
                if trk.label != det["label"]:
                    continue
                angle_diff = abs(trk.direction_deg - det["direction_deg"]) % 360
                angle_diff = min(angle_diff, 360 - angle_diff)
                score = trk.confidence - angle_diff / 180
                if score > best_score:
                    best_score, best_trk = score, trk
            d = dict(det)
            if best_trk and best_trk.age >= 2:
                d["velocity_mps"] = round(best_trk.velocity_mps(), 2)
                d["eta_s"] = round(best_trk.eta_seconds(), 1) if best_trk.eta_seconds() else None
                d["motion"] = best_trk.motion_label()
                d["track_age"] = best_trk.age
            enriched.append(d)
        return enriched

    def active_tracks(self) -> list[TrackState]:
        return [t for t in self._tracks if t.missed == 0]
