"""
Lightweight multi-object tracker using IoU matching across frames.
Computes velocity, acceleration, and collision ETA per tracked object.
No external dependencies — pure numpy.
"""
import logging
_log = logging.getLogger(__name__)

import time
import math
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


MAX_AGE = 5          # frames before track is dropped
IOU_THRESHOLD = 0.25 # min IoU to match detection to existing track
MAX_TRACKS = 6
MOTION_CONFIRM_FRAMES = 5
MOTION_MIN_HOLD_S = 1.2
MOTION_ENTER = 0.35
MOTION_EXIT = 0.18


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
    motion_state: str = "static"
    motion_votes: int = 0
    motion_changed_at: float = 0.0

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
        current = self.motion_state
        now = time.time()
        if v > 3.0:
            target = "closing-fast"
        elif v > MOTION_ENTER:
            target = "closing"
        elif v < -MOTION_ENTER:
            target = "receding"
        else:
            target = "static"

        if current != "static" and target != current and (now - self.motion_changed_at) < MOTION_MIN_HOLD_S:
            self.motion_votes = 0
            return current

        if target == current:
            self.motion_votes = 0
            return current

        if current in ("closing-fast", "closing") and target == "static":
            target = current if v > MOTION_EXIT else "static"
        elif current == "receding" and target == "static":
            target = current if v < -MOTION_EXIT else "static"

        self.motion_votes += 1
        if self.motion_votes < MOTION_CONFIRM_FRAMES:
            return current

        self.motion_votes = 0
        self.motion_state = target
        self.motion_changed_at = now
        return self.motion_state


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
                    # Fast-moving objects may have low IoU even when matching.
                    # Use center proximity as a secondary signal to avoid spawning
                    # duplicate tracks for the same object moving quickly.
                    if iou < IOU_THRESHOLD:
                        db = det["bbox"]
                        tb = trk.bbox
                        dcx = (db[0] + db[2]) / 2
                        dcy = (db[1] + db[3]) / 2
                        tcx = (tb[0] + tb[2]) / 2
                        tcy = (tb[1] + tb[3]) / 2
                        center_dist = math.hypot(dcx - tcx, dcy - tcy)
                        if center_dist < 44:  # ~20% of 224px image
                            iou = IOU_THRESHOLD  # just enough to match
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
                trk.motion_state = "static"
                trk.motion_votes = 0
                trk.motion_changed_at = time.time()
                self._tracks.append(trk)
                self._next_id += 1

        # Prune dead tracks; prioritize approaching over receding
        self._tracks = [t for t in self._tracks if t.missed < MAX_AGE]

        def _priority(t: TrackState) -> float:
            v = t.velocity_mps()
            if v > 0.5:     # clearly approaching
                mul = 1.6
            elif v > 0.1:   # weakly approaching / static
                mul = 1.1
            elif v < -1.0:  # fast-receding — deprioritize strongly
                mul = 0.2
            elif v < -0.3:  # gently receding
                mul = 0.5
            else:
                mul = 1.0
            return t.confidence * mul

        self._tracks = sorted(self._tracks, key=_priority, reverse=True)[:MAX_TRACKS]

        # Build enriched output
        return self._enrich(detections)

    def _enrich(self, detections: list[dict]) -> list[dict]:
        """Add velocity/eta/motion_label to each detection from matched track."""
        # Build label index once — O(N+M) instead of O(N×M)
        trk_by_label: dict[str, list] = {}
        for trk in self._tracks:
            trk_by_label.setdefault(trk.label, []).append(trk)

        enriched = []
        for det in detections:
            best_trk = None
            best_score = -1
            for trk in trk_by_label.get(det["label"], []):
                angle_diff = abs(trk.direction_deg - det["direction_deg"]) % 360
                angle_diff = min(angle_diff, 360 - angle_diff)
                score = trk.confidence - angle_diff / 180
                if score > best_score:
                    best_score, best_trk = score, trk
            d = dict(det)
            if best_trk and best_trk.age >= 2:
                v = best_trk.velocity_mps()
                d["velocity_mps"] = round(v, 2)
                d["eta_s"] = round(best_trk.eta_seconds(), 1) if best_trk.eta_seconds() else None
                d["motion"] = best_trk.motion_label()
                d["track_age"] = best_trk.age
                # Lead correction: bbox is from ~2 frames ago at 10fps → 0.2s lag.
                # Approaching objects appear further than they are by v*lag.
                if v > 0.1 and "distance_m" in d:
                    d["distance_m"] = round(max(0.3, d["distance_m"] - v * 0.08), 2)
            enriched.append(d)
        return enriched

    def active_tracks(self) -> list[TrackState]:
        return [t for t in self._tracks if t.missed == 0]
