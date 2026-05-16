"""
Mac Scene Service — runs on Mac (port 8083).
Pi sends depth sectors + YOLO + beam every ~0.5s.
Returns wall map, ranked escape paths, merged scene objects, Gemma 4 alert.

Architecture:
  /analyze  → instant (~2ms): path planning + tracker + cached Gemma scene
  background thread → calls Gemma 4 every ~1.5s, updates scene cache

Run:
  python tools/mac_scene_service.py
  python tools/mac_scene_service.py --port 8083 --gemma http://localhost:8081
"""

import argparse
import asyncio
import json
import math
import threading
import time
from typing import Optional

try:
    from flask import Flask, request, jsonify
except ImportError:
    raise SystemExit("pip install flask")

try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    import websockets
    _HAS_WS = True
except ImportError:
    _HAS_WS = False

PORT    = 8083
WS_PORT = 8084
GEMMA_URL = "http://localhost:8081"

app = Flask(__name__)

# ── WebSocket server state ────────────────────────────────────────────────────

_ws_clients: set = set()
_ws_loop: Optional[asyncio.AbstractEventLoop] = None
_ws_last_scene_hash: dict = {}  # per-client dedup: skip sending unchanged scenes

# ── Gemma scene builder ───────────────────────────────────────────────────────

# CJK compact format — matches what the fine-tuned model was trained on
_SCENE_SYSTEM = (
    "Output 8 CJK chars encoding 2 closest objects. Format: T1O1L1U1 T2O2L2U2 (no spaces). "
    "Angle: T(tens) O(ones), deg=T*19+O. "
    "T alphabet: 一二三四五六七八九十甲乙丙丁戊己庚辛壬 (idx 0-18). "
    "O alphabet: 子丑寅卯辰巳午未申酉戌亥角亢氐房心尾箕 (idx 0-18). "
    "L: 车卡巴人自物 (car truck bus person bike obstacle). "
    "U: 危急中远 (crit<2m, high<4m, med<8m, low>=8m). "
    "Pick 2 closest objects. No other text."
)

# Decode tables (mirror of spatial_mapper.py)
_T_CHARS = list("一二三四五六七八九十甲乙丙丁戊己庚辛壬")
_O_CHARS = list("子丑寅卯辰巳午未申酉戌亥角亢氐房心尾箕")
_LABELS_CJK  = "车卡巴人自物"
_URG_CJK     = "危急中远"
_LABEL_NAMES = ["car", "truck", "bus", "person", "bike", "obstacle"]
_URG_NAMES   = ["critical", "high", "medium", "low"]
_URG_DIST    = [1.0, 3.0, 6.0, 12.0]

_gemma_lock   = threading.Lock()
_gemma_cache  = {"scene": "", "ts": 0.0, "input_hash": 0}
_gemma_latest = {"sector_depths": [], "scene_objects": [], "beam_scan": [], "heading": 0.0}
_gemma_latest_lock = threading.Lock()
_GEMMA_UPDATE_HZ  = 1.0 / 1.5   # ~0.67 calls/sec
_SCENE_UNCHANGED_TTL = 4.0       # reuse scene if input hash same within this window


def _input_hash(sector_depths, scene_objects, beam_scan) -> int:
    def _urg(d): return 0 if d < 2 else 1 if d < 4 else 2 if d < 8 else 3
    det_sig = tuple(sorted(
        (o["label"], round(o["angle_deg"] / 15) * 15, _urg(o["dist_m"]))
        for o in scene_objects if o["dist_m"] < 15
    ))
    beam_sig = tuple(
        round(b.get("direction_deg", 0) / 45) * 45
        for b in beam_scan if b.get("energy", 0) > 0.08
    )
    sector_sig = tuple(
        (round(s["angle_deg"] / 10) * 10, s.get("safe", True))
        for s in sector_depths
    )
    return hash((det_sig, beam_sig, sector_sig))


def _build_scene_prompt(sector_depths, scene_objects, beam_scan, heading) -> str:
    """Build compact text prompt matching the CJK fine-tune's expected input format."""
    parts = [f"heading={heading:.0f}°"]

    if scene_objects:
        obj_strs = []
        for o in scene_objects[:4]:
            approach = " cross" if o.get("crossing") else (" appr" if o.get("approaching") else "")
            snd = ""
            # Check if beam scan annotates this object
            for b in beam_scan:
                bdir = float(b.get("direction_deg", 0))
                delta = abs((bdir - o["angle_deg"] + 180) % 360 - 180)
                if delta < 35 and b.get("energy", 0) > 0.06:
                    snd = f" snd={b.get('label','noise')}"
                    break
            obj_strs.append(
                f"{o['label']}@{o['angle_deg']:.0f}° {o['dist_m']:.1f}m{approach}{snd}"
            )
        parts.append("visual=[" + ", ".join(obj_strs) + "]")

    top_beams = [b for b in beam_scan if b.get("energy", 0) > 0.08]
    if top_beams:
        b_strs = [f"{b.get('label','noise')}@{b['direction_deg']:.0f}°" for b in top_beams[:2]]
        parts.append("audio=[" + ",".join(b_strs) + "]")

    return " | ".join(parts)


def _decode_cjk(buf: str, fallback_objects: list[dict]) -> str:
    """Decode CJK compact output → human-readable scene alert string."""
    _T_SET = set(_T_CHARS)
    _O_SET = set(_O_CHARS)
    _L_SET = set(_LABELS_CJK)
    _U_SET = set(_URG_CJK)

    chars = [c for c in buf if c in _T_SET | _O_SET | _L_SET | _U_SET]
    decoded = []
    i = 0
    while i + 3 < len(chars) and len(decoded) < 2:
        t, o, l, u = chars[i], chars[i+1], chars[i+2], chars[i+3]
        try:
            deg = float(min(359, _T_CHARS.index(t) * 19 + _O_CHARS.index(o)))
        except ValueError:
            i += 1
            continue
        if l in _L_SET and u in _U_SET:
            li = _LABELS_CJK.index(l)
            ui = _URG_CJK.index(u)
            decoded.append({
                "label": _LABEL_NAMES[li],
                "angle": deg,
                "urgency": _URG_NAMES[ui],
                "dist": _URG_DIST[ui],
            })
            i += 4
        else:
            i += 1

    if not decoded:
        # Fallback: build from tracker objects if CJK parse failed
        for obj in fallback_objects[:2]:
            decoded.append({
                "label": obj["label"],
                "angle": obj["angle_deg"],
                "urgency": obj["urgency"],
                "dist": obj["dist_m"],
            })

    if not decoded:
        return ""

    parts = []
    for d in decoded:
        direction = "ahead" if abs((d["angle"] + 180) % 360 - 180) < 22 else \
                    "right"  if 0 < d["angle"] < 180 else "left"
        parts.append(f"{d['label']} {d['dist']:.0f}m {direction} [{d['urgency']}]")

    return " · ".join(parts)


def _call_gemma(prompt: str, gemma_url: str, fallback_objects: list[dict]) -> str:
    if not _HAS_REQUESTS:
        return ""
    body = {
        "model": "gemma",
        "messages": [
            {"role": "system", "content": _SCENE_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": 12,
        "temperature": 0.0,
        "stream": False,
        "stop": ["<end_of_turn>", "</s>"],
        "cache_prompt": True,
    }
    try:
        resp = _req.post(f"{gemma_url}/v1/chat/completions", json=body, timeout=3.0)
        if resp.status_code == 200:
            choice = resp.json().get("choices", [{}])[0]
            raw = (choice.get("message", {}).get("content") or "").strip()
            return _decode_cjk(raw, fallback_objects)
    except Exception as e:
        print(f"  [gemma] {e}")
    return ""


def _gemma_worker(gemma_url: str):
    """Background thread: calls Gemma at ~0.67Hz, updates _gemma_cache."""
    while True:
        time.sleep(1.0 / _GEMMA_UPDATE_HZ)
        with _gemma_latest_lock:
            sectors  = list(_gemma_latest["sector_depths"])
            objs     = list(_gemma_latest["scene_objects"])
            beams    = list(_gemma_latest["beam_scan"])
            heading  = float(_gemma_latest["heading"])

        ih = _input_hash(sectors, objs, beams)
        with _gemma_lock:
            unchanged = (ih == _gemma_cache["input_hash"]
                         and (time.time() - _gemma_cache["ts"]) < _SCENE_UNCHANGED_TTL)
        if unchanged:
            continue

        prompt = _build_scene_prompt(sectors, objs, beams, heading)
        scene  = _call_gemma(prompt, gemma_url, objs)
        if scene:
            with _gemma_lock:
                _gemma_cache["scene"]      = scene
                _gemma_cache["ts"]         = time.time()
                _gemma_cache["input_hash"] = ih
            print(f"  [SCENE] {scene}")
            # Push alert immediately to all connected Pi clients
            ws_broadcast_threadsafe({
                "type":        "alert",
                "gemma_scene": scene,
                "ts":          time.time(),
            })


def _get_cached_scene() -> str:
    with _gemma_lock:
        age = time.time() - _gemma_cache["ts"]
        return _gemma_cache["scene"] if age < 8.0 else ""


# ── persistent object tracker ─────────────────────────────────────────────────

_tracker_lock = threading.Lock()
_tracked: dict[str, dict] = {}
TRACK_TTL    = 2.0           # drop ghost tracks fast
ANGLE_TOL    = 40.0          # degrees — merge tracks within this band
EMA_ALPHA    = 0.15          # smoother velocity (less reactive to YOLO noise)
ANGLE_ALPHA  = 0.4           # angle EMA weight
V_DIST_MAX   = 4.0           # m/s — cap velocity magnitude
V_ANGLE_MAX  = 60.0          # deg/s — cap angular velocity

COAST_THRESHOLD_S = 0.5      # wait 500ms before coasting (not 200ms)
PREDICT_STEPS     = [0.5, 1.0]   # fewer ghost dots


def _track_update(detections: list[dict]) -> list[dict]:
    now = time.time()
    with _tracker_lock:
        # Expire tracks with no real detection for TRACK_TTL seconds
        for k in [k for k, v in _tracked.items() if now - v["last_seen"] > TRACK_TTL]:
            del _tracked[k]

        # Match incoming detections to tracks (label + angular proximity)
        for det in detections:
            label = str(det.get("label", "obstacle"))
            angle = float(det.get("direction_deg", det.get("angle_deg", 0.0)))
            dist  = float(det.get("distance_m",   det.get("dist_m",  5.0)))
            conf  = float(det.get("confidence", 0.7))
            if conf > 1.0: conf /= 100.0
            if conf < 0.25:  # skip low-confidence detections
                continue

            best_k, best_d = None, float("inf")
            for k, t in _tracked.items():
                if k.split("|")[0] != label:
                    continue
                delta = abs((t["angle"] - angle + 180) % 360 - 180)
                if delta < best_d and delta < ANGLE_TOL:
                    best_d, best_k = delta, k

            if best_k is None:
                _tracked[f"{label}|{int(now*100)%10000}"] = {
                    "label": label, "angle": angle, "dist": dist,
                    "v_dist": 0.0, "v_angle": 0.0,
                    "last_seen": now, "born": now, "count": 1, "conf": conf,
                }
            else:
                t  = _tracked[best_k]
                dt = max(now - t["last_seen"], 0.05)
                raw_vd = (dist - t["dist"]) / dt
                t["v_dist"]  = (1-EMA_ALPHA)*t["v_dist"]  + EMA_ALPHA*max(-V_DIST_MAX, min(V_DIST_MAX, raw_vd))
                da = (angle - t["angle"] + 180) % 360 - 180
                raw_va = da / dt
                t["v_angle"] = (1-EMA_ALPHA)*t["v_angle"] + EMA_ALPHA*max(-V_ANGLE_MAX, min(V_ANGLE_MAX, raw_va))
                t["angle"]   = (t["angle"] + da * ANGLE_ALPHA) % 360
                t["dist"]    = (1-EMA_ALPHA)*t["dist"] + EMA_ALPHA*dist
                t["last_seen"] = now
                t["count"]  += 1
                t["conf"]    = conf

        result = []
        for t in _tracked.values():
            # Coast: extrapolate position from last real detection using velocity
            dt_coast = now - t["last_seen"]
            coasting  = dt_coast > COAST_THRESHOLD_S
            if coasting:
                cur_angle = (t["angle"] + t["v_angle"] * dt_coast) % 360
                cur_dist  = max(0.1, t["dist"] + t["v_dist"] * dt_coast)
            else:
                cur_angle = t["angle"]
                cur_dist  = t["dist"]

            # Approach: strongly negative v_dist AND low angular speed (not just crossing)
            v_angle_abs = abs(t["v_angle"])
            # Crossing guard: if angular speed > 20 dps and radial speed is small,
            # object is lateral/parallel — reduce approach sensitivity
            crossing = v_angle_abs > 20.0 and abs(t["v_dist"]) < 0.5
            approach_thresh = -0.30 if crossing else -0.15  # harder to flag as approaching when crossing
            approaching = t["v_dist"] < approach_thresh
            eta = cur_dist / max(abs(t["v_dist"]), 0.1) if approaching else 99.0

            # Urgency: distance-based floor + approach ETA boost, damped when crossing
            if cur_dist < 2 or (approaching and eta < 2.0):
                urgency = "critical"
            elif cur_dist < 4 or (approaching and eta < 4.0):
                urgency = "high"
            elif cur_dist < 6 or (approaching and eta < 6.0) or (crossing and cur_dist < 5):
                urgency = "medium"
            else:
                urgency = "low"

            # Trajectory prediction: linear extrapolation of angle + dist
            predicted = []
            for future_dt in PREDICT_STEPS:
                p_dist = cur_dist + t["v_dist"] * future_dt
                if p_dist <= 0:
                    break  # object reaches us — stop predicting
                predicted.append({
                    "dt_s":      future_dt,
                    "angle_deg": round((cur_angle + t["v_angle"] * future_dt) % 360, 1),
                    "dist_m":    round(p_dist, 2),
                })

            result.append({
                "label":       t["label"],
                "angle_deg":   round(cur_angle,    1),
                "dist_m":      round(cur_dist,     2),
                "v_dist_mps":  round(t["v_dist"],  2),
                "v_angle_dps": round(t["v_angle"], 2),
                "approaching": approaching,
                "crossing":    crossing,
                "eta_s":       round(min(eta, 99.0), 1),
                "urgency":     urgency,
                "confidence":  round(t["conf"],    2),
                "age_s":       round(now - t["born"], 1),
                "count":       t["count"],
                "coasting":    coasting,
                "coast_s":     round(dt_coast, 2),
                "predicted":   predicted,
            })

    result.sort(key=lambda o: (
        {"critical": 0, "high": 1, "medium": 2, "low": 3}[o["urgency"]],
        o["dist_m"],
    ))
    return result


# ── YOLO-sector depth fusion ──────────────────────────────────────────────────

def _fuse_yolo_into_sectors(sector_depths: list[dict],
                             detections: list[dict]) -> list[dict]:
    """
    If a YOLO detection falls within a sector's angular span and its distance
    is less than the sector's stereo reading, use the YOLO distance.
    YOLO is more accurate at <3m; stereo is better for peripheral/wide coverage.
    """
    if not detections or not sector_depths:
        return sector_depths

    sectors = sorted(sector_depths, key=lambda s: s["angle_deg"])
    if len(sectors) < 2:
        return sectors

    step = abs(sectors[1]["angle_deg"] - sectors[0]["angle_deg"]) / 2.0

    fused = []
    for s in sectors:
        center = s["angle_deg"]
        s_min  = s["min_depth_m"]
        for det in detections:
            det_angle = float(det.get("direction_deg", det.get("angle_deg", 0)))
            det_dist  = float(det.get("distance_m", det.get("dist_m", 99)))
            delta = abs((det_angle - center + 180) % 360 - 180)
            if delta <= step and det_dist < s_min:
                s_min = det_dist
        fused.append({**s, "min_depth_m": round(s_min, 2),
                      "safe": s_min > WALL_DEPTH_M})
    return fused


# ── wall detection ────────────────────────────────────────────────────────────

WALL_DEPTH_M  = 2.0
CLEAR_DEPTH_M = 3.5


def _detect_walls(sector_depths: list[dict]) -> tuple[list[dict], list[dict]]:
    if not sector_depths:
        return [], []

    sectors = sorted(sector_depths, key=lambda s: s["angle_deg"])
    step = (abs(sectors[1]["angle_deg"] - sectors[0]["angle_deg"])
            if len(sectors) > 1 else 15.0)

    def _blocked(s): return float(s["min_depth_m"]) < WALL_DEPTH_M

    groups: list[dict] = []
    for s in sectors:
        b = _blocked(s)
        if not groups or groups[-1]["blocked"] != b:
            groups.append({"blocked": b, "sectors": [s]})
        else:
            groups[-1]["sectors"].append(s)

    walls, corridors = [], []
    for g in groups:
        secs   = g["sectors"]
        start  = float(secs[0]["angle_deg"])
        end    = float(secs[-1]["angle_deg"])
        width  = abs(end - start) + step
        center = (start + end) / 2.0
        if g["blocked"]:
            walls.append({
                "start_deg":   round(start,  1),
                "end_deg":     round(end,    1),
                "center_deg":  round(center, 1),
                "min_depth_m": round(min(float(s["min_depth_m"]) for s in secs), 2),
                "width_deg":   round(width,  1),
            })
        else:
            clearance = min(float(s["min_depth_m"]) for s in secs)
            corridors.append({
                "start_deg":   round(start,    1),
                "end_deg":     round(end,      1),
                "center_deg":  round(center,   1),
                "clearance_m": round(clearance, 2),
                "width_deg":   round(width,    1),
            })
    return walls, corridors


# ── detection-only path planning (no stereo) ─────────────────────────────────

DANGER_RESOLUTION = 5        # degrees per sample
DANGER_SIGMA_BASE = 22.0     # base Gaussian spread per object (degrees)
CORRIDOR_MIN_DEG  = 20.0     # minimum clear arc to count as a corridor
APPROACH_EXTRA    = 1.5      # extra spread multiplier for approaching objects


def _build_danger_field(scene_objects: list[dict], beam_scan: list[dict]) -> list[float]:
    """
    Build a 360°/5° = 72-element danger ring [0.0=clear .. 1.0=blocked].
    Objects contribute Gaussian bumps; approaching objects have wider spread.
    Beam energy adds softer bumps (unseen hazards).
    """
    n = 360 // DANGER_RESOLUTION
    danger = [0.0] * n

    urgency_peak = {"critical": 1.0, "high": 0.80, "medium": 0.55, "low": 0.25}

    for obj in scene_objects:
        center = obj["angle_deg"]
        dist   = obj["dist_m"]
        peak   = urgency_peak.get(obj["urgency"], 0.25)
        # Tighter spread for far objects, wider for close/approaching
        spread = DANGER_SIGMA_BASE * (1.0 + (1.0 / max(dist, 1.0)))
        if obj.get("approaching"):
            spread *= APPROACH_EXTRA
        for i in range(n):
            deg   = i * DANGER_RESOLUTION
            delta = abs((deg - center + 180) % 360 - 180)
            danger[i] = min(1.0, danger[i] + peak * math.exp(-0.5 * (delta / spread) ** 2))

    # Audio bumps (softer — unseen hazards)
    for b in beam_scan:
        energy = float(b.get("energy", 0))
        if energy < 0.07:
            continue
        center = float(b.get("direction_deg", 0))
        peak   = min(0.6, energy * 1.5)
        for i in range(n):
            deg   = i * DANGER_RESOLUTION
            delta = abs((deg - center + 180) % 360 - 180)
            danger[i] = min(1.0, danger[i] + peak * math.exp(-0.5 * (delta / 40.0) ** 2))

    return danger


def _corridors_from_danger(danger: list[float],
                            threshold: float = 0.35) -> list[dict]:
    """
    Find contiguous arcs below danger threshold → corridors.
    Returns corridors sorted by angular width (widest first).
    """
    n   = len(danger)
    res = DANGER_RESOLUTION
    corridors = []

    # Find clear runs (wrap-around aware via double array)
    in_clear = False
    start_i  = 0
    # scan twice to catch wrap-around corridors
    doubled  = list(range(n)) + list(range(n))
    runs: list[tuple[int,int]] = []

    for idx, i in enumerate(doubled):
        clear = danger[i] < threshold
        if clear and not in_clear:
            in_clear = True
            start_i  = idx
        elif not clear and in_clear:
            in_clear = False
            run_len  = idx - start_i
            if run_len * res >= CORRIDOR_MIN_DEG:
                runs.append((start_i % n, (idx - 1) % n, run_len))

    if in_clear:  # closed at end
        run_len = len(doubled) - start_i
        if run_len * res >= CORRIDOR_MIN_DEG:
            runs.append((start_i % n, (len(doubled) - 1) % n, run_len))

    # Deduplicate (wrap can produce same corridor twice)
    seen = set()
    for s, e, length in runs:
        key = (s, length)
        if key in seen:
            continue
        seen.add(key)
        width   = min(length * res, 355.0)  # cap at nearly full circle
        center  = ((s * res + width / 2.0)) % 360
        # clearance = 1 - mean danger in corridor
        idxs    = [(s + k) % n for k in range(min(length, n))]
        mean_d  = sum(danger[i] for i in idxs) / len(idxs)
        clearance_proxy = round((1.0 - mean_d) * 15.0, 1)  # map to ~meters proxy

        corridors.append({
            "start_deg":   round(s * res, 1),
            "end_deg":     round(((s + length - 1) % n) * res, 1),
            "center_deg":  round(center, 1),
            "clearance_m": clearance_proxy,
            "width_deg":   round(width, 1),
            "source":      "danger_field",
        })

    corridors.sort(key=lambda c: -c["width_deg"])
    return corridors[:6]   # top 6 widest corridors


# ── path planning ─────────────────────────────────────────────────────────────

_path_state_lock = threading.Lock()
_last_escape: dict = {"angle": None, "ts": 0.0}
HYSTERESIS_S       = 2.5   # hold escape direction for this long
HYSTERESIS_DEG     = 30.0  # only switch if new best is >30° different AND score much better


def _gaussian_penalty(angle: float, scene_objects: list[dict], sigma: float = 28.0) -> float:
    """Gaussian object penalty at a given angle. Returns [0,1] — 1.0 = fully clear."""
    penalty = 1.0
    urgency_floor = {"critical": 0.0, "high": 0.2, "medium": 0.55, "low": 0.85}
    for obj in scene_objects:
        delta  = abs((obj["angle_deg"] - angle + 180) % 360 - 180)
        weight = math.exp(-0.5 * (delta / sigma) ** 2)
        floor  = urgency_floor.get(obj["urgency"], 1.0)
        # penalty = min(penalty, lerp(floor, 1.0, 1-weight))
        effective = floor * weight + 1.0 * (1.0 - weight)
        penalty = min(penalty, effective)
    return max(0.0, penalty)


def _audio_penalty(angle: float, beam_scan: list[dict], sigma: float = 35.0) -> float:
    """Penalise corridors with loud sound sources (may indicate unseen hazard)."""
    penalty = 1.0
    for b in beam_scan:
        energy = float(b.get("energy", 0))
        if energy < 0.08:
            continue
        bdir  = float(b.get("direction_deg", 0))
        delta = abs((bdir - angle + 180) % 360 - 180)
        weight = math.exp(-0.5 * (delta / sigma) ** 2)
        # At full energy (1.0) the penalty floor is 0.4; scales with energy
        floor  = max(0.4, 1.0 - energy * 1.2)
        effective = floor * weight + 1.0 * (1.0 - weight)
        penalty = min(penalty, effective)
    return max(0.0, penalty)


def _plan_paths(corridors: list[dict], scene_objects: list[dict],
                beam_scan: list[dict]) -> list[dict]:
    if not corridors:
        return []

    paths = []
    for c in corridors:
        center    = c["center_deg"]
        clearance = c["clearance_m"]
        width     = c["width_deg"]

        # Sample penalty across corridor width every 5°
        sample_angles = [
            (c["start_deg"] + i * 5.0) % 360
            for i in range(max(1, int(width / 5)))
        ]
        obj_p  = min(_gaussian_penalty(a, scene_objects) for a in sample_angles)
        aud_p  = min(_audio_penalty(a, beam_scan) for a in sample_angles)

        clearance_score = 1.0 - math.exp(-clearance / 4.0)
        # Softer forward bias: 0.3 base + 0.7 cosine — never zeros out rear paths
        forward_bias    = 0.3 + 0.7 * (1.0 + math.cos(math.radians(center))) / 2.0
        width_score     = min(width / 45.0, 1.0) ** 0.5   # sqrt: penalise narrow harder

        score = clearance_score * forward_bias * width_score * obj_p * aud_p

        c_norm = center % 360
        if   c_norm < 22.5 or c_norm >= 337.5: direction = "ahead"
        elif c_norm < 67.5:                      direction = "slight-right"
        elif c_norm < 112.5:                     direction = "right"
        elif c_norm < 157.5:                     direction = "rear-right"
        elif c_norm < 202.5:                     direction = "behind"
        elif c_norm < 247.5:                     direction = "rear-left"
        elif c_norm < 292.5:                     direction = "left"
        else:                                    direction = "slight-left"

        paths.append({
            "angle_deg":   round(center,    1),
            "clearance_m": round(clearance, 2),
            "width_deg":   round(width,     1),
            "score":       round(score,     4),
            "direction":   direction,
        })

    paths.sort(key=lambda p: -p["score"])
    return paths


def _apply_hysteresis(paths: list[dict]) -> tuple[float, list[dict]]:
    """
    Sticky escape direction: keep last direction unless:
      - It's been HYSTERESIS_S seconds, OR
      - Best new path differs by >HYSTERESIS_DEG and scores >30% better.
    Returns (escape_deg, paths_with_selected_flag).
    """
    if not paths:
        return 0.0, paths

    now = time.time()
    with _path_state_lock:
        last_angle = _last_escape["angle"]
        last_ts    = _last_escape["ts"]

    best = paths[0]
    timeout = (now - last_ts) > HYSTERESIS_S

    if last_angle is None or timeout:
        chosen = best
    else:
        # Find the path closest to last escape direction
        sticky = min(paths, key=lambda p: abs((p["angle_deg"] - last_angle + 180) % 360 - 180))
        delta  = abs((best["angle_deg"] - last_angle + 180) % 360 - 180)
        score_gain = best["score"] - sticky["score"]
        # Switch if much better AND significantly different angle
        if delta > HYSTERESIS_DEG and score_gain > 0.15:
            chosen = best
        else:
            chosen = sticky

    with _path_state_lock:
        _last_escape["angle"] = chosen["angle_deg"]
        _last_escape["ts"]    = now

    for p in paths:
        p["selected"] = p is chosen
    return round(chosen["angle_deg"], 1), paths


def _build_summary(scene_objects, walls, safe_paths, escape_deg) -> str:
    parts = []
    if scene_objects:
        top = scene_objects[0]
        approach = " CROSS" if top.get("crossing") else (" APPR" if top.get("approaching") else "")
        parts.append(f"{top['label']} {top['dist_m']:.1f}m @{top['angle_deg']:.0f}°{approach} [{top['urgency']}]")
        if len(scene_objects) > 1:
            parts.append(f"+{len(scene_objects)-1}")
    else:
        parts.append("clear")
    if walls:
        parts.append(f"{len(walls)} wall(s)")
    if safe_paths:
        sel = next((p for p in safe_paths if p.get("selected")), safe_paths[0])
        parts.append(f"escape→{sel['direction']} ({escape_deg:.0f}°)")
    return " | ".join(parts)


# ── Core analyze logic (shared by HTTP + WebSocket) ──────────────────────────

def _process_analyze(data: dict) -> dict:
    t0 = time.monotonic()
    sector_depths = data.get("sector_depths", [])
    detections    = data.get("detections",    [])
    beam_scan     = data.get("beam_scan",     [])
    heading_deg   = float(data.get("heading_deg", 0.0))

    # 1. Fuse audio-only sources into detection list
    merged_dets = list(detections)
    det_angles  = {float(d.get("direction_deg", d.get("angle_deg", 0))) for d in detections}
    for b in beam_scan:
        if float(b.get("energy", 0)) < 0.08:
            continue
        bdir = float(b.get("direction_deg", 0))
        nearest_delta = min(
            (abs((a - bdir + 180) % 360 - 180) for a in det_angles), default=999
        )
        if nearest_delta > 40:
            merged_dets.append({
                "label": b.get("label", "sound_source"),
                "direction_deg": bdir,
                "distance_m": 8.0,
                "confidence": min(float(b.get("energy", 0)) * 4, 0.6),
            })

    # 2. Persistent tracker
    scene_objects = _track_update(merged_dets)

    # 3. Fuse YOLO into stereo sectors (YOLO wins at <3m)
    fused_sectors = _fuse_yolo_into_sectors(sector_depths, detections)

    # 4. Wall detection
    walls, corridors = _detect_walls(fused_sectors)

    # Fallback: no stereo → build corridors from detection danger field
    if not corridors:
        danger = _build_danger_field(scene_objects, beam_scan)
        corridors = _corridors_from_danger(danger)

    # 5. Path planning with Gaussian penalties + audio penalty
    safe_paths = _plan_paths(corridors, scene_objects, beam_scan)

    # 6. Hysteresis sticky escape direction
    escape_deg, safe_paths = _apply_hysteresis(safe_paths)

    # 7. Summary + cached Gemma scene
    summary     = _build_summary(scene_objects, walls, safe_paths, escape_deg)
    gemma_scene = _get_cached_scene()

    # Feed latest state to Gemma background worker
    with _gemma_latest_lock:
        _gemma_latest["sector_depths"] = fused_sectors
        _gemma_latest["scene_objects"] = scene_objects
        _gemma_latest["beam_scan"]     = beam_scan
        _gemma_latest["heading"]       = heading_deg

    return {
        "type":          "scene",
        "wall_map":       walls,
        "safe_paths":     safe_paths,
        "escape_dir_deg": escape_deg,
        "scene_objects":  scene_objects,
        "n_walls":        len(walls),
        "n_safe_paths":   len(safe_paths),
        "summary":        summary,
        "gemma_scene":    gemma_scene,
        "latency_ms":     round((time.monotonic() - t0) * 1000, 1),
    }


# ── WebSocket broadcast helpers ───────────────────────────────────────────────

async def _ws_broadcast(msg: dict) -> None:
    if not _ws_clients:
        return
    raw  = json.dumps(msg)
    dead = set()
    for ws in list(_ws_clients):
        try:
            await ws.send(raw)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


def ws_broadcast_threadsafe(msg: dict) -> None:
    """Push msg to all connected Pi WS clients from any thread."""
    if _ws_loop and _ws_clients:
        asyncio.run_coroutine_threadsafe(_ws_broadcast(msg), _ws_loop)


# ── WebSocket server ──────────────────────────────────────────────────────────

def _scene_hash(result: dict) -> str:
    """Coarse hash of scene_objects for dedup — rounds dist to 0.5m, angle to 5°."""
    objs = result.get("scene_objects", [])
    key = tuple(
        (o["label"], round(o["angle_deg"] / 5) * 5, round(o["dist_m"] * 2) / 2, o["urgency"])
        for o in objs
    )
    return str(hash(key))


async def _ws_handler(websocket) -> None:
    _ws_clients.add(websocket)
    try:
        remote = getattr(websocket, "remote_address", None) or "unknown"
    except Exception:
        remote = "unknown"
    print(f"[ws] Pi connected from {remote}", flush=True)
    loop = asyncio.get_running_loop()
    try:
        async for raw in websocket:
            try:
                data = json.loads(raw)
            except Exception:
                continue
            msg_type = data.get("type", "sensor")
            if msg_type == "sensor":
                try:
                    result = await loop.run_in_executor(None, _process_analyze, data)
                    payload = json.dumps(result)
                    await websocket.send(payload)
                except Exception as e:
                    print(f"[ws] handler error: {e}", flush=True)
    except Exception as e:
        print(f"[ws] connection error: {e}", flush=True)
    finally:
        _ws_clients.discard(websocket)
        print(f"[ws] Pi disconnected from {remote}", flush=True)


async def _ws_serve(host: str, port: int) -> None:
    async with websockets.serve(_ws_handler, host, port):
        print(f"[mac_scene_service] WS → ws://{host}:{port}")
        await asyncio.Future()  # run forever


def _start_ws_server(host: str = "0.0.0.0", port: int = WS_PORT) -> None:
    global _ws_loop
    if not _HAS_WS:
        print("[mac_scene_service] websockets not installed — WS server disabled (pip install websockets)")
        return
    _ws_loop = asyncio.new_event_loop()
    t = threading.Thread(
        target=lambda: _ws_loop.run_until_complete(_ws_serve(host, port)),
        daemon=True,
    )
    t.start()


# ── Flask endpoints ───────────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    data   = request.get_json(force=True, silent=True) or {}
    result = _process_analyze(data)
    return jsonify(result)


@app.route("/health")
def health():
    with _tracker_lock:
        n_tracks = len(_tracked)
    with _gemma_lock:
        scene_age = round(time.time() - _gemma_cache["ts"], 1)
        scene_preview = _gemma_cache["scene"][:60]
    return jsonify({
        "status": "ok",
        "tracks": n_tracks,
        "scene_age_s": scene_age,
        "scene": scene_preview,
    })


@app.route("/reset", methods=["POST"])
def reset():
    with _tracker_lock:
        _tracked.clear()
    with _gemma_lock:
        _gemma_cache.update({"scene": "", "ts": 0.0, "input_hash": 0})
    with _path_state_lock:
        _last_escape.update({"angle": None, "ts": 0.0})
    return jsonify({"status": "reset"})


# ── startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",    type=int, default=PORT)
    parser.add_argument("--ws-port", type=int, default=WS_PORT, dest="ws_port")
    parser.add_argument("--host",    default="0.0.0.0")
    parser.add_argument("--gemma",   default=GEMMA_URL, help="llama-server URL")
    args = parser.parse_args()

    ws_port = getattr(args, "ws_port", WS_PORT)

    # Start Gemma background worker
    gw = threading.Thread(target=_gemma_worker, args=(args.gemma,), daemon=True)
    gw.start()

    # Start WebSocket server
    _start_ws_server(host=args.host, port=ws_port)

    print(f"[mac_scene_service] HTTP → http://{args.host}:{args.port}  WS → ws://{args.host}:{ws_port}  gemma={args.gemma}")
    if not _HAS_REQUESTS:
        print("  WARNING: requests not installed — Gemma scene disabled")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
