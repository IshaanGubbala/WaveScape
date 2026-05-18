"""
WaveScape live data server.
Serves ui/ static files and streams pipeline state via SSE at /stream.
Integrates with web_ui.py shared state.

Run alongside pipeline:
  python -m output.live_server   (port 8090)
"""
import json
import logging
import math
import os
import time
import threading
from collections import defaultdict
from pathlib import Path

_log = logging.getLogger(__name__)

try:
    from flask import Flask, Response, abort
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

import output.web_ui as _web_ui

UI_DIR = Path(__file__).parent.parent / "ui"
PORT = int(os.environ.get("UI_PORT", 8090))
STREAM_HZ = int(os.environ.get("STREAM_HZ", 10))   # push rate; reduce to 8 on Pi to save CPU

_CAM_HALF_FOV_DEG = 33.0  # must match process/vision.py CAM_HALF_FOV_DEG

app = Flask(__name__, static_folder=None)

# ── helpers ──────────────────────────────────────────────────────────────────

_STATIC_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".jsx": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _static_response(filename: str):
    path = (UI_DIR / filename).resolve()
    if UI_DIR.resolve() not in path.parents and path != UI_DIR.resolve():
        abort(404)
    if not path.is_file():
        abort(404)
    return Response(
        path.read_bytes(),
        content_type=_STATIC_MIME.get(path.suffix.lower(), "application/octet-stream"),
        headers={"Cache-Control": "no-cache"},
    )

def _beam_energy_from_scan(beam_scan: list) -> dict:
    """Map beamform_scan results [{direction_deg, energy}] → {front,right,back,left}."""
    mapping = {"front": 35, "right": 145, "back": 215, "left": 325}
    out = {k: 0.0 for k in mapping}
    if not beam_scan:
        return out
    total_e = sum(b.get("energy", 0) for b in beam_scan) or 1.0
    for b in beam_scan:
        deg = b.get("direction_deg", 0)
        e = b.get("energy", 0) / total_e
        # assign to closest named direction
        best, best_d = "front", 999
        for name, center in mapping.items():
            d = abs((deg - center + 180) % 360 - 180)
            if d < best_d:
                best, best_d = name, d
        out[best] = max(out[best], min(1.0, e * 4))  # scale 0-1
    return out


def _objects_to_threats(objects: list, yolo_detections: list, source: str = "GEMMA") -> list:
    """Merge spatial map objects with YOLO detections into WaveScape threat format."""
    threats = []
    _id = 0

    # From spatial map (Gemma-fused, have direction + distance)
    # Dedup by (label, 20° angle bin) so multiple people at different angles all show
    seen_slots: set = set()  # (label, angle_bin)
    for obj in objects:
        _id += 1
        dist = float(obj.get("distance_m", obj.get("dist_m", obj.get("distance", 5.0))))
        angle = float(obj.get("direction_deg", obj.get("angle_deg", obj.get("dir", 0.0))))
        urgency = obj.get("urgency", "medium")
        if not urgency or urgency == "none":
            urgency = "low" if dist > 8 else "medium" if dist > 4 else "high" if dist > 2 else "critical"
        label = obj.get("label", obj.get("class", "OBJECT")).upper()
        seen_slots.add((label, int(angle / 15)))
        _type = "person" if any(w in label.lower() for w in ["person","pedestrian","human"]) \
            else "vehicle" if any(w in label.lower() for w in ["car","truck","vehicle","bike","motor"]) \
            else "obstacle"
        icon = {"person": "PERSON", "vehicle": "VEHICLE", "obstacle": "OBJ"}.get(_type, "OBJ")
        vel = float(obj.get("v_dist_mps", obj.get("velocity", obj.get("closure_ms", 0.0))))
        v_ang = float(obj.get("v_angle_dps", 0.0))
        threats.append({
            "id": f"T-{_id:04d}",
            "type": _type,
            "icon": icon,
            "label": label,
            "audioClass": obj.get("audio_class", ""),
            "visualClass": obj.get("visual_class", label.lower()),
            "distance": round(dist, 2),
            "angle": round(angle % 360, 1),
            "vDist": round(vel, 3),
            "vAngle": round(v_ang, 3),
            "confidence": int((obj.get("confidence", 0.75) * 100)
                              if obj.get("confidence", 0.75) < 2.0
                              else obj.get("confidence", 75)),
            "velocity": round(vel, 2),
            "urgency": urgency,
            "source": source,
            "serverTs": int(time.time() * 1000),
            "bornAt": int(time.time() * 1000),
            "ttl": 5000,
            "eta": round(dist / max(0.3, abs(vel)), 1) if vel < 0 else 99.0,
            "coasting":   obj.get("coasting", False),
            "crossing":   obj.get("crossing", False),
            "approaching":obj.get("approaching", False),
            "receding":   obj.get("receding", False),
            "predicted":  obj.get("predicted", []),
        })

    # Add YOLO-only detections not already in spatial map (dedup by angle bin)
    for det in yolo_detections[:8]:
        label = str(det.get("label", det.get("class_name", "OBJECT"))).upper()
        angle = float(det.get("direction_deg", det.get("angle_deg", det.get("direction", 0.0))))
        slot = (label, int(angle / 15))
        if slot in seen_slots:
            continue

        # Skip confirmed fast-receding objects — no longer a threat
        v_mps = float(det.get("velocity_mps", 0.0))
        motion = det.get("motion", "")
        if motion == "receding" and v_mps < -0.5:
            continue

        seen_slots.add(slot)
        _id += 1
        conf = det.get("confidence", det.get("conf", 0.7))
        if conf <= 1.0:
            conf = int(conf * 100)
        dist = float(det.get("distance_m", det.get("dist", 4.0)))
        urgency = "low" if dist > 8 else "medium" if dist > 4 else "high" if dist > 2 else "critical"
        _type = "person" if "person" in label.lower() else "vehicle" \
            if any(w in label.lower() for w in ["car","truck","bus","bike"]) else "obstacle"
        icon = {"person": "PERSON", "vehicle": "VEHICLE", "obstacle": "OBJ"}.get(_type, "OBJ")
        approaching = v_mps > 0.3
        receding    = v_mps < -0.3
        eta = round(dist / max(0.1, v_mps), 1) if approaching else 99.0
        threats.append({
            "id": f"T-{_id:04d}",
            "type": _type,
            "icon": icon,
            "label": label,
            "audioClass": "",
            "visualClass": label.lower(),
            "distance": round(dist, 2),
            "angle": round(angle % 360, 1),
            "vDist": round(v_mps, 3),
            "vAngle": 0.0,
            "confidence": int(conf),
            "velocity": round(v_mps, 2),
            "urgency": urgency,
            "source": "YOLO",
            "bornAt": int(time.time() * 1000),
            "ttl": 2000,
            "eta": eta,
            "approaching": approaching,
            "receding":    receding,
            "coasting":    False,
            "crossing":    False,
            "predicted":   [],
        })

    return threats


def _log_to_events(log_entries) -> list:
    events = []
    for i, entry in enumerate(list(log_entries)[:40]):
        if not isinstance(entry, dict):
            continue
        urgency = entry.get("urgency", "medium")
        angle = float(entry.get("direction", entry.get("angle", 0.0)))
        dist = float(entry.get("distance", entry.get("dist", 5.0)))
        events.append({
            "id": f"E-{i:04d}",
            "type": "obstacle",
            "icon": "⚠",
            "label": entry.get("label", entry.get("class", "EVENT")).upper(),
            "audioClass": entry.get("audio_class", ""),
            "visualClass": entry.get("visual_class", ""),
            "distance": round(dist, 2),
            "angle": round(angle % 360, 1),
            "confidence": int(entry.get("confidence", 75)),
            "velocity": float(entry.get("velocity", 0.5)),
            "urgency": urgency,
            "source": entry.get("source", "FUSION"),
            "bornAt": int(entry.get("ts", time.time()) * 1000),
            "ttl": 999999,
            "eta": round(dist / max(0.3, float(entry.get("velocity", 0.5))), 1),
        })
    return events


MAC_SCENE_TTL_S = 5.0  # treat Mac scene stale after this


def _dedup_threats(threats: list) -> list:
    """Remove near-duplicates (same label, <12° apart). Keep highest urgency."""
    out = []
    urgency_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    for t in threats:
        merged = False
        for existing in out:
            if existing["label"] != t["label"]:
                continue
            delta = abs((t["angle"] - existing["angle"] + 180) % 360 - 180)
            if delta < 12:
                # Keep the one with higher urgency (lower rank)
                if urgency_rank.get(t["urgency"], 3) < urgency_rank.get(existing["urgency"], 3):
                    existing.update(t)
                merged = True
                break
        if not merged:
            out.append(dict(t))
    return out


_cpu_prev: list = []  # [user, nice, system, idle, ...]
_sys_stat_cache: dict = {"temp": 0, "cpu_pct": 0, "ts": 0.0}
_SYS_STAT_TTL = 1.0  # seconds — avoid file I/O on every 10Hz SSE tick

def _read_cpu_temp() -> int:
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read().strip()) // 1000
    except Exception:
        return 0

def _read_cpu_pct() -> int:
    global _cpu_prev
    try:
        fields = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        if _cpu_prev:
            delta = [fields[i] - _cpu_prev[i] for i in range(len(fields))]
            total = sum(delta)
            idle  = delta[3]
            pct   = int(100 * (total - idle) / total) if total else 0
        else:
            pct = 0
        _cpu_prev = fields
        return pct
    except Exception:
        return 0

def _cached_sys_stat() -> tuple:
    now = time.time()
    if now - _sys_stat_cache["ts"] < _SYS_STAT_TTL:
        return _sys_stat_cache["temp"], _sys_stat_cache["cpu_pct"]
    temp    = _read_cpu_temp()
    cpu_pct = _read_cpu_pct()
    _sys_stat_cache.update({"temp": temp, "cpu_pct": cpu_pct, "ts": now})
    return temp, cpu_pct


def _enrich_with_mac(gemma_objects: list, mac_objects: list) -> list:
    """Apply Mac tracker velocity/predictions to Pi Gemma objects.
    Gemma gives semantic labels + positions. Mac gives smooth velocity/ETA/predictions.
    Match by label proximity (within 45°). Gemma position wins; Mac kinematics win.
    """
    enriched = []
    for obj in gemma_objects:
        g_label = str(obj.get("label", "")).lower()
        g_angle = float(obj.get("direction_deg", obj.get("angle_deg", 0.0)))
        best_mac = None
        best_delta = 45.0  # max match angle
        for m in mac_objects:
            if m.get("label", "").lower() != g_label:
                continue
            delta = abs((m.get("angle_deg", 0) - g_angle + 180) % 360 - 180)
            if delta < best_delta:
                best_delta, best_mac = delta, m
        result = dict(obj)
        if best_mac:
            # Mac kinematics — keep Gemma's semantic position but add motion
            result["v_dist_mps"]  = best_mac.get("v_dist_mps", 0.0)
            result["v_angle_dps"] = best_mac.get("v_angle_dps", 0.0)
            result["approaching"] = best_mac.get("approaching", False)
            result["receding"]    = best_mac.get("receding", False)
            result["crossing"]    = best_mac.get("crossing", False)
            result["coasting"]    = best_mac.get("coasting", False)
            result["predicted"]   = best_mac.get("predicted", [])
            result["eta_s"]       = best_mac.get("eta_s", 99.0)
            # Mac urgency upgrade only (never downgrade Gemma's semantic urgency)
            urg_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            g_urg = obj.get("urgency", "medium")
            m_urg = best_mac.get("urgency", "medium")
            if urg_rank.get(m_urg, 3) < urg_rank.get(g_urg, 3):
                result["urgency"] = m_urg
        enriched.append(result)

    # Append Mac-only objects (no Gemma match) so tracker sees all detected threats
    matched_mac_ids = set()
    for obj in gemma_objects:
        g_label = str(obj.get("label", "")).lower()
        g_angle = float(obj.get("direction_deg", obj.get("angle_deg", 0.0)))
        for m in mac_objects:
            if m.get("label", "").lower() != g_label:
                continue
            if abs((m.get("angle_deg", 0) - g_angle + 180) % 360 - 180) < 15.0:
                matched_mac_ids.add(id(m))
                break
    for m in mac_objects:
        if id(m) not in matched_mac_ids:
            enriched.append({
                "label":       m.get("label", "OBJECT").upper(),
                "direction_deg": m.get("angle_deg", 0.0),
                "distance_m":  m.get("dist_m", 5.0),
                "urgency":     m.get("urgency", "low"),
                "confidence":  m.get("confidence", 0.5),
                "v_dist_mps":  m.get("v_dist_mps", 0.0),
                "v_angle_dps": m.get("v_angle_dps", 0.0),
                "approaching": m.get("approaching", False),
                "receding":    m.get("receding", False),
                "crossing":    m.get("crossing", False),
                "coasting":    m.get("coasting", False),
                "predicted":   m.get("predicted", []),
                "eta_s":       m.get("eta_s", 99.0),
                "source":      "MAC",
            })
    return enriched


def _yolo_correct_angles(objects: list, yolo_detections: list) -> list:
    """Replace Gemma's LLM-guessed direction_deg with YOLO's pixel-accurate bearing.
    Gemma angle estimates can be wildly off (LLM guesses). YOLO derives angle from
    bbox center pixel — pixel-precise. Match purely by label, pair greedily by confidence.
    """
    yolo_by_label = defaultdict(list)
    for y in yolo_detections:
        lbl   = str(y.get("label", y.get("class_name", ""))).lower()
        angle = y.get("direction_deg")  # None if absent — avoids 999 sentinel collisions
        dist  = float(y.get("distance_m", y.get("dist", 0)))
        conf  = float(y.get("confidence", y.get("conf", 0.5)))
        if angle is not None:
            yolo_by_label[lbl].append({"angle": float(angle), "dist": dist, "conf": conf})
    # Sort each label's detections by confidence desc so best match goes first
    for lbl in yolo_by_label:
        yolo_by_label[lbl].sort(key=lambda x: -x["conf"])

    # Track how many Gemma objects of each label we've already processed
    label_used = defaultdict(int)
    for obj in objects:
        g_label = str(obj.get("label", "")).lower()
        candidates = yolo_by_label.get(g_label, [])
        idx = label_used[g_label]
        if idx < len(candidates):
            best = candidates[idx]
            label_used[g_label] += 1
            obj["direction_deg"] = round(best["angle"], 1)
            if best["dist"] > 0:
                obj["distance_m"] = round(best["dist"], 2)
    return objects


def _build_frame() -> dict:
    """Snapshot all shared pipeline state into WaveScape wire format."""
    with _web_ui._spatial_lock:
        spatial = dict(_web_ui._spatial_map)
    with _web_ui._yolo_lock:
        yolo = dict(_web_ui._yolo_live)
    with _web_ui._perf_lock:
        perf = dict(_web_ui._perf_state)
    with _web_ui._stereo_lock:
        stereo = dict(_web_ui._stereo_state)
    with _web_ui._log_lock:
        log = list(_web_ui._log)
    with _web_ui._mac_lock:
        mac = dict(_web_ui._mac_scene)

    mac_fresh = (time.time() - mac.get("ts", 0)) < MAC_SCENE_TTL_S
    mac_objects = mac.get("scene_objects", []) if mac_fresh else []

    GEMMA_TTL_S = 4.0
    gemma_ts = spatial.get("ts", 0)
    gemma_fresh = (time.time() - gemma_ts) < GEMMA_TTL_S
    gemma_objects = spatial.get("objects", []) if gemma_fresh else []

    # Gemma 4 on Pi is the irreplaceable semantic authority.
    # Mac enriches Gemma objects with velocity/predictions (kinematic layer only).
    # Fallback: raw YOLO bboxes until Gemma loads — Mac never acts as primary authority.
    raw_yolo = yolo.get("detections", [])
    if gemma_objects:
        primary_objects = _yolo_correct_angles(_enrich_with_mac(gemma_objects, mac_objects), raw_yolo)
        src = "GEMMA"
    else:
        # Gemma stale: YOLO-direct. Drop confirmed fast-receding objects before display.
        primary_objects = [
            d for d in raw_yolo
            if not (d.get("motion") == "receding" and float(d.get("velocity_mps", 0)) < -0.5)
        ]
        src = "YOLO"

    threats = _objects_to_threats(primary_objects, raw_yolo, source=src)
    threats = _dedup_threats(threats)

    beams = _beam_energy_from_scan(spatial.get("beam_scan", []))

    # Escape: Mac planner primary (has full sector map), fall back to stereo
    mac_escape = mac.get("escape_dir_deg") if mac_fresh else None
    heading = float(mac_escape if mac_escape is not None
                    else stereo.get("escape_dir_deg", 0.0))

    fps = perf.get("fps", 0.0)
    _temp, _cpu_pct = _cached_sys_stat()

    # Normalized bbox detections for camera overlay (0-1 relative to 224×224 frame)
    _FW, _FH = 224.0, 224.0
    yolo_dets = []
    for d in yolo.get("detections", [])[:12]:
        bb = d.get("bbox")
        if bb and len(bb) == 4:
            x1, y1, x2, y2 = bb
            yolo_dets.append({
                "label":  d.get("label", "?").upper(),
                "conf":   round(float(d.get("confidence", 0)), 2),
                "dist":   round(float(d.get("distance_m", 0)), 1),
                "urgency": d.get("urgency", "low"),
                "bbox_n": [round(x1/_FW,4), round(y1/_FH,4),
                           round(x2/_FW,4), round(y2/_FH,4)],
            })

    return {
        "threats":   threats,
        "events":    _log_to_events(log),
        "beams":     beams,
        "heading":   round(heading, 1),
        "yolo_dets": yolo_dets,
        "stat": {
            "gemma":    round(spatial.get("gemma_ms", 0) / 1000.0, 2),
            "pipeline": round(fps, 1),
            "yolo":     int(perf.get("yolo_ms", 0)),
            "temp":     _temp,
            "cpu_pct":  _cpu_pct,
        },
        "ts": time.time(),
    }


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return _static_response("index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return _static_response(filename)


@app.route("/stream")
def stream():
    def generate():
        while True:
            try:
                frame = _build_frame()
                yield f"data: {json.dumps(frame)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            time.sleep(1.0 / STREAM_HZ)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/api/snapshot")
def snapshot():
    return json.dumps(_build_frame()), 200, {"Content-Type": "application/json",
                                              "Access-Control-Allow-Origin": "*"}


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of cam0 frame with YOLO boxes drawn."""
    try:
        import cv2 as _cv2
        import numpy as _np
        _CV2_OK = True
    except ImportError:
        _CV2_OK = False

    def _blank_jpeg():
        if _CV2_OK:
            img = _np.zeros((240, 320, 3), dtype=_np.uint8)
            _cv2.putText(img, "NO FRAME", (80, 120),
                         _cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 100), 2)
            _, buf = _cv2.imencode(".jpg", img, [_cv2.IMWRITE_JPEG_QUALITY, 70])
            return buf.tobytes()
        return b""

    def generate():
        while True:
            try:
                with _web_ui._frames_lock:
                    frame = _web_ui._frames.get(0)
                if frame is not None and _CV2_OK:
                    # frame is RGB (already has boxes from update_frame)
                    bgr = _cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR)
                    # Scale down to 320×240 for low bandwidth
                    h, w = bgr.shape[:2]
                    th, tw = 240, 320
                    if w > tw or h > th:
                        scale = min(tw / w, th / h)
                        bgr = _cv2.resize(bgr, (int(w * scale), int(h * scale)))
                    _, buf = _cv2.imencode(".jpg", bgr, [_cv2.IMWRITE_JPEG_QUALITY, 75])
                    jpeg = buf.tobytes()
                else:
                    jpeg = _blank_jpeg()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            except Exception:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _blank_jpeg() + b"\r\n"
            time.sleep(1.0 / 15)  # 15fps feed

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Access-Control-Allow-Origin": "*"},
    )


def start(port: int = PORT, debug: bool = False):
    if not FLASK_AVAILABLE:
        print("[live_server] Flask not installed, UI server disabled.")
        return
    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False),
        daemon=True,
    )
    t.start()
    _log.info("WaveScape UI → http://0.0.0.0:%d", port)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True, use_reloader=False)
