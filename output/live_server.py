"""
WaveScape live data server.
Serves ui/ static files and streams pipeline state via SSE at /stream.
Integrates with web_ui.py shared state.

Run alongside pipeline:
  python -m output.live_server   (port 8090)
"""
import json
import math
import os
import time
import threading
from pathlib import Path

try:
    from flask import Flask, Response, send_from_directory
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

import output.web_ui as _web_ui

UI_DIR = Path(__file__).parent.parent / "ui"
PORT = 8090
STREAM_HZ = 10   # push rate

app = Flask(__name__, static_folder=None)

# ── helpers ──────────────────────────────────────────────────────────────────

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
    seen_labels = set()
    for obj in objects:
        _id += 1
        dist = float(obj.get("distance_m", obj.get("dist_m", obj.get("distance", 5.0))))
        angle = float(obj.get("direction_deg", obj.get("angle_deg", obj.get("dir", 0.0))))
        urgency = obj.get("urgency", "medium")
        if not urgency or urgency == "none":
            urgency = "low" if dist > 8 else "medium" if dist > 4 else "high" if dist > 2 else "critical"
        label = obj.get("label", obj.get("class", "OBJECT")).upper()
        seen_labels.add(label)
        _type = "person" if any(w in label.lower() for w in ["person","pedestrian","human"]) \
            else "vehicle" if any(w in label.lower() for w in ["car","truck","vehicle","bike","motor"]) \
            else "obstacle"
        icon = {"person": "👤", "vehicle": "🚗", "obstacle": "⚠"}.get(_type, "⚠")
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
                              if obj.get("confidence", 0.75) <= 1.0
                              else obj.get("confidence", 75)),
            "velocity": round(vel, 2),
            "urgency": urgency,
            "source": source,
            "serverTs": int(time.time() * 1000),
            "bornAt": int(time.time() * 1000),
            "ttl": 5000,
            "eta": round(dist / max(0.3, abs(vel)), 1) if vel < 0 else 99.0,
            "coasting": obj.get("coasting", False),
            "crossing": obj.get("crossing", False),
            "predicted": obj.get("predicted", []),
        })

    # Add YOLO-only detections not already in spatial map
    for det in yolo_detections[:4]:
        label = str(det.get("label", det.get("class_name", "OBJECT"))).upper()
        if label in seen_labels:
            continue
        _id += 1
        conf = det.get("confidence", det.get("conf", 0.7))
        if conf <= 1.0:
            conf = int(conf * 100)
        dist = float(det.get("distance_m", det.get("dist", 4.0)))
        angle = float(det.get("direction_deg", det.get("angle_deg", det.get("direction", 0.0))))
        urgency = "low" if dist > 8 else "medium" if dist > 4 else "high" if dist > 2 else "critical"
        _type = "person" if "person" in label.lower() else "vehicle" \
            if any(w in label.lower() for w in ["car","truck","bus","bike"]) else "obstacle"
        icon = {"person": "👤", "vehicle": "🚗", "obstacle": "⚠"}.get(_type, "⚠")
        threats.append({
            "id": f"T-{_id:04d}",
            "type": _type,
            "icon": icon,
            "label": label,
            "audioClass": "",
            "visualClass": label.lower(),
            "distance": round(dist, 2),
            "angle": round(angle % 360, 1),
            "confidence": int(conf),
            "velocity": 0.5,
            "urgency": urgency,
            "source": "YOLO",
            "bornAt": int(time.time() * 1000),
            "ttl": 2000,
            "eta": round(dist / 0.5, 1),
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
    """Remove near-duplicates (same label, <20° apart). Keep highest urgency."""
    out = []
    urgency_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    for t in threats:
        merged = False
        for existing in out:
            if existing["label"] != t["label"]:
                continue
            delta = abs((t["angle"] - existing["angle"] + 180) % 360 - 180)
            if delta < 20:
                # Keep the one with higher urgency (lower rank)
                if urgency_rank.get(t["urgency"], 3) < urgency_rank.get(existing["urgency"], 3):
                    existing.update(t)
                merged = True
                break
        if not merged:
            out.append(dict(t))
    return out


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

    # Prefer Mac tracker objects (smooth EMA) over Gemma 15°-bucketed objects.
    # Fall back to Gemma spatial map when Mac is stale.
    if mac_objects:
        primary_objects = mac_objects
    else:
        primary_objects = spatial.get("objects", [])

    src = "MAC" if mac_objects else "GEMMA"
    threats = _objects_to_threats(primary_objects, yolo.get("detections", []), source=src)
    threats = _dedup_threats(threats)

    beams = _beam_energy_from_scan(spatial.get("beam_scan", []))

    # Escape direction: prefer Mac planner, fall back to stereo
    mac_escape = mac.get("escape_dir_deg") if mac_fresh else None
    heading = float(mac_escape if mac_escape is not None
                    else stereo.get("escape_dir_deg", 0.0))

    fps = perf.get("fps", 0.0)

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
            "temp":     45,
            "batt":     100,
        },
        "ts": time.time(),
    }


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(UI_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(UI_DIR, filename)


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
    print(f"[live_server] WaveScape UI → http://0.0.0.0:{port}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True, use_reloader=False)
