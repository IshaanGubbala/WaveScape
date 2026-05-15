"""
ThreatDetect live dashboard — radar + haptic log + mic beamform panel.
"""
import html
import time
import math
import threading
from collections import deque

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    import numpy as np

MAX_LOG = 50
SAMPLE_RATE = 22050

_log: deque = deque(maxlen=MAX_LOG)
_log_lock = threading.Lock()

_haptic_signal = {"ts": 0.0, "urgency": "none", "direction": 0.0}
_signal_lock = threading.Lock()
_last_sent_ts = 0.0

_frames: dict = {0: None, 1: None}
_frames_lock = threading.Lock()

_spatial_map = {"objects": [], "dominant_hazard": {}, "beam_scan": [], "ts": 0.0}
_spatial_lock = threading.Lock()

_yolo_live = {"detections": [], "ts": 0.0}
_yolo_lock = threading.Lock()

_stereo_state: dict = {"sector_depths": [], "escape_dir_deg": 0.0, "min_depth_m": 99.0, "latency_ms": 0.0, "ts": 0.0}
_stereo_lock = threading.Lock()

_perf_state: dict = {"frame_ms": 0.0, "fps": 0.0, "yolo_ms": 0.0, "stereo_ms": 0.0, "flow_ms": 0.0, "ts": 0.0}
_perf_lock = threading.Lock()

# Radar canvas
_MAP_SIZE = 400
_MAP_CX   = _MAP_SIZE // 2
_MAP_CY   = _MAP_SIZE // 2
_RANGE_M  = 12.0
_PX_PER_M = (_MAP_SIZE // 2 - 20) / _RANGE_M

_URGENCY_BGR = {
    "low":      (133, 120, 108),  # muted warm gray
    "medium":   (65, 140, 202),   # soft blue
    "high":     (61, 82, 214),    # deeper blue
    "critical": (41, 49, 179),    # indigo
}
_TEAL_BGR  = (176, 129, 67)    # warm accent
_RING_BGR  = (215, 215, 212)   # outline-variant
_SPOKE_BGR = (235, 233, 228)
_RADAR_BG  = (250, 248, 246)  # warm paper white


def _make_beep(urgency: str, direction_deg: float):
    freq = {"critical": 1200, "high": 900, "medium": 660, "low": 440}.get(urgency, 880)
    dur  = {"critical": 0.06, "high": 0.09, "medium": 0.12, "low": 0.18}.get(urgency, 0.09)
    n = int(SAMPLE_RATE * dur)
    t = np.linspace(0, dur, n, False)
    wave = np.sin(2 * np.pi * freq * t) * np.exp(-t * (8 / dur)) * 0.5
    pan = max(-1.0, min(1.0, math.sin(math.radians(direction_deg))))
    angle = (pan + 1) * math.pi / 4
    stereo = np.column_stack([wave * math.cos(angle), wave * math.sin(angle)]).astype(np.float32)
    return SAMPLE_RATE, stereo


def log_audio_event(labels: list[tuple[str, float]], beam_scan: list[dict]):
    pass  # audio panel built from beam_scan directly


def update_frame(cam_idx: int, frame_bgr, detections: list = None):
    if not CV2_AVAILABLE or frame_bgr is None:
        return
    img = frame_bgr.copy()
    if detections:
        for d in detections:
            bbox = d.get("bbox")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                label = f"{d.get('label','?')} {d.get('confidence',0):.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    with _frames_lock:
        _frames[cam_idx] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def log_haptic(source: str, direction_deg: float, urgency: str, confidence: float, pattern: str):
    intensity = {"low": 3, "medium": 6, "high": 9, "critical": 10}.get(urgency, 0)
    bar  = "█" * intensity + "░" * (10 - intensity)
    side = "L" if direction_deg > 180 else "R"
    ts   = time.strftime("%H:%M:%S")
    line = f"{ts}  {source:8s}  {side} {direction_deg:5.1f}°  {urgency:8s}  {bar}  {confidence:.2f}"
    with _log_lock:
        _log.appendleft(line)
    with _signal_lock:
        _haptic_signal["ts"]        = time.time()
        _haptic_signal["urgency"]   = urgency
        _haptic_signal["direction"] = direction_deg


def update_yolo_live(detections: list):
    with _yolo_lock:
        _yolo_live["detections"] = detections
        _yolo_live["ts"] = time.time()


def update_stereo(sdepth: dict):
    with _stereo_lock:
        _stereo_state.update(sdepth)
        _stereo_state["ts"] = time.time()


def update_perf(frame_ms: float, fps: float, yolo_ms: float = 0.0, stereo_ms: float = 0.0, flow_ms: float = 0.0):
    with _perf_lock:
        _perf_state["frame_ms"] = frame_ms
        _perf_state["fps"] = fps
        _perf_state["yolo_ms"] = yolo_ms
        _perf_state["stereo_ms"] = stereo_ms
        _perf_state["flow_ms"] = flow_ms
        _perf_state["ts"] = time.time()


def update_spatial(smap, beam_scan: list = None):
    with _spatial_lock:
        _spatial_map["objects"]         = getattr(smap, "objects", [])
        _spatial_map["dominant_hazard"] = getattr(smap, "dominant_hazard", {})
        _spatial_map["beam_scan"]       = beam_scan or []
        _spatial_map["ts"]              = time.time()


def update_threat(threat_type, confidence, direction_deg, distance_m, urgency, haptic_pattern, natural_language):
    if urgency in ("medium", "high", "critical") and confidence > 0.5:
        log_haptic("GEMMA", direction_deg, urgency, confidence, haptic_pattern)


def _dir_to_xy(direction_deg: float, dist_m: float):
    rad = math.radians(direction_deg)
    px  = dist_m * _PX_PER_M
    return int(_MAP_CX + math.sin(rad) * px), int(_MAP_CY - math.cos(rad) * px)


def _render_map() -> np.ndarray:
    if not CV2_AVAILABLE:
        return np.zeros((_MAP_SIZE, _MAP_SIZE, 3), dtype=np.uint8)

    canvas = np.full((_MAP_SIZE, _MAP_SIZE, 3), _RADAR_BG[0], dtype=np.uint8)
    canvas[:, :] = _RADAR_BG

    # Sweep sector
    sweep_deg = (time.time() * 40) % 360
    sweep_pts = [(_MAP_CX, _MAP_CY)]
    for d in range(int(sweep_deg) - 45, int(sweep_deg) + 1, 2):
        rad = math.radians(d % 360)
        px  = int(_RANGE_M * _PX_PER_M)
        sweep_pts.append((int(_MAP_CX + math.sin(rad) * px),
                          int(_MAP_CY - math.cos(rad) * px)))
    sw_ov = canvas.copy()
    cv2.fillPoly(sw_ov, [np.array(sweep_pts, dtype=np.int32)], (230, 240, 220))
    cv2.addWeighted(canvas, 0.82, sw_ov, 0.18, 0, canvas)
    lead_rad = math.radians(sweep_deg)
    cv2.line(canvas, (_MAP_CX, _MAP_CY),
             (int(_MAP_CX + math.sin(lead_rad) * _RANGE_M * _PX_PER_M),
              int(_MAP_CY - math.cos(lead_rad) * _RANGE_M * _PX_PER_M)),
             _TEAL_BGR, 1)

    # Range rings + labels
    for r_m in (3, 6, 12):
        r_px = int(r_m * _PX_PER_M)
        cv2.circle(canvas, (_MAP_CX, _MAP_CY), r_px, _RING_BGR, 1)
        cv2.putText(canvas, f"{r_m}m", (_MAP_CX + r_px + 3, _MAP_CY - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, _RING_BGR, 1)

    # Dashed spokes + NESW
    for deg in (0, 90, 180, 270):
        rad = math.radians(deg)
        mpx = int(_RANGE_M * _PX_PER_M)
        for s in range(8, mpx, 12):
            cv2.line(canvas,
                     (int(_MAP_CX + math.sin(rad) * s), int(_MAP_CY - math.cos(rad) * s)),
                     (int(_MAP_CX + math.sin(rad) * min(s+7, mpx)),
                      int(_MAP_CY - math.cos(rad) * min(s+7, mpx))),
                     _SPOKE_BGR, 1)
    for deg, lbl in ((0,"N"),(90,"E"),(180,"S"),(270,"W")):
        lx, ly = _dir_to_xy(deg, _RANGE_M * 0.90)
        cv2.putText(canvas, lbl, (lx-4, ly+4), cv2.FONT_HERSHEY_SIMPLEX, 0.27, _RING_BGR, 1)

    with _spatial_lock:
        objects = list(_spatial_map["objects"])
        hazard  = dict(_spatial_map["dominant_hazard"])
        beams   = list(_spatial_map["beam_scan"])

    # Beam wedges
    max_energy = max((b.get("energy", 0) for b in beams), default=1e-6)
    for b in beams:
        energy = b.get("energy", 0)
        if energy < 0.008:
            continue
        bdir  = b["direction_deg"]
        blen  = min(energy / max_energy * _RANGE_M * 0.68, _RANGE_M * 0.65)
        wedge = [(_MAP_CX, _MAP_CY)]
        for dd in range(-5, 6, 2):
            wx, wy = _dir_to_xy(bdir + dd, blen)
            wedge.append((wx, wy))
        ov2 = canvas.copy()
        cv2.fillPoly(ov2, [np.array(wedge, dtype=np.int32)], (220, 200, 180))
        cv2.addWeighted(canvas, 1 - min(0.3, energy*3.5), ov2, min(0.3, energy*3.5), 0, canvas)
        if b.get("label"):
            lx2, ly2 = _dir_to_xy(bdir, blen * 0.78)
            cv2.putText(canvas, b["label"][:5], (lx2-14, ly2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, _TEAL_BGR, 1)

    # YOLO live — hollow circles
    with _yolo_lock:
        yolo_dets = list(_yolo_live["detections"])
    for det in yolo_dets:
        dir_deg = float(det.get("direction_deg", det.get("dir_deg", 0)))
        dist_m  = max(0.5, min(float(det.get("distance_m", det.get("dist_m", 8))), _RANGE_M-0.5))
        ox, oy  = _dir_to_xy(dir_deg, dist_m)
        cv2.circle(canvas, (ox, oy), 4, (160, 150, 125), 1)

    # Gemma objects
    for i, obj in enumerate(objects):
        dir_deg = float(obj.get("dir", obj.get("direction_deg", 0)))
        dist_m  = max(0.5, min(float(obj.get("dist_m", obj.get("distance_m", 6))), _RANGE_M-0.5))
        urgency = hazard.get("urgency", "low") if i == 0 else "low"
        color   = _URGENCY_BGR.get(urgency, _TEAL_BGR)
        ox, oy  = _dir_to_xy(dir_deg, dist_m)
        ov3 = canvas.copy()
        cv2.circle(ov3, (ox, oy), 13, color, -1)
        cv2.addWeighted(canvas, 0.90, ov3, 0.10, 0, canvas)
        cv2.circle(canvas, (ox, oy), 5, color, -1)
        cv2.circle(canvas, (ox, oy), 7, color, 1)
        label = obj.get("label","?")[:7]
        sound = obj.get("sound","")
        cv2.putText(canvas, label+(f"/{sound[:4]}" if sound and sound!="none" else ""),
                    (ox+10, oy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.27, color, 1)

    if hazard and "dir" in hazard:
        hdir  = float(hazard["dir"])
        hcol  = _URGENCY_BGR.get(hazard.get("urgency","low"), _TEAL_BGR)
        hx, hy = _dir_to_xy(hdir, 2.8)
        cv2.arrowedLine(canvas, (_MAP_CX,_MAP_CY), (hx,hy), hcol, 2, tipLength=0.33)

    # Stereo depth sector overlay (forward FOV ±33°)
    with _stereo_lock:
        s_sectors = list(_stereo_state.get("sector_depths", []))
        s_escape = _stereo_state.get("escape_dir_deg", None)
        s_min = _stereo_state.get("min_depth_m", 99.0)
        s_ts = _stereo_state.get("ts", 0.0)

    if s_sectors and (time.time() - s_ts) < 2.0:
        SECTOR_DRAW_M = 4.0
        half_sector_deg = 5.5
        for s in s_sectors:
            angle = float(s["angle_deg"])  # relative to ahead; 0=forward
            depth = min(float(s["min_depth_m"]), SECTOR_DRAW_M)
            frac = depth / SECTOR_DRAW_M  # 0=near red, 1=far green
            bgr_s = (30, int(160 * frac), int(160 * (1 - frac)))
            pts = [(_MAP_CX, _MAP_CY)]
            for dd in range(int(-half_sector_deg * 2), int(half_sector_deg * 2) + 1, 2):
                wa = (angle + dd * 0.5) % 360
                wx, wy = _dir_to_xy(wa, depth)
                pts.append((wx, wy))
            ov_s = canvas.copy()
            cv2.fillPoly(ov_s, [np.array(pts, dtype=np.int32)], bgr_s)
            cv2.addWeighted(canvas, 0.72, ov_s, 0.28, 0, canvas)
        # Escape direction arrow — bold green
        if s_escape is not None:
            ex, ey = _dir_to_xy(float(s_escape) % 360, 3.2)
            cv2.arrowedLine(canvas, (_MAP_CX, _MAP_CY), (ex, ey), (30, 190, 60), 3, tipLength=0.28)
            cv2.putText(canvas, f"go {s_escape:.0f}°", (ex + 6, ey - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.24, (20, 140, 40), 1)
        # Min depth label in corner
        if s_min < 3.0:
            cv2.putText(canvas, f"min {s_min:.1f}m", (8, _MAP_SIZE - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (30, 60, 200) if s_min < 1.5 else (30, 130, 200), 1)

    # YOU diamond
    diamond = np.array([
        [_MAP_CX, _MAP_CY-8], [_MAP_CX+6, _MAP_CY],
        [_MAP_CX, _MAP_CY+8], [_MAP_CX-6, _MAP_CY],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [diamond], (31, 28, 23))

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _beam_panel_text() -> str:
    """Build plain-text mic beam panel: direction, energy bar, distance est, label."""
    with _spatial_lock:
        beams = list(_spatial_map["beam_scan"])

    if not beams:
        return "(no mic data)"

    # Energy → rough distance estimate (same as spatial_mapper._energy_to_dist)
    ENERGY_TO_DIST = [(0.15, 1.5), (0.08, 3.0), (0.04, 6.0), (0.02, 9.0), (0.01, 12.0)]
    def e2d(e):
        for thr, d in ENERGY_TO_DIST:
            if e >= thr:
                return d
        return 15.0

    max_e = max(b.get("energy", 0) for b in beams) or 1e-6
    lines = []
    for b in beams:
        deg    = b.get("direction_deg", 0)
        energy = b.get("energy", 0)
        label  = b.get("label") or "—"
        dist   = e2d(energy)
        bar_w  = int(energy / max_e * 20)
        bar    = "█" * bar_w + "░" * (20 - bar_w)
        lines.append(
            f"{deg:>5.0f}°  [{bar}]  {energy:.4f}  ~{dist:4.1f}m  {label}"
        )

    header = f"{'DIR':>5}   {'ENERGY (louder →)':22}  {'RMS':6}  {'DIST':6}  LABEL"
    sep    = "─" * 56
    return "\n".join([header, sep] + lines)


def _get_frame():
    with _frames_lock:
        f0 = _frames.get(0)
        f1 = _frames.get(1)
    if f0 is None and f1 is None:
        return None
    if f0 is None:
        return f1
    if f1 is None:
        return f0
    if not CV2_AVAILABLE:
        return f0
    h = max(f0.shape[0], f1.shape[0])
    if f0.shape[0] != h:
        f0 = cv2.resize(f0, (int(f0.shape[1] * h / f0.shape[0]), h))
    if f1.shape[0] != h:
        f1 = cv2.resize(f1, (int(f1.shape[1] * h / f1.shape[0]), h))
    return np.hstack([f0, f1])


def _direction_label(deg: float) -> str:
    deg = deg % 360
    if deg < 45 or deg >= 315:
        return "Front"
    if deg < 135:
        return "Right"
    if deg < 225:
        return "Behind"
    return "Left"


def _direction_to_split(deg: float) -> float:
    deg = deg % 360
    if deg <= 90:
        return 0.5 + (deg / 90) * 0.5
    if deg <= 180:
        return 1.0 - ((deg - 90) / 90) * 0.5
    if deg <= 270:
        return 0.5 - ((deg - 180) / 90) * 0.5
    return ((deg - 270) / 90) * 0.5


def _energy_to_distance(energy: float) -> float:
    if energy >= 0.15:
        return 1.5
    if energy >= 0.08:
        return 3.0
    if energy >= 0.04:
        return 6.0
    if energy >= 0.02:
        return 9.0
    if energy >= 0.01:
        return 12.0
    return 15.0


def _build_stats_html() -> str:
    with _spatial_lock:
        beams = list(_spatial_map["beam_scan"])
        objects = list(_spatial_map["objects"])
        hazard = dict(_spatial_map["dominant_hazard"])
    with _yolo_lock:
        detections = list(_yolo_live["detections"])
    with _signal_lock:
        sig = dict(_haptic_signal)

    active = len(objects) if objects else sum(
        1 for d in detections
        if float(d.get("confidence", d.get("magnitude", 0.0))) >= 0.2
    )
    peak = max(beams, key=lambda b: b.get("energy", 0), default=None)
    peak_value = f"{peak['direction_deg']:.0f}° · {peak['energy']:.3f}" if peak else "Idle"
    urgency = sig["urgency"].title() if sig.get("ts") else "Idle"
    direction = f"{_direction_label(sig['direction'])} · {sig['direction']:.0f}°" if sig.get("ts") else "Waiting"
    pipeline = "Vision + audio + Gemma"
    source = "Real hardware" if CV2_AVAILABLE else "CPU fallback"
    hazard_label = hazard.get("urgency", "none").title() if hazard else "Idle"
    cards = [
        ("Active threats", str(active), "Camera + spatial map"),
        ("Beam peak", peak_value, f"{len(beams)} beam directions"),
        ("Latest alert", urgency, direction),
        ("Pipeline", source, pipeline),
    ]
    if hazard:
        cards.append(("Dominant hazard", hazard_label, f"{hazard.get('dir', 0):.0f}° in view"))

    return (
        "<section class='td-stat-grid'>"
        + "".join(
            f"<div class='td-stat-card'>"
            f"<div class='td-stat-label'>{html.escape(label)}</div>"
            f"<div class='td-stat-value'>{html.escape(value)}</div>"
            f"<div class='td-stat-caption'>{html.escape(caption)}</div>"
            f"</div>"
            for label, value, caption in cards
        )
        + "</section>"
    )


def _build_audio_html() -> str:
    with _spatial_lock:
        beams = list(_spatial_map["beam_scan"])

    rows = []
    if beams:
        max_energy = max((b.get("energy", 0) for b in beams), default=1e-6) or 1e-6
        for b in beams:
            deg = float(b.get("direction_deg", 0))
            energy = float(b.get("energy", 0))
            label = str(b.get("label") or _direction_label(deg))
            dist = _energy_to_distance(energy)
            pct = max(6, int((energy / max_energy) * 100))
            rows.append(
                f"<div class='td-bar-item'>"
                f"<div class='td-bar-head'><span>{deg:.0f}°</span><span>{energy:.3f}</span></div>"
                f"<div class='td-bar-track'><div class='td-bar-fill' style='width:{pct}%'></div></div>"
                f"<div class='td-bar-foot'>{html.escape(label)} · ~{dist:.1f}m</div>"
                f"</div>"
            )
        body = "".join(rows)
    else:
        body = "<div class='td-empty'>No beam data yet.</div>"

    return (
        "<section class='td-panel td-copy-card'>"
        "<div class='td-card-head'><span>Audio classification</span><span>Beam scan</span></div>"
        "<div class='td-card-body'>"
        f"{body}"
        "</div></section>"
    )


def _build_haptic_html() -> str:
    with _signal_lock:
        sig = dict(_haptic_signal)
    with _log_lock:
        lines = list(_log)

    if sig.get("ts"):
        urgency = sig["urgency"]
        duty = {"low": 0, "medium": 40, "high": 75, "critical": 100}.get(urgency, 0)
        right_frac = _direction_to_split(sig["direction"])
        left_duty = round(duty * (1.0 - right_frac))
        right_duty = round(duty * right_frac)
        direction = f"{_direction_label(sig['direction'])} · {sig['direction']:.0f}°"
        status = urgency.title()
        status_caption = "Last haptic event"
    else:
        left_duty = right_duty = 0
        direction = "Waiting for output"
        status = "Idle"
        status_caption = "No haptic activity yet"

    log_rows = []
    for line in lines[:4]:
        parts = line.split()
        label = html.escape(line)
        log_rows.append(f"<div class='td-log-row'><span>{label}</span></div>")
    if not log_rows:
        log_rows.append("<div class='td-empty'>No haptic events yet.</div>")

    return (
        "<section class='td-panel td-copy-card'>"
        "<div class='td-card-head'><span>Haptic motor status</span><span>Recent output</span></div>"
        "<div class='td-card-body'>"
        "<div class='td-motor-grid'>"
        f"<div class='td-motor-card'><span>Left temple</span><strong>{left_duty}%</strong><small>{html.escape(direction)}</small></div>"
        f"<div class='td-motor-card'><span>Right temple</span><strong>{right_duty}%</strong><small>{html.escape(status)}</small></div>"
        "</div>"
        f"<div class='td-status-line'><strong>{html.escape(status)}</strong><span>{html.escape(status_caption)}</span></div>"
        "<div class='td-log-list'>"
        + "".join(log_rows)
        + "</div>"
        "</div></section>"
    )


def _build_threats_html() -> str:
    with _spatial_lock:
        objects = list(_spatial_map["objects"])
        hazard = dict(_spatial_map["dominant_hazard"])
    with _yolo_lock:
        detections = list(_yolo_live["detections"])

    entries = []
    if objects:
        for i, obj in enumerate(objects):
            deg = float(obj.get("dir", obj.get("direction_deg", 0)))
            dist = float(obj.get("dist_m", obj.get("distance_m", 0.0)) or 0.0)
            urgency = hazard.get("urgency", "low") if i == 0 else "low"
            conf = float(obj.get("confidence", 0.85 if i == 0 else 0.65))
            entries.append({
                "label": str(obj.get("label", "object")),
                "source": "Gemma",
                "direction": deg,
                "distance": dist,
                "confidence": conf,
                "urgency": urgency,
            })
    else:
        for det in detections:
            conf = float(det.get("confidence", det.get("magnitude", 0.0)))
            if conf < 0.15:
                continue
            deg = float(det.get("direction_deg", det.get("dir_deg", 0)))
            dist = float(det.get("distance_m", det.get("dist_m", 0.0)) or 0.0)
            entries.append({
                "label": str(det.get("label", "object")),
                "source": "YOLO",
                "direction": deg,
                "distance": dist,
                "confidence": conf,
                "urgency": "medium" if conf > 0.5 else "low",
            })

    if not entries:
        body = "<div class='td-empty'>No active threats yet.</div>"
    else:
        rows = []
        for item in entries[:6]:
            side = _direction_label(item["direction"])
            dist = f"{item['distance']:.1f}m" if item["distance"] else "—"
            rows.append(
                "<div class='td-threat-row'>"
                "<div class='td-threat-copy'>"
                f"<div class='td-threat-label'>{html.escape(item['label'])}</div>"
                f"<div class='td-threat-meta-line'>{html.escape(item['source'])} · {html.escape(side)} · {item['direction']:.0f}° · {html.escape(dist)}</div>"
                "</div>"
                "<div class='td-threat-score'>"
                f"<span>{html.escape(item['urgency'].title())}</span>"
                f"<strong>{item['confidence'] * 100:.0f}%</strong>"
                "</div>"
                "</div>"
            )
        body = "".join(rows)

    return (
        "<section class='td-panel td-panel-wide td-copy-card'>"
        "<div class='td-card-head'><span>Active threats</span><span>Recent objects</span></div>"
        "<div class='td-card-body'>"
        f"{body}"
        "</div></section>"
    )


def _build_nav_html() -> str:
    with _stereo_lock:
        sectors = list(_stereo_state.get("sector_depths", []))
        escape = _stereo_state.get("escape_dir_deg", 0.0)
        min_d = _stereo_state.get("min_depth_m", 99.0)
        s_ts = _stereo_state.get("ts", 0.0)
    with _perf_lock:
        perf = dict(_perf_state)

    age = time.time() - s_ts
    if not sectors or age > 3.0:
        sector_html = "<div class='td-empty'>No stereo depth data. Run with --stereo-sim or --dual-cam.</div>"
    else:
        NEAR = 1.5
        FAR = 3.0
        cells = []
        for s in sectors:
            d = s["min_depth_m"]
            a = s["angle_deg"]
            if d < NEAR:
                color = "#e53e3e"
                label = "BLOCKED"
            elif d < FAR:
                color = "#d97706"
                label = f"{d:.1f}m"
            else:
                color = "#0f9d8a"
                label = f"{d:.1f}m"
            dir_lbl = "L" if a < -2 else ("R" if a > 2 else "FWD")
            cells.append(
                f"<div class='td-nav-sector' style='border-top:3px solid {color}'>"
                f"<div class='td-nav-deg'>{a:+.0f}°</div>"
                f"<div class='td-nav-depth' style='color:{color}'>{label}</div>"
                f"<div class='td-nav-dir'>{dir_lbl}</div>"
                f"</div>"
            )
        escape_label = "Forward" if abs(escape % 360) < 20 or abs((escape % 360) - 360) < 20 else \
                       ("Right" if 0 < (escape % 360) < 180 else "Left")
        min_color = "#e53e3e" if min_d < NEAR else ("#d97706" if min_d < FAR else "#0f9d8a")
        sector_html = (
            f"<div class='td-nav-grid'>{''.join(cells)}</div>"
            f"<div class='td-nav-escape'>"
            f"<span>Escape</span><strong style='color:#0f9d8a'>{escape_label} · {escape:.0f}°</strong>"
            f"<span style='color:{min_color}'>nearest {min_d:.2f}m</span>"
            f"</div>"
        )

    # Perf rows
    def ms_bar(ms, cap):
        pct = min(100, int(ms / cap * 100))
        color = "#e53e3e" if pct > 80 else ("#d97706" if pct > 50 else "#0f9d8a")
        return (
            f"<div class='td-bar-track' style='margin-top:4px'>"
            f"<div class='td-bar-fill' style='width:{pct}%;background:{color}'></div>"
            f"</div>"
        )

    fps_color = "#0f9d8a" if perf["fps"] > 7 else ("#d97706" if perf["fps"] > 4 else "#e53e3e")
    perf_html = (
        f"<div class='td-perf-grid'>"
        f"<div class='td-perf-item'><span>FPS</span><strong style='color:{fps_color}'>{perf['fps']:.1f}</strong>{ms_bar(1000/max(perf['fps'],0.1), 200)}</div>"
        f"<div class='td-perf-item'><span>Frame</span><strong>{perf['frame_ms']:.0f}ms</strong>{ms_bar(perf['frame_ms'], 150)}</div>"
        f"<div class='td-perf-item'><span>YOLO</span><strong>{perf['yolo_ms']:.0f}ms</strong>{ms_bar(perf['yolo_ms'], 100)}</div>"
        f"<div class='td-perf-item'><span>Stereo</span><strong>{perf['stereo_ms']:.0f}ms</strong>{ms_bar(perf['stereo_ms'], 60)}</div>"
        f"</div>"
    )

    return (
        "<section class='td-panel td-copy-card'>"
        "<div class='td-card-head'><span>Navigation</span><span>Stereo depth sectors</span></div>"
        "<div class='td-card-body'>"
        f"{sector_html}"
        "<div class='td-card-head' style='margin:12px -15px 10px;padding:8px 15px'><span>Pipeline speed</span><span>Live timings</span></div>"
        f"{perf_html}"
        "</div></section>"
    )


def _poll():
    global _last_sent_ts
    with _signal_lock:
        sig_ts = _haptic_signal["ts"]
        urgency = _haptic_signal["urgency"]
        direction = _haptic_signal["direction"]

    map_img = _render_map()
    frame = _get_frame()
    stats_html = _build_stats_html()
    audio_html = _build_audio_html()
    haptic_html = _build_haptic_html()
    threats_html = _build_threats_html()
    nav_html = _build_nav_html()

    if sig_ts > _last_sent_ts:
        _last_sent_ts = sig_ts
        beep = _make_beep(urgency, direction)
    else:
        beep = None

    return frame, map_img, stats_html, audio_html, haptic_html, threats_html, nav_html, beep


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {
  box-sizing: border-box;
}

:root {
  --bg: #f5f2ec;
  --bg-2: #fcfbf8;
  --surface: rgba(255, 255, 255, 0.94);
  --surface-strong: rgba(255, 255, 255, 1);
  --surface-tint: rgba(248, 249, 251, 0.92);
  --border: rgba(15, 23, 42, 0.09);
  --border-strong: rgba(15, 23, 42, 0.16);
  --text-1: #0f1724;
  --text-2: #243244;
  --text-3: #5b697d;
  --accent: #0f9d8a;
  --accent-2: #65ead6;
  --accent-soft: rgba(15, 157, 138, 0.08);
  --good: #0f766e;
  --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
  --font: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', Inter, ui-sans-serif, system-ui, sans-serif;
  --shadow: 0 18px 42px rgba(15, 23, 42, 0.06);
  --shadow-strong: 0 26px 70px rgba(15, 23, 42, 0.10);
}

html,
body {
  min-height: 100%;
  background:
    radial-gradient(circle at 18% 12%, rgba(15, 157, 138, 0.10), transparent 28%),
    radial-gradient(circle at 82% 8%, rgba(37, 99, 235, 0.07), transparent 24%),
    radial-gradient(circle at 50% 100%, rgba(56, 189, 248, 0.06), transparent 28%),
    linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%) !important;
}

body,
.gradio-container {
  background: linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%) !important;
  color: var(--text-1) !important;
  font-family: var(--font) !important;
}

#td-shell {
  color: var(--text-1);
}

.gradio-container,
.gradio-container * {
  -webkit-font-smoothing: antialiased;
  text-rendering: geometricPrecision;
}

.gradio-container {
  max-width: none !important;
  min-height: 100vh;
}

.gradio-container > .main > .wrap {
  max-width: 1240px !important;
  padding: 20px 18px 30px !important;
}

footer,
.built-with {
  display: none !important;
}

.block {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}

label.svelte-1b6s6im,
.label-wrap,
span.svelte-1gfkn6j {
  display: none !important;
}

#td-shell {
  display: grid;
  gap: 16px;
  position: relative;
  isolation: isolate;
}

#td-shell::before,
#td-shell::after {
  content: "";
  position: fixed;
  inset: auto;
  pointer-events: none;
  z-index: -1;
}

#td-shell::before {
  top: 90px;
  right: -120px;
  width: 340px;
  height: 340px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(15, 157, 138, 0.10) 0%, rgba(15, 157, 138, 0.00) 68%);
  filter: blur(2px);
}

#td-shell::after {
  left: -90px;
  bottom: 60px;
  width: 280px;
  height: 280px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(37, 99, 235, 0.08) 0%, rgba(37, 99, 235, 0.00) 70%);
  filter: blur(2px);
}

#td-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 15px 18px;
  border: 1px solid var(--border);
  border-radius: 22px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 248, 246, 0.94));
  box-shadow: var(--shadow);
  backdrop-filter: blur(16px);
}

.td-brand {
  display: flex;
  align-items: center;
  gap: 14px;
  min-width: 0;
}

.td-mark {
  display: grid;
  place-items: center;
  width: 42px;
  height: 42px;
  border-radius: 15px;
  background: linear-gradient(180deg, rgba(15, 157, 138, 0.14), rgba(101, 234, 214, 0.10));
  border: 1px solid rgba(15, 157, 138, 0.14);
  color: var(--accent);
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.08em;
}

.td-kicker {
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: var(--text-3);
}

.td-title {
  margin-top: 2px;
  font-size: 1.02rem;
  font-weight: 700;
  color: var(--text-1);
}

.td-live {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(15, 118, 110, 0.14);
  background: rgba(15, 118, 110, 0.06);
  color: var(--good);
  font-size: 0.67rem;
  font-weight: 700;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  white-space: nowrap;
}

.td-live-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: var(--good);
  box-shadow: 0 0 0 6px rgba(15, 118, 110, 0.10);
  animation: blink 1.4s infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.35; transform: scale(0.92); }
}

#td-hero {
  display: grid;
  grid-template-columns: minmax(0, 1.18fr) minmax(300px, 0.82fr);
  gap: 16px;
  align-items: stretch;
  padding: 0 4px;
}

.td-hero-copy h1 {
  margin-top: 10px;
  max-width: 17ch;
  font-size: clamp(2.05rem, 3vw, 3.35rem);
  line-height: 1.02;
  letter-spacing: -0.05em;
  color: var(--text-1);
}

.td-hero-copy p {
  margin-top: 12px;
  max-width: 54ch;
  color: var(--text-2);
  font-size: 0.98rem;
  line-height: 1.6;
}

.td-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 12px;
  border-radius: 999px;
  border: 1px solid rgba(15, 157, 138, 0.14);
  background: var(--accent-soft);
  color: var(--accent);
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.22em;
  text-transform: uppercase;
}

.td-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 18px;
}

.td-chip,
.td-pill {
  display: inline-flex;
  align-items: center;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255, 255, 255, 0.94);
  color: var(--text-2);
  font-size: 0.76rem;
  font-weight: 600;
}

.td-pill {
  background: rgba(255, 255, 255, 0.86);
}

.td-hero-board {
  min-width: min(100%, 340px);
  display: grid;
}

.td-board-shell {
  display: grid;
  gap: 12px;
  padding: 18px;
  border-radius: 24px;
  border: 1px solid rgba(15, 23, 42, 0.10);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(250, 249, 246, 0.96));
  box-shadow: var(--shadow-strong);
  position: relative;
  overflow: hidden;
}

.td-board-shell::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    linear-gradient(135deg, rgba(15, 157, 138, 0.10), transparent 38%),
    radial-gradient(circle at top right, rgba(101, 234, 214, 0.12), transparent 24%);
  pointer-events: none;
}

.td-board-top {
  position: relative;
  z-index: 1;
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: center;
}

.td-board-top span {
  font-size: 0.64rem;
  font-weight: 700;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  color: var(--text-3);
}

.td-board-title {
  position: relative;
  z-index: 1;
  font-size: 1.35rem;
  font-weight: 700;
  letter-spacing: -0.04em;
  color: var(--text-1);
}

.td-board-copy {
  position: relative;
  z-index: 1;
  color: var(--text-2);
  font-size: 0.9rem;
  line-height: 1.55;
}

.td-board-grid {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.td-mini-tile {
  padding: 12px 12px 13px;
  border-radius: 18px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  background: rgba(255, 255, 255, 0.82);
}

.td-mini-tile span {
  display: block;
  color: var(--text-3);
  font-size: 0.63rem;
  font-weight: 700;
  letter-spacing: 0.22em;
  text-transform: uppercase;
}

.td-mini-tile strong {
  display: block;
  margin-top: 8px;
  color: var(--text-1);
  font-size: 0.96rem;
  font-weight: 650;
  line-height: 1.25;
}

.td-mini-tile small {
  display: block;
  margin-top: 5px;
  color: var(--text-2);
  font-size: 0.78rem;
  line-height: 1.35;
}

.td-stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}

.td-stat-card {
  padding: 15px;
  border-radius: 22px;
  border: 1px solid var(--border);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.99), rgba(250, 248, 244, 0.95));
  box-shadow: 0 12px 26px rgba(15, 23, 42, 0.05);
  position: relative;
  overflow: hidden;
}

.td-stat-card:first-child {
  background:
    linear-gradient(180deg, rgba(15, 157, 138, 0.09), rgba(255, 255, 255, 0.98));
  border-color: rgba(15, 157, 138, 0.16);
}

.td-stat-label {
  color: var(--text-3);
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.22em;
  text-transform: uppercase;
}

.td-stat-value {
  margin-top: 8px;
  font-size: 1.05rem;
  font-weight: 650;
  letter-spacing: -0.02em;
}

.td-stat-caption {
  margin-top: 5px;
  color: var(--text-2);
  font-size: 0.82rem;
  line-height: 1.35;
}

#td-main-grid,
#td-secondary-grid {
  display: grid;
  gap: 12px;
}

#td-main-grid {
  grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr) minmax(0, 0.9fr);
}

#td-secondary-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

#td-stats-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.td-panel {
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: 24px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 1), rgba(250, 248, 244, 0.97));
  box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
  color: var(--text-1);
  position: relative;
}

.td-copy-card {
  height: 100%;
}

.td-panel::before {
  content: "";
  position: absolute;
  inset: 0 auto auto 0;
  width: 100%;
  height: 1px;
  background: linear-gradient(90deg, rgba(15, 157, 138, 0.24), rgba(37, 99, 235, 0.10), transparent 65%);
  pointer-events: none;
}

.td-panel-wide {
  margin-top: 2px;
}

.td-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 13px 16px 11px;
  border-bottom: 1px solid var(--border);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(247, 246, 243, 0.95));
}

.td-card-head span:first-child {
  font-size: 0.91rem;
  font-weight: 650;
  letter-spacing: -0.01em;
  color: var(--text-1);
}

.td-card-head span:last-child {
  color: var(--text-3);
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.22em;
  text-transform: uppercase;
}

.td-card-body {
  padding: 15px;
  color: var(--text-1);
}

#td-video-feed,
#td-radar-feed {
  width: 100% !important;
}

#td-video-feed .image-container,
#td-radar-feed .image-container {
  border: 1px solid rgba(15, 23, 42, 0.06) !important;
  border-radius: 18px !important;
  overflow: hidden !important;
  background: #ffffff !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
}

#td-video-feed img,
#td-radar-feed img {
  display: block;
  width: 100% !important;
  height: 360px !important;
  object-fit: cover;
}

#td-radar-feed img {
  object-fit: contain;
  background: linear-gradient(180deg, rgba(37, 99, 235, 0.015), rgba(56, 189, 248, 0.03));
}

.td-empty {
  padding: 10px 2px;
  color: var(--text-2);
  font-size: 0.86rem;
}

.td-bar-item {
  display: grid;
  gap: 7px;
  padding: 12px 0;
  border-bottom: 1px solid rgba(15, 23, 42, 0.06);
}

.td-bar-item:last-child {
  border-bottom: none;
  padding-bottom: 0;
}

.td-bar-head,
.td-threat-meta-line,
.td-status-line {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}

.td-bar-head {
  font-size: 0.84rem;
  font-weight: 650;
  color: var(--text-1);
}

.td-bar-head span:last-child {
  color: var(--text-2);
  font-family: var(--mono);
  font-size: 0.78rem;
}

.td-bar-track {
  height: 7px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.07);
  overflow: hidden;
}

.td-bar-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
}

.td-bar-foot {
  color: var(--text-2);
  font-size: 0.77rem;
  letter-spacing: 0.01em;
}

.td-motor-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.td-motor-card {
  padding: 13px;
  border-radius: 18px;
  border: 1px solid var(--border);
  background: rgba(255, 255, 255, 0.95);
}

.td-motor-card span,
.td-log-row span {
  display: block;
}

.td-motor-card span {
  color: var(--text-3);
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.22em;
  text-transform: uppercase;
}

.td-motor-card strong {
  display: block;
  margin-top: 8px;
  font-size: 1.05rem;
  font-weight: 650;
}

.td-motor-card small {
  display: block;
  margin-top: 5px;
  color: var(--text-2);
  font-size: 0.8rem;
}

.td-status-line {
  align-items: baseline;
  margin-top: 10px;
  padding: 10px 12px;
  border-radius: 16px;
  background: rgba(15, 157, 138, 0.06);
  border: 1px solid rgba(15, 157, 138, 0.10);
}

.td-status-line strong {
  font-size: 0.95rem;
  font-weight: 650;
}

.td-status-line span {
  color: var(--text-3);
  font-size: 0.78rem;
}

.td-log-list {
  display: grid;
  gap: 7px;
  margin-top: 10px;
  max-height: 195px;
  overflow: auto;
}

.td-log-row {
  padding: 10px 12px;
  border-radius: 14px;
  border: 1px solid rgba(15, 23, 42, 0.06);
  background: rgba(255, 255, 255, 0.95);
  font-family: var(--mono);
  font-size: 0.76rem;
  color: var(--text-1);
  line-height: 1.45;
}

.td-threat-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px;
  border-radius: 18px;
  border: 1px solid rgba(15, 23, 42, 0.06);
  background: rgba(255, 255, 255, 0.96);
}

.td-threat-row + .td-threat-row {
  margin-top: 10px;
}

.td-threat-label {
  font-size: 0.96rem;
  font-weight: 650;
  letter-spacing: -0.01em;
  color: var(--text-1);
}

.td-threat-meta-line {
  margin-top: 5px;
  color: var(--text-2);
  font-size: 0.78rem;
  flex-wrap: wrap;
}

.td-threat-score {
  text-align: right;
  white-space: nowrap;
}

.td-threat-score span {
  display: block;
  color: var(--text-3);
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.22em;
  text-transform: uppercase;
}

.td-threat-score strong {
  display: block;
  margin-top: 6px;
  font-size: 0.98rem;
  font-weight: 650;
  color: var(--text-1);
}

.td-nav-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 6px;
  margin-bottom: 10px;
}

.td-nav-sector {
  padding: 8px 6px;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.92);
  text-align: center;
}

.td-nav-deg {
  font-size: 0.65rem;
  color: var(--text-3);
  font-weight: 700;
  font-family: var(--mono);
}

.td-nav-depth {
  font-size: 0.82rem;
  font-weight: 700;
  margin-top: 4px;
}

.td-nav-dir {
  font-size: 0.60rem;
  color: var(--text-3);
  margin-top: 2px;
  letter-spacing: 0.15em;
}

.td-nav-escape {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  border-radius: 14px;
  background: rgba(15,157,138,0.06);
  border: 1px solid rgba(15,157,138,0.12);
  font-size: 0.82rem;
  margin-top: 2px;
}

.td-nav-escape span {
  color: var(--text-3);
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}

.td-nav-escape strong {
  font-weight: 650;
}

.td-perf-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}

.td-perf-item {
  padding: 10px 10px 12px;
  border-radius: 14px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.92);
}

.td-perf-item span {
  display: block;
  color: var(--text-3);
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.20em;
  text-transform: uppercase;
}

.td-perf-item strong {
  display: block;
  margin-top: 6px;
  font-size: 0.88rem;
  font-weight: 650;
  font-family: var(--mono);
}

.td-footer {
  padding: 2px 4px 0;
  color: var(--text-3);
  font-size: 0.74rem;
}

@media (max-width: 980px) {
  .gradio-container > .main > .wrap {
    padding: 16px 12px 24px !important;
  }

  #td-nav,
  #td-main-grid,
  #td-secondary-grid,
  #td-stats-strip,
  .td-motor-grid,
  .td-threat-row {
    grid-template-columns: 1fr;
  }

  #td-video-feed img,
  #td-radar-feed img {
    height: 250px !important;
  }
}

@media (max-width: 640px) {
  #td-nav,
  .td-panel,
  .td-stat-card {
    border-radius: 18px;
  }

  .td-card-body,
  .td-card-head,
  .td-stat-card {
    padding-left: 12px;
    padding-right: 12px;
  }

  #td-video-feed img,
  #td-radar-feed img {
    height: 220px !important;
  }

  .td-chip-row {
    margin-top: 14px;
  }
}
"""

_SHELL_OPEN_HTML = """
<div id="td-shell">
  <header id="td-nav">
    <div class="td-brand">
      <div class="td-mark">TD</div>
      <div>
        <div class="td-kicker">ThreatDetect</div>
        <div class="td-title">Edge threat monitoring</div>
      </div>
    </div>
    <div class="td-live"><div class="td-live-dot"></div>Live pipeline</div>
  </header>
  <section id="td-stats-strip">
"""

_STATS_CLOSE_MAIN_OPEN_HTML = """
  </section>
  <section id="td-main-grid">
    <section class="td-panel" id="td-video-card">
      <div class="td-card-head">
        <span>Camera stream</span>
        <span>YOLO v8 · ONNX</span>
      </div>
      <div class="td-card-body">
"""

_MAIN_GRID_SEP_HTML = """
      </div>
    </section>
    <section class="td-panel" id="td-radar-card">
      <div class="td-card-head">
        <span>Spatial radar</span>
        <span>Gemma + beam scan</span>
      </div>
      <div class="td-card-body">
"""

_MAIN_GRID_SEP2_HTML = """
      </div>
    </section>
    <section class="td-panel td-copy-card" id="td-nav-card">
      <div class="td-card-body" style="padding-top:0">
"""

_MAIN_GRID_CLOSE_HTML = """
      </div>
    </section>
  </section>
  <section id="td-secondary-grid">
"""

_SECONDARY_SEP1_HTML = """
"""

_SECONDARY_SEP2_HTML = """
"""

_SECONDARY_GRID_CLOSE_HTML = """
  </section>
"""

_FOOTER_HTML = """
  <div class="td-footer">ThreatDetect live dashboard · stereo nav enabled.</div>
</div>
"""


def launch(server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False):
    if not GRADIO_AVAILABLE:
        raise RuntimeError("pip install gradio")

    with gr.Blocks(title="ThreatDetect") as demo:
        gr.HTML(_SHELL_OPEN_HTML)
        stats = gr.HTML(_build_stats_html(), elem_id="td-stats-html")
        gr.HTML(_STATS_CLOSE_MAIN_OPEN_HTML)
        video = gr.Image(type="numpy", show_label=False, height=360, elem_id="td-video-feed")
        gr.HTML(_MAIN_GRID_SEP_HTML)
        map_img = gr.Image(type="numpy", show_label=False, height=360, elem_id="td-radar-feed")
        gr.HTML(_MAIN_GRID_SEP2_HTML)
        nav_html = gr.HTML(_build_nav_html(), elem_id="td-nav-html")
        gr.HTML(_MAIN_GRID_CLOSE_HTML)
        audio_html = gr.HTML(_build_audio_html(), elem_id="td-audio-html")
        gr.HTML(_SECONDARY_SEP1_HTML)
        haptic_html = gr.HTML(_build_haptic_html(), elem_id="td-haptic-html")
        gr.HTML(_SECONDARY_SEP2_HTML)
        threats_html = gr.HTML(_build_threats_html(), elem_id="td-threats-html")
        gr.HTML(_SECONDARY_GRID_CLOSE_HTML)
        gr.HTML(_FOOTER_HTML)
        beep = gr.Audio(show_label=False, autoplay=True, visible=False)
        timer = gr.Timer(0.05)
        timer.tick(fn=_poll, outputs=[video, map_img, stats, audio_html, haptic_html, threats_html, nav_html, beep])

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        css=_CSS,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    launch()
