"""
ThreatDetect live dashboard — radar + haptic log + mic beamform panel.
"""
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


def _poll():
    global _last_sent_ts
    with _log_lock:
        lines = list(_log)
    with _signal_lock:
        sig_ts    = _haptic_signal["ts"]
        urgency   = _haptic_signal["urgency"]
        direction = _haptic_signal["direction"]

    log_text  = "\n".join(lines) if lines else "(no events yet)"
    beam_text = _beam_panel_text()
    map_img   = _render_map()
    frame     = _get_frame()

    if sig_ts > _last_sent_ts:
        _last_sent_ts = sig_ts
        beep = _make_beep(urgency, direction)
    else:
        beep = None

    return frame, map_img, log_text, beam_text, beep


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {
  box-sizing: border-box;
}

:root {
  --bg: #fbfbf8;
  --bg-2: #f3f6f8;
  --surface: rgba(255, 255, 255, 0.90);
  --surface-strong: rgba(255, 255, 255, 0.98);
  --surface-soft: rgba(255, 255, 255, 0.72);
  --border: rgba(15, 23, 42, 0.08);
  --border-strong: rgba(15, 23, 42, 0.12);
  --text-1: #111827;
  --text-2: #475569;
  --text-3: #94a3b8;
  --accent: #2563eb;
  --accent-2: #38bdf8;
  --accent-soft: rgba(37, 99, 235, 0.08);
  --good: #0f766e;
  --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
  --font: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', Inter, ui-sans-serif, system-ui, sans-serif;
  --shadow: 0 18px 60px rgba(15, 23, 42, 0.06);
  --shadow-strong: 0 24px 80px rgba(15, 23, 42, 0.10);
}

html,
body {
  min-height: 100%;
  background:
    radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 30%),
    radial-gradient(circle at top right, rgba(56, 189, 248, 0.06), transparent 26%),
    radial-gradient(circle at bottom right, rgba(15, 118, 110, 0.05), transparent 28%),
    linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%) !important;
}

body,
.gradio-container {
  background: linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%) !important;
  color: var(--text-1) !important;
  font-family: var(--font) !important;
}

.gradio-container {
  max-width: none !important;
  min-height: 100vh;
}

.gradio-container > .main > .wrap {
  max-width: 1480px !important;
  padding: 28px 20px 40px !important;
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
  gap: 12px;
}

#td-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 16px;
  border: 1px solid var(--border);
  border-radius: 24px;
  background: var(--surface);
  box-shadow: var(--shadow);
  backdrop-filter: blur(20px);
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
  width: 44px;
  height: 44px;
  border-radius: 16px;
  background: linear-gradient(180deg, rgba(37, 99, 235, 0.12), rgba(14, 165, 233, 0.08));
  border: 1px solid rgba(37, 99, 235, 0.12);
  color: var(--accent);
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 0.08em;
}

.td-kicker {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: var(--text-3);
}

.td-title {
  margin-top: 2px;
  font-size: 1rem;
  font-weight: 650;
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
  grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr);
  gap: 12px;
  align-items: stretch;
}

.td-hero-card,
.td-panel {
  border: 1px solid var(--border);
  border-radius: 28px;
  background: var(--surface-strong);
  box-shadow: var(--shadow);
  backdrop-filter: blur(20px);
}

.td-hero-card {
  padding: 18px 20px;
}

.td-hero-copy h1 {
  margin-top: 8px;
  max-width: 15ch;
  font-size: clamp(1.85rem, 2.9vw, 3.1rem);
  line-height: 1.02;
  letter-spacing: -0.04em;
}

.td-hero-copy p {
  margin-top: 10px;
  max-width: 50ch;
  color: var(--text-2);
  font-size: 0.94rem;
  line-height: 1.5;
}

.td-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 12px;
  border-radius: 999px;
  border: 1px solid rgba(37, 99, 235, 0.12);
  background: var(--accent-soft);
  color: var(--accent);
  font-size: 0.67rem;
  font-weight: 700;
  letter-spacing: 0.22em;
  text-transform: uppercase;
}

.td-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.td-pill {
  display: inline-flex;
  align-items: center;
  padding: 7px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255, 255, 255, 0.72);
  color: var(--text-2);
  font-size: 0.76rem;
  font-weight: 600;
}

.td-hero-metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  align-content: start;
}

.td-metric {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 14px;
  border-radius: 20px;
  border: 1px solid var(--border);
  background: var(--surface-strong);
}

.td-metric span,
.td-card-head span:last-child {
  color: var(--text-3);
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.22em;
  text-transform: uppercase;
}

.td-metric strong {
  font-size: 0.96rem;
  font-weight: 650;
  letter-spacing: -0.02em;
  line-height: 1.25;
}

#td-dashboard {
  display: grid;
  gap: 12px;
}

.td-section-title {
  display: flex;
  align-items: baseline;
  justify-content: flex-start;
  gap: 10px;
  padding: 0 4px;
}

.td-section-title strong {
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  color: var(--text-3);
}

.td-section-title span {
  color: var(--text-2);
  font-size: 0.84rem;
}

#td-media-grid,
#td-bottom-grid {
  display: grid;
  gap: 12px;
}

#td-media-grid {
  grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.9fr);
}

#td-bottom-grid {
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
}

.td-panel {
  overflow: hidden;
}

.td-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 14px 10px;
  border-bottom: 1px solid var(--border);
  background: rgba(255, 255, 255, 0.56);
}

.td-card-head span:first-child {
  font-size: 0.92rem;
  font-weight: 650;
  letter-spacing: -0.01em;
}

.td-card-body {
  padding: 14px;
}

#td-video-feed,
#td-radar-feed {
  width: 100% !important;
}

#td-video-feed .image-container,
#td-radar-feed .image-container {
  border: 1px solid rgba(15, 23, 42, 0.06) !important;
  border-radius: 22px !important;
  overflow: hidden !important;
  background: #ffffff !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
}

#td-video-feed img,
#td-radar-feed img {
  display: block;
  width: 100% !important;
  height: 280px !important;
  object-fit: cover;
}

#td-radar-feed img {
  object-fit: contain;
  background: linear-gradient(180deg, rgba(37, 99, 235, 0.015), rgba(56, 189, 248, 0.03));
}

#td-haptic-box textarea,
#td-beam-box textarea {
  width: 100% !important;
  min-height: 220px !important;
  border: 1px solid rgba(15, 23, 42, 0.08) !important;
  border-radius: 20px !important;
  background: rgba(255, 255, 255, 0.94) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72) !important;
  color: var(--text-2) !important;
  font-family: var(--mono) !important;
  font-size: 0.82rem !important;
  line-height: 1.55 !important;
  padding: 14px 16px !important;
  resize: none !important;
  outline: none !important;
}

#td-haptic-box textarea {
  min-height: 220px !important;
}

#td-beam-box textarea {
  min-height: 220px !important;
}

#td-haptic-box textarea::-webkit-scrollbar,
#td-beam-box textarea::-webkit-scrollbar {
  width: 6px;
}

#td-haptic-box textarea::-webkit-scrollbar-thumb,
#td-beam-box textarea::-webkit-scrollbar-thumb {
  background: rgba(148, 163, 184, 0.45);
  border-radius: 999px;
}

.td-footer {
  padding: 2px 4px 0;
  color: var(--text-3);
  font-size: 0.74rem;
}

@media (max-width: 980px) {
  .gradio-container > .main > .wrap {
    padding: 14px 12px 22px !important;
  }

  #td-nav {
    flex-direction: column;
    align-items: flex-start;
  }

  #td-hero,
  #td-media-grid,
  #td-bottom-grid {
    grid-template-columns: 1fr;
  }

  #td-video-feed img,
  #td-radar-feed img {
    height: 260px !important;
  }

  .td-hero-metrics {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 640px) {
  .td-hero-card,
  .td-panel,
  #td-nav {
    border-radius: 22px;
  }

  .td-card-body,
  .td-card-head,
  .td-hero-card {
    padding-left: 14px;
    padding-right: 14px;
  }

  #td-video-feed img,
  #td-radar-feed img {
    height: 230px !important;
  }
}
"""

_SHELL_OPEN_HTML = """
<div id="td-shell">
  <div id="td-nav">
    <div class="td-brand">
      <div class="td-mark">TD</div>
      <div>
        <div class="td-kicker">ThreatDetect</div>
        <div class="td-title">Calm, high-signal monitoring</div>
      </div>
    </div>
    <div class="td-live"><div class="td-live-dot"></div>Live pipeline</div>
  </div>

  <section id="td-hero">
    <div class="td-hero-card td-hero-copy">
      <div class="td-eyebrow">Light mode first · Apple-clean · subtle cyber</div>
      <h1>Situational awareness without the visual noise.</h1>
      <p>
        Video, spatial radar, haptic output, and mic beams are arranged as quiet cards so the
        signal stays easy to scan even when the pipeline is busy.
      </p>
      <div class="td-pills">
        <span class="td-pill">Vision feed</span>
        <span class="td-pill">Audio beams</span>
        <span class="td-pill">Spatial map</span>
        <span class="td-pill">Haptic timeline</span>
      </div>
    </div>

    <div class="td-hero-card td-hero-metrics">
      <div class="td-metric">
        <span>Mode</span>
        <strong>Real hardware</strong>
      </div>
      <div class="td-metric">
        <span>Signal path</span>
        <strong>Vision + audio + Gemma</strong>
      </div>
      <div class="td-metric">
        <span>Palette</span>
        <strong>Warm light glass</strong>
      </div>
    </div>
  </section>

  <section id="td-dashboard">
    <div class="td-section-title">
      <strong>Perception</strong>
      <span>Camera feed and spatial radar</span>
    </div>

    <div id="td-media-grid">
      <section class="td-panel" id="td-video-card">
        <div class="td-card-head">
          <span>Camera feed</span>
          <span>YOLO v8</span>
        </div>
        <div class="td-card-body">
"""

_AFTER_VIDEO_HTML = """
        </div>
      </section>

      <section class="td-panel" id="td-radar-card">
        <div class="td-card-head">
          <span>Spatial radar</span>
          <span>Gemma + beam scan</span>
        </div>
        <div class="td-card-body">
"""

_AFTER_RADAR_HTML = """
        </div>
      </section>
    </div>

    <div class="td-section-title">
      <strong>Response</strong>
      <span>Haptic events and microphone directionality</span>
    </div>

    <div id="td-bottom-grid">
      <section class="td-panel" id="td-haptic-card">
        <div class="td-card-head">
          <span>Haptic events</span>
          <span>Recent output</span>
        </div>
        <div class="td-card-body">
"""

_AFTER_HAPTIC_HTML = """
        </div>
      </section>

      <section class="td-panel" id="td-beam-card">
        <div class="td-card-head">
          <span>Mic beams</span>
          <span>Direction • energy • distance</span>
        </div>
        <div class="td-card-body">
"""

_FOOTER_HTML = """
        </div>
      </section>
    </div>

    <div class="td-footer">ThreatDetect live dashboard · tuned for light mode.</div>
  </section>
</div>
"""


def launch(server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False):
    if not GRADIO_AVAILABLE:
        raise RuntimeError("pip install gradio")

    with gr.Blocks(title="ThreatDetect", css=_CSS) as demo:
        gr.HTML(_SHELL_OPEN_HTML)
        video = gr.Image(type="numpy", show_label=False, height=340, elem_id="td-video-feed")
        gr.HTML(_AFTER_VIDEO_HTML)
        map_img = gr.Image(type="numpy", show_label=False, height=340, elem_id="td-radar-feed")
        gr.HTML(_AFTER_RADAR_HTML)
        haptic_box = gr.Textbox(
            show_label=False,
            lines=10,
            interactive=False,
            max_lines=14,
            elem_id="td-haptic-box"
        )
        gr.HTML(_AFTER_HAPTIC_HTML)
        beam_box = gr.Textbox(
            show_label=False,
            lines=10,
            interactive=False,
            max_lines=10,
            elem_id="td-beam-box"
        )
        gr.HTML(_FOOTER_HTML)
        beep  = gr.Audio(show_label=False, autoplay=True, visible=False)
        timer = gr.Timer(0.05)
        timer.tick(fn=_poll, outputs=[video, map_img, haptic_box, beam_box, beep])

    demo.launch(server_name=server_name, server_port=server_port, share=share)


if __name__ == "__main__":
    launch()
