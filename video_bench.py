"""
Full pipeline video benchmark with optimization variants.

Optimizations tested:
  - frame_skip: run YOLO every Nth frame, propagate detections via flow
  - model: yolov8n vs yolo11n
  - imgsz: 224 → 160
  - async_yolo: YOLO runs in background thread, pipeline never waits

Usage:
    python video_bench.py                    # run all comparison variants
    python video_bench.py --single yolo11n   # single run
    python video_bench.py --video /path.mp4
"""
import sys, os, time, math, threading, argparse, queue
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from fusion import (
    ThreatMemory, build_prompt, CascadeRouter, SensorSnapshot,
    ThreatAnticipator, ANTICIPATION_THRESHOLD, StreamParser, ThreatAlert,
)
from fusion.tracker import ObjectTracker
from process.vision import THREAT_CLASSES, bbox_to_direction, estimate_distance, IMG_W, IMG_H
from ultralytics import YOLO

LLAMA_URL = os.environ.get("LLAMA_URL", "http://localhost:8081")
DEFAULT_IMAGE = "/home/ishaan/.local/lib/python3.13/site-packages/ultralytics/assets/bus.jpg"


# ── Frame generator ────────────────────────────────────────────────────────────

def make_frames(source: str, n: int):
    if os.path.exists(source) and source.endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(source)
        frames = []
        while len(frames) < n:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(f, (IMG_W, IMG_H)))
        cap.release()
        return frames
    # Animate image
    src = cv2.imread(source)
    if src is None:
        raise FileNotFoundError(source)
    sh, sw = src.shape[:2]
    frames = []
    for i in range(n):
        t = i / max(n - 1, 1)
        scale = 1.0 + 0.15 * t
        sc = cv2.resize(src, (int(sw * scale), int(sh * scale)))
        x0 = int(t * max(sc.shape[1] - IMG_W, 0))
        y0 = int(0.3 * max(sc.shape[0] - IMG_H, 0))
        crop = sc[y0:y0 + IMG_H, x0:x0 + IMG_W]
        if crop.shape[0] < IMG_H or crop.shape[1] < IMG_W:
            crop = cv2.resize(crop, (IMG_W, IMG_H))
        frames.append(crop)
    return frames


# ── YOLO helpers ───────────────────────────────────────────────────────────────

def yolo_detect(model, frame, imgsz):
    results = model(frame, imgsz=imgsz, verbose=False, conf=0.35)
    dets = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            if label not in THREAT_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append({
                "label": label,
                "confidence": round(conf, 3),
                "direction_deg": round(bbox_to_direction((x1 + x2) / 2), 1),
                "distance_m": estimate_distance(label, x2 - x1),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
            })
    return dets


def propagate_detections(dets, flow):
    """Move bboxes by mean optical flow displacement — used on YOLO-skip frames."""
    if not dets or flow is None:
        return dets
    updated = []
    fh, fw = flow.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(fw, x2), min(fh, y2)
        roi = flow[y1c:y2c, x1c:x2c]
        if roi.size == 0:
            updated.append(d)
            continue
        dx = float(np.mean(roi[:, :, 0]))
        dy = float(np.mean(roi[:, :, 1]))
        nx1, ny1 = x1 + dx, y1 + dy
        nx2, ny2 = x2 + dx, y2 + dy
        cx = (nx1 + nx2) / 2
        dd = {**d,
              "bbox": [round(nx1), round(ny1), round(nx2), round(ny2)],
              "direction_deg": round(bbox_to_direction(cx), 1),
              "confidence": round(d["confidence"] * 0.92, 3)}  # slight decay
        updated.append(dd)
    return updated


def optical_flow(prev_gray, gray):
    if prev_gray is None:
        return None, []
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=2, winsize=15,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
    )
    return flow, []


# ── Async YOLO worker ──────────────────────────────────────────────────────────

class AsyncYOLO:
    """YOLO runs in background thread. Main loop never waits — uses last result."""
    def __init__(self, model, imgsz):
        self.model = model
        self.imgsz = imgsz
        self._q = queue.Queue(maxsize=1)
        self._result = []
        self._lock = threading.Lock()
        self._busy = False
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _worker(self):
        while True:
            frame = self._q.get()
            if frame is None:
                break
            dets = yolo_detect(self.model, frame, self.imgsz)
            with self._lock:
                self._result = dets
                self._busy = False

    def submit(self, frame):
        if not self._busy:
            self._busy = True
            try:
                self._q.put_nowait(frame.copy())
            except queue.Full:
                self._busy = False

    def get(self):
        with self._lock:
            return list(self._result)

    def stop(self):
        try:
            self._q.put(None, timeout=2)
            self._t.join(timeout=5)
        except Exception:
            pass


# ── Core benchmark runner ──────────────────────────────────────────────────────

def run_variant(frames, model_name, imgsz, frame_skip, async_yolo_flag, target_fps=10):
    model = YOLO(model_name)
    # Warm up
    _ = yolo_detect(model, frames[0], imgsz)

    memory = ThreatMemory()
    router = CascadeRouter(memory)
    tracker = ObjectTracker()
    anticipator = ThreatAnticipator.load_default()

    gemma_calls = 0
    reservoir_escalations = 0

    def on_early(a): pass
    def on_complete(a): pass
    parser = StreamParser(on_early, on_complete, server_url=LLAMA_URL)

    t_yolo_total = 0.0
    t_yolo_frames = 0
    t_flow = t_track = t_router = t_res = 0.0

    prev_gray = None
    last_dets = []
    flow_full = None

    if async_yolo_flag:
        ayolo = AsyncYOLO(model, imgsz)

    t_wall = time.monotonic()
    interval = 1.0 / target_fps

    for i, frame in enumerate(frames):
        t0 = time.monotonic()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical flow (every frame)
        t1 = time.perf_counter()
        flow_full, _ = optical_flow(prev_gray, gray)
        t_flow += (time.perf_counter() - t1) * 1000
        prev_gray = gray

        # YOLO (sync every Nth frame, or async)
        run_yolo_this_frame = (i % frame_skip == 0)

        if async_yolo_flag:
            ayolo.submit(frame)
            dets = ayolo.get()
            if flow_full is not None and not run_yolo_this_frame:
                dets = propagate_detections(dets, flow_full)
            # No measurable yolo time in main thread
        elif run_yolo_this_frame:
            t1 = time.perf_counter()
            dets = yolo_detect(model, frame, imgsz)
            dt = (time.perf_counter() - t1) * 1000
            t_yolo_total += dt
            t_yolo_frames += 1
            last_dets = dets
        else:
            # Propagate last detections via flow
            dets = propagate_detections(last_dets, flow_full) if flow_full is not None else last_dets

        if not async_yolo_flag:
            last_dets = dets

        # Tracker
        t1 = time.perf_counter()
        dets = tracker.update(dets)
        t_track += (time.perf_counter() - t1) * 1000

        # Router
        snap = SensorSnapshot(detections=dets, audio_labels=[], heading_deg=0.0, pitch_deg=0.0, flow_vectors=[])
        t1 = time.perf_counter()
        tier = router.route(snap)
        t_router += (time.perf_counter() - t1) * 1000

        # Reservoir
        t1 = time.perf_counter()
        p = anticipator.predict(snap)
        t_res += (time.perf_counter() - t1) * 1000

        if tier == 0 and p > ANTICIPATION_THRESHOLD:
            now = time.time()
            if (now - router._last_gemma_call) >= 0.8:
                router._last_gemma_call = now
                router._gemma_call_count += 1
                tier = 2
                reservoir_escalations += 1

        if tier == 2:
            gemma_calls += 1
            prompt = build_prompt(detections=dets, audio_labels=[], heading_deg=0.0,
                                  pitch_deg=0.0, flow_vectors=[], history_summary=memory.summary_tokens())
            parser.infer(prompt)

        elapsed = time.monotonic() - t0
        time.sleep(max(0, interval - elapsed))

    if async_yolo_flag:
        ayolo.stop()

    wall = time.monotonic() - t_wall
    n = len(frames)
    actual_fps = n / wall

    yolo_mean = t_yolo_total / t_yolo_frames if t_yolo_frames > 0 else 0
    # For async, yolo runs off-thread so latency impact = 0 on main loop

    return {
        "fps": actual_fps,
        "yolo_ms": yolo_mean,
        "yolo_frames": t_yolo_frames,
        "flow_ms": t_flow / n,
        "track_ms": t_track / n,
        "router_ms": t_router / n,
        "res_ms": t_res / n,
        "gemma_calls": gemma_calls,
        "res_escalations": reservoir_escalations,
        "frames": n,
        "async": async_yolo_flag,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

VARIANTS = [
    # (label,              model,        imgsz, skip, async)
    ("v8n-160  sync     ", "yolov8n.pt", 160,   1,    False),
    ("v8n-160  skip2    ", "yolov8n.pt", 160,   2,    False),
    ("v11n-160 sync     ", "yolo11n.pt", 160,   1,    False),
    ("v11n-160 skip2    ", "yolo11n.pt", 160,   2,    False),
    ("v12n-160 sync     ", "yolo12n.pt", 160,   1,    False),
    ("v12n-160 skip2    ", "yolo12n.pt", 160,   2,    False),
    ("v26n-160 sync     ", "yolo26n.pt", 160,   1,    False),
    ("v26n-160 skip2    ", "yolo26n.pt", 160,   2,    False),
    ("v26n-160 async    ", "yolo26n.pt", 160,   1,    True),
    ("v11n-160 async    ", "yolo11n.pt", 160,   1,    True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_IMAGE)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--url", default=LLAMA_URL)
    ap.add_argument("--single", default=None, help="Run single variant by label substring")
    args = ap.parse_args()

    import fusion.stream_parser as sp
    sp.LLAMA_URL = args.url

    print(f"Generating {args.frames} frames from: {args.source}")
    frames = make_frames(args.source, args.frames)
    print(f"  {len(frames)} frames ready ({IMG_W}x{IMG_H})\n")

    variants = VARIANTS
    if args.single:
        variants = [v for v in VARIANTS if args.single.lower() in v[0].lower()]
        if not variants:
            print(f"No variant matching '{args.single}'")
            return

    results = {}
    for label, model, imgsz, skip, async_flag in variants:
        print(f"▶ {label.strip()} | imgsz={imgsz} skip={skip} async={async_flag}")
        r = run_variant(frames, model, imgsz, skip, async_flag, target_fps=args.fps)
        results[label] = r
        yolo_str = f"{r['yolo_ms']:6.1f}ms ({r['yolo_frames']}fr)" if not r['async'] else "  off-thread"
        print(f"  FPS={r['fps']:.1f}  YOLO={yolo_str}  res={r['res_ms']:.4f}ms  "
              f"gemma={r['gemma_calls']}  res_esc={r['res_escalations']}\n")

    print("\n" + "═" * 90)
    print(f"{'Variant':<28} {'FPS':>5}  {'YOLO/f':>9}  {'Flow/f':>7}  {'Res/f':>9}  {'Gemma':>6}  {'ResEsc':>7}")
    print("─" * 90)
    for label, r in results.items():
        yolo_str = f"{r['yolo_ms']:7.1f}ms" if not r['async'] else "  async  "
        print(f"{label:<28} {r['fps']:>5.1f}  {yolo_str}  {r['flow_ms']:>5.1f}ms  "
              f"{r['res_ms']:>7.4f}ms  {r['gemma_calls']:>6}  {r['res_escalations']:>7}")
    print("═" * 90)


if __name__ == "__main__":
    main()
