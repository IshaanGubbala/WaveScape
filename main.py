"""
ThreatDetect main pipeline.
Cascade: Vision+Audio (always-on) → Router → Tier-1 haptic (YOLO direct)
         → SpatialMapper (Gemma every 3s, video + directional mic beams)
         → Predictor (10fps extrapolation between Gemma calls)
"""
import time
import threading
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from process import MockVisionProcessor, MockAudioProcessor
from fusion import (
    ThreatMemory, ThreatPredictor, MotionVector,
    CascadeRouter, SensorSnapshot,
    ThreatAnticipator, ANTICIPATION_THRESHOLD,
    SpatialMapper, SpatialMap,
    StereoDepthEstimator,
    MacSceneOffload,
)
from fusion.tracker import ObjectTracker
from output.haptic import HapticController
import output.web_ui as web_ui
import output.live_server as live_server

try:
    from process import VisionProcessor, DualVisionProcessor, AudioProcessor
    REAL_HW = True
except ImportError:
    REAL_HW = False

LLAMA_URL = os.environ.get("LLAMA_URL", "http://localhost:8081")
USE_MOCK = os.environ.get("USE_MOCK", "1") == "1"
TARGET_FPS = 10


_haptic = HapticController()


def _sim_stereo_frame(frame, shift_px: int = 5):
    """Synthesize fake right-camera frame by shifting left.
    shift_px=5 at 96px → disparity≈5 → depth≈0.96m (triggers obstacle haptic).
    Exposed border filled with edge column to avoid SGBM artifacts."""
    import cv2 as _cv2
    import numpy as _np
    f = _cv2.resize(frame, (96, 96))
    shifted = _np.roll(f, -shift_px, axis=1)
    shifted[:, -shift_px:] = shifted[:, -shift_px - 1 : -shift_px]
    return shifted


def _infer_pattern(direction_deg: float, urgency: str, llm_pattern: str) -> str:
    """Fall back to direction-based pattern when LLM outputs 'none'.
    Boundaries: ±22.5° around forward/behind = center; otherwise left/right.
    Motor intensities are smoothly interpolated by HapticController._direction_to_split."""
    if llm_pattern and llm_pattern != "none":
        return llm_pattern
    if urgency in ("high", "critical"):
        d = direction_deg % 360
        if 22.5 <= d < 157.5:
            return "rapid_right"
        elif 202.5 <= d < 337.5:
            return "rapid_left"
        else:
            return "rapid_center"   # within ±22.5° of ahead or behind
    return "slow_pulse" if urgency == "medium" else "none"


def haptic_fire(direction: float, urgency: str, confidence: float, source: str = "", pattern: str = "none"):
    """Fire haptic motor (GPIO on Pi, console on Mac)."""
    effective_pattern = _infer_pattern(direction, urgency, pattern)
    _haptic.fire(direction, urgency, effective_pattern)
    web_ui.log_haptic(source, direction, urgency, confidence, effective_pattern)


def main(mock: bool = True, dual_cam: bool = False, yolo_model: str = None, video_path: str = None, stereo_sim: bool = False):
    import platform
    _default_model = "/Users/ishaangubbala/Documents/airesearch/yolov8n.pt" if platform.system() == "Darwin" else "yolo26n.pt"
    _yolo = yolo_model or _default_model

    print("=" * 60)
    print("THREATDETECT — starting pipeline")
    print(f"  server: {LLAMA_URL}")
    print(f"  mode:   {'MOCK' if mock else 'REAL HARDWARE'}")
    if not mock:
        print(f"  yolo:   {_yolo}")
    print("=" * 60)

    # Components
    _imgsz = 96 if str(_yolo).endswith(".onnx") else 160
    if mock:
        vision = MockVisionProcessor()
    elif dual_cam:
        vision = DualVisionProcessor(model_size=_yolo, imgsz=_imgsz)
    else:
        vision = VisionProcessor(model_size=_yolo, imgsz=_imgsz, async_yolo=True)
    # Try MCP3008 SPI ADC first (Pi 5 + MAX4466 mic array), fall back to
    # sounddevice (Mac built-in mic / I2S), then mock.
    audio = None
    try:
        from process.audio_mcp3008 import MCP3008AudioProcessor
        audio = MCP3008AudioProcessor(num_mics=4, sample_rate=10000)
        import time as _t; _t.sleep(0.3)  # warm ring buffer
        probe = audio.capture_window(duration_ms=50)
        if probe is None or probe.size == 0:
            raise RuntimeError("no MCP3008 samples")
        print(f"  audio:  MCP3008 SPI (4ch @ 10000Hz)")
    except Exception as e:
        print(f"  audio:  MCP3008 unavailable ({e}), trying sounddevice...")
        try:
            audio = AudioProcessor()
            audio.capture_window(duration_ms=50)
            print(f"  audio:  real (sounddevice, {audio._channels}ch @ {audio._rate}Hz)")
        except Exception:
            audio = MockAudioProcessor()
            print("  audio:  mock (no input device)")
    memory = ThreatMemory()
    router = CascadeRouter(memory)
    tracker = ObjectTracker()
    anticipator = ThreatAnticipator.load_default()

    def on_spatial_map(smap: SpatialMap, beam_scan: list = None):
        web_ui.update_spatial(smap, beam_scan or [])
        if smap.is_empty():
            return
        direction = smap.dominant_direction()
        urgency = smap.dominant_urgency()
        if direction is not None and urgency in ("medium", "high", "critical"):
            haptic_fire(direction, urgency, 0.85, "SPATIAL")
        if smap.summary:
            print(f"    ↳ {smap.summary}")

    spatial_mapper = SpatialMapper(on_spatial_map, server_url=LLAMA_URL)
    mac_offload = MacSceneOffload()

    def haptic_cb(direction, urgency, confidence):
        haptic_fire(direction, urgency, confidence, "PREDICT")

    predictor = ThreatPredictor(haptic_cb, update_hz=TARGET_FPS)
    predictor.start()

    live_server.start()

    stereo = StereoDepthEstimator() if (dual_cam or stereo_sim) else None

    # Video file source (overrides camera)
    _video_cap = None
    if video_path:
        import cv2 as _cv2
        _video_cap = _cv2.VideoCapture(video_path)
        if not _video_cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        # Cap loop FPS to video's native FPS so we don't burn CPU reading faster than source
        _vid_fps = _video_cap.get(_cv2.CAP_PROP_FPS)
        if 1 <= _vid_fps <= 120:
            _loop_fps = _vid_fps if 1 <= _vid_fps <= 120 else TARGET_FPS
            print(f"  video: {video_path}  (native {_loop_fps:.1f}fps)")
        else:
            _loop_fps = TARGET_FPS
            print(f"  video: {video_path}")
    else:
        _loop_fps = TARGET_FPS

    print(f"\nRunning at {_loop_fps:.1f}fps sensor loop. Ctrl+C to stop.\n")
    frame_count = 0
    interval = 1.0 / _loop_fps
    _next_frame_time = time.monotonic()
    import collections as _col
    _frame_times = _col.deque(maxlen=30)
    _last_yolo_ms = 0.0
    _last_stereo_ms = 0.0
    # Haptic rate limiting: don't re-fire same source within cooldown
    _haptic_last: dict[str, float] = {}
    _haptic_dir:  dict[str, float] = {}
    HAPTIC_COOLDOWN = {"TIER1": 0.6, "STEREO": 1.0, "AUDIO": 1.5}
    HAPTIC_DIR_THRESH = 20.0  # re-fire if direction changes by this many degrees

    def _should_haptic(source: str, direction: float) -> bool:
        now = time.monotonic()
        last_t = _haptic_last.get(source, 0.0)
        last_d = _haptic_dir.get(source, -999.0)
        cooldown = HAPTIC_COOLDOWN.get(source, 0.5)
        delta_dir = abs((direction - last_d + 180) % 360 - 180)
        if now - last_t > cooldown or delta_dir > HAPTIC_DIR_THRESH:
            _haptic_last[source] = now
            _haptic_dir[source] = direction
            return True
        return False

    try:
        while True:
            t0 = time.monotonic()
            frame_count += 1

            # Video file read (loops at end) — sync to video's native FPS
            if _video_cap is not None:
                now = time.monotonic()
                sleep_s = _next_frame_time - now
                if sleep_s > 0:
                    time.sleep(sleep_s)
                _next_frame_time = max(time.monotonic(), _next_frame_time) + interval
                ret, frame = _video_cap.read()
                if not ret:
                    _video_cap.set(0, 0)  # loop
                    _next_frame_time = time.monotonic() + interval
                    ret, frame = _video_cap.read()
                if not ret:
                    break
                import cv2 as _cv2
                from process.vision import IMG_W, IMG_H
                frame = _cv2.resize(frame, (IMG_W, IMG_H))
                _t_frame_start = time.monotonic()
                detections, flow = vision.process_frame(frame)
                web_ui.update_frame(0, frame, detections)
                web_ui.update_yolo_live(detections)
                snap = SensorSnapshot(detections=detections, audio_labels=[], heading_deg=0.0, pitch_deg=0.0, flow_vectors=flow)
                tier = router.route(snap)
                p_threat = anticipator.predict(snap)
                audio_window = audio.capture_window(duration_ms=50)
                beam_scan = audio.beamform_scan(audio_window)
                audio_labels_vid = audio.classify(audio_window) if frame_count % 5 == 0 else []
                if beam_scan and audio_labels_vid:
                    beam_scan[0]["label"] = audio_labels_vid[0][0]
                web_ui.log_audio_event(audio_labels_vid, beam_scan)
                if beam_scan and beam_scan[0]["energy"] > 0.03:
                    top = beam_scan[0]
                    if _should_haptic("AUDIO", top["direction_deg"]):
                        haptic_fire(top["direction_deg"], "medium", top["energy"], "AUDIO")
                if frame_count % 10 == 0:
                    det_str = " ".join(f"{d['label']}@{d['confidence']:.2f}" for d in detections) or "nothing"
                    beam_str = ", ".join(f"{b['direction_deg']}°={b['energy']:.3f}" for b in beam_scan[:3]) or "quiet"
                    print(f"  [frame {frame_count:4d}] tier={tier} dets=[{det_str}] mic=[{beam_str}]")
                if tier >= 1 and detections:
                    d = max(detections, key=lambda x: x["confidence"])
                    if _should_haptic("TIER1", d["direction_deg"]):
                        haptic_fire(d["direction_deg"], "high", d["confidence"], "TIER1")
                if stereo_sim and stereo is not None and frame_count % 2 == 0:
                    sdepth = stereo.compute(frame, _sim_stereo_frame(frame))
                    _last_stereo_ms = sdepth.get("latency_ms", 0.0)
                    web_ui.update_stereo(sdepth)
                    mac_offload.update(sdepth["sector_depths"], detections, beam_scan)
                    if sdepth["min_depth_m"] < 1.5:
                        if _should_haptic("STEREO", sdepth["escape_dir_deg"]):
                            haptic_fire(sdepth["escape_dir_deg"], "high",
                                        1.0 - sdepth["min_depth_m"] / 1.5, "STEREO")
                    if frame_count % 20 == 0:
                        sectors = " ".join(
                            f"{'.' if s['safe'] else 'X'}{s['angle_deg']:.0f}°={s['min_depth_m']:.1f}m"
                            for s in sdepth["sector_depths"]
                        )
                        print(f"    [stereo-sim] escape={sdepth['escape_dir_deg']:.0f}° "
                              f"min={sdepth['min_depth_m']:.2f}m {sdepth['latency_ms']:.0f}ms | {sectors}")
                _frame_ms = (time.monotonic() - _t_frame_start) * 1000
                _frame_times.append(_frame_ms)
                _fps = 1000.0 / (sum(_frame_times) / len(_frame_times)) if _frame_times else 0.0
                web_ui.update_perf(_frame_ms, _fps, yolo_ms=0.0, stereo_ms=_last_stereo_ms)
                mac_offload.update([], detections, beam_scan)
                if mac_offload.is_fresh():
                    web_ui.update_mac_scene(mac_offload.get_result())
                if spatial_mapper.should_update(beam_scan):
                    print(f"\n  [frame {frame_count:4d}] SPATIAL → Gemma")
                    spatial_mapper.update(detections, beam_scan, frame=frame, heading_deg=0.0,
                                  audio_b64=audio.capture_wav_b64(duration_ms=2000))
                continue

            # Capture
            if dual_cam:
                frame0, frame1 = vision.read_frame()
                detections, flow = vision.process_frame(frame0, frame1)
                frame = frame0
                web_ui.update_frame(0, frame0, detections)
                if frame1 is not None:
                    web_ui.update_frame(1, frame1, detections)
                    # Stereo depth every other frame (SGBM ~30ms on Pi)
                    if stereo is not None and frame_count % 2 == 0:
                        sdepth = stereo.compute(frame0, frame1)
                        _last_stereo_ms = sdepth.get("latency_ms", 0.0)
                        web_ui.update_stereo(sdepth)
                        mac_offload.update(sdepth["sector_depths"], detections, beam_scan)
                        if sdepth["min_depth_m"] < 1.5:
                            haptic_fire(
                                sdepth["escape_dir_deg"], "high",
                                1.0 - sdepth["min_depth_m"] / 1.5,
                                "STEREO",
                            )
                        if frame_count % 20 == 0:
                            sectors = " ".join(
                                f"{'.' if s['safe'] else 'X'}{s['angle_deg']:.0f}°={s['min_depth_m']:.1f}m"
                                for s in sdepth["sector_depths"]
                            )
                            print(f"    [stereo] escape={sdepth['escape_dir_deg']:.0f}° "
                                  f"min={sdepth['min_depth_m']:.2f}m {sdepth['latency_ms']:.0f}ms | {sectors}")
            else:
                frame = vision.read_frame()
                if frame is None:
                    time.sleep(interval)
                    continue
                detections, flow = vision.process_frame(frame)
                web_ui.update_frame(0, frame, detections)
            audio_window = audio.capture_window(duration_ms=50)
            beam_scan = audio.beamform_scan(audio_window)
            audio_labels = audio.classify(audio_window) if frame_count % 5 == 0 else []
            if beam_scan and audio_labels:
                beam_scan[0]["label"] = audio_labels[0][0]
            web_ui.log_audio_event(audio_labels, beam_scan)
            if beam_scan and beam_scan[0]["energy"] > 0.08:
                top = beam_scan[0]
                haptic_fire(top["direction_deg"], "medium", top["energy"], "AUDIO")

            # Track objects across frames — adds velocity, ETA, motion_label
            detections = tracker.update(detections)
            web_ui.update_yolo_live(detections)

            # Snapshot
            snap = SensorSnapshot(
                detections=detections,
                audio_labels=audio_labels,
                heading_deg=0.0,
                pitch_deg=0.0,
                flow_vectors=flow,
            )

            # Route — Tier 1 only (Gemma now runs as periodic spatial mapper)
            tier = router.route(snap)
            p_threat = anticipator.predict(snap)

            # Heartbeat every 10 frames
            if frame_count % 10 == 0:
                det_str = " ".join(f"{d['label']}@{d['confidence']:.2f}" for d in detections) or "nothing"
                beam_str = ", ".join(f"{b['direction_deg']}°={b['energy']:.3f}" for b in beam_scan[:3]) or "quiet"
                print(f"  [frame {frame_count:4d}] tier={tier} dets=[{det_str}] mic=[{beam_str}]")

            # Tier 1: direct haptic from high-confidence YOLO detection
            if tier >= 1 and detections:
                d = max(detections, key=lambda x: x["confidence"])
                haptic_fire(d["direction_deg"], "high", d["confidence"], "TIER1")

            # Mac scene offload: send YOLO + beam scan every 20 frames (no stereo path)
            if stereo is None and frame_count % 20 == 0:
                mac_offload.update([], detections, beam_scan)

            # Spatial mapper: periodic Gemma call (every 3s) with frame + real mic beams
            if spatial_mapper.should_update(beam_scan):
                print(f"\n  [frame {frame_count:4d}] SPATIAL → Gemma")
                spatial_mapper.update(detections, beam_scan, frame=frame, heading_deg=0.0,
                                  audio_b64=audio.capture_wav_b64(duration_ms=2000))

            # Predictor: 10Hz haptic extrapolation between Gemma calls
            if flow:
                best = max(flow, key=lambda f: f["magnitude"])
                motion = MotionVector(
                    direction_deg=best["direction_deg"],
                    speed_mps=best["magnitude"] * 15,
                    label=best["label"],
                    confidence=best["magnitude"],
                    origin_distance_m=detections[0]["distance_m"] if detections else 5.0,
                    captured_at=time.time(),
                )
                predictor.update_motion(motion)

            _frame_ms = (time.monotonic() - t0) * 1000
            _frame_times.append(_frame_ms)
            _fps = 1000.0 / (sum(_frame_times) / len(_frame_times)) if _frame_times else 0.0
            web_ui.update_perf(_frame_ms, _fps, stereo_ms=_last_stereo_ms)

    except KeyboardInterrupt:
        predictor.stop()
        _haptic.cleanup()
        stats = router.stats()
        print(f"\n{'='*60}")
        print(f"Stopped after {frame_count} frames")
        print(f"Gemma calls: {stats['gemma_calls']} ({stats['gemma_calls']/max(frame_count,1)*100:.1f}% of frames)")
        vision.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Use real hardware")
    parser.add_argument("--dual-cam", action="store_true", help="Use both Pi cameras for 360° coverage")
    parser.add_argument("--url", default=LLAMA_URL, help="llama-server URL")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio web dashboard on port 7860")
    parser.add_argument("--yolo-model", default=None, help="YOLO model path (default: yolo26n.pt on Pi, yolov8n.pt on Mac)")
    parser.add_argument("--video", default=None, help="Path to video file (loops); overrides camera input")
    parser.add_argument("--stereo-sim", action="store_true",
                        help="Simulate stereo by shifting each frame 5px; tests depth pipeline without dual cam")
    args = parser.parse_args()

    import fusion.spatial_mapper as sm
    sm.LLAMA_URL = args.url

    if args.ui:
        import threading
        t = threading.Thread(target=web_ui.launch, kwargs={"server_name": "0.0.0.0", "server_port": 7860}, daemon=True)
        t.start()
        print("  web UI: http://0.0.0.0:7860")

    main(mock=not args.real, dual_cam=args.dual_cam, yolo_model=args.yolo_model, video_path=args.video, stereo_sim=args.stereo_sim)
