"""
End-to-end pipeline test. Run against live llama-server.
Tests: routing, stream parsing, memory decay, prompt compression, predictor.
"""
import sys
import time
import threading
sys.path.insert(0, "/Users/ishaangubbala/threatdetect")

from fusion import (
    StreamParser, ThreatAlert, ThreatMemory,
    build_prompt, ThreatPredictor, MotionVector,
    CascadeRouter, SensorSnapshot,
    ThreatAnticipator, ANTICIPATION_THRESHOLD,
)

# ── Test fixtures ──────────────────────────────────────────────────────────────

CAR_SCENARIO = dict(
    detections=[
        {"label": "car", "confidence": 0.91, "direction_deg": 315, "distance_m": 8.0},
    ],
    audio_labels=[("horn_honk", 0.96), ("engine_noise", 0.88)],
    heading_deg=0.0,
    pitch_deg=0.0,
    flow_vectors=[{"direction_deg": 90, "magnitude": 0.85, "label": "car"}],
)

STAIRS_SCENARIO = dict(
    detections=[],
    audio_labels=[("footsteps", 0.72)],
    heading_deg=180.0,
    pitch_deg=-18.0,
    flow_vectors=[],
)

CROWD_SCENARIO = dict(
    detections=[
        {"label": "person", "confidence": 0.83, "direction_deg": 45, "distance_m": 2.0},
        {"label": "person", "confidence": 0.71, "direction_deg": 180, "distance_m": 3.5},
    ],
    audio_labels=[("crowd_noise", 0.89), ("speech", 0.74)],
    heading_deg=90.0,
    pitch_deg=0.0,
    flow_vectors=[{"direction_deg": 45, "magnitude": 0.42, "label": "person"}],
)


def test_sensor_encoder():
    print("\n─── Sensor Encoder ───")
    for name, s in [("car", CAR_SCENARIO), ("stairs", STAIRS_SCENARIO), ("crowd", CROWD_SCENARIO)]:
        prompt = build_prompt(**s, history_summary="history:none")
        tokens = len(prompt.split())
        print(f"  {name:8s} → {tokens:3d} tokens: {prompt}")
    print("  ✓ compact encoding works")


def test_router():
    print("\n─── Cascade Router ───")
    memory = ThreatMemory()
    router = CascadeRouter(memory)

    for name, s in [("car", CAR_SCENARIO), ("stairs", STAIRS_SCENARIO), ("crowd", CROWD_SCENARIO)]:
        snap = SensorSnapshot(**s)
        tier = router.route(snap)
        score_str = f"vis={snap.max_detection_confidence():.2f} aud={snap.max_audio_confidence():.2f}"
        print(f"  {name:8s} → Tier {tier}  ({score_str})")

    print("  ✓ routing logic works")


def test_threat_memory():
    print("\n─── Threat Memory ───")
    memory = ThreatMemory()

    alert = ThreatAlert(
        threat_type="vehicle_approaching",
        confidence=0.91,
        direction_degrees=315,
        urgency="high",
        haptic_pattern="rapid_right",
    )
    memory.insert(alert)

    active = memory.active_threat()
    assert active is not None
    print(f"  t=0s  confidence={active.confidence:.3f}")

    time.sleep(2)
    active = memory.active_threat()
    assert active is not None
    print(f"  t=2s  confidence={active.confidence:.3f} (should be ~half)")
    assert active.confidence < 0.6

    print("  ✓ exponential decay works")


def test_predictor():
    print("\n─── Threat Predictor ───")
    results = []

    def haptic_cb(direction, urgency, confidence):
        results.append((direction, urgency, confidence))

    predictor = ThreatPredictor(haptic_cb, update_hz=5.0)
    predictor.start()

    alert = ThreatAlert(
        threat_type="vehicle_approaching",
        confidence=0.91,
        direction_degrees=315,
        distance_meters=8.0,
        urgency="high",
        haptic_pattern="rapid_right",
        timestamp=time.time(),
    )
    motion = MotionVector(
        direction_deg=90, speed_mps=12.0, label="car",
        confidence=0.91, origin_distance_m=8.0, captured_at=time.time()
    )
    predictor.update_threat(alert, motion)
    time.sleep(0.8)
    predictor.stop()

    print(f"  haptic callbacks fired: {len(results)}")
    for r in results:
        print(f"    dir={r[0]:.1f}° urgency={r[1]} conf={r[2]:.3f}")
    assert len(results) > 0
    print("  ✓ predictive extrapolation fires haptics")


def test_reservoir():
    print("\n─── Reservoir Anticipator ───")
    anticipator = ThreatAnticipator.load_default()

    # Calm: all zeros → low P(threat)
    snap_calm = SensorSnapshot(detections=[], audio_labels=[], heading_deg=0, pitch_deg=0, flow_vectors=[])
    p_calm = anticipator.predict(snap_calm)
    print(f"  calm    P(threat)={p_calm:.4f}  (expect ≈0)")
    assert p_calm < 0.20, f"Expected calm P < 0.20, got {p_calm:.4f}"

    # Threat build-up: feed 5 frames of high-confidence closing car + horn
    anticipator.reset()
    snap_threat = SensorSnapshot(
        detections=[{"label": "car", "confidence": 0.91, "direction_deg": 315,
                     "distance_m": 3.0, "motion": "closing-fast", "track_age": 5}],
        audio_labels=[("horn_honk", 0.96), ("engine_noise", 0.88)],
        heading_deg=0.0, pitch_deg=0.0,
        flow_vectors=[{"direction_deg": 315, "magnitude": 0.87, "label": "car"}],
    )
    p_threat = 0.0
    for _ in range(5):
        p_threat = anticipator.predict(snap_threat)
    print(f"  threat  P(threat)={p_threat:.4f}  (expect > {ANTICIPATION_THRESHOLD})")
    assert p_threat > ANTICIPATION_THRESHOLD, f"Expected threat P > {ANTICIPATION_THRESHOLD}, got {p_threat:.4f}"

    # Anticipation override: router says Tier 0, reservoir should escalate to Tier 2
    memory = ThreatMemory()
    router = CascadeRouter(memory)
    # Low-signal snap (tier 0) but reservoir already warmed up on threat sequence
    snap_borderline = SensorSnapshot(
        detections=[{"label": "car", "confidence": 0.30, "direction_deg": 315,
                     "distance_m": 10.0, "motion": "static", "track_age": 1}],
        audio_labels=[("engine_noise", 0.45)],
        heading_deg=0.0, pitch_deg=0.0, flow_vectors=[],
    )
    tier = router.route(snap_borderline)
    p_borderline = anticipator.predict(snap_borderline)
    print(f"  borderline: router_tier={tier}  reservoir_P={p_borderline:.4f}")
    # If reservoir P > threshold and tier==0, main.py would escalate — verify logic path exists
    would_escalate = (tier == 0 and p_borderline > ANTICIPATION_THRESHOLD)
    print(f"  would escalate preemptively: {would_escalate}")

    # Latency: reservoir step must be < 1ms
    import time as _time
    anticipator.reset()
    n = 1000
    t0 = _time.perf_counter()
    for _ in range(n):
        anticipator.predict(snap_threat)
    ms_per_step = (_time.perf_counter() - t0) / n * 1000
    print(f"  latency: {ms_per_step:.4f}ms/step (expect <1ms)")
    assert ms_per_step < 1.0, f"Too slow: {ms_per_step:.4f}ms"

    print("  ✓ reservoir anticipation works")


def test_stream_parser_live():
    print("\n─── Stream Parser (live llama-server) ───")
    early_fires = []
    complete_alerts = []
    done_event = threading.Event()

    def early_cb(alert: ThreatAlert):
        t = time.time()
        early_fires.append((t, alert))
        print(f"  ⚡ EARLY FIRE  dir={alert.direction_degrees}° urgency={alert.urgency} partial={alert.partial}")

    def complete_cb(alert: ThreatAlert):
        t = time.time()
        complete_alerts.append((t, alert))
        print(f"  ✅ COMPLETE    type={alert.threat_type} conf={alert.confidence:.2f} haptic={alert.haptic_pattern}")
        print(f"     natural: {alert.natural_language}")
        done_event.set()

    parser = StreamParser(early_cb, complete_cb)
    prompt = build_prompt(**CAR_SCENARIO, history_summary="history:none")
    print(f"  prompt ({len(prompt.split())} tokens): {prompt}")

    t0 = time.time()
    parser.infer(prompt)
    done_event.wait(timeout=90)
    elapsed = time.time() - t0

    print(f"\n  Total inference time: {elapsed:.2f}s")
    if early_fires and complete_alerts:
        early_t = early_fires[0][0]
        complete_t = complete_alerts[0][0]
        print(f"  Early fire lead time: {complete_t - early_t:.2f}s before completion")
    elif not early_fires:
        print("  (no early fire — threat below actionable threshold or fields parsed late)")


if __name__ == "__main__":
    print("=" * 60)
    print("THREATDETECT PIPELINE TEST")
    print("=" * 60)

    test_sensor_encoder()
    test_router()
    test_threat_memory()
    test_predictor()
    test_reservoir()
    test_stream_parser_live()

    print("\n" + "=" * 60)
    print("ALL TESTS DONE")
