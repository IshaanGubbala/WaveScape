"""
Reservoir-Gated Anticipatory Inference.

Echo State Network runs at 10Hz on sensor features.
Predicts router threat score 3 frames ahead (300ms), triggering Gemma
proactively so it's already running when YOLO confirms the threat.

Improvements over v1:
  - 9 input features (was 6): adds closing_speed, min_eta, delta_vis
  - 200 reservoir neurons (was 100): better slow build-up memory
  - LOOKAHEAD=3 frames (was 2): 300ms more warmup for Gemma
  - Online readout update: W_out adapts from each real Gemma completion
    (recursive gradient descent on readout weights, lr=0.01)

Training: run train_reservoir.py offline on Mac → reservoir_weights.npz (~160KB)
Deploy: load .npz on Pi 5. <0.2ms per step.
"""
import numpy as np
import threading
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .router import SensorSnapshot

INPUT_DIM = 9
RESERVOIR_SIZE = 200
SPECTRAL_RADIUS = 0.95   # slightly higher → longer memory for slow build-ups
INPUT_SCALING = 0.5
SPARSITY = 0.9

LOOKAHEAD = 3            # predict 3 frames (300ms) ahead
ONLINE_LR = 0.01         # readout learning rate for online updates
STATE_BUFFER_LEN = 60    # 6 seconds of state history for online updates

WEIGHTS_PATH = Path(__file__).parent / "reservoir_weights.npz"
ANTICIPATION_THRESHOLD = 0.42

# Idea 5: action head class indices
URGENCY_CLASSES = ("low", "medium", "high", "critical")
PATTERN_CLASSES = ("none", "slow_pulse", "rapid_left", "rapid_right", "rapid_center")

# Tier 1.5 fast-haptic gate: reservoir must clear both p_threat and class confidence
RBI_THREAT_THRESHOLD = 0.50
RBI_CLASS_CONFIDENCE = 0.55   # 1 - normalized entropy of softmax


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def _norm_entropy(p: np.ndarray) -> float:
    """Entropy normalized to [0,1] by dividing by log(K) — uniform = 1, peaked = 0."""
    eps = 1e-9
    K = len(p)
    h = -float(np.sum(p * np.log(p + eps)))
    return h / float(np.log(K)) if K > 1 else 0.0


def extract_features(snap: "SensorSnapshot", prev_vis: float = 0.0) -> np.ndarray:
    """
    9-feature vector from SensorSnapshot.
    prev_vis: vis_conf from last frame (for delta_vis computation).
    """
    vis = snap.max_detection_confidence()
    aud = snap.max_audio_confidence()
    mot = snap.max_motion_magnitude()
    n_det = min(len(snap.detections) / 5.0, 1.0)
    pitch_unsafe = float(snap.is_stairs_likely())

    # Closing flag — any object labeled closing or closing-fast
    is_closing = float(
        any(d.get("motion") in ("closing", "closing-fast") for d in snap.detections)
    )

    # Max closing speed (mps) among tracked closing objects, normalised to [0,1]
    closing_speeds = [
        d.get("velocity_mps", 0.0) or 0.0
        for d in snap.detections
        if d.get("motion") in ("closing", "closing-fast")
    ]
    closing_speed_norm = min(max(closing_speeds, default=0.0) / 10.0, 1.0)

    # Min ETA across all tracked objects — closer to 1 = more urgent
    etas = [d.get("eta_s") for d in snap.detections if d.get("eta_s") is not None]
    min_eta_norm = 1.0 - min(etas, default=10.0) / 10.0
    min_eta_norm = float(np.clip(min_eta_norm, 0.0, 1.0))

    # Rate of change in visual confidence (positive = threat building)
    delta_vis = float(np.clip(vis - prev_vis, -1.0, 1.0))

    return np.array(
        [vis, aud, mot, n_det, is_closing, pitch_unsafe,
         closing_speed_norm, min_eta_norm, delta_vis],
        dtype=np.float32,
    )


class EchoStateNetwork:
    def __init__(self, seed: int = 42):
        rng = np.random.RandomState(seed)
        N = RESERVOIR_SIZE

        self.W_in = rng.uniform(-INPUT_SCALING, INPUT_SCALING, (N, INPUT_DIM)).astype(np.float32)

        W = rng.uniform(-1, 1, (N, N))
        mask = rng.rand(N, N) > SPARSITY
        W *= mask
        sr = np.max(np.abs(np.linalg.eigvals(W)))
        if sr > 0:
            W *= SPECTRAL_RADIUS / sr
        self.W_res = W.astype(np.float32)

        self.W_out: np.ndarray | None = None
        # Idea 5 multi-head readouts: urgency (4-class) + pattern (5-class)
        self.W_out_u: np.ndarray | None = None     # shape (N, 4)
        self.W_out_h: np.ndarray | None = None     # shape (N, 5)
        self._state = np.zeros(N, dtype=np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
        pre = self.W_in @ x + self.W_res @ self._state
        self._state = np.tanh(pre)
        return self._state.copy()

    def predict(self, x: np.ndarray) -> float:
        state = self.step(x)
        if self.W_out is None:
            return 0.0
        return float(np.clip(self.W_out @ state, 0.0, 1.0))

    def reset(self) -> None:
        self._state[:] = 0.0

    def save(self, path: Path | str) -> None:
        kw = {"W_in": self.W_in, "W_res": self.W_res, "W_out": self.W_out}
        if self.W_out_u is not None:
            kw["W_out_u"] = self.W_out_u
        if self.W_out_h is not None:
            kw["W_out_h"] = self.W_out_h
        np.savez(path, **kw)

    def load(self, path: Path | str) -> None:
        data = np.load(path)
        self.W_in = data["W_in"].astype(np.float32)
        self.W_res = data["W_res"].astype(np.float32)
        self.W_out = data["W_out"].astype(np.float32)
        self.W_out_u = data["W_out_u"].astype(np.float32) if "W_out_u" in data.files else None
        self.W_out_h = data["W_out_h"].astype(np.float32) if "W_out_h" in data.files else None
        self._state = np.zeros(RESERVOIR_SIZE, dtype=np.float32)


class ThreatAnticipator:
    """
    Live wrapper around ESN.

    Call predict(snap) each 10Hz frame → returns P(threat in 3 frames).
    Call online_update(confidence, gemma_ts) after each Gemma completion
    → updates W_out from ground truth. System calibrates to real environment.
    """

    def __init__(self, esn: EchoStateNetwork):
        self._esn = esn
        self._prev_vis: float = 0.0
        # Rolling buffer: (timestamp, state_vector) pairs
        self._state_buffer: deque = deque(maxlen=STATE_BUFFER_LEN)
        self._lock = threading.Lock()

    def predict(self, snap: "SensorSnapshot") -> float:
        import time
        x = extract_features(snap, self._prev_vis)
        self._prev_vis = snap.max_detection_confidence()

        with self._lock:
            state = self._esn.step(x)
            p = float(np.clip(self._esn.W_out @ state, 0.0, 1.0)) if self._esn.W_out is not None else 0.0
            self._state_buffer.append((time.monotonic(), state.copy()))

        return p

    def predict_action(self, snap: "SensorSnapshot") -> dict:
        """
        Idea 5 — Reservoir as fast action policy.
        Returns {p_threat, urgency, pattern, confidence, ready}.
        ready=True iff multi-head readouts are loaded AND class confidence above gate.
        Uses CURRENT reservoir state (no extra step) — call AFTER predict().
        """
        if self._esn.W_out_u is None or self._esn.W_out_h is None:
            return {"ready": False, "p_threat": 0.0, "urgency": "low",
                    "pattern": "none", "confidence": 0.0}

        with self._lock:
            state = self._esn._state.copy()

        p = float(np.clip(self._esn.W_out @ state, 0.0, 1.0)) if self._esn.W_out is not None else 0.0
        u_logits = self._esn.W_out_u.T @ state    # (4,)
        h_logits = self._esn.W_out_h.T @ state    # (5,)

        u_probs = _softmax(u_logits)
        h_probs = _softmax(h_logits)
        u_idx = int(np.argmax(u_probs))
        h_idx = int(np.argmax(h_probs))

        # Class confidence = 1 - normalized entropy. Higher = more peaked.
        u_conf = 1.0 - _norm_entropy(u_probs)
        h_conf = 1.0 - _norm_entropy(h_probs)
        conf = float(min(u_conf, h_conf))

        return {
            "ready": True,
            "p_threat": p,
            "urgency": URGENCY_CLASSES[u_idx],
            "pattern": PATTERN_CLASSES[h_idx],
            "confidence": conf,
        }

    def online_update(self, ground_truth_confidence: float, gemma_ts: float) -> None:
        """
        Called after Gemma completes. ground_truth_confidence is the LLM's
        threat confidence (0–1). gemma_ts is time.monotonic() when Gemma fired.

        Finds the reservoir state from LOOKAHEAD frames before the Gemma call
        and does a gradient step: W_out += lr * error * state.
        """
        if self._esn.W_out is None:
            return

        target_ts = gemma_ts - LOOKAHEAD * 0.1  # 0.1s per frame at 10Hz

        with self._lock:
            # Find buffered state closest to target_ts
            if not self._state_buffer:
                return
            best = min(self._state_buffer, key=lambda s: abs(s[0] - target_ts))
            _, state = best

            pred = float(self._esn.W_out @ state)
            error = ground_truth_confidence - pred
            self._esn.W_out = self._esn.W_out + ONLINE_LR * error * state

    def reset(self) -> None:
        with self._lock:
            self._esn.reset()
            self._prev_vis = 0.0
            self._state_buffer.clear()

    @classmethod
    def load_default(cls) -> "ThreatAnticipator":
        esn = EchoStateNetwork()
        if WEIGHTS_PATH.exists():
            esn.load(WEIGHTS_PATH)
        else:
            print(f"[reservoir] WARNING: no weights at {WEIGHTS_PATH} — run fusion/train_reservoir.py")
        return cls(esn)
