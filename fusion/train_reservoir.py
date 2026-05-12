"""
Offline training for the improved reservoir threat anticipator (v2).
Run once on Mac, saves reservoir_weights.npz alongside this file.

Changes from v1:
  - INPUT_DIM=9 (was 6): closing_speed, min_eta, delta_vis added
  - RESERVOIR_SIZE=200 (was 100)
  - LOOKAHEAD=3 (was 2)
  - Richer synthetic sequences: slow_build, multi_threat, audio_then_visual

Usage:
    cd /Users/ishaangubbala/threatdetect
    PYTHONPATH=/Users/ishaangubbala/threatdetect python -m fusion.train_reservoir
"""
import numpy as np
from pathlib import Path
from .reservoir import (
    EchoStateNetwork, WEIGHTS_PATH,
    INPUT_DIM, RESERVOIR_SIZE, LOOKAHEAD,
)

WARMUP = 25
N_SEQUENCES = 800
SEQ_LEN = 50
RIDGE_ALPHA = 1e-4


def _router_score(x: np.ndarray) -> np.ndarray:
    """Matches CascadeRouter._score. Uses first 6 features."""
    return np.clip(0.45 * x[:, 0] + 0.30 * x[:, 1] + 0.15 * x[:, 2] + 0.10 * x[:, 5], 0, 1)


def generate_sequences(seed: int = 42):
    rng = np.random.RandomState(seed)
    X_seqs, y_seqs = [], []

    kinds = [
        "calm", "sudden_car", "gradual_car", "oscillating",
        "audio_spike", "slow_build", "multi_threat", "audio_then_visual",
        "closing_fast", "near_miss",
    ]

    for i in range(N_SEQUENCES):
        kind = kinds[i % len(kinds)]
        x = rng.normal(0.04, 0.02, (SEQ_LEN, INPUT_DIM)).clip(0, 1)
        # cols: [vis, aud, mot, n_det, is_closing, pitch, closing_spd, min_eta_n, delta_vis]

        if kind == "calm":
            pass

        elif kind == "sudden_car":
            start = rng.randint(30, 42)
            x[start:, 0] = rng.normal(0.88, 0.06, SEQ_LEN - start).clip(0, 1)
            x[start:, 4] = 1.0
            x[start:, 6] = rng.normal(0.7, 0.1, SEQ_LEN - start).clip(0, 1)
            x[start:, 7] = rng.normal(0.85, 0.08, SEQ_LEN - start).clip(0, 1)

        elif kind == "gradual_car":
            start = rng.randint(5, 20)
            for t in range(start, SEQ_LEN):
                ramp = (t - start) / max(SEQ_LEN - start, 1)
                x[t, 0] = float(np.clip(ramp * 0.9 + rng.normal(0, 0.03), 0, 1))
                x[t, 4] = float(ramp > 0.4)
                x[t, 6] = float(np.clip(ramp * 0.8, 0, 1))
                x[t, 7] = float(np.clip(ramp * 0.9, 0, 1))
            # delta_vis reflects the ramp
            x[1:, 8] = np.diff(x[:, 0]).clip(-1, 1)

        elif kind == "oscillating":
            t = np.linspace(0, 4 * np.pi, SEQ_LEN)
            x[:, 0] = np.clip(0.3 + 0.25 * np.sin(t) + rng.normal(0, 0.02, SEQ_LEN), 0, 1)
            x[:, 1] = np.clip(0.2 + 0.20 * np.cos(t) + rng.normal(0, 0.02, SEQ_LEN), 0, 1)

        elif kind == "audio_spike":
            start = rng.randint(10, 25)
            x[start:, 1] = rng.normal(0.91, 0.05, SEQ_LEN - start).clip(0, 1)
            x[start:, 0] = rng.normal(0.40, 0.10, SEQ_LEN - start).clip(0, 1)

        elif kind == "slow_build":
            start = rng.randint(0, 10)
            for t in range(start, SEQ_LEN):
                ramp = ((t - start) / max(SEQ_LEN - start, 1)) ** 0.7
                x[t, 0] = float(np.clip(ramp * 0.85 + rng.normal(0, 0.04), 0, 1))
                x[t, 3] = float(np.clip(ramp * 0.6, 0, 1))
                x[t, 6] = float(np.clip(ramp * 0.6, 0, 1))
                x[t, 7] = float(np.clip(ramp * 0.7, 0, 1))
            x[1:, 8] = np.diff(x[:, 0]).clip(-1, 1)

        elif kind == "multi_threat":
            # Two objects, one arrives early, one escalates later
            s1 = rng.randint(5, 15)
            x[s1:, 0] = rng.normal(0.55, 0.08, SEQ_LEN - s1).clip(0, 1)
            x[s1:, 3] = 0.4
            s2 = rng.randint(25, 35)
            x[s2:, 0] = rng.normal(0.88, 0.05, SEQ_LEN - s2).clip(0, 1)
            x[s2:, 4] = 1.0
            x[s2:, 6] = rng.normal(0.75, 0.1, SEQ_LEN - s2).clip(0, 1)
            x[s2:, 7] = rng.normal(0.9, 0.05, SEQ_LEN - s2).clip(0, 1)

        elif kind == "audio_then_visual":
            # Horn heard, then car comes into frame
            sa = rng.randint(5, 15)
            x[sa:, 1] = rng.normal(0.88, 0.06, SEQ_LEN - sa).clip(0, 1)
            sv = sa + rng.randint(8, 18)
            if sv < SEQ_LEN:
                x[sv:, 0] = rng.normal(0.82, 0.07, SEQ_LEN - sv).clip(0, 1)
                x[sv:, 4] = 1.0
                x[sv:, 6] = rng.normal(0.65, 0.1, SEQ_LEN - sv).clip(0, 1)
                x[sv:, 7] = rng.normal(0.8, 0.08, SEQ_LEN - sv).clip(0, 1)

        elif kind == "closing_fast":
            start = rng.randint(20, 32)
            x[start:, 0] = rng.normal(0.85, 0.07, SEQ_LEN - start).clip(0, 1)
            x[start:, 4] = 1.0
            x[start:, 6] = rng.normal(0.90, 0.06, SEQ_LEN - start).clip(0, 1)
            x[start:, 7] = rng.normal(0.95, 0.04, SEQ_LEN - start).clip(0, 1)
            x[start:, 2] = rng.normal(0.80, 0.08, SEQ_LEN - start).clip(0, 1)

        elif kind == "near_miss":
            peak = rng.randint(15, 30)
            for t in range(SEQ_LEN):
                dist = abs(t - peak)
                r = max(0, 1 - dist / 12)
                x[t, 0] = float(np.clip(r * 0.92 + rng.normal(0, 0.04), 0, 1))
                x[t, 4] = float(r > 0.5)
                x[t, 6] = float(np.clip(r * 0.85, 0, 1))
                x[t, 7] = float(np.clip(r * 0.9, 0, 1))
            x[1:, 8] = np.diff(x[:, 0]).clip(-1, 1)

        x = x.clip(0, 1).astype(np.float32)

        score = _router_score(x)
        y = np.zeros(SEQ_LEN, dtype=np.float32)
        y[:-LOOKAHEAD] = score[LOOKAHEAD:]
        y[-LOOKAHEAD:] = score[-1]

        X_seqs.append(x)
        y_seqs.append(y)

    return X_seqs, y_seqs


def collect_states(esn, X_seqs, y_seqs):
    all_states, all_targets = [], []
    for x, y in zip(X_seqs, y_seqs):
        esn.reset()
        states = []
        for t in range(len(x)):
            s = esn.step(x[t])
            if t >= WARMUP:
                states.append(s)
        all_states.append(np.stack(states))
        all_targets.append(y[WARMUP:])
    return np.vstack(all_states), np.concatenate(all_targets)


def ridge_solve(states, targets):
    N = states.shape[1]
    A = states.T @ states + RIDGE_ALPHA * np.eye(N)
    b = states.T @ targets
    return np.linalg.solve(A, b).astype(np.float32)


def evaluate(esn, X_seqs, y_seqs):
    preds, trues = [], []
    for x, y in zip(X_seqs[:80], y_seqs[:80]):
        esn.reset()
        prev_vis = 0.0
        for t in range(len(x)):
            p = float(np.clip(esn.W_out @ esn.step(x[t]), 0, 1)) if esn.W_out is not None else 0.0
            if t >= WARMUP:
                preds.append(p)
                trues.append(y[t])
    preds, trues = np.array(preds), np.array(trues)
    THRESH = 0.38
    tp = int(np.sum((preds > THRESH) & (trues > THRESH)))
    fn = int(np.sum((preds <= THRESH) & (trues > THRESH)))
    fp = int(np.sum((preds > THRESH) & (trues <= THRESH)))
    return {
        "mse": float(np.mean((preds - trues) ** 2)),
        "accuracy": float(np.mean((preds > THRESH) == (trues > THRESH))),
        "threat_recall": tp / max(tp + fn, 1),
        "false_alarm_rate": fp / max(len(preds), 1),
        "n_eval": len(preds),
    }


def train():
    print(f"Generating {N_SEQUENCES} sequences (10 scenario types)...")
    X_seqs, y_seqs = generate_sequences(seed=42)
    print(f"  {len(X_seqs)} seqs × {SEQ_LEN} steps = {len(X_seqs)*SEQ_LEN} frames")
    print(f"  INPUT_DIM={INPUT_DIM}  RESERVOIR_SIZE={RESERVOIR_SIZE}  LOOKAHEAD={LOOKAHEAD} frames")

    esn = EchoStateNetwork(seed=42)

    print("Collecting reservoir states...")
    states, targets = collect_states(esn, X_seqs, y_seqs)
    print(f"  State matrix: {states.shape}")

    print("Fitting readout (Ridge regression)...")
    esn.W_out = ridge_solve(states, targets)

    m = evaluate(esn, X_seqs, y_seqs)
    print(f"\nResults:")
    print(f"  MSE:              {m['mse']:.5f}")
    print(f"  Accuracy@0.38:    {m['accuracy']*100:.1f}%")
    print(f"  Threat recall:    {m['threat_recall']*100:.1f}%")
    print(f"  False alarm rate: {m['false_alarm_rate']*100:.1f}%")
    print(f"  Eval frames:      {m['n_eval']}")

    esn.save(WEIGHTS_PATH)
    size_kb = WEIGHTS_PATH.stat().st_size / 1024
    print(f"\nSaved: {WEIGHTS_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    train()
