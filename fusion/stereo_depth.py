"""
Stereo depth estimation from dual IMX708 cameras via OpenCV SGBM.

No calibration file required: assumes cameras are approximately parallel
(valid for rigid glasses-mount with small angular deviation).

Geometry:
  cam0 = left  (330° offset = 30° left of forward)
  cam1 = right (30°  offset = 30° right of forward)
  Baseline: ~65mm interpupillary distance on glasses frame.

Output: per-sector min depth + safest escape direction for haptic.
"""

import math
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Physical constants
BASELINE_M = 0.111          # 111mm baseline between lens centers
CAM_HALF_FOV_DEG = 30.5     # 122° total coverage / 2 cameras = 61° per cam, half = 30.5°

# Processing resolution — same as YOLO input, no extra resize needed
STEREO_W = 96
STEREO_H = 96

# Focal length in pixels at STEREO_W resolution
FOCAL_PX = (STEREO_W / 2) / math.tan(math.radians(CAM_HALF_FOV_DEG))

# SGBM parameters (tuned for 96×96 input, 65mm baseline, 1-5m range)
_NUM_DISP = 32   # divisible by 16; covers disparities 1-32px → depth 0.15-4.8m
_BLOCK = 5       # must be odd
_P1 = 8  * 3 * _BLOCK ** 2
_P2 = 32 * 3 * _BLOCK ** 2

# Navigation
N_SECTORS = 6           # divide FOV into N angular slices
NEAR_M = 1.5            # obstacle within this range → sector flagged as blocked
FAR_M = 3.0             # beyond this → sector considered safe


def _px_col_to_deg(col: float) -> float:
    """Convert pixel column (0..STEREO_W) to angle relative to straight-ahead."""
    return (col / STEREO_W - 0.5) * 2 * CAM_HALF_FOV_DEG


_CALIB_PATHS = [
    "/home/ishaan/threatdetect/stereo_calib.npz",
    "stereo_calib.npz",
]


def _load_calib() -> dict | None:
    import os
    for p in _CALIB_PATHS:
        if os.path.exists(p):
            d = dict(np.load(p))
            print(f"  [stereo] loaded calibration: {p}  baseline={d['baseline_mm']:.1f}mm  rms={float(d['rms']):.3f}px")
            return d
    return None


class StereoDepthEstimator:
    """
    One-shot stereo depth from a rectified (or approximately parallel) camera pair.
    Loads stereo_calib.npz if available (produced by tools/calibrate_stereo.py).
    Falls back to uncalibrated (assumes parallel cameras) if no file found.
    """

    def __init__(self, baseline_m: float = BASELINE_M):
        self._sgbm: object | None = None
        self._calib = _load_calib()
        if self._calib is not None:
            self.baseline_m = float(self._calib["baseline_mm"]) / 1000.0
            # Focal length from rectified projection matrix P0 (element [0,0])
            self._focal_px = float(self._calib["P0"][0, 0]) * (STEREO_W / 224.0)
        else:
            self.baseline_m = baseline_m
            self._focal_px = FOCAL_PX

    def _build(self):
        self._sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=_NUM_DISP,
            blockSize=_BLOCK,
            P1=_P1,
            P2=_P2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute(self, frame0: np.ndarray, frame1: np.ndarray) -> dict:
        """
        Args:
            frame0: left  camera BGR frame (cam0, any resolution)
            frame1: right camera BGR frame (cam1, any resolution)

        Returns:
            depth_map:      (STEREO_H, STEREO_W) float32 depth in meters; 0 = invalid
            sector_depths:  list of N dicts — angle_deg, min_depth_m, safe
            escape_dir_deg: angle (0=ahead, 90=right, 270=left) to safest open sector
            min_depth_m:    nearest valid obstacle distance
            latency_ms:     wall-clock SGBM time
        """
        if not CV2_AVAILABLE:
            return _null_result()
        if self._sgbm is None:
            self._build()

        import time
        t0 = time.monotonic()

        # Resize to stereo resolution
        f0 = cv2.resize(frame0, (STEREO_W, STEREO_H))
        f1 = cv2.resize(frame1, (STEREO_W, STEREO_H))

        # Apply rectification maps if calibration is loaded
        if self._calib is not None:
            sx = STEREO_W / 224.0
            sy = STEREO_H / 224.0
            # Scale maps from calibration resolution (224×224) to STEREO_W×STEREO_H
            mx0 = self._calib["map0x"] * sx
            my0 = self._calib["map0y"] * sy
            mx1 = self._calib["map1x"] * sx
            my1 = self._calib["map1y"] * sy
            f0 = cv2.remap(f0, mx0.astype(np.float32), my0.astype(np.float32), cv2.INTER_LINEAR)
            f1 = cv2.remap(f1, mx1.astype(np.float32), my1.astype(np.float32), cv2.INTER_LINEAR)

        gl = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

        # SGBM: 16-bit fixed-point output, divide by 16 for float disparity
        disp16 = self._sgbm.compute(gl, gr)
        disp = disp16.astype(np.float32) / 16.0

        # Disparity → metric depth:  Z = f * B / d
        depth = np.zeros_like(disp)
        valid = disp > 0.5
        depth[valid] = (self._focal_px * self.baseline_m) / disp[valid]
        depth[(depth > 10.0) | (depth < 0.1)] = 0.0  # clamp to plausible range

        latency_ms = (time.monotonic() - t0) * 1000

        # --- Sector analysis ---
        sector_px = STEREO_W // N_SECTORS
        sector_depths = []
        for i in range(N_SECTORS):
            x0 = i * sector_px
            x1 = x0 + sector_px
            col = depth[:, x0:x1]
            valid_d = col[col > 0.1]
            if valid_d.size >= 5:
                min_d = float(np.percentile(valid_d, 10))  # robust near-obstacle estimate
            else:
                min_d = FAR_M + 1.0  # treat empty sector as open
            center_col = (x0 + x1) / 2.0
            angle = _px_col_to_deg(center_col)
            sector_depths.append({
                "angle_deg": round(angle, 1),
                "min_depth_m": round(min_d, 2),
                "safe": min_d > NEAR_M,
            })

        # --- Escape direction ---
        # Prefer sectors closest to straight-ahead among safe ones;
        # if all blocked, pick the one with most clearance.
        safe = [s for s in sector_depths if s["safe"]]
        if safe:
            # Among safe sectors, pick the one closest to 0°
            best = min(safe, key=lambda s: abs(s["angle_deg"]))
        else:
            best = max(sector_depths, key=lambda s: s["min_depth_m"])

        # Convert angle-relative-to-ahead → 0-360 haptic convention
        escape_haptic = best["angle_deg"] % 360  # 0=ahead, positive=right, negative wraps to ~359

        valid_all = depth[depth > 0.1]
        global_min = float(valid_all.min()) if valid_all.size > 0 else 99.0

        return {
            "depth_map": depth,
            "sector_depths": sector_depths,
            "escape_dir_deg": round(escape_haptic, 1),
            "min_depth_m": round(global_min, 2),
            "latency_ms": round(latency_ms, 1),
        }


def _null_result() -> dict:
    return {
        "depth_map": np.zeros((STEREO_H, STEREO_W), dtype=np.float32),
        "sector_depths": [],
        "escape_dir_deg": 0.0,
        "min_depth_m": 99.0,
        "latency_ms": 0.0,
    }
