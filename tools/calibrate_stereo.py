"""
Stereo calibration for dual IMX708 cameras on Pi 5.

Usage (run on Pi):
    python3 tools/calibrate_stereo.py [--squares 25] [--cols 9] [--rows 6] [--captures 20]

Hold a flat checkerboard in front of both cameras. Script auto-captures every 2 seconds
(listen for countdown beeps in terminal). Move board to different angles/distances between
captures. Need at least 10 good pairs (both cameras see all corners).

Output: stereo_calib.npz in ~/threatdetect/ — loaded automatically by StereoDepthEstimator.
"""

import time
import sys
import argparse
import numpy as np
import cv2

try:
    from picamera2 import Picamera2
    PICAM = True
except ImportError:
    PICAM = False


CAPTURE_INTERVAL_S = 2.5
IMG_W, IMG_H = 224, 224


def open_cameras():
    cam0 = Picamera2(0)
    cam1 = Picamera2(1)
    cfg0 = cam0.create_preview_configuration(main={"size": (IMG_W, IMG_H), "format": "RGB888"})
    cfg1 = cam1.create_preview_configuration(main={"size": (IMG_W, IMG_H), "format": "RGB888"})
    cam0.configure(cfg0)
    cam1.configure(cfg1)
    cam0.start()
    cam1.start()
    time.sleep(1.0)
    return cam0, cam1


def capture_pair(cam0, cam1):
    rgb0 = cam0.capture_array()
    rgb1 = cam1.capture_array()
    bgr0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR)
    bgr1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR)
    bgr0 = cv2.flip(bgr0, -1)  # cameras mounted upside-down
    bgr1 = cv2.flip(bgr1, -1)
    return bgr0, bgr1


def find_corners(img, pattern_size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if ok:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)
    return ok, corners


def run(n_captures: int, square_mm: float, cols: int, rows: int, out_path: str):
    pattern = (cols, rows)
    obj_pts_one = np.zeros((cols * rows, 3), np.float32)
    obj_pts_one[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj_pts_one *= square_mm  # scale to real-world mm

    obj_pts = []
    img_pts0 = []
    img_pts1 = []
    good = 0

    if not PICAM:
        print("ERROR: picamera2 not available. Run on Pi.")
        sys.exit(1)

    print(f"Opening cameras...")
    cam0, cam1 = open_cameras()
    print(f"Cameras ready.")
    print(f"Hold checkerboard ({cols}×{rows} inner corners, {square_mm}mm squares) in view of both cameras.")
    print(f"Will auto-capture every {CAPTURE_INTERVAL_S}s × {n_captures} captures. Move board between captures!")
    print()

    for i in range(n_captures):
        countdown = CAPTURE_INTERVAL_S
        print(f"  Pose {i+1}/{n_captures} — capturing in {countdown:.0f}s... ", end="", flush=True)
        time.sleep(CAPTURE_INTERVAL_S)

        bgr0, bgr1 = capture_pair(cam0, cam1)
        ok0, c0 = find_corners(bgr0, pattern)
        ok1, c1 = find_corners(bgr1, pattern)

        if ok0 and ok1:
            good += 1
            obj_pts.append(obj_pts_one)
            img_pts0.append(c0)
            img_pts1.append(c1)
            # Save debug images
            debug0 = bgr0.copy()
            debug1 = bgr1.copy()
            cv2.drawChessboardCorners(debug0, pattern, c0, True)
            cv2.drawChessboardCorners(debug1, pattern, c1, True)
            cv2.imwrite(f"/tmp/calib_cam0_{i:02d}.jpg", debug0)
            cv2.imwrite(f"/tmp/calib_cam1_{i:02d}.jpg", debug1)
            print(f"GOOD ({good} total)")
        else:
            miss = ("cam0" if not ok0 else "") + (" cam1" if not ok1 else "")
            print(f"MISSED — board not found in {miss.strip()}")

    cam0.stop()
    cam1.stop()

    if good < 8:
        print(f"\nOnly {good} good pairs — need at least 8. Re-run with board clearly visible.")
        sys.exit(1)

    print(f"\nCalibrating with {good} pairs...")

    # Individual calibrations (needed as initial estimate)
    h, w = IMG_H, IMG_W
    ret0, K0, D0, _, _ = cv2.calibrateCamera(obj_pts, img_pts0, (w, h), None, None)
    ret1, K1, D1, _, _ = cv2.calibrateCamera(obj_pts, img_pts1, (w, h), None, None)
    print(f"  Individual RMS: cam0={ret0:.3f}px  cam1={ret1:.3f}px")

    flags = cv2.CALIB_FIX_INTRINSIC
    ret, K0, D0, K1, D1, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_pts0, img_pts1,
        K0, D0, K1, D1,
        (w, h),
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    )
    print(f"  Stereo RMS: {ret:.3f}px  (target <1.0)")

    baseline_mm = float(np.linalg.norm(T))
    print(f"  Recovered baseline: {baseline_mm:.1f}mm")

    # Rectification maps
    R0, R1, P0, P1, Q, _, _ = cv2.stereoRectify(
        K0, D0, K1, D1, (w, h), R, T,
        alpha=0,  # no black borders
    )
    map0x, map0y = cv2.initUndistortRectifyMap(K0, D0, R0, P0, (w, h), cv2.CV_32FC1)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)

    np.savez(
        out_path,
        K0=K0, D0=D0, K1=K1, D1=D1,
        R=R, T=T, R0=R0, R1=R1, P0=P0, P1=P1, Q=Q,
        map0x=map0x, map0y=map0y, map1x=map1x, map1y=map1y,
        rms=ret, baseline_mm=baseline_mm,
    )
    print(f"\nSaved: {out_path}")
    print(f"Debug images: /tmp/calib_cam*.jpg")
    print(f"\nBaseline: {baseline_mm:.1f}mm  RMS: {ret:.3f}px")
    if ret < 1.0:
        print("Calibration quality: GOOD")
    elif ret < 2.0:
        print("Calibration quality: ACCEPTABLE (more poses would help)")
    else:
        print("Calibration quality: POOR — re-run with better board coverage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--squares", type=float, default=25.0, help="Square size in mm")
    parser.add_argument("--cols", type=int, default=9, help="Inner corners columns")
    parser.add_argument("--rows", type=int, default=6, help="Inner corners rows")
    parser.add_argument("--captures", type=int, default=20, help="Number of auto-capture attempts")
    parser.add_argument("--out", default="/home/ishaan/threatdetect/stereo_calib.npz")
    args = parser.parse_args()
    run(args.captures, args.squares, args.cols, args.rows, args.out)
