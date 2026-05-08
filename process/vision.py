"""
Vision processing: YOLO-nano detection + optical flow motion vectors.
Outputs detections in the format expected by sensor_encoder.
"""
import time
import math
import threading
import queue
import numpy as np

try:
    import cv2
    from ultralytics import YOLO
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

# Classes YOLO we care about for threat detection
THREAT_CLASSES = {
    "car", "truck", "bus", "motorcycle", "bicycle",
    "person", "dog", "traffic light", "stop sign",
    "stairs",  # not in COCO but added for context
}

# Pi Camera Module 3 (measured): 66° horizontal FOV → ±33° half-angle
CAM_HALF_FOV_DEG = 33.0

# Computed from geometry: f = (IMG_W/2) / tan(half_fov)
FOCAL_LENGTH_PX = round((224 / 2) / math.tan(math.radians(CAM_HALF_FOV_DEG)))
KNOWN_WIDTHS_M = {
    "car": 1.8, "truck": 2.4, "bus": 2.5, "motorcycle": 0.8,
    "bicycle": 0.6, "person": 0.5, "dog": 0.4,
}
IMG_W = 224
IMG_H = 224
IMG_CENTER_X = IMG_W / 2


def estimate_distance(label: str, bbox_w_px: float) -> float:
    """Estimate distance using known object width + focal length."""
    known_w = KNOWN_WIDTHS_M.get(label, 0.5)
    if bbox_w_px < 1:
        return 99.0
    return round((known_w * FOCAL_LENGTH_PX) / bbox_w_px, 1)


def bbox_to_direction(cx_px: float, half_fov: float = CAM_HALF_FOV_DEG) -> float:
    """Map horizontal bbox center to direction degrees (0=ahead, 90=right, 270=left)."""
    offset = (cx_px - IMG_CENTER_X) / IMG_CENTER_X  # -1 to 1
    return (offset * half_fov) % 360


YOLO_SKIP_N = 3  # run YOLO every Nth frame; flow propagates bboxes in between

class VisionProcessor:
    """
    YOLO + optical flow vision processor.
    async_yolo=True (default): YOLO runs in background thread, sampled every YOLO_SKIP_N frames.
    Main loop never waits — optical flow propagates last detections each skipped frame.
    """

    def __init__(
        self,
        model_size: str = "yolo26n.pt",
        imgsz: int = 160,
        camera_index: int = 0,
        async_yolo: bool = True,
    ):
        self.model_size = model_size
        self.imgsz = imgsz
        self.camera_index = camera_index
        self.async_yolo = async_yolo
        # ONNX at 96px is less accurate → lower conf threshold to catch more objects
        self._conf = 0.10 if str(model_size).endswith(".onnx") else 0.15
        self._model = None
        self._cap = None
        self._picam = None
        self._prev_gray = None
        self._last_flow = None
        self._frame_count = 0

        # Async YOLO state
        self._yolo_q: queue.Queue = queue.Queue(maxsize=1)
        self._yolo_result: list = []
        self._yolo_lock = threading.Lock()
        self._yolo_busy = False
        self._yolo_thread: threading.Thread | None = None

    def _load_model(self):
        if not CV2_AVAILABLE:
            raise RuntimeError("opencv + ultralytics not installed")
        if self._model is None:
            # NCNN models are a directory ending in _ncnn_model
            self._model = YOLO(self.model_size, task="detect")
            if self.async_yolo:
                self._yolo_thread = threading.Thread(target=self._yolo_worker, daemon=True)
                self._yolo_thread.start()

    def _yolo_worker(self):
        while True:
            frame = self._yolo_q.get()
            if frame is None:
                break
            dets = self._run_yolo_sync(frame)
            with self._yolo_lock:
                self._yolo_result = dets
                self._yolo_busy = False

    def _run_yolo_sync(self, frame: np.ndarray) -> list[dict]:
        results = self._model(frame, imgsz=self.imgsz, verbose=False, conf=self._conf)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self._model.names[cls_id]
                if label not in THREAT_CLASSES:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                w = x2 - x1
                detections.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "direction_deg": round(bbox_to_direction(cx), 1),
                    "distance_m": estimate_distance(label, w),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                })
        return detections

    def _propagate(self, dets: list[dict], flow: np.ndarray) -> list[dict]:
        """Shift bboxes by mean optical flow displacement. Slight confidence decay."""
        if not dets or flow is None:
            return dets
        fh, fw = flow.shape[:2]
        out = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            roi = flow[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
            if roi.size == 0:
                out.append(d)
                continue
            dx = float(np.mean(roi[:, :, 0]))
            dy = float(np.mean(roi[:, :, 1]))
            nx1, ny1, nx2, ny2 = x1 + dx, y1 + dy, x2 + dx, y2 + dy
            out.append({**d,
                "bbox": [round(nx1), round(ny1), round(nx2), round(ny2)],
                "direction_deg": round(bbox_to_direction((nx1 + nx2) / 2), 1),
                "confidence": round(d["confidence"] * 0.95, 3),
            })
        return out

    def _open_camera(self):
        if PICAMERA2_AVAILABLE and self._picam is None:
            self._picam = Picamera2()
            cfg = self._picam.create_preview_configuration(
                main={"size": (IMG_W, IMG_H), "format": "RGB888"}
            )
            self._picam.configure(cfg)
            self._picam.start()
        elif not PICAMERA2_AVAILABLE and (self._cap is None or not self._cap.isOpened()):
            import platform
            backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
            self._cap = cv2.VideoCapture(self.camera_index, backend)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
            self._cap.set(cv2.CAP_PROP_FPS, 30)

    def process_frame(self, frame: np.ndarray) -> tuple[list[dict], list[dict]]:
        """
        Returns (detections, flow_vectors).
        Async mode: submits frame to YOLO thread, returns flow-propagated last result.
        Sync mode: blocks on YOLO, returns fresh detections.
        """
        self._load_model()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical flow (runs every frame, ~15ms)
        dense_flow = None
        if self._prev_gray is not None:
            dense_flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                pyr_scale=0.5, levels=1, winsize=11,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
            )
        self._prev_gray = gray
        self._last_flow = dense_flow

        if self.async_yolo:
            # Submit frame every YOLO_SKIP_N frames; flow propagates bboxes between calls
            self._frame_count += 1
            if not self._yolo_busy and self._frame_count % YOLO_SKIP_N == 0:
                self._yolo_busy = True
                try:
                    self._yolo_q.put_nowait(frame.copy())
                except queue.Full:
                    self._yolo_busy = False
            with self._yolo_lock:
                detections = list(self._yolo_result)
            if dense_flow is not None:
                detections = self._propagate(detections, dense_flow)
        else:
            detections = self._run_yolo_sync(frame)

        # Build flow_vectors from dense flow over detection bboxes
        flow_vectors = []
        if dense_flow is not None:
            fh, fw = dense_flow.shape[:2]
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                roi = dense_flow[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
                if roi.size == 0:
                    continue
                mean_vx = float(np.mean(roi[:, :, 0]))
                mean_vy = float(np.mean(roi[:, :, 1]))
                magnitude = math.sqrt(mean_vx**2 + mean_vy**2) / 10.0
                if magnitude > 0.1:
                    direction = math.degrees(math.atan2(mean_vx, -mean_vy)) % 360
                    flow_vectors.append({
                        "direction_deg": round(direction, 1),
                        "magnitude": round(min(magnitude, 1.0), 3),
                        "label": det["label"],
                    })

        return detections, flow_vectors

    def read_frame(self) -> np.ndarray | None:
        self._open_camera()
        if self._picam is not None:
            rgb = self._picam.capture_array()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return cv2.flip(bgr, -1)   # cameras mounted upside-down → 180° rotate
        ret, frame = self._cap.read()
        if not ret:
            return None
        return cv2.resize(frame, (IMG_W, IMG_H))

    def release(self):
        if self.async_yolo and self._yolo_thread:
            try:
                self._yolo_q.put(None, timeout=2)
                self._yolo_thread.join(timeout=5)
            except Exception:
                pass
        if self._picam:
            self._picam.stop()
        if self._cap:
            self._cap.release()


class MockVisionProcessor:
    """Drop-in mock for testing without camera."""

    SCENARIOS = [
        (  # Car approaching fast from NW
            [{"label": "car", "confidence": 0.91, "direction_deg": 315, "distance_m": 8.0, "bbox": [10, 60, 90, 140]}],
            [{"direction_deg": 90, "magnitude": 0.87, "label": "car"}],
        ),
        (  # Person stepping into path ahead
            [{"label": "person", "confidence": 0.83, "direction_deg": 0, "distance_m": 2.5, "bbox": [80, 40, 140, 180]}],
            [{"direction_deg": 5, "magnitude": 0.32, "label": "person"}],
        ),
        (  # Clear
            [], [],
        ),
    ]

    def __init__(self):
        self._idx = 0

    def process_frame(self, frame=None):
        result = self.SCENARIOS[self._idx % len(self.SCENARIOS)]
        self._idx += 1
        return result

    def read_frame(self):
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    def release(self):
        pass


class DualVisionProcessor:
    """
    Two Pi cameras giving 360° coverage.
    Camera 0 (index=0): forward hemisphere, heading_offset=0°.
    Camera 1 (index=1): rear hemisphere, heading_offset=180°.

    Single shared async YOLO thread — both cameras feed the same queue,
    tagged with their heading offset so direction_deg is corrected before merge.
    Falls back gracefully if camera 1 not yet plugged in (single-camera mode).

    read_frame() returns (frame0, frame1_or_None).
    process_frame(frame0, frame1) returns merged (detections, flow_vectors).
    """

    def __init__(
        self,
        model_size: str = "yolo26n.pt",
        imgsz: int = 160,
        cam0_offset_deg: float = 330.0,  # 30° left of forward: 66° FOV, 6° front stitch
        cam1_offset_deg: float = 30.0,   # 30° right: total coverage 126° (297°→63°)
    ):                                    # 5° stitch overlap at 0° (straight ahead)
        self._model_size = model_size
        self._imgsz = imgsz
        self._cam0_offset = cam0_offset_deg
        self._cam1_offset = cam1_offset_deg
        self._model = None

        self._picam0: object | None = None
        self._picam1: object | None = None
        self._cam1_available = False
        self._conf = 0.10 if str(model_size).endswith(".onnx") else 0.15

        self._prev_gray0 = None
        self._prev_gray1 = None

        # Shared async YOLO queue: items are (frame, heading_offset)
        self._yolo_q: queue.Queue = queue.Queue(maxsize=2)
        self._results: dict[float, list] = {cam0_offset_deg: [], cam1_offset_deg: []}
        self._results_lock = threading.Lock()
        self._busy0 = self._busy1 = False
        self._thread: threading.Thread | None = None

    def _load_model(self):
        if not CV2_AVAILABLE:
            raise RuntimeError("opencv + ultralytics not installed")
        if self._model is None:
            self._model = YOLO(self._model_size)
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _worker(self):
        while True:
            item = self._yolo_q.get()
            if item is None:
                break
            frame, offset = item
            results = self._model(frame, imgsz=self._imgsz, verbose=False, conf=self._conf)
            dets = []
            for r in results:
                for box in r.boxes:
                    label = self._model.names[int(box.cls[0])]
                    if label not in THREAT_CLASSES:
                        continue
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2
                    # Apply camera heading offset to direction
                    raw_dir = bbox_to_direction(cx)
                    direction = (raw_dir + offset) % 360
                    dets.append({
                        "label": label,
                        "confidence": round(conf, 3),
                        "direction_deg": round(direction, 1),
                        "distance_m": estimate_distance(label, x2 - x1),
                        "bbox": [round(x1), round(y1), round(x2), round(y2)],
                        "camera": int(offset // 90),
                    })
            with self._results_lock:
                self._results[offset] = dets
                if offset == self._cam0_offset:
                    self._busy0 = False
                else:
                    self._busy1 = False

    def _open_cameras(self):
        if not PICAMERA2_AVAILABLE:
            return
        if self._picam0 is None:
            self._picam0 = Picamera2(0)
            cfg = self._picam0.create_preview_configuration(
                main={"size": (IMG_W, IMG_H), "format": "RGB888"}
            )
            self._picam0.configure(cfg)
            self._picam0.start()

        if self._picam1 is None and not self._cam1_available:
            try:
                self._picam1 = Picamera2(1)
                cfg = self._picam1.create_preview_configuration(
                    main={"size": (IMG_W, IMG_H), "format": "RGB888"}
                )
                self._picam1.configure(cfg)
                self._picam1.start()
                self._cam1_available = True
                print("[DualVision] Camera 1 online — 360° coverage active")
            except Exception:
                self._picam1 = None
                self._cam1_available = False

    def _capture(self, picam) -> np.ndarray:
        rgb = picam.capture_array()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return cv2.flip(bgr, -1)   # cameras mounted upside-down → 180° rotate at capture

    def _flow_and_propagate(
        self, frame: np.ndarray, prev_gray, last_dets: list, heading_offset: float
    ) -> tuple[np.ndarray, list, list]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dense_flow = None
        if prev_gray is not None:
            dense_flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=1, winsize=11,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
            )

        # Propagate detections via flow
        dets = last_dets
        if dense_flow is not None and dets:
            fh, fw = dense_flow.shape[:2]
            updated = []
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                roi = dense_flow[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
                if roi.size == 0:
                    updated.append(d)
                    continue
                dx = float(np.mean(roi[:, :, 0]))
                dy = float(np.mean(roi[:, :, 1]))
                nx1, ny1, nx2, ny2 = x1 + dx, y1 + dy, x2 + dx, y2 + dy
                raw_dir = bbox_to_direction((nx1 + nx2) / 2)
                updated.append({**d,
                    "bbox": [round(nx1), round(ny1), round(nx2), round(ny2)],
                    "direction_deg": round((raw_dir + heading_offset) % 360, 1),
                    "confidence": round(d["confidence"] * 0.95, 3),
                })
            dets = updated

        # Flow vectors over detection bboxes
        flow_vecs = []
        if dense_flow is not None:
            fh, fw = dense_flow.shape[:2]
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                roi = dense_flow[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
                if roi.size == 0:
                    continue
                vx = float(np.mean(roi[:, :, 0]))
                vy = float(np.mean(roi[:, :, 1]))
                mag = math.sqrt(vx**2 + vy**2) / 10.0
                if mag > 0.1:
                    raw_dir = math.degrees(math.atan2(vx, -vy)) % 360
                    flow_vecs.append({
                        "direction_deg": round((raw_dir + heading_offset) % 360, 1),
                        "magnitude": round(min(mag, 1.0), 3),
                        "label": d["label"],
                    })
        return gray, dets, flow_vecs

    def read_frame(self) -> tuple[np.ndarray, np.ndarray | None]:
        self._open_cameras()
        frame0 = self._capture(self._picam0) if self._picam0 else np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        frame1 = self._capture(self._picam1) if self._cam1_available else None
        return frame0, frame1

    def process_frame(
        self, frame0: np.ndarray, frame1: np.ndarray | None
    ) -> tuple[list[dict], list[dict]]:
        self._load_model()

        # Submit to async YOLO thread
        if not self._busy0:
            self._busy0 = True
            try:
                self._yolo_q.put_nowait((frame0.copy(), self._cam0_offset))
            except queue.Full:
                self._busy0 = False

        if frame1 is not None and not self._busy1:
            self._busy1 = True
            try:
                self._yolo_q.put_nowait((frame1.copy(), self._cam1_offset))
            except queue.Full:
                self._busy1 = False

        # Get last YOLO results and propagate with flow
        with self._results_lock:
            last0 = list(self._results[self._cam0_offset])
            last1 = list(self._results[self._cam1_offset]) if self._cam1_available else []

        self._prev_gray0, dets0, flow0 = self._flow_and_propagate(
            frame0, self._prev_gray0, last0, self._cam0_offset
        )
        if frame1 is not None:
            self._prev_gray1, dets1, flow1 = self._flow_and_propagate(
                frame1, self._prev_gray1, last1, self._cam1_offset
            )
        else:
            dets1, flow1 = [], []

        return dets0 + dets1, flow0 + flow1

    def release(self):
        if self._thread:
            try:
                self._yolo_q.put(None, timeout=2)
                self._thread.join(timeout=5)
            except Exception:
                pass
        for cam in (self._picam0, self._picam1):
            if cam:
                try:
                    cam.stop()
                except Exception:
                    pass
