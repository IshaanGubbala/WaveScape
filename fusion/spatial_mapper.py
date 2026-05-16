"""
Spatial scene mapper using Gemma vision + directional mic beamforming.
Called every ~3s (not per-threat). Produces a persistent world model
that drives haptic output between YOLO Tier-1 events.
"""
import json
import re
import time
import threading
import requests
from dataclasses import dataclass, field
from typing import Callable, Optional

LLAMA_URL = "http://192.168.68.65:8081"  # Mac Metal (LAN offload, ~160ms vs ~5s Pi CPU)
SPATIAL_UPDATE_INTERVAL = 0.0  # back-to-back; naturally rate-limited by inference (~1.5s)
GEMMA_TIMEOUT_S = 10.0          # force-release busy lock after this many seconds
GEMMA_HTTP_TIMEOUT = 9.0        # HTTP request timeout (slightly under busy timeout)
SCENE_UNCHANGED_TTL = 2.0       # skip Gemma if scene unchanged within this window

# Smooth-angle CJK codebook — 8 tokens total, 1° precision
# Format: T1 O1 L1 U1 T2 O2 L2 U2  (no spaces), deg = T*19 + O
_T_CHARS = list("一二三四五六七八九十甲乙丙丁戊己庚辛壬")  # tens (0-18)
_O_CHARS = list("子丑寅卯辰巳午未申酉戌亥角亢氐房心尾箕")  # ones (0-18)
_LABELS_CJK = "车卡巴人自物"  # car truck bus person bike obstacle
_URG_CJK = "危急中远"          # crit high med low
_LABEL_NAMES = ["car", "truck", "bus", "person", "bike", "obstacle"]
_URG_NAMES = ["critical", "high", "medium", "low"]
_URG_DIST = [1.0, 3.0, 6.0, 12.0]

SPATIAL_SYSTEM_PROMPT = (
    "Output 8 CJK chars encoding 2 closest objects. Format: T1O1L1U1 T2O2L2U2 (no spaces). "
    "Angle: T(tens) O(ones), deg=T*19+O. "
    "T alphabet: 一二三四五六七八九十甲乙丙丁戊己庚辛壬 (idx 0-18). "
    "O alphabet: 子丑寅卯辰巳午未申酉戌亥角亢氐房心尾箕 (idx 0-18). "
    "L: 车卡巴人自物 (car truck bus person bike obstacle). "
    "U: 危急中远 (crit<2m, high<4m, med<8m, low>=8m). "
    "Pick 2 closest objects. No other text."
)

# Minified GBNF — forces dense JSON, no whitespace, ~2× faster generation
_SPATIAL_GRAMMAR = (
    'root ::= "{\\"objects\\":[" objects "],\\"dominant_hazard\\":" hazard ",\\"summary\\":\\"" summary "\\"}" | "{}"\n'
    'objects ::= "" | object ("," object)*\n'
    'object ::= "{\\"dir\\":" int ",\\"label\\":\\"" label "\\",\\"dist_m\\":" number ",\\"sound\\":\\"" sound "\\",\\"moving\\":" boolean "}"\n'
    'hazard ::= "{}" | "{\\"dir\\":" int ",\\"urgency\\":\\"" urgency "\\"}"\n'
    'label ::= "car" | "truck" | "bus" | "person" | "bike" | "obstacle" | "sound_source"\n'
    'sound ::= "engine" | "horn" | "footsteps" | "speech" | "none"\n'
    'urgency ::= "critical" | "high" | "medium" | "low"\n'
    'boolean ::= "true" | "false"\n'
    'int ::= "-"? [0-9]+\n'
    'number ::= [0-9]+ ("." [0-9]*)?\n'
    'summary ::= [^"]*\n'
)


@dataclass
class SpatialMap:
    objects: list[dict] = field(default_factory=list)
    dominant_hazard: dict = field(default_factory=dict)
    summary: str = ""
    timestamp: float = field(default_factory=time.time)

    def is_empty(self) -> bool:
        return not self.objects and not self.dominant_hazard

    def dominant_direction(self) -> Optional[float]:
        h = self.dominant_hazard
        if h and "dir" in h:
            return float(h["dir"])
        if self.objects:
            return float(self.objects[0]["dir"])
        return None

    def dominant_urgency(self) -> str:
        return self.dominant_hazard.get("urgency", "low")


_ENERGY_TO_DIST = [(0.15, 1.5), (0.08, 3.0), (0.04, 6.0), (0.02, 9.0), (0.01, 12.0)]

def _energy_to_dist(energy: float) -> float:
    for threshold, dist in _ENERGY_TO_DIST:
        if energy >= threshold:
            return dist
    return 15.0


def _scene_hash(detections: list[dict], beam_scan: list[dict]) -> int:
    """Stable hash of detection scene. Bucket angle to 15°, dist to urgency tier."""
    def _urg(dist: float) -> int:
        if dist < 2: return 0
        if dist < 4: return 1
        if dist < 8: return 2
        return 3

    det_items = tuple(sorted(
        (d.get("label", "?"),
         round(float(d.get("direction_deg", 0)) / 15) * 15,
         _urg(float(d.get("distance_m", 99))))
        for d in detections
        if float(d.get("distance_m", 99)) < 15.0
    ))
    beam_items = tuple(sorted(
        (round(float(b.get("direction_deg", 0)) / 45) * 45,)
        for b in beam_scan if float(b.get("energy", 0)) > 0.08
    ))
    return hash((det_items, beam_items))


def _fuse_audio_visual(detections: list[dict], beam_scan: list[dict]) -> list[dict]:
    """Cross-correlate beam peaks with visual detections.
    - Visual det near a beam peak → annotate with sound label.
    - Strong beam with no nearby visual → inject audio-only detection.
    Only meaningful on Pi with real stereo mic array.
    """
    fused = [dict(d) for d in detections]
    ANGLE_TOL = 45.0   # degrees — beam must be within this of visual det
    ENERGY_MIN = 0.08   # ignore ambient noise (Mac mono mic ~0.04 baseline)

    significant = [b for b in beam_scan if b.get("energy", 0) >= ENERGY_MIN]
    significant.sort(key=lambda b: b["energy"], reverse=True)

    for beam in significant[:4]:
        bdir = float(beam["direction_deg"])
        blabel = beam.get("label", "noise")
        benergy = beam["energy"]

        # Find nearest visual detection by angular distance
        best_det = None
        best_delta = float("inf")
        for det in fused:
            det_dir = float(det.get("direction_deg", det.get("dir_deg", 0)))
            delta = abs((det_dir - bdir + 180) % 360 - 180)
            if delta < best_delta:
                best_delta = delta
                best_det = det

        if best_det is not None and best_delta <= ANGLE_TOL:
            # Annotate existing detection with sound
            if best_det.get("sound", "none") in ("none", "", None):
                best_det["sound"] = blabel
        else:
            # No visual match — inject audio-only object
            fused.append({
                "label": "sound_source",
                "direction_deg": bdir,
                "dir_deg": bdir,
                "distance_m": _energy_to_dist(benergy),
                "dist_m": _energy_to_dist(benergy),
                "confidence": min(benergy * 5, 0.6),
                "sound": blabel,
                "audio_only": True,
            })

    return fused


def _build_spatial_prompt(detections: list[dict], beam_scan: list[dict],
                           heading_deg: float = 0.0) -> str:
    fused = _fuse_audio_visual(detections, beam_scan)

    def _priority(d):
        dist = float(d.get("distance_m") or d.get("dist_m") or 99)
        vel  = float(d.get("velocity_mps") or 0)  # positive = approaching
        eta  = float(d.get("eta_seconds") or 99)
        # Score: lower = more urgent. Approaching objects get boosted.
        approaching_bonus = -20 if vel > 0.3 else 0   # shorten effective dist
        return dist + approaching_bonus

    # Keep urgent (<8m) and approaching (vel>0.3), sort by priority score
    relevant = [d for d in fused if
                (float(d.get("distance_m") or d.get("dist_m") or 99) < 8.0)
                or (float(d.get("velocity_mps") or 0) > 0.3)]
    if not relevant:
        relevant = fused  # fallback: all objects

    relevant.sort(key=_priority)

    det_parts = []
    for d in relevant[:6]:
        dist = d.get("distance_m") or d.get("dist_m") or "?"
        tag = "[audio]" if d.get("audio_only") else ""
        sound = d.get("sound", "")
        sound_str = f" snd={sound}" if sound and sound != "none" else ""
        det_parts.append(
            f"{tag}{d.get('label','?')}@{d.get('direction_deg', d.get('dir_deg', 0)):.0f}°"
            f" {dist}m{sound_str}"
        )

    parts = [f"heading={heading_deg:.0f}°"]
    if det_parts:
        parts.append("visual=[" + ", ".join(det_parts) + "]")

    return " | ".join(parts)


def _chars_to_deg(t: str, o: str) -> Optional[float]:
    try:
        return float(min(359, _T_CHARS.index(t) * 19 + _O_CHARS.index(o)))
    except ValueError:
        return None


def _parse_spatial_map(buf: str, detections: list[dict] = None) -> SpatialMap:
    """Decode Gemma output. Primary: 8-char CJK smooth-angle format.
    Fallback: ASCII 'label@deg°' format emitted by Q4_0 CPU inference."""
    _T_SET = set(_T_CHARS)
    _O_SET = set(_O_CHARS)
    _L_SET = set(_LABELS_CJK)
    _U_SET = set(_URG_CJK)

    # --- Primary: CJK 8-char format (Mac Metal, Q8+) ---
    chars = [c for c in buf if c in _T_SET | _O_SET | _L_SET | _U_SET]
    if len(chars) >= 4:
        objects = []
        summary_parts = []
        i = 0
        while i + 3 < len(chars) and len(objects) < 2:
            t, o, l, u = chars[i], chars[i+1], chars[i+2], chars[i+3]
            deg = _chars_to_deg(t, o)
            if deg is not None and l in _L_SET and u in _U_SET:
                li = _LABELS_CJK.index(l)
                ui = _URG_CJK.index(u)
                objects.append({
                    "dir": deg,
                    "label": _LABEL_NAMES[li],
                    "dist_m": _URG_DIST[ui],
                    "urgency": _URG_NAMES[ui],
                })
                summary_parts.append(f"{_LABEL_NAMES[li]}@{deg:.0f}°")
                i += 4
            else:
                i += 1
        if objects:
            hazard = {"dir": objects[0]["dir"], "urgency": objects[0]["urgency"]}
            return SpatialMap(objects=objects, dominant_hazard=hazard, summary=" | ".join(summary_parts))

    # --- Fallback: ASCII 'label@deg°' format (Q4_0 CPU Pi inference) ---
    import re
    matches = re.findall(r'([a-z_]+)@(\d+(?:\.\d+)?)[°\s]', buf.lower())
    if not matches:
        return SpatialMap()

    objects = []
    summary_parts = []
    for raw_label, deg_str in matches[:2]:
        deg = float(deg_str)
        label = raw_label.strip("_")
        # Cross-reference with YOLO detections for distance
        dist = 8.0
        if detections:
            # Try angle-match first, fall back to label-match (Q4_0 dirs may be off)
            for d in detections:
                det_dir = float(d.get("direction_deg", d.get("dir_deg", 0)))
                if abs((det_dir - deg + 180) % 360 - 180) < 60:
                    dist = float(d.get("distance_m", 8.0))
                    break
            else:
                for d in detections:
                    if label.lower() in d.get("label", "").lower():
                        dist = float(d.get("distance_m", 8.0))
                        break
        urgency = ("critical" if dist < 2 else "high" if dist < 4
                   else "medium" if dist < 8 else "low")
        objects.append({"dir": deg, "label": label, "dist_m": dist, "urgency": urgency})
        summary_parts.append(f"{label}@{deg:.0f}°")

    if not objects:
        return SpatialMap()
    hazard = {"dir": objects[0]["dir"], "urgency": objects[0]["urgency"]}
    return SpatialMap(objects=objects, dominant_hazard=hazard, summary=" | ".join(summary_parts))


class SpatialMapper:
    """
    Periodic Gemma spatial mapping. Decoupled from per-frame threat pipeline.
    Calls Gemma every SPATIAL_UPDATE_INTERVAL seconds with latest frame + audio beam scan.
    Fires map_cb(SpatialMap, beam_scan) on each completed update.
    """

    def __init__(self, map_cb: Callable,
                 server_url: str = LLAMA_URL):
        self.map_cb = map_cb
        self.server_url = server_url
        self._busy = threading.Event()
        self._busy_set_at: float = 0.0
        self._last_call: float = 0.0
        self._current_map: SpatialMap = SpatialMap()
        self._map_lock = threading.Lock()
        self._kv_cache: dict = {}
        self._cache_lock = threading.Lock()
        self._energy_history: list[float] = []
        self._last_audio_trigger: float = 0.0
        self._last_scene_hash: int = 0
        self._last_scene_time: float = 0.0
        self._err_count: int = 0
        self._last_err_time: float = 0.0

    @property
    def current_map(self) -> SpatialMap:
        with self._map_lock:
            return self._current_map

    def _audio_spike(self, beam_scan: list[dict]) -> bool:
        """True if sudden loud sound — energy > 3× rolling avg and cooldown elapsed."""
        if not beam_scan:
            return False
        peak = max(b.get("energy", 0) for b in beam_scan)
        self._energy_history.append(peak)
        if len(self._energy_history) > 30:
            self._energy_history.pop(0)
        if len(self._energy_history) < 5:
            return False
        avg = sum(self._energy_history[:-1]) / (len(self._energy_history) - 1)
        cooldown_ok = (time.time() - self._last_audio_trigger) >= 3.0
        if peak > max(avg * 3.0, 0.05) and cooldown_ok:
            self._last_audio_trigger = time.time()
            print(f"  [AUDIO SPIKE] peak={peak:.3f} avg={avg:.3f} → immediate spatial update")
            return True
        return False

    def should_update(self, beam_scan: list[dict] = None) -> bool:
        if self._busy.is_set():
            if time.time() - self._busy_set_at > GEMMA_TIMEOUT_S:
                print("  [SPATIAL] busy timeout — force-releasing lock")
                self._busy.clear()
            else:
                return False
        # Exponential backoff on repeated errors (cap at 60s)
        if self._err_count > 0:
            backoff = min(60.0, 2.0 * self._err_count)
            if time.time() - self._last_err_time < backoff:
                return False
        interval_due = (time.time() - self._last_call) >= SPATIAL_UPDATE_INTERVAL
        audio_triggered = self._audio_spike(beam_scan or [])
        return interval_due or audio_triggered

    def update(self, detections: list[dict], beam_scan: list[dict],
               frame=None, heading_deg: float = 0.0, audio_b64: str = None) -> None:
        """Non-blocking. Drops if previous call still running."""
        if self._busy.is_set():
            return
        self._busy.set()
        self._busy_set_at = time.time()
        self._last_call = time.time()
        t = threading.Thread(
            target=self._infer,
            args=(detections, beam_scan, frame, heading_deg, audio_b64),
            daemon=True,
        )
        t.start()

    def _server_has_vision(self) -> bool:
        # CJK fine-tune is text-only. Force text path even if Gemma 4 server
        # advertises mmproj capability (we don't load mmproj for CJK).
        return False

    @staticmethod
    def _frame_to_b64(frame) -> str:
        import cv2, base64
        h, w = frame.shape[:2]
        if w > 512:
            frame = cv2.resize(frame, (512, int(h * 512 / w)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf).decode("utf-8")

    def _build_messages(self, prompt: str, frame, audio_b64: str = None) -> list:
        msgs = [{"role": "system", "content": SPATIAL_SYSTEM_PROMPT}]

        content = []
        if frame is not None and self._server_has_vision():
            b64 = self._frame_to_b64(frame)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        if audio_b64:
            content.append({"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}})
        content.append({"type": "text", "text": prompt})

        msgs.append({"role": "user", "content": content if len(content) > 1 else prompt})
        return msgs

    def _infer(self, detections: list[dict], beam_scan: list[dict],
               frame, heading_deg: float, audio_b64: str = None) -> None:
        self._last_beam_scan = beam_scan

        # Scene-diff gate: skip LLM if scene unchanged within TTL window
        scene_h = _scene_hash(detections, beam_scan)
        now = time.time()
        if (scene_h == self._last_scene_hash
                and (now - self._last_scene_time) < SCENE_UNCHANGED_TTL):
            with self._map_lock:
                cached = self._current_map
            print(f"  [SPATIAL] scene unchanged — reusing cached map")
            self.map_cb(cached, beam_scan)
            return

        prompt = _build_spatial_prompt(detections, beam_scan, heading_deg)
        print(f"  [GEMMA IN] {prompt[:120]}")
        messages = self._build_messages(prompt, frame, None)  # text-only model, no audio

        body = {
            "model": "gemma",
            "messages": messages,
            "max_tokens": 12,
            "stop": ["<end_of_turn>", "</s>"],
            "temperature": 0.0,
            "stream": False,
            "cache_prompt": True,
        }

        buf = ""
        try:
            resp = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=body,
                timeout=GEMMA_HTTP_TIMEOUT,
            )
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            buf = msg.get("content") or msg.get("reasoning_content") or ""
            if resp.status_code != 200:
                print(f"  [GEMMA ERR] {resp.status_code}: {resp.text[:200]}")
            print(f"  [GEMMA RAW] status={resp.status_code} content={repr(msg.get('content'))} finish={choice.get('finish_reason')}")

            print(f"  [SPATIAL tail] {repr(buf[-300:])}")
            spatial_map = _parse_spatial_map(buf, detections)
            spatial_map.timestamp = time.time()

            with self._map_lock:
                self._current_map = spatial_map

            if buf.strip():
                with self._cache_lock:
                    self._kv_cache = {"prompt": prompt, "response": buf.strip()}
                self._last_scene_hash = scene_h
                self._last_scene_time = time.time()

            print(f"  [SPATIAL] {spatial_map.summary or '(empty)'}"
                  f"  objs={len(spatial_map.objects)}"
                  f"  hazard={spatial_map.dominant_hazard}")
            self.map_cb(spatial_map, beam_scan)

        except Exception as e:
            self._err_count += 1
            self._last_err_time = time.time()
            if self._err_count <= 2 or self._err_count % 60 == 0:
                print(f"  [SPATIAL] error: {e}  (×{self._err_count})")
        else:
            self._err_count = 0  # reset on success
        finally:
            self._busy.clear()
