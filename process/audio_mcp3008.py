"""
MCP3008 SPI ADC audio capture for Pi 5.
Drop-in replacement for AudioProcessor (process/audio.py) when using
analog electret mics (MAX4466) instead of I2S INMP441.

Wiring (Pi physical pins → MCP3008):
  Pi pin 17 (3.3V)  → VDD, VREF
  Pi pin 25 (GND)   → AGND, DGND
  Pi pin 23 (SCLK)  → CLK
  Pi pin 21 (MISO)  → DOUT
  Pi pin 19 (MOSI)  → DIN
  Pi pin 24 (CE0)   → CS

Mics (MAX4466 modules) → MCP3008:
  mic 0 → CH0
  mic 1 → CH1
  mic 2 → CH2
  mic 3 → CH3
  (each MAX4466: VCC=3.3V, GND=GND, OUT=CHn)

Place mics in known geometry (linear or square) for delay-and-sum beamforming.
"""
import math
import threading
import time
import numpy as np
from typing import Optional

try:
    import spidev
    SPIDEV_AVAILABLE = True
except ImportError:
    SPIDEV_AVAILABLE = False

SAMPLE_RATE = 10000      # 10kHz per channel — voice/clap/horn fine
NUM_MICS = 4
# Clockwise from front-right: mic1=FR, mic2=BR, mic3=BL, mic4=FL
# MCP3008 channels (0-indexed): user channels 5,8,6,3 → 0-indexed 4,7,5,2
MIC_CHANNELS = [4, 7, 5, 2]  # FR, BR, BL, FL
SPEED_OF_SOUND = 343.0

# X-formation mic positions (x=right, y=forward, meters from center)
# Width: 6.878in = 174.74mm, Depth: 2.986in = 75.84mm
# CH2=FL, CH4=FR, CH5=BL, CH7=BR
_MIC_POS = {
    2: (-0.08737,  0.03792),   # FL front-left  (325°)
    4: ( 0.08737,  0.03792),   # FR front-right  (35°)
    5: (-0.08737, -0.03792),   # BL back-left   (215°)
    7: ( 0.08737, -0.03792),   # BR back-right  (145°)
}
RING_BUFFER_SECONDS = 2
SPI_BUS = 0              # /dev/spidev0.0 (header SPI, dtparam=spi=on)
SPI_DEVICE = 0           # CE0
SPI_SPEED = 1_350_000    # MCP3008 max ~1.35MHz at 3.3V

# Software gain trim per channel (1.0 = no change, <1.0 = attenuate).
# Compensates for MAX4466 trimpot inaccessible — tune if channels saturate.
_CHANNEL_GAIN = {
    2: 1.0,   # FL — dead, placeholder
    4: 1.0,   # FR
    5: 1.0,   # BL
    7: 1.0,   # BR
}


class MCP3008AudioProcessor:
    """SPI ADC-based audio capture. Same API as AudioProcessor."""

    def __init__(self, num_mics: int = NUM_MICS, sample_rate: int = SAMPLE_RATE,
                 model_path: Optional[str] = None):
        if not SPIDEV_AVAILABLE:
            raise RuntimeError("spidev not installed: pip install spidev")
        self.num_mics = num_mics
        self._rate = sample_rate
        self._channels = MIC_CHANNELS
        self._spi = spidev.SpiDev()
        self._spi.open(SPI_BUS, SPI_DEVICE)
        self._spi.max_speed_hz = SPI_SPEED
        self._spi.mode = 0

        self._dead: set = self._quick_rail_check()
        self._active = [c for c in MIC_CHANNELS if c not in self._dead]
        print(f"  mic detect: active={self._active} dead={sorted(self._dead)}")

        self._ring = np.zeros((sample_rate * RING_BUFFER_SECONDS, num_mics),
                              dtype=np.float32)
        self._write_idx = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()


    def _quick_rail_check(self) -> set:
        """50 fast reads per channel. Dead if mean stuck near rail (|mean|>0.80 normalized)."""
        dead = set()
        for ch in MIC_CHANNELS:
            vals = []
            for _ in range(50):
                r = self._spi.xfer2([1, (8 + ch) << 4, 0])
                vals.append(((r[1] & 3) << 8) | r[2])
            mean = (sum(vals) / len(vals) - 512) / 512.0
            if abs(mean) > 0.80:
                dead.add(ch)
        return dead

    def _read_channel(self, ch: int) -> int:
        r = self._spi.xfer2([1, (8 + ch) << 4, 0])
        return ((r[1] & 0x03) << 8) | r[2]

    def _sample_loop(self):
        period = 1.0 / self._rate
        next_t = time.monotonic()
        N = self._ring.shape[0]
        while not self._stop.is_set():
            samples = np.empty(self.num_mics, dtype=np.float32)
            for i, ch in enumerate(MIC_CHANNELS):
                raw = self._read_channel(ch)
                samples[i] = (raw - 512) / 512.0 * _CHANNEL_GAIN.get(ch, 1.0) if ch not in self._dead else 0.0
            with self._lock:
                self._ring[self._write_idx] = samples
                self._write_idx = (self._write_idx + 1) % N
            next_t += period
            sleep_t = next_t - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.monotonic()   # we're behind, reset

    def capture_window(self, duration_ms: int = 500) -> Optional[np.ndarray]:
        """Return last `duration_ms` of audio as (N, channels) float32 array, AC-coupled."""
        n = int(self._rate * duration_ms / 1000)
        with self._lock:
            N = self._ring.shape[0]
            if n > N:
                n = N
            end = self._write_idx
            start = (end - n) % N
            if start < end:
                buf = self._ring[start:end].copy()
            else:
                buf = np.concatenate([self._ring[start:], self._ring[:end]]).copy()
        # AC-couple: remove per-channel DC offset
        buf -= buf.mean(axis=0, keepdims=True)
        return buf

    def capture_wav_b64(self, duration_ms: int = 2000) -> Optional[str]:
        # CJK fine-tuned spatial model is text-only (no mmproj). Returning None
        # forces text-only path with KV cache prefix. Audio handled via beam_scan.
        return None

    def classify(self, audio: np.ndarray) -> list[tuple[str, float]]:
        """Simple RMS-based threat detection (no TFLite model needed)."""
        if audio is None or audio.size == 0:
            return []
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        rms = float(np.sqrt(np.mean(mono ** 2)))
        if rms > 0.05:
            return [("unknown_loud_sound", min(rms * 10, 0.99))]
        return []

    def beamform_scan(self, audio: np.ndarray,
                      directions: list[float] = None) -> list[dict]:
        """Delay-and-sum beamform across active mics (skip DEAD_CHANNELS).
        Returns sorted list of {direction_deg, energy} per direction."""
        if audio is None or audio.size == 0:
            return []
        if directions is None:
            directions = [35, 145, 215, 325]

        N, M = audio.shape if audio.ndim == 2 else (audio.shape[0], 1)
        if M == 1:
            return []

        active_mics = [m for m in range(M) if m not in self._dead]
        if not active_mics:
            return []

        scan = []
        for deg in directions:
            rad = math.radians(deg)
            # beam unit vector (x=right, y=forward)
            bx, by = math.sin(rad), math.cos(rad)
            summed = np.zeros(N, dtype=np.float32)
            for mi, ch in enumerate(active_mics):
                # 2D delay: project mic position onto beam direction
                mx, my = _MIC_POS.get(ch, (0.0, 0.0))
                delay_s = (mx * bx + my * by) / SPEED_OF_SOUND
                shift = int(round(delay_s * self._rate))
                col = mi  # column index in audio array (active mics only)
                if shift >= 0:
                    summed[shift:] += audio[:N - shift, col] if shift > 0 else audio[:, col]
                else:
                    summed[:N + shift] += audio[-shift:, col]
            summed /= len(active_mics)

            rms = float(np.sqrt(np.mean(summed ** 2)))
            scan.append({"direction_deg": deg, "energy": round(rms, 5)})

        scan.sort(key=lambda x: x["energy"], reverse=True)
        return scan

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._spi.close()


class MockMCP3008Processor:
    """Mock when spidev unavailable or no MCP3008 wired."""
    def __init__(self, *a, **kw):
        self._rate = SAMPLE_RATE

    def capture_window(self, duration_ms=500):
        return np.zeros((int(SAMPLE_RATE * duration_ms / 1000), NUM_MICS),
                        dtype=np.float32)

    def capture_wav_b64(self, duration_ms=2000):
        return None

    def classify(self, audio=None):
        return []

    def beamform_scan(self, audio=None, directions=None):
        return []

    def close(self):
        pass
