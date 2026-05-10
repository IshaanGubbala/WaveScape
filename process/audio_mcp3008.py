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
NUM_MICS = 4             # MCP3008 channels 0-3
MIC_SPACING_M = 0.04     # 4cm between adjacent mics on glasses frame
SPEED_OF_SOUND = 343.0
RING_BUFFER_SECONDS = 2
SPI_BUS = 0              # /dev/spidev0.0 (header SPI, dtparam=spi=on)
SPI_DEVICE = 0           # CE0
SPI_SPEED = 1_350_000    # MCP3008 max ~1.35MHz at 3.3V


class MCP3008AudioProcessor:
    """SPI ADC-based audio capture. Same API as AudioProcessor."""

    def __init__(self, num_mics: int = NUM_MICS, sample_rate: int = SAMPLE_RATE,
                 model_path: Optional[str] = None):
        if not SPIDEV_AVAILABLE:
            raise RuntimeError("spidev not installed: pip install spidev")
        self.num_mics = num_mics
        self._rate = sample_rate
        self._channels = num_mics
        self._spi = spidev.SpiDev()
        self._spi.open(SPI_BUS, SPI_DEVICE)
        self._spi.max_speed_hz = SPI_SPEED
        self._spi.mode = 0

        self._dead: set = self._quick_rail_check()
        self._active = [c for c in range(num_mics) if c not in self._dead]
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
        for ch in range(self.num_mics):
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
            for c in range(self.num_mics):
                raw = self._read_channel(c)
                samples[c] = (raw - 512) / 512.0 if c not in self._dead else 0.0
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
        """Return last `duration_ms` of audio as (N, channels) float32 array."""
        n = int(self._rate * duration_ms / 1000)
        with self._lock:
            N = self._ring.shape[0]
            if n > N:
                n = N
            end = self._write_idx
            start = (end - n) % N
            if start < end:
                return self._ring[start:end].copy()
            return np.concatenate([self._ring[start:], self._ring[:end]]).copy()

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
            summed = np.zeros(N, dtype=np.float32)
            for m in active_mics:
                # delay per mic for this direction (linear array along x-axis)
                delay_per_mic_s = (MIC_SPACING_M * math.cos(math.radians(deg))) / SPEED_OF_SOUND
                delay_per_mic_samples = delay_per_mic_s * self._rate
                shift = int(round(m * delay_per_mic_samples))
                if shift >= 0:
                    summed[shift:] += audio[:N - shift, m] if shift > 0 else audio[:, m]
                else:
                    summed[:N + shift] += audio[-shift:, m]
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
