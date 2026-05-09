"""
Audio processing: I2S mic capture + delay-and-sum beamforming + TFLite classification.
Falls back to mock when hardware unavailable.
"""
import time
import math
import threading
import numpy as np
from typing import Optional

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        TFLiteInterpreter = tf.lite.Interpreter
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False

SAMPLE_RATE = 48000   # Pi I2S (googlevoicehat) native; Mac falls back via sounddevice negotiation
WINDOW_MS = 500
HOP_MS = 250
N_MELS = 64
MIC_SPACING_M = 0.08   # ~8cm between INMP441 mics on glasses frame
SPEED_OF_SOUND = 343.0

# Threat-relevant YAMNet class indices (subset)
THREAT_LABELS = {
    "Vehicle horn, car horn, honking": "horn_honk",
    "Siren": "siren",
    "Engine": "engine_noise",
    "Motorcycle": "motorcycle",
    "Footsteps, footfall": "footsteps",
    "Crowd": "crowd_noise",
    "Speech": "speech",
    "Dog": "dog_bark",
    "Alarm": "alarm",
    "Screaming": "scream",
}


def delay_and_sum_beamform(
    left: np.ndarray, right: np.ndarray, direction_deg: float = 0.0
) -> np.ndarray:
    """
    Delay-and-sum beamforming from two omni mics → synthetic cardioid.
    direction_deg: 0=front, 90=right, 270=left
    """
    delay_s = (MIC_SPACING_M * math.cos(math.radians(direction_deg))) / SPEED_OF_SOUND
    delay_samples = int(delay_s * SAMPLE_RATE)
    if delay_samples >= 0:
        left_shifted = np.pad(left, (delay_samples, 0))[: len(left)]
        return (left_shifted + right) / 2.0
    else:
        right_shifted = np.pad(right, (-delay_samples, 0))[: len(right)]
        return (left + right_shifted) / 2.0


def extract_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    if not LIBROSA_AVAILABLE:
        return np.zeros((N_MELS, 32), dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        fmax=8000,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


class AudioProcessor:
    def __init__(self, model_path: Optional[str] = None, device_index: Optional[int] = None):
        self.model_path = model_path
        self.device_index = device_index
        self._interpreter = None
        self._labels: list[str] = []
        self._channels = 1
        self._rate = SAMPLE_RATE
        self._stream = None
        self._stream_buf = []
        self._stream_lock = threading.Lock()
        if SOUNDDEVICE_AVAILABLE:
            try:
                info = sd.query_devices(device_index, "input")
                self._channels = min(2, int(info["max_input_channels"]))
                if info.get("default_samplerate"):
                    self._rate = int(info["default_samplerate"])
            except Exception:
                pass
            self._open_stream()

    def _open_stream(self):
        def _cb(indata, frames, time, status):
            with self._stream_lock:
                self._stream_buf.append(indata.copy())
                # Keep only last 2s worth
                max_frames = self._rate * 2
                total = sum(b.shape[0] for b in self._stream_buf)
                while total > max_frames and self._stream_buf:
                    total -= self._stream_buf[0].shape[0]
                    self._stream_buf.pop(0)
        try:
            self._stream = sd.InputStream(
                samplerate=self._rate, channels=self._channels,
                dtype="float32", device=self.device_index,
                blocksize=1024, callback=_cb,
            )
            self._stream.start()
        except Exception as e:
            print(f"  [MIC] stream open failed: {e}", flush=True)
            self._stream = None

    def _load_model(self):
        if self._interpreter is not None or not TFLITE_AVAILABLE or not self.model_path:
            return
        self._interpreter = TFLiteInterpreter(model_path=self.model_path)
        self._interpreter.allocate_tensors()

    def _load_yamnet_labels(self, label_path: str):
        with open(label_path) as f:
            self._labels = [line.strip() for line in f]

    def capture_wav_b64(self, duration_ms: int = 2000) -> Optional[str]:
        """Return last `duration_ms` of audio encoded as base64 WAV for Gemma."""
        import io, wave, base64
        audio = self.capture_window(duration_ms)
        if audio is None:
            return None
        mono = audio[:, 0]
        pcm16 = np.clip(mono * 32767, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._rate)
            wf.writeframes(pcm16.tobytes())
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def capture_window(self, duration_ms: int = WINDOW_MS) -> Optional[np.ndarray]:
        if not SOUNDDEVICE_AVAILABLE or self._stream is None:
            return None
        samples_needed = int(self._rate * duration_ms / 1000)
        with self._stream_lock:
            chunks = list(self._stream_buf)
        if not chunks:
            return None
        audio = np.concatenate(chunks, axis=0)[-samples_needed:]
        if audio.shape[0] < samples_needed // 2:
            return None
        if self._channels == 1:
            return np.column_stack([audio[:, 0], audio[:, 0]])
        return audio

    def classify(self, audio_stereo: np.ndarray) -> list[tuple[str, float]]:
        """Returns list of (label, confidence) sorted by confidence desc."""
        if audio_stereo is None:
            return []

        left = audio_stereo[:, 0] if audio_stereo.ndim == 2 else audio_stereo
        right = audio_stereo[:, 1] if audio_stereo.ndim == 2 else audio_stereo
        mono = delay_and_sum_beamform(left, right)

        self._load_model()
        if self._interpreter is None:
            rms = float(np.sqrt(np.mean(mono**2)))
            if rms > 0.05:
                return [("unknown_loud_sound", min(rms * 10, 0.99))]
            return []

        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()
        mel = extract_mel_spectrogram(mono)
        inp = mel[np.newaxis, :, :, np.newaxis]
        self._interpreter.set_tensor(input_details[0]["index"], inp)
        self._interpreter.invoke()
        scores = self._interpreter.get_tensor(output_details[0]["index"])[0]

        results = []
        for i, score in enumerate(scores):
            if i >= len(self._labels):
                break
            label = self._labels[i]
            mapped = THREAT_LABELS.get(label)
            if mapped and score > 0.2:
                results.append((mapped, round(float(score), 3)))

        return sorted(results, key=lambda x: x[1], reverse=True)[:5]

    def beamform_scan(self, audio_stereo: np.ndarray,
                      directions: list[float] = None) -> list[dict]:
        """Scan 4 diagonal directions (35° off forward/back axis), return sorted list of {direction_deg, energy, label}.
        Highest energy = dominant sound source direction."""
        if directions is None:
            directions = [35, 145, 215, 325]
        if audio_stereo is None:
            return []
        left = audio_stereo[:, 0] if audio_stereo.ndim == 2 else audio_stereo
        right = audio_stereo[:, 1] if audio_stereo.ndim == 2 else audio_stereo

        scan = []
        for deg in directions:
            beam = delay_and_sum_beamform(left, right, deg)
            rms = float(np.sqrt(np.mean(beam ** 2)))
            scan.append({"direction_deg": deg, "energy": round(rms, 5)})

        scan.sort(key=lambda x: x["energy"], reverse=True)
        return scan


class MockAudioProcessor:
    """Mock for environments where sounddevice is unavailable (e.g. headless Pi test).
    Returns silence — no fake threat data."""

    def __init__(self):
        pass

    def capture_window(self, duration_ms=500):
        return np.zeros((int(16000 * duration_ms / 1000), 2), dtype=np.float32)

    def classify(self, audio=None):
        return []

    def beamform_scan(self, audio=None, directions=None) -> list[dict]:
        return []
