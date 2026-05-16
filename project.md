# ThreatDetect — Full Project Reference

**Date**: 2026-05-14  
**Status**: Functional pipeline on Pi 5. Light-mode Gradio UI refreshed. Stereo depth available. Hackathon submission pending.  
**Target**: Gemma4 Good hackathon — 3 award tracks: best wearable, best resource-constrained hardware, best Unsloth fine-tuned Gemma 4 model.

---

## Concept

Wearable real-time threat detection for blind/deaf users. Detects vehicles, pedestrians, and environmental hazards using camera(s) + mic array. Outputs directional haptic alerts via vibration motors on the temples.

**Problem statement**: Visually or hearing-impaired pedestrians lack spatial awareness of approaching threats. Commercial solutions (white canes, guide dogs) don't detect fast-moving traffic or warn by direction. This system provides sub-second directional haptic feedback with a $60 hardware cost.

**Key insight**: Full LLM spatial mapping runs only every 2s on a slow CPU. Fast Tier-1 haptic (YOLO + proximity) fires in <1 frame. ThreatPredictor bridges the gap at 10Hz. From the user's perspective, haptic never blinks out.

---

## Hardware

### Raspberry Pi 5 (8GB RAM)
- BCM2712 (4× Cortex-A76 @ 2.4GHz)
- SSH: `ishaan@192.168.68.107` (LAN) | `ishaan@100.99.239.20` (Tailscale), password: `gubbala`
- SSH flag: `-o StrictHostKeyChecking=no`
- Code: `~/threatdetect/` | OS: Raspberry Pi OS, kernel 6.12.75+rpt-rpi-2712

### Cameras
- 2× Pi Camera Module 3 (IMX708, 66° diagonal FOV, ~33° horizontal half-FOV)
- `cam0` = 330° azimuth (front-left), `cam1` = 30° azimuth (front-right)
- 126° total coverage, 6° front overlap
- Capture resolution: 224×224 natively (no resize step)
- YOLO input: imgsz=96 (fixed ONNX shape, downscaled at inference)
- Stereo: 96×96 (SGBM at same resolution as YOLO, no extra resize)
- Cameras mounted upside-down on glasses frame → `cv2.flip(frame, -1)` at capture

### Haptic Motors
- Left temple: **GPIO 24** | Right temple: **GPIO 25**
- PWM at 200Hz via `lgpio`
- Direction→split: smooth piecewise-linear  
  - 0°/180° ahead/behind = 50/50 both motors  
  - 90° right = 100% right motor  
  - 270° left = 100% left motor

### Microphone Array
- **MCP3008 SPI ADC** + 4× MAX4466 amplified electret mics
- SPI: `/dev/spidev10.0` (CE0), speed: 1.35MHz
- Sample rate: 10kHz per channel × 4 channels
- Pi wiring: pin 17 (3.3V), 25 (GND), 23 (SCLK), 21 (MISO), 19 (MOSI), 24 (CE0)
- Mic spacing: ~4cm (`MIC_SPACING_M = 0.04`)
- Channel 1 hardware dead (stuck near rail) — auto-detected at startup, zeroed
- Beamforming: 4 diagonal beams at 35°, 145°, 215°, 325° (active mics only)
- `main.py` auto-detects: MCP3008 → sounddevice → mock

> **I2S INMP441**: Software fully working, hardware SD pin fault. All I2S code in `~/dmic_fd/`. Not active.

### Glasses Frame (Stereo Baseline)
- Interpupillary distance: ~65mm baseline between lens centers
- Cameras parallel (rigid mount, small angular deviation ≤5°)
- Stereo calibration stored at `~/threatdetect/stereo_calib.npz` (if run)
- Calibration tool: `tools/calibrate_stereo.py` — uses 9×6 checkerboard, 25mm squares, 20 auto-captures

---

## Models

### YOLO (always-on, every 3rd frame)
- **Pi model**: `yolo26n.onnx` (FP32, 9.3MB, imgsz=96)
  - INT8 QDQ version exists (`yolo26n_int8.onnx`, 2.9MB) but **broken on ORT 1.26 ARM** (outputs all zeros, max_conf=0.0). Do not use. Backup FP32 at `~/threatdetect/yolo26n_fp32_backup.onnx`.
- **Mac model**: `yolov8n.pt` (PyTorch, dev only)
- Input: `imgsz=96`, conf threshold 0.10, 9 threat classes
- Runtime: direct `ort.InferenceSession` (bypasses ultralytics Python overhead)
- Session options: `ORT_ENABLE_ALL`, `ORT_SEQUENTIAL`, `intra_threads=2`
- Inference: **17ms FP32 on Pi** (was 307ms PyTorch, 70ms ultralytics-wrapped ONNX)
- Async background thread, `queue.Queue(maxsize=1)` + `put_nowait` (drops stale frames)
- `YOLO_SKIP_N=3` with optical flow bbox propagation between frames
- Confidence decay: 0.95× per skipped frame (prevents stale tracks from sustaining haptic)

### Gemma 4 E2B (spatial mapper, every 2s)
- Base: `gemma-4-E2B-it` (2B params, Gemma 4 architecture)
- Fine-tuned with LoRA (rank=16, scale=16, lr=1e-4, 5000 iters, MLX on Mac M5)
- **Active model**: `e2b-smooth-q4_0.gguf` (3.2GB Q4_0)
  - Pi: `~/models/gemma4-e2b/e2b-smooth-q4_0.gguf`
  - Mac: `/Users/ishaangubbala/threatdetect/finetune/e2b-smooth-q4_0.gguf`
- Served via `llama-server` on port 8081
- Inference: ~0.18s Mac Metal, **~3s Pi** (Q4_0, cores 2-3, with pipeline running)
- **Template**: `~/gemma4-nothink.jinja` — MUST use Gemma 4's `<|turn>` tokens (NOT Gemma 3's `<start_of_turn>`). Wrong template → 111° systematic direction bias.
- Config: `-c 256 -t 2 --mlock -ngl 0 --parallel 1 --log-disable`
- `--flash-attn on` BREAKS Q4_0 output on ARM. Never use on Pi.
- **Q4_K_M**: built at `~/models/gemma4-e2b/e2b-smooth-q4km.gguf` but 2× slower on ARM (no NEON dequant path). Not active.

### Model History
| Model | Status | Notes |
|-------|--------|-------|
| `gemma-3-1b-spatial` | abandoned | too small, hallucinated |
| `e2b-spatial` (CSV) | abandoned | verbose output, slow |
| `e2b-cjk` | abandoned | 8.8° avg error (biased training data) |
| `e2b-ultra` | abandoned | model collapse — LoRA can't learn 8640-class arbitrary codebook |
| `e2b-smooth` | **ACTIVE** | smooth angle encoding, 1° precision, 8 tokens/output |

---

## CJK Smooth Encoding

The fine-tuned Gemma model outputs exactly **8 CJK characters** for 2 nearest objects.

```
Output format:  T1 O1 L1 U1 T2 O2 L2 U2

angle_deg = T_index × 19 + O_index    (range 0–359, 1° precision)

T_CHARS = "一二三四五六七八九十甲乙丙丁戊己庚辛壬"   (19 chars, tens: idx 0–18)
O_CHARS = "子丑寅卯辰巳午未申酉戌亥角亢氐房心尾箕"   (19 chars, ones: idx 0–18)
L_CHARS = "车卡巴人自物"   → car truck bus person bike obstacle
U_CHARS = "危急中远"       → critical(<2m) high(<4m) medium(<8m) low(>=8m)
```

**Why CJK?** Each char = 1 BPE token in Gemma. ASCII digits split: `"135"` = 3 tokens. CJK = 8 tokens total for 2 objects with 1° precision vs 20+ ASCII tokens.

**Why T×19+O arithmetic (not direct lookup)?** An earlier design used 1 arbitrary char per object from 8640-char codebook — model collapsed (LoRA can't learn pure lookup). T×19+O is learnable because it's arithmetic. LoRA learns "increase O → +1°, increase T → +19°".

**Why 19?** 19×19 = 361 ≥ 360° coverage with only 38 unique characters needed.

---

## Pipeline Architecture

```
Camera(s) (224×224 @ ~10fps Pi / ~49fps Mac)
    │
    ▼
VisionProcessor / DualVisionProcessor
    │  YOLO detection → label, conf, bbox, direction_deg, distance_m
    │  Optical flow → bbox propagation between YOLO frames
    │  Optional: StereoDepthEstimator → depth_map, sector_depths, escape_dir_deg
    │
    ▼
ObjectTracker (IoU matching → velocity_mps, eta_s, motion_label)
    │
    ▼
CascadeRouter
    │  score = 0.50×vis + 0.25×aud + 0.15×motion + 0.10×pitch + prox_boost
    │  prox_boost: <3m→+0.20, <6m→+0.10, <12m→+0.05
    │
    ├── Tier 0 (score < 0.18): no action
    ├── Tier 1 (score ≥ 0.62): direct haptic, no LLM wait
    └── Tier 2 (score ≥ 0.18, cooldown 0.8s): → SpatialMapper
    │
    ├──► AudioProcessor (MCP3008 SPI 10kHz, 4-beam scan every 5 frames)
    │       Audio spike (>3× rolling avg) → immediate SpatialMapper trigger
    │
    ├──► SpatialMapper (every 2s OR on audio spike)
    │        Fuses YOLO dets + audio beams → text prompt
    │        Priority: <8m OR velocity>0.3 m/s, top 6 objects
    │        Prompt → Gemma → 8 CJK chars → SpatialMap
    │        Scene-diff gate (urgency tier + 15° bucket hash, TTL=6s)
    │        haptic_fire() + web_ui.update_spatial()
    │
    ├──► ThreatPredictor (10Hz, background thread)
    │        Last MotionVector from flow → extrapolates threat position
    │        haptic_cb at 10Hz between Gemma calls
    │
    ├──► ThreatMemory (exponential decay, t½=2s)
    │        Sustains threat during 3s Gemma inference gap
    │        Decayed conf > 0.35 → fire tier-1 haptic without new LLM call
    │
    └──► HapticController (GPIO 24/25 PWM 200Hz or console fallback)
             web_ui.log_haptic()
```

---

## Stereo Depth (fusion/stereo_depth.py)

Added in session 2026-05-14 to support dual-cam navigation.

**Purpose**: Metric depth per angular sector → safest escape direction for navigation haptic.

**Algorithm**: OpenCV SGBM (Semi-Global Block Matching)
- Input: 96×96 grayscale stereo pair
- `numDisparities=32`, `blockSize=5` (odd)
- P1=600, P2=2400
- Output: 16-bit fixed-point disparity → float at /16

**Depth formula**: `Z = focal_px × baseline_m / disparity`
- `FOCAL_PX = (96/2) / tan(33°) ≈ 74px` at 96px width
- `BASELINE_M = 0.065` (65mm interpupillary)
- Valid range: 0.1–10m (clamp outside)

**Sector analysis**: 6 sectors across ±33° FOV, 10th-percentile robust depth per sector
- `NEAR_M = 1.5`: sector blocked if min_depth < 1.5m
- `FAR_M = 3.0`: sector safe if min_depth > 3.0m

**Escape direction**: safest open sector closest to straight-ahead. If all blocked, highest clearance.

**Latency**: 1–4ms on Pi 5, ~0ms on Mac (SGBM is fast at 96×96).

**Calibration**: Auto-loads `stereo_calib.npz` from `~/threatdetect/` or `./`.
- If loaded: uses `P0[0,0]` for focal length, applies `cv2.remap` rectification before SGBM
- If not loaded: assumes parallel cameras (usable but depth less accurate)
- Tool: `python3 tools/calibrate_stereo.py --squares 25 --cols 9 --rows 6 --captures 20`

**Stereo sim mode** (`--stereo-sim`): shifts frame 5px right → constant disparity → ~0.96m fake depth. Used for testing pipeline on single-camera setup.

---

## Fusion Modules (fusion/)

| Module | Purpose |
|--------|---------|
| `spatial_mapper.py` | Periodic Gemma spatial mapping, CJK decode, audio-visual fusion, scene-diff gate |
| `tracker.py` | IoU multi-object tracker, velocity/ETA/motion labels, max_tracks=6, MAX_AGE=5 |
| `router.py` | Cascade routing tier 0/1/2, proximity boost, score formula |
| `threat_memory.py` | Short-term threat persistence, exponential decay t½=2s |
| `predictor.py` | 10Hz haptic extrapolation from last MotionVector (background thread) |
| `reservoir.py` | Echo State Network (200 nodes) threat anticipator |
| `lsai.py` | Logit-bias steering for Gemma (17 concept vocabularies) |
| `stream_parser.py` | Streaming JSON parser for Gemma output |
| `sensor_encoder.py` | Builds text prompt from SensorSnapshot |
| `inference_profile.py` | URGENT/STANDARD/EXPLORATORY Gemma call profiles |
| `stereo_depth.py` | SGBM stereo depth, sector analysis, escape direction, optional calibration |

---

## Process Modules (process/)

| Module | Purpose |
|--------|---------|
| `vision.py` | `VisionProcessor` (single cam), `DualVisionProcessor` (dual cam), `MockVisionProcessor` |
| `audio.py` | `AudioProcessor` (sounddevice), `MockAudioProcessor`; 4 diagonal beams |
| `audio_mcp3008.py` | `MCP3008AudioProcessor`; SPI ADC 10kHz, 4ch, beamform via phase delay |

**DualVisionProcessor**: runs both cameras, calls `StereoDepthEstimator.compute()` on each frame pair.
**Optical flow**: `levels=1, winsize=11` (levels=2 negligible improvement at 224px).

---

## Output Modules (output/)

| Module | Purpose |
|--------|---------|
| `haptic.py` | GPIO PWM via lgpio; GPIO 24=left, 25=right; direction→split piecewise-linear |
| `web_ui.py` | Gradio dashboard at port 7860 |

### Web UI (current layout, post 2026-05-14)
- Gradio at port 7860, JS polling every 50ms (20fps effective)
- Light-mode hero board with warm-white surfaces, teal accents, and stronger text contrast
- **Main grid**: video feed (left) | spatial radar (right)
- **Secondary grid**: audio beams | haptic status
- **Stats strip**: compact live cards for active threats, beam peak, latest alert, and pipeline mode
- Radar: 400×400px canvas, sweep animation, YOLO dots (hollow), Gemma objects (filled), audio beam wedges
- Threat list: active objects with direction, distance, confidence, and urgency
- Audio: stereo beep on haptic events (panned by direction)

---

## main.py — Key Flags

```bash
--real              # use real cameras (Picamera2), else mock
--dual-cam          # open both cameras (requires cam0 + cam1)
--stereo-sim        # fake stereo by shifting single frame 5px (test mode)
--ui                # start Gradio web UI at port 7860
--video PATH        # use video file instead of live camera
--yolo-model PATH   # path to YOLO model (.onnx or .pt)
```

---

## Pi Start Commands

### llama-server (always start first)
```bash
~/start_llama.sh
# Expands to:
GGML_VULKAN_DISABLE=1 nohup taskset -c 2,3 ~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma4-e2b/e2b-smooth-q4_0.gguf \
  -ngl 0 --mlock -c 256 -t 2 --port 8081 --host 0.0.0.0 \
  --parallel 1 \
  --chat-template-file /home/ishaan/gemma4-nothink.jinja \
  --log-disable > /tmp/llama.log 2>&1 &
```

### Pipeline (video test + stereo sim + UI)
```bash
cd ~/threatdetect && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  nohup taskset -c 0,1 python3 -u main.py \
  --real --stereo-sim --ui \
  --yolo-model ~/threatdetect/yolo26n.onnx \
  --video /home/ishaan/test.mp4 >> /tmp/pipeline.log 2>&1 &
```

### Pipeline (dual cam, real stereo)
```bash
cd ~/threatdetect && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  nohup taskset -c 0,1 python3 -u main.py \
  --real --dual-cam --ui \
  --yolo-model ~/threatdetect/yolo26n.onnx >> /tmp/pipeline.log 2>&1 &
```

### Health check
```bash
curl http://localhost:8081/health   # → {"status":"ok"}
tail -f /tmp/pipeline.log
tail -f /tmp/llama.log
pgrep -fa python3  # confirm single pipeline instance
```

> **SSH quirk**: Use `kill PID` not `pkill -9 -f pattern` — pkill over SSH drops the connection.

### Stereo calibration (run once with checkerboard)
```bash
python3 tools/calibrate_stereo.py --squares 25 --cols 9 --rows 6 --captures 20
# Saves: ~/threatdetect/stereo_calib.npz
# Debug images: /tmp/calib_cam*.jpg
# Target RMS: <1.0px
```

---

## Mac Dev Commands

### llama-server (Mac Metal)
```bash
nohup /opt/homebrew/bin/llama-server \
  -m /Users/ishaangubbala/threatdetect/finetune/e2b-smooth-q4_0.gguf \
  -c 2048 --port 8081 --host 0.0.0.0 --parallel 1 \
  --chat-template-file ~/gemma4-nothink.jinja --log-disable > /tmp/llama_mac.log 2>&1 &
# --flash-attn OK on Mac Metal (not on Pi Q4_0)
```

### Pipeline (Mac, video + stereo sim)
```bash
cd /Users/ishaangubbala/threatdetect
/Users/ishaangubbala/miniconda/bin/python3 main.py --real --stereo-sim --ui \
  --yolo-model /Users/ishaangubbala/Documents/airesearch/yolov8n.pt \
  --video /Users/ishaangubbala/Downloads/15124736_2560_1440_30fps.mp4
# NOTE: use miniconda python (has cv2/ultralytics). /usr/local/bin/python3 does not.
```

---

## Fine-Tuning

### Active config: `finetune/lora_e2b_cjk_smooth.yaml`
```yaml
model: "/Users/ishaangubbala/models/gemma-4-E2B-it"
data: ".../finetune/data_cjk_smooth"
fine_tune_type: lora
lora_parameters: {rank: 16, scale: 16.0, dropout: 0.05}
keys: [q_proj, k_proj, v_proj, o_proj]
iters: 5000
learning_rate: 1.0e-4
batch_size: 4
adapter_path: ".../adapters_e2b_cjk_smooth"
```

### Data generator: `finetune/gen_cjk_smooth.py`
- Samples T and O independently (0–18 each) → uniform angle distribution
- Previous bug: `random.randint(0,359)` → biased mod-19 → 8.8° avg error
- Fixed: independent sampling → ~8.5° avg error (haptic motors can't convey <15°)

### Fuse + quantize pipeline
```bash
# 1. Fuse LoRA into base weights (MLX on Mac)
mlx_lm.fuse --model /path/to/gemma-4-E2B-it \
  --adapter-path adapters_e2b_cjk_smooth \
  --save-path e2b-smooth-fused --dequantize

# 2. Convert to GGUF
python3 convert_hf_to_gguf.py e2b-smooth-fused \
  --outfile e2b-smooth-f16.gguf --outtype f16

# 3. Quantize to Q4_0
llama-quantize e2b-smooth-f16.gguf e2b-smooth-q4_0.gguf Q4_0
```

Adapter path: `/Users/ishaangubbala/threatdetect/finetune/adapters_e2b_cjk_smooth/`  
Base model: `/Users/ishaangubbala/models/gemma-4-E2B-it`

---

## Key Design Decisions

**Why CJK encoding?** 8 tokens total for 2 objects with 1° precision. JSON takes 40–80 tokens. 5–10× faster generation on CPU-constrained Pi.

**Why T×19+O (not direct lookup)?** LoRA can't learn 8640-class arbitrary codebook — model collapsed. Arithmetic encoding is learnable: model extrapolates between training examples.

**Why no GBNF grammar?** Pi's llama-server build hangs on GBNF constraints. Robust CJK parser (with ASCII fallback) handles freeform output instead.

**Why SpatialMapper + YOLO cascade?** LLM inference too slow for every frame. YOLO runs every frame at 10fps for Tier-1 haptic. Gemma runs every 2s for spatial context. ThreatPredictor bridges at 10Hz.

**Why MCP3008 instead of I2S?** Pi 5 DW-I2S RX requires concurrent TX to generate clocks. INMP441 SD pin has hardware fault. MCP3008 + MAX4466 analog path works reliably.

**Why stereo at 96×96?** Matches YOLO input resolution — no extra resize. SGBM is O(W×H×D), so small resolution critical for real-time. At 96px width and 65mm baseline, 32 disparity levels cover 0.15–4.8m — exactly the obstacle-detection range needed.

**Why Q4_0 over Q4_K_M?** ARM's NEON path only vectorizes Q4_0 dequantization (`vdotq_s32`). Q4_K_M falls back to scalar = 2× slower on BCM2712.

**Why scene-diff gate?** Minor distance drift (3.3m → 3.5m) shouldn't trigger new Gemma call. Hash on urgency tier + 15° angle bucket → stable hash within tier → 60–80% cache hits in real deployment.

---

## Current Status (2026-05-14)

### Working
- Full pipeline: YOLO + MCP3008 audio + Gemma CJK spatial mapping + haptic output
- Pi llama-server: Q4_0 + `<|turn>` template → CJK output, ±10° direction accuracy
- YOLO ONNX FP32: 17ms direct ORT, async thread, YOLO_SKIP_N=3 with flow propagation
- Stereo depth: SGBM 1–4ms on Pi, sector analysis, escape direction, UI overlay
- Stereo sim mode: `--stereo-sim` flag, 5px shift → ~0.96m fake depth, verified end-to-end
- MCP3008: ch0, ch2, ch3 active (ch1 dead). Beamforming functional.
- Web UI: Gradio port 7860, light hero-board layout (video | radar, audio | haptic, threats), strong light-mode contrast

### Known Issues
- `llama-server` on port 8081 must be running separately; when it is down the UI still loads but Gemma-backed panels stay empty and the logs show connection refused.
- **INT8 ONNX broken on Pi ORT 1.26**: `yolo26n_int8.onnx` outputs all zeros (max_conf=0.0). Root cause: QDQ per-tensor quantization incompatibility with ORT 1.26 ARM. Active model is FP32 (17ms). INT8 was 6.3ms — restore if ORT upgraded.
- **Stereo calibration not done**: Real dual-cam stereo works but depth unreliable without `stereo_calib.npz`. Run calibration tool with checkerboard.
- **cam1 cable**: Repaired (was loose FPC connector). Dual-cam stereo physically functional.
- **MCP3008 ch1 dead**: hardware fault. Zeroed at startup. Beamforming uses 3 channels.
- **I2S INMP441**: software working, hardware SD pin fault. Not active.
- **Radar frozen under CPU contention**: Gradio websocket drops under heavy Pi load. Fix: browser refresh. Root cause: Pi CPU fully saturated (pipeline + UI on 2 cores).

### Next Steps
- Stereo calibration: run `calibrate_stereo.py` with 9×6 checkerboard, 20 poses
- Hackathon submission: writeup, video demo, HF Hub upload of fine-tuned model
- Consider ORT upgrade on Pi to re-enable INT8 (6.3ms vs 17ms)

---

## Hackathon Award Tracks

1. **Best wearable/mobile** — intelligent task routing YOLO→Gemma, real haptic output, <1s Tier-1 response, 10Hz continuous feedback
2. **Best resource-constrained hardware** — real-time pipeline on Pi 5 ($80), 3.2GB model, ~3s Gemma latency, 49× YOLO speedup from baseline
3. **Best Unsloth fine-tuned Gemma 4** — CJK smooth LoRA, custom data generator with uniform angle distribution, 8-token spatial encoding, ±10° accuracy

---

## File Tree

```
threatdetect/
├── main.py                        # Pipeline loop, argparse, cam/audio init
├── project.md                     # This file
├── CONTEXT.md                     # Original context doc (may lag project.md)
├── PERF.md                        # Performance optimization record
├── fusion/
│   ├── spatial_mapper.py          # Gemma spatial mapper, CJK decode, audio-visual fusion
│   ├── tracker.py                 # IoU tracker + velocity/ETA
│   ├── router.py                  # Cascade routing tier 0/1/2
│   ├── threat_memory.py           # Threat persistence + exponential decay
│   ├── predictor.py               # 10Hz haptic extrapolation
│   ├── reservoir.py               # ESN threat anticipator (200 nodes)
│   ├── lsai.py                    # Logit-bias steering
│   ├── stereo_depth.py            # SGBM stereo depth + sector nav
│   ├── inference_profile.py       # URGENT/STANDARD/EXPLORATORY profiles
│   ├── stream_parser.py           # Streaming JSON parser
│   └── sensor_encoder.py          # Text prompt builder
├── process/
│   ├── vision.py                  # VisionProcessor, DualVisionProcessor
│   ├── audio.py                   # AudioProcessor (sounddevice)
│   └── audio_mcp3008.py           # MCP3008AudioProcessor (SPI ADC)
├── output/
│   ├── haptic.py                  # GPIO PWM haptic driver
│   └── web_ui.py                  # Gradio dashboard (port 7860)
├── tools/
│   └── calibrate_stereo.py        # Stereo calibration (run on Pi w/ checkerboard)
├── finetune/
│   ├── lora_e2b_cjk_smooth.yaml   # Active LoRA config
│   ├── gen_cjk_smooth.py          # Training data generator
│   ├── e2b-smooth-q4_0.gguf       # Active model (3.2GB, Mac copy)
│   └── adapters_e2b_cjk_smooth/   # LoRA adapter weights
└── yolo26n_int8.onnx              # INT8 ONNX (BROKEN on ORT 1.26 Pi ARM)
```
