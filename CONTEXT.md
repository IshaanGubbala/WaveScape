# ThreatDetect — Full Project Context

## What It Is

Wearable real-time threat detection system for blind/deaf users. Runs on a Raspberry Pi 5 worn on the body. Detects vehicles, pedestrians, and environmental hazards using a camera + mic array, then outputs directional haptic alerts via vibration motors on the temples.

Target: **Gemma4 Good hackathon** (3 award tracks: best wearable, best resource-constrained hardware, best Unsloth fine-tuned Gemma 4 model).

---

## Hardware

### Raspberry Pi 5 (8GB RAM)
- SSH: `ishaan@192.168.68.107` (LAN) | `ishaan@100.99.239.20` (Tailscale)
- Password: `gubbala`
- SSH flag required: `-o StrictHostKeyChecking=no`
- Code: `~/threatdetect/`
- OS: Raspberry Pi OS, kernel 6.12.75+rpt-rpi-2712

### Cameras
- 2× Pi Camera Module 3 (IMX708, 66° FOV each)
- `cam0` = 330° (front-left), `cam1` = 30° (front-right)
- 126° total coverage, 6° front overlap
- Resolution used: 224×224 (captured natively, no resize), `imgsz=96` for YOLO ONNX

### Haptic Motors
- Left temple: **GPIO 24**
- Right temple: **GPIO 25**
- PWM at 200Hz via `lgpio`
- Direction-to-split: smooth piecewise-linear — 0°/180° = 50/50, 90° = all right, 270° = all left

### Microphone Array
- **MCP3008 SPI ADC** + 4× MAX4466 amplified electret mics
- SPI: `/dev/spidev10.0` (CE0)
- Sample rate: 10kHz per channel × 4 channels
- Beamforming: 4 diagonal beams at 35°, 145°, 215°, 325°
- Pi wiring: pin 17 (3.3V), 25 (GND), 23 (SCLK), 21 (MISO), 19 (MOSI), 24 (CE0)
- Mics → MCP3008 CH0–CH3, spacing ~4cm (`MIC_SPACING_M = 0.04`)
- `main.py` auto-detects: tries MCP3008 → sounddevice → mock

> **I2S INMP441 status**: Software fully working, hardware wiring fault on SD pin. All I2S work preserved in `~/dmic_fd/`. Not used in current stack.

---

## Models

### YOLO (always-on vision)
- Model: `yolo26n.onnx` (Pi, ONNX export) / `yolov8n.pt` (Mac dev)
- Input: 224×224 capture, `imgsz=96` (ONNX fixed shape), conf threshold 0.10
- Output: label, confidence, bbox, direction_deg, distance_m
- Inference: 70ms median on Pi 5 (vs 307ms PyTorch baseline)
- Async: background thread, YOLO_SKIP_N=3, optical flow propagates bboxes between frames
- ~9–11fps pipeline (video file), 14.3fps theoretical cap

### Gemma 4 E2B (spatial mapper)
- Base: `gemma-4-E2B-it` (2B params)
- Fine-tuned with LoRA (rank=16, scale=16, lr=1e-4, 5000 iters, MLX on Mac M5)
- **Active model**: `e2b-smooth-q4_0.gguf` (3.2GB Q4_0)
  - Path on Pi: `~/models/gemma4-e2b/e2b-smooth-q4_0.gguf`
  - Path on Mac: `/Users/ishaangubbala/threatdetect/finetune/e2b-smooth-q4_0.gguf`
- Served via `llama-server` on port 8081
- Inference: ~0.18s on Mac Metal, **~3s on Pi 5** (Q4_0, cores 2-3, pipeline running)
- Template: `~/gemma4-nothink.jinja` with `<|turn>` Gemma 4 tokens (NOT `<start_of_turn>`)
- Config: `-c 512 -t 2 --mlock -ngl 0 --parallel 1 --log-disable`
- **Q4_K_M built but NOT used**: 2× slower on ARM (no NEON path); at `~/models/gemma4-e2b/e2b-smooth-q4km.gguf`

### Model History (all in `finetune/`)
| Model | Status | Notes |
|-------|--------|-------|
| `gemma-3-1b-spatial` | abandoned | too small, hallucinated |
| `e2b-spatial` (CSV format) | abandoned | verbose output, slow |
| `e2b-cjk` | abandoned | O-char distribution bias (8.8° avg error) |
| `e2b-ultra` | abandoned | model collapse — LoRA can't learn 8640-class arbitrary codebook |
| `e2b-smooth` | **ACTIVE** | smooth angle encoding, 1° precision, 8 tokens/output |

---

## CJK Smooth Encoding

The active model outputs exactly 8 CJK characters: `T1 O1 L1 U1 T2 O2 L2 U2` for 2 closest objects.

```
angle_deg = T_index * 19 + O_index   (range 0–359, 1° precision)

T_CHARS = "一二三四五六七八九十甲乙丙丁戊己庚辛壬"  (tens, idx 0–18)
O_CHARS = "子丑寅卯辰巳午未申酉戌亥角亢氐房心尾箕"  (ones, idx 0–18)
L_CHARS = "车卡巴人自物"   → car truck bus person bike obstacle
U_CHARS = "危急中远"       → critical(<2m) high(<4m) medium(<8m) low(>=8m)
```

System prompt spells out both alphabets explicitly. Parser decodes via index lookup.

---

## Pipeline Architecture

```
Camera (224×224 @ ~49fps Mac / ~10fps Pi)
    │
    ▼
VisionProcessor (YOLO → detections: label, conf, bbox, dir, dist)
    │
    ▼
ObjectTracker (IoU matching → adds velocity_mps, eta_s, motion_label)
    │
    ▼
CascadeRouter (score = 0.5×vis + 0.25×aud + 0.15×motion + 0.1×pitch + prox_boost)
    │            Tier 0: no action
    │            Tier 1: direct haptic (score ≥ 0.62)
    │            Tier 2: was Gemma — now handled by SpatialMapper
    │
    ├──► AudioProcessor (50ms window, beamform_scan, classify every 5 frames)
    │
    ├──► SpatialMapper (every 2s OR on audio spike)
    │        │  Fuses YOLO + audio beams
    │        │  Priority filter: objects <8m OR velocity>0.3 m/s
    │        │  Sorts by (dist - 20 if approaching)
    │        │  Sends text prompt to Gemma → decodes 8 CJK chars → SpatialMap
    │        └──► haptic_fire() + web_ui.update_spatial()
    │
    ├──► ThreatPredictor (10Hz extrapolation between Gemma calls)
    │        Fires haptic based on last MotionVector
    │
    └──► HapticController (GPIO PWM or console fallback)
             web_ui.log_haptic()
```

### Key thresholds (`router.py`)
- Tier 1 (direct haptic): score ≥ 0.62
- Tier 2 (Gemma): score ≥ 0.18, cooldown 0.8s
- Proximity boost: <3m → +0.20, <6m → +0.10, <12m → +0.05

### FPS
- Mac dev: ~49fps (uncapped, no sleep throttle)
- Pi 5 camera: ~10fps
- Gemma fires: every 2s (or on audio spike >3× rolling avg)
- Web UI poll: 50ms (20fps)

---

## Fusion Modules (`fusion/`)

| Module | Purpose |
|--------|---------|
| `spatial_mapper.py` | Periodic Gemma spatial mapping, audio-visual fusion, CJK decode |
| `tracker.py` | IoU multi-object tracker, velocity/ETA/motion labels |
| `router.py` | Cascade routing score → tier 0/1/2 |
| `threat_memory.py` | Short-term threat persistence between Gemma calls |
| `predictor.py` | 10Hz haptic extrapolation from last MotionVector |
| `reservoir.py` | Echo State Network threat anticipator (200 nodes) |
| `lsai.py` | Logit-bias steering for Gemma (17 concept vocabularies) |
| `stream_parser.py` | Streaming JSON parser for Gemma output |
| `sensor_encoder.py` | Builds text prompt from SensorSnapshot |
| `inference_profile.py` | URGENT/STANDARD/EXPLORATORY Gemma call profiles |

---

## Output Modules (`output/`)

| Module | Purpose |
|--------|---------|
| `haptic.py` | GPIO PWM via lgpio; GPIO 24=left, 25=right; direction→split mapping |
| `web_ui.py` | Gradio dashboard: radar map + video feed + haptic log + mic beams |

### Web UI
- Gradio at port 7860
- Radar: 400×400px canvas, sweep animation, YOLO dots (hollow), Gemma objects (filled), beam wedges
- Poll rate: 50ms (20fps)
- Audio: stereo beep on haptic events (panned by direction)

---

## Process Modules (`process/`)

| Module | Purpose |
|--------|---------|
| `vision.py` | `VisionProcessor` (single cam), `DualVisionProcessor` (dual cam), `MockVisionProcessor` |
| `audio.py` | `AudioProcessor` (sounddevice), `MockAudioProcessor`; 4 diagonal beams |
| `audio_mcp3008.py` | `MCP3008AudioProcessor`; SPI ADC, 4ch 10kHz, beamform via phase delay |

---

## Pi Server Commands

### Start llama-server (smooth model)
```bash
~/start_llama.sh
# Contents:
GGML_VULKAN_DISABLE=1 nohup taskset -c 2,3 ~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma4-e2b/e2b-smooth-q4_0.gguf \
  -ngl 0 --mlock -c 512 -t 2 --port 8081 --host 0.0.0.0 \
  --parallel 1 \
  --chat-template-file /home/ishaan/gemma4-nothink.jinja \
  --log-disable > /tmp/llama.log 2>&1 &
```

> **Note**: llama.cpp was rebuilt without Vulkan (`-DGGML_VULKAN=OFF`) because the Pi 5's GPU shared memory was too small. Original Vulkan `.so` files backed up in `~/vulkan_backup/`.

> **Critical**: `~/gemma4-nothink.jinja` MUST use Gemma 4's `<|turn>` token format (NOT Gemma 3's `<start_of_turn>`). Wrong template → ASCII garbage output instead of CJK.

### Start pipeline (video test)
```bash
nohup /tmp/run_pipeline.sh &>/dev/null &
# Contents:
cd /home/ishaan/threatdetect
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 exec nohup \
  taskset -c 0,1 python3 -u main.py --real --ui \
  --yolo-model /home/ishaan/threatdetect/yolo26n.onnx \
  --video /home/ishaan/test.mp4 >> /tmp/pipeline.log 2>&1
```

### Start pipeline (dual cam, real hardware)
```bash
cd ~/threatdetect && OMP_NUM_THREADS=2 nohup taskset -c 0,1 python3 -u main.py \
  --real --dual-cam --ui \
  --yolo-model /home/ishaan/threatdetect/yolo26n.onnx > /tmp/pipeline.log 2>&1 &
```

> **SSH quirk**: Use `kill PID` not `pkill -9 -f pattern` — pkill over SSH drops the connection.

### Check health
```bash
curl http://localhost:8081/health   # → {"status":"ok"}
tail -f /tmp/pipeline.log
tail -f /tmp/llama.log
```

---

## Mac Dev Commands

### Start llama-server (Mac Metal)
```bash
nohup /opt/homebrew/bin/llama-server \
  -m /Users/ishaangubbala/threatdetect/finetune/e2b-smooth-q4_0.gguf \
  -c 2048 --port 8081 --host 0.0.0.0 --parallel 1 \
  --chat-template-file ~/gemma4-nothink.jinja --log-disable > /tmp/llama_mac.log 2>&1 &
# Note: Mac uses Homebrew llama-server (Metal GPU). Same nothink.jinja template required.
# --flash-attn ON is fine on Mac Metal. Do NOT use with Q4_0 on Pi CPU (breaks output).
```

### Start pipeline (Mac, video)
```bash
cd /Users/ishaangubbala/threatdetect
python3 main.py --real --ui \
  --yolo-model /Users/ishaangubbala/Documents/airesearch/yolov8n.pt \
  --video /Users/ishaangubbala/Downloads/15124736_2560_1440_30fps.mp4 \
  > /tmp/pipeline.log 2>&1 &
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

### Data generation: `finetune/gen_cjk_smooth.py`
- Samples T and O independently (0–18 each) to ensure uniform angle distribution
- Previous bug: `random.randint(0,359)` → biased mod-19 distribution → 8.8° avg error
- Fixed: independent sampling → ~8.5° avg error (acceptable — haptic motors can't convey <15°)

### Fuse + quantize pipeline
```bash
# Fuse LoRA into base weights
mlx_lm.fuse --model /path/to/gemma-4-E2B-it \
  --adapter-path adapters_e2b_cjk_smooth \
  --save-path e2b-smooth-fused --dequantize

# Convert to GGUF
python3 convert_hf_to_gguf.py e2b-smooth-fused \
  --outfile e2b-smooth-f16.gguf --outtype f16

# Quantize to Q4_0
llama-quantize e2b-smooth-f16.gguf e2b-smooth-q4_0.gguf Q4_0
```

---

## Key Design Decisions

**Why CJK encoding?** Dense token output — 8 tokens total for 2 objects with 1° angle precision. Standard JSON output takes 40–80 tokens. 5–10× faster generation on CPU-constrained Pi 5.

**Why smooth (T×19+O) instead of ultra (1 char per object)?** Ultra encoding requires an 8640-class arbitrary lookup that LoRA attention layers can't learn — model collapsed to repetitive output. Smooth encoding is learnable because angle is arithmetic (T×19+O), not a lookup table.

**Why remove GBNF grammar?** Pi's llama-server build hangs on GBNF constraints. Robust CJK parser handles freeform output instead.

**Why spatial mapper instead of per-frame Gemma?** LLM inference is too slow for every frame. Spatial mapper decouples: YOLO runs every frame at 10fps for fast Tier-1 haptic, Gemma runs every 2s for spatial context. ThreatPredictor extrapolates between Gemma calls at 10Hz.

**Why MCP3008 instead of I2S?** Pi 5's DW-I2S RX requires concurrent TX to generate clocks. INMP441 SD pin had a hardware wiring fault and chip may be damaged. MCP3008 + MAX4466 analog path works reliably.

---

## Current Status (2026-05-13)

### Working
- Full pipeline: YOLO + MCP3008 audio + Gemma CJK spatial mapping + haptic output
- Pi llama-server: Q4_0 + `<|turn>` template → CJK output, ±10° direction accuracy
- ONNX YOLO: 70ms median, async thread, frame skip N=3 with flow propagation
- MCP3008: ch0, ch2, ch3 active (ch1 dead, hardware fault). Beamforming functional.
- Web UI: Gradio at port 7860, radar + video + haptic log

### Known Issues
- **cam1 hardware timeout**: dual-cam mode fails because cam1 FPC connector is loose/damaged. Use single-cam (`--real`, no `--dual-cam`).
- **MCP3008 ch1 dead**: hardware fault (stuck near rail). Auto-detected and zeroed on startup. Beamform uses 3 active channels.
- **I2S INMP441**: software working, hardware SD wire fault. All I2S code in `~/dmic_fd/`. Not used.
- **Q4_K_M slower**: built at `~/models/gemma4-e2b/e2b-smooth-q4km.gguf` but 2× slower than Q4_0 on ARM (no NEON dequant path). Not active.
- **flash-attn on Pi**: `--flash-attn on` breaks Q4_0 output on ARM CPU. Do NOT use.

### Next Steps
- Hackathon submission: writeup, video demo, HF Hub upload of fine-tuned model
- cam1 connector check (hardware)

---

## Hackathon Awards Targeted

1. **Best wearable/mobile** — intelligent task routing between YOLO (Tier 1) and Gemma (Tier 2/spatial)
2. **Best resource-constrained hardware** — real-time pipeline on Pi 5, 3.2GB Q4_0 model, <2s Gemma latency
3. **Best Unsloth fine-tuned Gemma 4** — CJK smooth LoRA, custom data generator, domain-specific compression

---

## File Tree (key files)

```
threatdetect/
├── main.py                     # Main pipeline loop
├── CONTEXT.md                  # This file
├── fusion/
│   ├── spatial_mapper.py       # Gemma spatial mapper, CJK decode, audio-visual fusion
│   ├── tracker.py              # IoU tracker + velocity/ETA
│   ├── router.py               # Cascade routing tier 0/1/2
│   ├── predictor.py            # 10Hz haptic extrapolation
│   ├── reservoir.py            # ESN threat anticipator
│   └── lsai.py                 # Logit-bias steering
├── process/
│   ├── vision.py               # VisionProcessor, DualVisionProcessor
│   ├── audio.py                # AudioProcessor (sounddevice)
│   └── audio_mcp3008.py        # MCP3008AudioProcessor (SPI ADC)
├── output/
│   ├── haptic.py               # GPIO PWM haptic driver
│   └── web_ui.py               # Gradio dashboard
└── finetune/
    ├── lora_e2b_cjk_smooth.yaml  # Active LoRA config
    ├── gen_cjk_smooth.py         # Training data generator
    └── e2b-smooth-q4_0.gguf      # Active model (3.2GB)
```
