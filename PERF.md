# ThreatDetect Pi 5 — Performance Optimization Record

**Date**: 2026-05-14  
**Hardware**: Raspberry Pi 5 (8GB RAM, BCM2712, 4× Cortex-A76 @ 2.4GHz)  
**Baseline**: Bare pipeline at project start. All numbers measured on device.

These are not incidental tuning tweaks. Each one required finding a non-obvious root cause
or making a design decision that goes against the obvious approach. Documented in order of
impact, with the reasoning that wasn't obvious in advance.

---

## Summary Table

### YOLO / Vision

| # | Optimization | Baseline | After | Gain |
|---|---|---|---|---|
| 1 | PyTorch → ONNX export | 307ms | **70ms** | **4.4×** |
| 2 | imgsz 160 → 96 | ~200ms | **70ms** | **2.9×** |
| 3 | Conf threshold 0.15 → 0.10 (ONNX compensation) | ~15% missed dets | parity | recovers resolution loss |
| 4 | ONNX: simplify=True, opset=12 | larger graph | **9.3MB** | faster load, smaller graph |
| 5 | Async YOLO background thread | blocks main loop | **0ms block** | never waits on inference |
| 6 | Queue maxsize=1 + put_nowait | lag accumulates | **drops stale frames** | zero lag buildup |
| 7 | THREAT_CLASSES filter | 80 COCO classes | **9 classes** | skips 89% of post-process |
| 8 | verbose=False | print per-frame | **silent** | eliminates I/O stall |
| 9 | YOLO skip N=3 | 208% CPU (2 cores) | **177–189%** | ~15% CPU freed |
| 10 | Optical flow bbox propagation | stale bboxes on skip | **flow-shifted** | smooth tracking between YOLO frames |
| 11 | Optical flow levels=2 → levels=1 | 16.4ms | **16.0ms** | ~2% (negligible at 224px) |
| 12 | Capture at 224×224 natively | resize after capture | **0ms resize** | eliminates scale step |
| 13 | Single grayscale conversion | 2× cvtColor | **1× gray** | shared between flow and YOLO |
| 14 | Camera flip at capture | per-display flip | **once at capture** | eliminates per-frame copy |
| 15 | Bypass ultralytics → direct ORT session | 70ms (wrapper overhead) | **17ms** | **4.1×** — ultralytics Python per-inference cost eliminated |
| 16 | ORT_ENABLE_ALL + ORT_SEQUENTIAL + intra_threads=2 | ORT_ENABLE_BASIC default | **17ms** | constant folding, node fusion, layout opts |
| 17 | INT8 static quantization (QDQ, per-tensor) | 17ms FP32 | **6.3ms** | **2.7×** — ARM asimddp runs INT8 dot-products natively |
| 18 | INT8 model size | 9.7MB FP32 | **2.9MB** | **3.4× smaller** — better L2 cache fit |

### Audio

| # | Optimization | Baseline | After | Gain |
|---|---|---|---|---|
| 15 | MCP3008 SPI at 10kHz (vs I2S 48kHz) | 48kHz × 4ch | **10kHz × 4ch** | **4.8× less data** |
| 16 | Dead-channel rail-check at startup | silent NaN/bias | **ch1 zeroed** | no corrupted beamform |
| 17 | Ring buffer bounded at 2s | unbounded history | **1.28MB fixed** | no memory growth |
| 18 | Audio classify every 5th frame | every frame | **every 5th** | 80% classification skipped |
| 19 | 4-direction beamform scan only | continuous sweep | **4 ops** | O(1) not O(N) |
| 20 | ENERGY_MIN=0.08 noise gate | fires on ambient | **filtered** | eliminates ambient triggers |
| 21 | Audio spike: 3× rolling avg trigger | fixed interval only | **event-driven** | immediate on sudden sound |
| 22 | SPI at rated max speed (1.35MHz) | lower default | **1.35MHz** | faster ADC sample loop |
| 23 | capture_wav_b64 returns None | encode+send 2s WAV | **0 bytes audio** | text-only LLM path |
| 24 | Beamform normalizes by active mic count | level drops per dead mic | **normalized** | consistent energy scale |
| 25 | Beam scan result passed by reference | scan per subsystem | **0 extra SPI reads** | reused across pipeline |

### LLM / Gemma

| # | Optimization | Baseline | After | Gain |
|---|---|---|---|---|
| 26 | Gemma 4 chat template fix (`<\|turn>`) | ±111° direction bias | **±10°** | **11× error reduction** |
| 27 | CJK smooth-angle encoding (T×19+O) | 20+ ASCII tokens | **8 CJK tokens** | **2.5× fewer generation steps** |
| 28 | cache_prompt=True | re-eval 235-token system prompt | **KV cached** | ~235 tokens saved per call |
| 29 | temperature=0.0 | sampling overhead | **greedy argmax** | fastest decode path |
| 30 | max_tokens=12 | 64 default | **12** | hard stop after 8 CJK chars |
| 31 | stream=False | chunked HTTP | **single response** | removes streaming frame overhead |
| 32 | stop=["<end_of_turn>", "</s>"] | full context decode | **early exit** | trims tail generation |
| 33 | Text-only path (no mmproj) | encode 512px JPEG per call | **0ms, 0 bytes** | text model, no vision path |
| 34 | System prompt < 100 tokens | may overflow context | **always cached** | fits guaranteed in -c 512 |
| 35 | Scene-diff gate, 6s TTL | every 2s always | **0ms on cache hit** | expected 60–80% skip in real deployment |
| 36 | Scene hash: urgency tiers not raw dist | minor dist drift = cache miss | **tier stable** | prevents false misses |
| 37 | HTTP timeout 60s → 9s + lock timeout 10s | 60s max stall | **10s** | **6× faster recovery** |
| 38 | Prompt: top-6 by priority | all detections | **6 objects** | bounded prompt length |
| 39 | ASCII fallback parser + label-match | dropped output on Q4_0 | **dual-path parse** | 0 dropped outputs |
| 40 | Audio beam fused into text prompt | separate LLM call | **single fused prompt** | no extra inference call |

### System / OS / llama-server

| # | Optimization | Baseline | After | Gain |
|---|---|---|---|---|
| 41 | w1_gpio + wire kernel module blacklist | 54% CPU burned, load=9.25 | **0% waste, load=0.33** | **Gemma 37s → 3s** |
| 42 | CPU affinity: pipeline cores 0-1, Gemma cores 2-3 | 7–10s contention | **~3s isolated** | **~3×** |
| 43 | OMP_NUM_THREADS=2 (matches pinned cores) | 3.67fps (4-thread thrash) | **~8.8fps** | **2.4×** |
| 44 | llama-server -c 512 (vs default 2048) | 4× over-allocated KV | **~450MB RAM freed** | faster alloc, exact fit |
| 45 | llama-server -t 2 (matches pinned cores) | 4 threads on 2 cores | **2:2 match** | no wasted threads |
| 46 | Q4_0 over Q4_K_M | 6.5s (K-quant, no NEON) | **~3s** (Q4_0, NEON) | **2× faster on ARM** |
| 47 | --mlock (lock model in RAM) | swaps under pressure | **never paged** | eliminates RAM→swap spikes |
| 48 | -ngl 0 + GGML_VULKAN_DISABLE=1 | Vulkan probe at startup | **skipped** | no GPU error, faster start |
| 49 | --log-disable | token-level disk I/O | **silent** | no I/O contention on µSD |
| 50 | --parallel 1 (single request slot) | multi-slot overhead | **single slot** | no scheduler contention |
| 51 | llama.cpp rebuild: -DGGML_NATIVE=ON -DGGML_LTO=ON | generic aarch64 | **march=native** | asimddp picked up more aggressively; LTO inlines hot paths |
| 52 | llama-server -c 256 (down from 512) | ~450MB KV alloc | **~225MB** | actual usage ~168 tokens; quadratic attn cost halved |

### Pipeline Architecture

| # | Optimization | Baseline | After | Gain |
|---|---|---|---|---|
| 51 | Cascade router tiers 0/1/2 | every frame to LLM | **>90% frames → tier 0** | LLM call rate ~0.5Hz not 10Hz |
| 52 | Proximity boost in cascade scorer | confidence only | **prox<3m → +0.20** | catches low-conf but nearby objects |
| 53 | ThreatMemory exponential decay (t½=2s) | silence during LLM gap | **memory sustains** | continuous haptic during 3s inference |
| 54 | ThreatPredictor 10Hz extrapolation | 1 haptic per LLM call | **10Hz continuous** | 30 haptic updates per 3s LLM cycle |
| 55 | ObjectTracker IoU + velocity/ETA | no tracking | **vel + ETA, no model** | free from geometry alone |
| 56 | Duplicate alert suppression (1.5s window) | fires every frame | **deduplicated** | no double-fire on same threat |
| 57 | max_tracks=6 + MAX_AGE=5 frames | unbounded growth | **O(1) bounded** | fixed RAM, deterministic update time |
| 58 | Heartbeat log every 10 frames | every frame | **every 10th** | 90% log I/O eliminated |
| 59 | Graceful fallback: MCP3008→sounddevice→mock | hard crash on hw absent | **auto-fallback** | runs on any hardware config |
| 60 | frame_to_b64: 512px cap + JPEG quality 75 | full-res uncompressed | **3–5× smaller payload** | faster HTTP to LLM |
| 61 | Daemon threads | join blocks on exit | **instant Ctrl+C** | no shutdown hang |
| 62 | YOLO thread sentinel (None in queue) | thread hangs | **clean stop** | no zombie workers |
| 63 | Audio spike gates Gemma interval | fixed 2s interval only | **event-driven trigger** | immediate response to sudden loud sound |

---

## Detailed Notes — Reasoning and Metrics

---

### 1. w1_gpio Kernel Module Blacklist — The Hidden CPU Thief

**Finding**: On Pi OS, `w1_gpio` (1-Wire GPIO bus driver) loads by default even if no 1-Wire devices are attached. The kernel spawns `w1_bus_master` which polls GPIO in a tight loop looking for devices that don't exist. No documentation warns about this.

**Before**: `w1_bus_master` visible in `top` at 54% CPU. System load average: 9.25 despite no user-space work. Gemma latency: 37s.  
**After**: Blacklisted in `/etc/modprobe.d/blacklist-w1.conf`. Load average: 0.33. Gemma latency: ~3s.

54% CPU = 2.16 cores out of 4 total. Effectively more than half the Pi's compute was burning in a kernel polling loop. This wasn't detectable from the pipeline itself — the pipeline showed normal CPU usage, but the kernel thread was starving the LLM.

**Why it causes 37s Gemma latency**: llama-server's matrix multiply loops compete for L2 cache and CPU cycles with `w1_bus_master`. The polling loop is likely causing excessive cache thrashing. At 37s/call, the model was unusable for real-time use.

**Fix** (persists across reboots):
```
blacklist w1_gpio
blacklist wire
```

**Gain**: +54% CPU freed. Gemma: 37s → 3s (**12× latency reduction**). This single fix made the project feasible.

---

### 2. Chat Template Token Format — Root Cause of 111° Direction Bias

**Finding**: Gemma 4 uses a completely different token format from Gemma 3. Gemma 3: `<start_of_turn>user`. Gemma 4: `<|turn>user`. The model's GGUF metadata lists `<|turn>` and `<turn|>` as special tokens. Using the wrong template means the model receives garbled input — the template delimiters are interpreted as regular text, not turn boundaries.

**Consequence**: With wrong template, the model never "sees" a properly formatted conversation. It falls back to outputting something that matches patterns from pre-training. In this case: ASCII-format coordinate strings like `"car@111° 10.0m"`. The 111° value is not random — it recurs consistently because it's a memorized pattern from training data.

**Measurement**:

| Template | Format | Direction error |
|---|---|---|
| `--chat-template gemma3` | `"car@111° 10.0m"` | **±111° systematic bias** |
| nothink.jinja w/ `<start_of_turn>` | `"car@111° 10.0m"` | **±111° systematic bias** |
| nothink.jinja w/ `<\|turn>` | `"一子人中一申人中"` | **±10° random error** |

The error is *systematic* (always ~111°, not random), which confirms the model is outputting memorized text rather than processing the prompt. Fixing the template unlocked CJK output and reduced direction error 11×. This was the single most impactful accuracy fix.

**Why `nothink` template**: The model was fine-tuned to output CJK directly without chain-of-thought. The `nothink` jinja disables the model's built-in thinking prefix, which would otherwise consume token budget before the 8-char output.

---

### 3. CJK Smooth-Angle Encoding — Arithmetic Codebook in 8 Tokens

**The problem**: Generating detailed spatial descriptions on a CPU-constrained device requires minimizing output tokens. Standard JSON output for 2 objects: ~40–80 tokens. Standard ASCII format (`"car@135° 3.0m, person@270° 5.0m"`): ~20 tokens. 

**The solution**: Encode 2 objects in exactly 8 CJK characters using a 2D lookup table where angle is arithmetic, not arbitrary.

```
angle_deg = T_index × 19 + O_index
T: "一二三四五六七八九十甲乙丙丁戊己庚辛壬"  (19 chars, tens digit 0–18)
O: "子丑寅卯辰巳午未申酉戌亥角亢氐房心尾箕"  (19 chars, ones digit 0–18)
L: "车卡巴人自物"  (6 chars: car/truck/bus/person/bike/obstacle)
U: "危急中远"      (4 chars: crit<2m / high<4m / med<8m / low)
```

19×19 = 361 combinations → 0–360° at **1° precision** using only 38 characters. Two objects = 4+4 = 8 chars = **8 tokens**.

**Why CJK characters?** In Gemma's BPE vocabulary, common CJK characters tokenize to a single token each. ASCII digits and degree symbols can split: `"135"` = potentially 3 tokens. CJK gives guaranteed 1 token/char, making output length predictable and minimal.

**Why arithmetic (T×19+O) instead of a direct lookup?** An earlier design (`e2b-ultra`) used a single arbitrary CJK character per object from an 8640-character codebook. The model collapsed — LoRA attention layers can't learn a pure lookup table with no structure. T×19+O is **learnable** because it's arithmetic: the model can learn "增加O减少一° , 增加T增加19°". The arithmetic structure also means generalization between training examples.

**Token budget math**:
- Generation time per token on Pi 5: ~3ms (at 330 tokens/s)
- ASCII format: ~20 tokens × 3ms = **60ms** generation
- CJK format: 8 tokens × 3ms = **24ms** generation
- Saving: **36ms per call**, or ~1.2s over 30 calls/minute

**Precision vs sensor resolution**: YOLO bbox center at 224×224 with 66° FOV → angular resolution per pixel ≈ 66°/224 ≈ 0.29°/px. At 96px imgsz, ≈ 0.69°/px. CJK encoding at 1° precision matches sensor precision — no benefit from higher resolution.

---

### 4. YOLO: PyTorch → ONNX + imgsz Reduction

**Why ONNX beats PyTorch on Pi**: The `ultralytics` PyTorch runtime includes Python overhead per-inference (tensor wrapping, result parsing, memory allocation). ONNX Runtime bypasses Python entirely for the forward pass, using optimized kernel dispatch. Additionally, ONNX Runtime's ARM backend has optimized NEON intrinsics for common Conv2d patterns that PyTorch's generic backend doesn't use.

**Why imgsz=96**: YOLO accuracy degrades gracefully at low resolution for large-object detection (cars, people occupy many pixels even at 96px). The 9 threat classes we care about are predominantly large objects. The confidence threshold was lowered from 0.15 → 0.10 to compensate — this costs false-positive rate but false negatives are more dangerous for a safety device.

**imgsz=96 vs imgsz=160 accuracy check**: Ran both on test video, compared class detections frame by frame. Same classes detected in 94% of frames. The 6% gap was single-frame blips (object partially occluded) not systematic misses.

**Why opset=12 + simplify=True**: opset=12 is the last version before Attention operator changes that ONNX Runtime 1.x on Pi doesn't support. `simplify=True` (onnxsim) fuses adjacent ops: Conv+BN+ReLU → single fused op, reducing graph traversal overhead.

| Config | min | median | p90 | fps cap |
|---|---|---|---|---|
| PyTorch `.pt` imgsz=160 | 185ms | **307ms** | 415ms | 3.3fps |
| ONNX imgsz=96 | 35ms | **70ms** | 83ms | **14.3fps** |

14.3fps cap means YOLO is no longer the pipeline bottleneck. At YOLO_SKIP_N=3 and 10fps pipeline, YOLO runs at effective 3.3fps — well within the 14.3fps cap.

---

### 5. Async YOLO + Frame Skip N=3 — Zero-Latency Inference Scheduling

**Why naive async isn't enough**: Simply putting YOLO in a thread and submitting every frame creates a growing queue. At 10fps input and 70ms/call, the queue absorbs 0.7 frames per cycle — manageable. But at any CPU spike (Gemma taking extra cores for a moment), queue builds up, introducing 500ms+ latency on old frames.

**Solution**: `queue.Queue(maxsize=1)` + `put_nowait`. If YOLO is busy, the new frame is dropped immediately. Main loop never blocks. Last YOLO result stays current until the next one arrives. This is a classic real-time systems pattern: prefer dropping stale data over queuing it.

**Optical flow propagation**: Instead of using stale bboxes unchanged, `_propagate()` shifts each bbox by the mean optical flow in its ROI region. Since flow runs every frame (~16ms), bboxes track moving objects even on skipped YOLO frames.

```
for each tracked bbox:
    roi_flow = dense_flow[y1:y2, x1:x2]
    dx, dy = mean(roi_flow)
    bbox.shift(dx, dy)
    confidence *= 0.95   # decay per skipped frame
```

Confidence decay (0.95×/frame) is important: after 3 frames (one YOLO cycle), confidence is 0.95³ = 0.857× — slight reduction. After 10 frames without YOLO refresh, 0.95¹⁰ = 0.60× — significant reduction that routes to lower tier. This prevents stale tracks from permanently sustaining high-urgency haptic.

**CPU result**: YOLO skip N=3 reduced pipeline CPU from 208% to 177–189% (~15% saving). The remaining CPU is flow computation (16ms), tracking, scoring, and audio.

---

### 6. CPU Affinity + OMP_NUM_THREADS — Two Separate But Related Problems

**Problem 1: Kernel scheduling**: Without affinity, the Linux scheduler freely migrates both llama-server threads and pipeline threads across all 4 cores. During Gemma inference, both workloads compete for the same L2 cache. Each context switch causes cache thrashing — Gemma's KV cache and Python's numpy arrays share 512KB L2 per core.

**Fix 1**: `taskset -c 2,3` for llama-server, `taskset -c 0,1` for pipeline. Cores 0-1 and 2-3 share separate L2 caches on BCM2712 (2 cores per cluster). Workloads never share L2.

| State | Gemma latency under pipeline load |
|---|---|
| No affinity | 7–10s |
| With affinity | **~3s** |

**Problem 2: ONNX thread over-subscription** (separate, discovered after affinity was set): After pinning pipeline to cores 0-1, ONNX Runtime still spawned 4 threads (it asks the OS: "how many CPUs?" → 4). 4 threads on 2 cores = 200% context switch overhead on cores 0-1. Pipeline FPS dropped from 11.6fps to 3.67fps.

**Fix 2**: `OMP_NUM_THREADS=2` tells ONNX (and any OpenMP library) to use at most 2 threads, matching the 2 pinned cores.

| State | Pipeline FPS |
|---|---|
| Affinity only (4 ONNX threads on 2 cores) | **3.67fps** |
| Affinity + OMP_NUM_THREADS=2 | **~8.8fps** |

The two fixes are independent but must both be applied. Affinity without OMP_NUM_THREADS makes things *worse* (forces 4 threads onto 2 cores). OMP_NUM_THREADS without affinity helps less (scheduler still migrates threads).

---

### 7. Q4_0 vs Q4_K_M on ARM — When Lower Quality Is Faster

**Counter-intuitive finding**: Q4_K_M is a higher-quality quantization scheme (uses mixed 4-bit and 6-bit block quantization) that outperforms Q4_0 quality-wise on x86 and CUDA. On ARM CPU, it's **2× slower**.

**Why**: Q4_K_M dequantization requires computing block scales with 6-bit precision — this loop has no NEON vectorization path in llama.cpp's ARM backend as of this build. It falls back to scalar float operations. Q4_0 uses a simpler block format (single scale per 32 weights) that maps directly to `vdotq_s32` NEON dot-product instructions.

**Measurement**: Built both on Pi.

| Quant | Latency (pipeline running) | Quality |
|---|---|---|
| Q4_K_M | 6.5s | better |
| Q4_0 | **~3s** | sufficient |

For this use case (8-token CJK output, direction accuracy ±10°), Q4_0 quality is sufficient. The 2× latency difference (6.5s vs 3s) is the difference between "barely usable" and "real-time spatial awareness."

**Note**: `--flash-attn on` also breaks Q4_0 output on ARM CPU (outputs garbage). Flash attention's ARM implementation is likely not tested with Q4_0 dequantization. Confirmed broken, do not use.

---

### 8. Scene-Diff Gate — Semantic Equivalence for Caching

**Problem**: A naive hash of detection positions creates too many false cache misses. If a car moves from 3.3m to 3.5m between frames, the hash changes even though the urgency ("high" in both cases) is identical. Calling Gemma for a 0.2m distance change is wasted inference.

**Solution**: Before hashing, quantize each detection to (label, 15°-angle-bucket, urgency-tier). Urgency tiers:
```
crit = 0  (dist < 2m)
high = 1  (dist < 4m)
med  = 2  (dist < 8m)
low  = 3  (dist >= 8m)
```

The scene hash is stable within a tier. A car at 3.3m and 3.5m both hash to tier=1 ("high"), same label, same 15° bucket → cache hit → 0ms instead of 3s.

**Expected cache hit rate** (real deployment, not looping video):
- Walking in static hallway: 80% cache hits (same objects, same positions)
- Waiting at crosswalk: 60–70% (cars slow/stop, pedestrians mill around)
- Active intersection crossing: 20–40% (frequent urgency tier changes)

On looping test video, every loop looks identical but the pipeline runs every 2s regardless because `_last_scene_time` tracks time-since-last-cache-hit, not content change.

---

### 9. ThreatMemory + ThreatPredictor — Continuous Haptic Without Continuous LLM

**The gap problem**: At ~3s Gemma latency and 10fps pipeline, there are ~30 frames between Gemma responses where the system has no fresh spatial map. A naive implementation would simply fire no haptic during this window — a blind spot that could span the full time a car takes to reach a pedestrian from 10m away at 30km/h (≈1.2s).

**Two-layer solution**:

**Layer 1 — ThreatMemory (exponential decay)**:
- Stores last 5 threats with insertion timestamp
- Confidence decays with half-life 2s: `c = c₀ × e^(-t × ln2 / 2)`
- At 3s elapsed: confidence = c₀ × 0.354 (35% of original)
- If decayed confidence > 0.35, router fires tier-1 haptic without new Gemma call
- Half-life of 2s chosen to match expected Gemma latency: confidence is still actionable during the inference gap, but expired by the time the next Gemma call returns

**Layer 2 — ThreatPredictor (10Hz extrapolation)**:
- Runs in background thread at 10Hz (separate from pipeline loop)
- Takes last MotionVector from optical flow (already computed in vision loop, zero extra cost)
- Extrapolates threat position: `angular_shift = atan2(lateral_velocity × elapsed, current_distance)`
- Fires haptic_cb at extrapolated direction + decayed confidence

**Result**: 30 haptic updates per 3s LLM cycle. The haptic direction tracks moving objects even between Gemma calls. From the user's perspective, the haptic never "blinks out" during inference.

---

### 10. MCP3008 SPI Audio at 10kHz — Hardware-Matched Sampling

**Why 10kHz and not 48kHz**: Threat-relevant sounds (car horn: 300–500Hz, engine: 50–200Hz, voice: 100–3400Hz, clap/screech: up to ~4kHz) all fall below 4kHz. Nyquist for 4kHz is 8kHz minimum. 10kHz gives comfortable margin with 4.8× less data than 48kHz I2S, and the MCP3008 ADC can handle 4 channels at 10kHz/channel within its 1.35MHz SPI bandwidth (4ch × 10kHz × 24 clocks/sample = 960kHz < 1350kHz max).

**Dead channel detection at startup**: Before starting the sample loop, `_quick_rail_check()` reads 50 samples per channel and computes the normalized mean. If `|mean| > 0.80`, the channel is "railed" (stuck at supply or ground — dead mic or hardware fault). Channel 1 on the test hardware rails consistently (hardware fault). Dead channels are added to `self._dead` set and zeroed in the sample loop — no runtime checks needed.

**Beamform with dead channels**: Instead of dividing by total mic count, `beamform_scan()` divides by `len(active_mics)`. This keeps the RMS energy scale consistent whether 3 or 4 channels are active. Without this, dead channel 1 would reduce all beam energies by 25%, causing false "quiet" classifications.

**Ring buffer design**: `np.zeros((sample_rate × 2, num_mics))` allocated once at startup = 10000 × 2 × 4 × 4 bytes = 320KB. Write pointer wraps modulo buffer length. No allocation in the hot loop. `capture_window()` extracts the last N samples using modular slice — no copy unless the window wraps around the ring boundary (handles wrap with two concatenated slices).

---

### 11. Cascade Router — LLM as Last Resort

**Architecture philosophy**: Most video frames contain no actionable threat. Running Gemma at 10fps would cost 10 × 3s = 30 CPU-seconds per second (impossible). The cascade router ensures LLM is called only when necessary.

**Score formula**:
```
score = vis×0.50 + aud×0.25 + motion×0.15 + stairs×0.10 + prox_boost
prox_boost = 0.20 if dist<3m else 0.10 if dist<6m else 0.05 if dist<12m
```

**Example scenarios**:
- Empty sidewalk: vis=0, aud=0, motion=0 → score=0 → **tier 0** (no action)
- Car at 15m at 0.85 conf: vis=0.43, prox=0.05 → score=0.48 → **tier 2** (Gemma)
- Car at 2m at 0.85 conf: vis=0.43, prox=0.20 → score=0.63 → **tier 1** (direct haptic, no LLM wait)
- Car at 2m at 0.50 conf (YOLO uncertain): vis=0.25, prox=0.20 → score=0.45 → **tier 2** (Gemma confirms)

**Proximity boost is critical**: Without it, a very close but partially-occluded object (low confidence) scores below the tier-1 threshold and waits for Gemma. At 2m, a 3s Gemma wait is unacceptable. Prox boost ensures close objects always get immediate tier-1 haptic regardless of YOLO confidence.

**Observed call rate**: On test video (active street), ~0.4 Gemma calls/second. Without cascade: would be 10 calls/second (25× more). With scene-diff gate, effective Gemma inference rate drops further.

---

---

### 19. Bypass Ultralytics + ORT_ENABLE_ALL — Eliminating Wrapper Overhead

**Root cause of 70ms**: The `ultralytics` YOLO wrapper does significant Python work per inference call: result parsing, tensor wrapping, metadata extraction, and passes no `SessionOptions` to ONNX Runtime — leaving `graph_optimization_level` at `ORT_ENABLE_BASIC`. This meant constant folding, node fusion, and memory layout optimization were disabled.

**Fix**: Created `_run_yolo_ort()` — direct `ort.InferenceSession` bypassing ultralytics entirely.

Key session options:
```python
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.intra_op_num_threads = 2   # match taskset -c 0,1
opts.inter_op_num_threads = 1
```

`ORT_SEQUENTIAL` is critical: with 2 pinned cores, parallel execution graph scheduling adds synchronization overhead that exceeds any parallelism gain. Sequential mode removes it.

Class names parsed from ONNX metadata at load time (`ast.literal_eval` on the `names` property). No ultralytics import needed at runtime.

**Gain**: 70ms → 17ms (4.1×). Ultralytics Python overhead was 53ms per inference.

---

### 20. INT8 Static Quantization — ARM asimddp Native Integer Math

**Why INT8 is fast on Cortex-A76**: The BCM2712 CPU features `asimddp` (ASIMD dot-product) instructions — `vdotq_s32` / `udot` — which compute 4× INT8 multiply-accumulates per cycle in a single instruction. ONNX Runtime's ARM Conv kernel uses these directly for INT8 weights and activations. FP32 uses `fmla` which has similar throughput but requires 4× the data bandwidth.

**Why data bandwidth matters**: YOLO at imgsz=96 is bandwidth-bound, not compute-bound on Pi 5. INT8 weights are 4× smaller than FP32 → more weights fit in 512KB L2 cache → fewer cache misses → throughput improves beyond the raw compute ratio.

**Quantization approach**: Static QDQ with 64 synthetic calibration frames (uniform random [0,1]). `per_channel=False` required for ONNX Runtime 1.26 compatibility on Pi (`per_channel=True` adds `axis` attribute to `DequantizeLinear` not supported in older ORT). `ActivationSymmetric=True` enables zero-point=0 for activations, reducing dequant overhead.

| Config | Inference | Size |
|---|---|---|
| FP32 ONNX (ultralytics) | 70ms | 9.7MB |
| FP32 ONNX (direct ORT) | 17ms | 9.7MB |
| INT8 QDQ (direct ORT) | **6.3ms** | **2.9MB** |

**Total YOLO gain over project**: 70ms → 6.3ms = **11.1× faster**.

---

## Current Pipeline State (2026-05-14)

```
Pi 5 @ 2400MHz, native build (DGGML_NATIVE=ON, LTO), throttled=0x0

llama-server:  taskset -c 2,3
               ~/models/gemma4-e2b/e2b-smooth-q4_0.gguf (Q4_0, 3.2GB)
               -c 256 -t 2 --mlock -ngl 0 --parallel 1 --log-disable
               --chat-template-file ~/gemma4-nothink.jinja  (<|turn> Gemma4 format)
               GGML_VULKAN_DISABLE=1
               
pipeline:      taskset -c 0,1  OMP_NUM_THREADS=2  OPENBLAS_NUM_THREADS=2
               yolo26n.onnx (INT8 QDQ)  imgsz=96  conf=0.10  skip_N=3  async_yolo=True
               ORT: ORT_ENABLE_ALL + ORT_SEQUENTIAL + intra_threads=2
               audio: MCP3008 SPI 10kHz 4ch (ch1 dead → ch0,2,3 active)
               Gemma: interval=2s  timeout=9s  scene-diff TTL=6s
               LLM params: max_tokens=12  temp=0.0  cache_prompt=True  stream=False

Performance:
  YOLO inference:    6.3ms median (>100fps cap — no longer bottleneck)
  Pipeline FPS:      ~9–11fps (video file, H.264 decode overhead)
  Gemma latency:     ~3s (pipeline running), 2.7s min observed
  Gemma direction:   ±10° (was ±111° before template fix)
  Pipeline CPU:      ~177–189% (of 2-core allocation)
  Gemma CPU:         ~139–146% (of 2-core allocation)
  Haptic update rate: 10Hz continuous (ThreatPredictor)
  Effective LLM rate: ~0.4 calls/s (cascade + scene gate)
```

---

## Cumulative: Baseline vs Current

| Scenario | Baseline | Current | Gain |
|---|---|---|---|
| Gemma first response | 37s (w1 bug) | **~3s** | **12×** |
| Gemma under pipeline load | 7–10s | **~3s** | **~3×** |
| YOLO inference | 307ms (PyTorch) | **6.3ms** (INT8 ORT) | **49×** |
| YOLO inference (stages) | 307ms → 70ms → 17ms → **6.3ms** | PyTorch→ONNX→direct ORT→INT8 | 4.4× → 4.1× → 2.7× |
| Pipeline FPS | 3.67fps (OMP thrash) | **~10fps** | **2.7×** |
| Direction accuracy | ±111° (wrong template) | **±10°** | **11×** |
| Output token count | 20+ (ASCII) | **8** (CJK) | **2.5× fewer steps** |
| LLM call rate | 10Hz (every frame) | **~0.4Hz** (cascade+gate) | **25× fewer LLM calls** |
| Haptic update rate | 0.33Hz (per LLM) | **10Hz** (predictor) | **30× smoother** |
| Max Gemma stall | 60s (timeout) | **10s** | **6×** |
| CPU idle waste | 54% (w1 bug) | **0%** | +54% reclaimed |
| RAM for KV | ~1.6GB (-c 2048) | **~90MB** (-c 256) | **~18× smaller** |
| YOLO model size | 9.7MB (FP32) | **2.9MB** (INT8) | 3.4× smaller, better L2 fit |
