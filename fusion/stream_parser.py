"""
Real-time SSE stream parser for llama-server.
Fires haptic callbacks as soon as key JSON fields appear in the token stream.
Does not wait for full inference completion.

KV-Cache optimisation (Idea 1):
  - cache_prompt=True: llama.cpp retains KV state between calls.
  - 1-turn conversation history: system+prev turn are cached tokens;
    only the new user message needs prefill. ~3s savings/call on Pi 5.
  - DeltaEncoder marks changed fields with * so Gemma focuses attention.
"""
import json
import re
import time
import threading
import requests
from dataclasses import dataclass, field
from typing import Callable, Optional
from .inference_profile import InferenceProfile, STANDARD

LLAMA_URL = "http://localhost:8081"

EARLY_FIRE_FIELDS = {"direction_degrees", "urgency", "haptic_pattern"}

SYSTEM_PROMPT = """Threat detector for blind/deaf users. Fields marked * changed.
JSON only. direction_degrees:0=ahead,90=right,180=behind,270=left. urgency:low/medium/high/critical. haptic:none/slow_pulse/rapid_left/rapid_right/rapid_center. natural_language:≤7 words second-person eg "Car on your left."
Urgency rules: vehicle/person <3m=high, <6m=medium; approaching fast=critical. Static distant object=low."""


@dataclass
class ThreatAlert:
    threat_type: str = "none"
    confidence: float = 0.0
    direction_degrees: float = 0.0
    distance_meters: float = 0.0
    urgency: str = "low"
    haptic_pattern: str = "none"
    natural_language: str = ""
    timestamp: float = field(default_factory=time.time)
    partial: bool = True

    def is_actionable(self) -> bool:
        return (self.confidence > 0.6 and self.urgency in ("medium", "high", "critical")) or \
               (self.confidence > 0.8 and self.urgency != "low")


class StreamParser:
    """
    Streams tokens from llama-server, builds JSON incrementally.
    Calls early_fire_cb as soon as actionable fields appear — before inference ends.
    Calls complete_cb when full JSON is parsed.
    """

    def __init__(
        self,
        early_fire_cb: Callable[[ThreatAlert], None],
        complete_cb: Callable[[ThreatAlert], None],
        server_url: str = LLAMA_URL,
    ):
        self.early_fire_cb = early_fire_cb
        self.complete_cb = complete_cb
        self.server_url = server_url
        self._fired_fields: set[str] = set()
        self._kv_cache: dict = {}   # {"prompt": str, "response": str}
        self._cache_lock = threading.Lock()
        # Path A: cache draft token IDs per profile name (fetched once from /tokenize)
        self._draft_token_cache: dict[str, list[int]] = {}
        self._draft_cache_lock = threading.Lock()
        self._busy = threading.Event()   # set while inference in flight

    def infer(self, sensor_prompt: str, frame=None,
              profile: InferenceProfile = STANDARD,
              logit_bias: Optional[dict] = None) -> None:
        """Non-blocking. Drops call if inference already in flight (Pi can't queue).
        profile: InferenceProfile from select_profile() — gates max_tokens/temp/schema.
        logit_bias: optional {token_id: bias_value} from LogitSteerer (Idea 4)."""
        if self._busy.is_set():
            return   # drop — previous call still running
        self._busy.set()
        t = threading.Thread(
            target=self._stream,
            args=(sensor_prompt, frame, profile, logit_bias),
            daemon=True,
        )
        t.start()

    def _server_has_vision(self) -> bool:
        """Check if llama-server was started with --mmproj (vision capable)."""
        try:
            r = requests.get(f"{self.server_url}/props", timeout=2)
            props = r.json()
            return props.get("has_mmproj", False) or "mmproj" in str(props).lower()
        except Exception:
            return False

    def _get_draft_tokens(self, profile) -> list[int]:
        """Path A: fetch token IDs for JSON prefix from /tokenize. Cached per profile.
        Draft gives speculative decoder a head start on the fixed JSON opening tokens."""
        with self._draft_cache_lock:
            if profile.name in self._draft_token_cache:
                return self._draft_token_cache[profile.name]

        # JSON always starts with { then first field name from schema
        first_field = profile.schema_fields[0]
        prefix = '{' + f'"{first_field}":'
        try:
            r = requests.post(
                f"{self.server_url}/tokenize",
                json={"content": prefix, "add_special": False},
                timeout=3,
            )
            ids = r.json().get("tokens", [])
            with self._draft_cache_lock:
                self._draft_token_cache[profile.name] = ids
            return ids
        except Exception:
            return []

    @staticmethod
    def _frame_to_b64(frame) -> str:
        """Encode numpy BGR frame as base64 JPEG."""
        import cv2, base64
        # Downscale to 512px wide to keep tokens reasonable
        h, w = frame.shape[:2]
        if w > 512:
            scale = 512 / w
            frame = cv2.resize(frame, (512, int(h * scale)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf).decode("utf-8")

    def _build_messages(self, prompt: str, frame, profile: InferenceProfile) -> list:
        msgs: list = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Prepend previous (user, assistant) turn for KV-cache reuse.
        # Vision frames change every call (base64 payload differs) so history
        # only helps the text-only path — skip it when frame is attached.
        if frame is None:
            with self._cache_lock:
                cached = dict(self._kv_cache)
            if cached.get("prompt") and cached.get("response"):
                msgs.append({"role": "user", "content": cached["prompt"]})
                msgs.append({"role": "assistant", "content": cached["response"]})

        # Append profile schema hint so Gemma narrows to the requested fields
        full_prompt = f"{prompt} | {profile.schema_hint()}"

        if frame is None:
            msgs.append({"role": "user", "content": full_prompt})
        else:
            b64 = self._frame_to_b64(frame)
            msgs.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": full_prompt},
                ],
            })
        return msgs

    def _stream(self, prompt: str, frame=None,
                profile: InferenceProfile = STANDARD,
                logit_bias: Optional[dict] = None) -> None:
        alert = ThreatAlert()
        buf = ""
        reasoning_buf = ""
        fired_early = False

        # Vision path only if server loaded mmproj; text fallback otherwise
        use_vision = frame is not None and self._server_has_vision()
        messages = self._build_messages(prompt, frame if use_vision else None, profile)

        body = {
            "model": "gemma",
            "messages": messages,
            "max_tokens": profile.max_tokens,
            "temperature": profile.temperature,
            "stream": True,
            "cache_prompt": True,
            # No grammar: Gemma 4 thinking model outputs reasoning_content first,
            # grammar blocks those tokens causing inference stall. _parse_full
            # handles markdown fencing and partial JSON gracefully.
        }
        if logit_bias:
            body["logit_bias"] = {str(k): v for k, v in logit_bias.items()}

        try:
            resp = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=body,
                stream=True,
                timeout=300 if use_vision else 180,   # Pi Gemma EXPLORATORY can take >60s
            )

            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {}) or {}
                    # Gemma 4 thinking model: JSON lands in content after reasoning phase.
                    # Collect reasoning_content too as fallback for _parse_full.
                    token = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or ""
                    buf += token
                    reasoning_buf += reasoning
                    # scan both buffers for early-fire fields
                    scan = buf if buf.strip() else reasoning_buf
                    alert, fired_early = self._try_early_fire(scan, alert, fired_early)
                except (json.JSONDecodeError, KeyError):
                    continue

            # Complete parse — if content empty (thinking model), try reasoning_content
            parse_src = buf if buf.strip() else reasoning_buf
            alert = self._parse_full(parse_src, alert)
            alert.partial = False

            # Cache this turn for the next call (text path only).
            # Strips base64 images from cache — no point caching those.
            if not use_vision and buf.strip():
                with self._cache_lock:
                    self._kv_cache = {"prompt": prompt, "response": buf.strip()}

            self.complete_cb(alert)

        except Exception as e:
            print(f"[StreamParser] error: {e}")
        finally:
            self._fired_fields.clear()
            self._busy.clear()

    def _try_early_fire(
        self, buf: str, alert: ThreatAlert, fired: bool
    ) -> tuple[ThreatAlert, bool]:
        """Check if any early-fire fields are parseable in partial buffer.

        Fire strategy: direction_degrees appears 3rd in schema (after threat_type,
        confidence) — earliest reliable signal. Fire once we have direction +
        confidence > 0.5. Don't wait for urgency (arrives ~5 tokens later).
        """
        for field_name in EARLY_FIRE_FIELDS:
            if field_name in self._fired_fields:
                continue
            pattern = rf'"{field_name}"\s*:\s*("([^"]*)"|([\d.]+))'
            m = re.search(pattern, buf)
            if m:
                val = m.group(2) if m.group(2) is not None else m.group(3)
                self._fired_fields.add(field_name)
                setattr(alert, field_name, self._coerce(field_name, val))

        # Fire as soon as direction_degrees is known + confidence indicates threat
        if not fired and "direction_degrees" in self._fired_fields:
            conf = alert.confidence if alert.confidence > 0 else 0.7
            if conf > 0.5 or "urgency" in self._fired_fields:
                urgency = alert.urgency if alert.urgency else "medium"
                if urgency in ("medium", "high", "critical"):
                    alert.confidence = conf
                    alert.partial = True
                    self.early_fire_cb(alert)
                    fired = True

        return alert, fired

    def _parse_full(self, buf: str, alert: ThreatAlert) -> ThreatAlert:
        # Strip markdown code blocks
        buf = re.sub(r"```(?:json)?\s*", "", buf).strip()
        m = re.search(r"\{[^{}]*\}", buf, re.DOTALL)
        if not m:
            return alert
        try:
            data = json.loads(m.group())
            for k, v in data.items():
                if hasattr(alert, k):
                    setattr(alert, k, self._coerce(k, v))
        except json.JSONDecodeError:
            pass
        self._fired_fields.clear()
        return alert

    @staticmethod
    def _coerce(key: str, val):
        float_keys = {"confidence", "direction_degrees", "distance_meters"}
        if key in float_keys:
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0
        return val
