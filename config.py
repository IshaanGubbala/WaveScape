"""
Central runtime configuration for ThreatDetect.
All network addresses, ports, and tunable constants read from environment
with hardcoded fallbacks. Override via env or export before running.
"""
import os

# ── Network ────────────────────────────────────────────────────────────────────
LLAMA_URL    = os.environ.get("LLAMA_URL",    "http://localhost:8081")
MAC_HOST     = os.environ.get("MAC_HOST",     "100.75.147.6")    # Mac Tailscale IP
MAC_WS_URL   = os.environ.get("MAC_WS_URL",  f"ws://{MAC_HOST}:8084")
MAC_HTTP_URL = os.environ.get("MAC_HTTP_URL", f"http://{MAC_HOST}:8083")
UI_PORT      = int(os.environ.get("UI_PORT", 8090))

# ── Pipeline ───────────────────────────────────────────────────────────────────
TARGET_FPS = int(os.environ.get("TARGET_FPS", 10))
USE_MOCK   = os.environ.get("USE_MOCK", "1") == "1"

# ── Camera ─────────────────────────────────────────────────────────────────────
CAM_HALF_FOV_DEG = 33.0   # Pi Camera Module 3 measured ±33° horizontal
IMG_W            = 224
IMG_H            = 224

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
