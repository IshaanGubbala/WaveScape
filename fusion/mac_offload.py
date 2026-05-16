"""
Pi-side WebSocket client for Mac Scene Service (port 8084).
Persistent connection: Pi pushes sensor data, Mac pushes scene results + Gemma alerts.
Falls back to HTTP POST (/analyze on port 8083) if WebSocket unavailable.

Usage:
    from fusion.mac_offload import MacSceneOffload

    offload = MacSceneOffload()
    offload.update(sector_depths, detections, beam_scan, heading_deg=0.0)

    result = offload.get_result()   # None until first response
    if result:
        escape = result["escape_dir_deg"]
        objs   = result["scene_objects"]
"""

import asyncio
import json
import threading
import time
from typing import Optional

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    import websockets as _ws_lib
    _HAS_WS = True
except ImportError:
    _HAS_WS = False

MAC_HOST        = "100.75.147.6"   # Mac Tailscale IP (LAN unreachable when on different networks)
MAC_WS_URL      = f"ws://{MAC_HOST}:8084/ws"
MAC_HTTP_URL    = f"http://{MAC_HOST}:8083"
RESULT_TTL_S    = 4.0    # treat cached result stale after this
HTTP_TIMEOUT_S  = 2.5
UPDATE_MIN_S    = 0.1    # max 10 sensor pushes/sec even if called every frame
WS_RECONNECT_S  = 2.0
WS_PING_S       = 5.0


class MacSceneOffload:
    """
    Bidirectional Pi↔Mac channel.
    - Pi → Mac: sensor data (YOLO dets, beam, sectors) at up to 10Hz
    - Mac → Pi: scene results immediately after processing
    - Mac → Pi: Gemma alerts pushed as soon as background thread produces them
    Falls back to HTTP if websockets not installed or Mac WS unreachable.
    """

    def __init__(self, ws_url: str = MAC_WS_URL, http_url: str = MAC_HTTP_URL):
        self._ws_url   = ws_url
        self._http_url = http_url

        self._lock         = threading.Lock()
        self._result: Optional[dict] = None
        self._result_ts: float = 0.0
        self._last_send_ts: float = 0.0

        # WS asyncio state
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._send_queue: Optional[asyncio.Queue] = None
        self._ws_connected = False
        self._fail_count   = 0

        if _HAS_WS:
            t = threading.Thread(target=self._run_ws_loop, daemon=True)
            t.start()
        elif _HAS_REQUESTS:
            print("[mac_offload] websockets not installed — using HTTP fallback")
        else:
            print("[mac_offload] neither websockets nor requests installed — disabled")

    # ── public API (unchanged from HTTP version) ──────────────────────────────

    def update(self, sector_depths: list, detections: list,
               beam_scan: list, heading_deg: float = 0.0) -> None:
        """Non-blocking sensor push. Throttled to UPDATE_MIN_S."""
        now = time.time()
        with self._lock:
            if now - self._last_send_ts < UPDATE_MIN_S:
                return
            self._last_send_ts = now

        msg = {
            "type":          "sensor",
            "detections":    detections,
            "beam_scan":     beam_scan,
            "sector_depths": sector_depths,
            "heading_deg":   heading_deg,
        }

        if _HAS_WS and self._loop and self._send_queue and self._ws_connected:
            self._loop.call_soon_threadsafe(self._enqueue, msg)
        elif _HAS_REQUESTS:
            t = threading.Thread(target=self._http_post, args=(msg,), daemon=True)
            t.start()

    def get_result(self) -> Optional[dict]:
        with self._lock:
            return self._result

    def is_fresh(self, max_age_s: float = RESULT_TTL_S) -> bool:
        with self._lock:
            return (self._result is not None
                    and (time.time() - self._result_ts) < max_age_s)

    def get_escape_dir(self, fallback: float = 0.0) -> float:
        r = self.get_result()
        return float(r["escape_dir_deg"]) if r else fallback

    def get_summary(self) -> str:
        r = self.get_result()
        return r.get("summary", "") if r else ""

    def get_gemma_scene(self) -> str:
        r = self.get_result()
        return r.get("gemma_scene", "") if r else ""

    # ── WebSocket client ──────────────────────────────────────────────────────

    def _run_ws_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._ws_main())

    async def _ws_main(self) -> None:
        self._send_queue = asyncio.Queue(maxsize=20)
        while True:
            try:
                async with _ws_lib.connect(
                    self._ws_url,
                    ping_interval=WS_PING_S,
                    ping_timeout=WS_PING_S * 2,
                    open_timeout=5.0,
                ) as ws:
                    with self._lock:
                        self._ws_connected = True
                        self._fail_count   = 0
                    print(f"[mac_offload] WS connected → {self._ws_url}")

                    sender   = asyncio.create_task(self._sender(ws))
                    receiver = asyncio.create_task(self._receiver(ws))
                    done, pending = await asyncio.wait(
                        [sender, receiver],
                        return_when=asyncio.FIRST_EXCEPTION,
                    )
                    for t in pending:
                        t.cancel()
                    # surface exception if any
                    for t in done:
                        if t.exception():
                            raise t.exception()

            except Exception as e:
                with self._lock:
                    self._ws_connected = False
                    self._fail_count  += 1
                fc = self._fail_count
                if fc <= 3 or fc % 20 == 0:
                    print(f"[mac_offload] WS disconnected: {e}  (attempt {fc})")
                await asyncio.sleep(WS_RECONNECT_S)

    async def _sender(self, ws) -> None:
        while True:
            msg = await self._send_queue.get()
            await ws.send(json.dumps(msg))

    async def _receiver(self, ws) -> None:
        async for raw in ws:
            try:
                data = json.loads(raw)
            except Exception:
                continue
            msg_type = data.get("type")
            if msg_type in ("scene", "alert"):
                with self._lock:
                    # Merge alert into existing result so scene_objects persist
                    if msg_type == "alert" and self._result:
                        self._result["gemma_scene"] = data.get("gemma_scene", "")
                    else:
                        self._result = data
                    self._result_ts = time.time()
                if msg_type == "alert":
                    print(f"  [mac_offload] ALERT ← {data.get('gemma_scene','')}")
                else:
                    print(f"  [mac_offload] scene ← {data.get('summary','')}  "
                          f"objs={len(data.get('scene_objects',[]))}  "
                          f"lat={data.get('latency_ms',0):.0f}ms")

    def _enqueue(self, msg: dict) -> None:
        """Called on WS asyncio loop via call_soon_threadsafe."""
        if self._send_queue.full():
            try:
                self._send_queue.get_nowait()  # drop oldest
            except asyncio.QueueEmpty:
                pass
        try:
            self._send_queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    # ── HTTP fallback ─────────────────────────────────────────────────────────

    def _http_post(self, msg: dict) -> None:
        try:
            resp = _requests.post(
                f"{self._http_url}/analyze",
                json=msg,
                timeout=HTTP_TIMEOUT_S,
            )
            if resp.status_code == 200:
                data = resp.json()
                with self._lock:
                    self._result    = data
                    self._result_ts = time.time()
                print(f"  [mac_offload] HTTP ← {data.get('summary','')}  "
                      f"lat={data.get('latency_ms',0):.0f}ms")
        except Exception as e:
            with self._lock:
                self._fail_count += 1
            if self._fail_count <= 3 or self._fail_count % 20 == 0:
                print(f"  [mac_offload] HTTP failed: {e}")
