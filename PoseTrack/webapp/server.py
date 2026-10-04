"""
server.py — FastAPI application: control plane + live data plane
=================================================================

Routes
------
``GET  /``                        the dashboard page
``GET  /api/health``              liveness probe
``GET  /api/status``              full pipeline state and rolling metrics
``GET  /api/explain``             machine-readable description of every reported value
``POST /api/filter``              select the active temporal filter
``POST /api/framework``           select the pose-estimation framework
``POST /api/udp``                 enable / disable / retarget the Unity stream
``POST /api/reset``               reset filter state and rolling statistics
``POST /api/calibration/...``     reference-pose calibration wizard
``POST /api/logging/start|stop``  CSV session logging
``GET  /api/sessions``            list recorded session logs
``GET  /api/sessions/{name}``     download a session CSV
``GET  /api/sessions/{name}/summary``  statistics computed from a recorded session
``GET  /api/sessions/{name}/plot.png`` joint-angle time series rendered from a session
``GET  /api/sources``             cameras and recorded videos visible to the server
``POST /api/source``              start / stop a server-side camera or file source
``GET  /api/preview.mjpg``        annotated MJPEG preview of a server-side source
``WS   /ws/stream``               browser frames up, per-frame results down

The WebSocket carries binary JPEG frames from the page's webcam and returns
one JSON result per frame. Control actions go over REST so that every one of
them is scriptable with ``curl`` and testable without a browser.
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
import time
from contextlib import asynccontextmanager
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from config.config import Config, OUTPUTS_DIR, RAW_VIDEOS_DIR
from .explain import explain_payload
from .pipeline import FILTER_TYPES, LivePipeline, PipelineConfig
from .sources import ServerFrameSource, SourceError, encode_jpeg, probe_cameras

STATIC_DIR = Path(__file__).resolve().parent / "static"
WEB_OUTPUT_DIR = Path(OUTPUTS_DIR) / "web"

# Frameworks the UI offers. Availability is probed lazily: selecting one that
# is not installed returns the import error rather than failing silently.
FRAMEWORKS = ("mediapipe", "movenet_lightning", "movenet_thunder", "posenet")


class DashboardState:
    """Process-wide singleton tying the pipeline to its connected clients."""

    def __init__(self) -> None:
        cfg = PipelineConfig(
            framework="mediapipe",
            filter_type=Config.DEFAULT_FILTER_TYPE
            if Config.DEFAULT_FILTER_TYPE in FILTER_TYPES else "kalman",
            stream_hz=float(Config.STREAM_HZ),
            udp_host=Config.UDP_IP,
            udp_port=int(Config.UDP_PORT),
            udp_enabled=True,
        )
        self.pipeline = LivePipeline(cfg, WEB_OUTPUT_DIR)
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.server_source: ServerFrameSource | None = None
        self._latest_preview: bytes | None = None
        self._rtt = deque(maxlen=120)
        self._frame_lock = asyncio.Lock()

    # -- browser round-trip, measured by the client and reported back -----
    def record_rtt(self, ms: float) -> None:
        if 0.0 < ms < 10_000.0:
            self._rtt.append(float(ms))

    def rtt_summary(self) -> dict:
        if not self._rtt:
            return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        s = sorted(self._rtt)
        n = len(s)
        return {
            "n": n,
            "mean": round(sum(s) / n, 2),
            "p50": round(s[int(0.50 * (n - 1))], 2),
            "p95": round(s[int(0.95 * (n - 1))], 2),
            "max": round(s[-1], 2),
        }

    def latest_preview(self) -> bytes | None:
        return self._latest_preview

    def set_preview(self, jpeg: bytes) -> None:
        self._latest_preview = jpeg

    async def broadcast(self, payload: dict) -> None:
        """Push a message to every connected dashboard client."""
        dead = []
        text = json.dumps(payload)
        for ws in list(self.clients):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


state = DashboardState()


def _warm_up_estimator() -> None:
    try:
        state.pipeline.ensure_estimator()
    except Exception:
        pass  # recorded in pipeline.status()["estimator_error"]

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Capture the running loop on start; release camera and sockets on stop."""
    state.loop = asyncio.get_running_loop()
    WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Load the pose network in the background so the first frame does not
    # stall for seconds; a load failure is surfaced via /api/status.
    threading.Thread(target=_warm_up_estimator, daemon=True, name="EstimatorWarmup").start()
    yield
    if state.server_source is not None:
        state.server_source.stop()
    state.pipeline.close()


app = FastAPI(
    title="MonoArm — Monocular Arm Joint Angle Estimation",
    description=(
        "Real-time monocular estimation of shoulder and elbow joint angles, "
        "streamed to a Unity avatar over UDP and explained frame by frame."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(500, "Dashboard assets are missing.")
    return HTMLResponse(page.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Small inline icon so the browser does not log a 404 for every page load."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="6" fill="#141b24"/>'
        '<path d="M8 24 L14 12 L22 17" fill="none" stroke="#4ea8de" '
        'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="14" cy="12" r="2.6" fill="#4ade80"/>'
        '</svg>'
    )
    return Response(svg, media_type="image/svg+xml")


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "t": time.time()}


@app.get("/api/explain")
async def explain() -> dict:
    return explain_payload()


@app.get("/api/status")
async def status() -> dict:
    st = state.pipeline.status()
    st["frameworks"] = list(FRAMEWORKS)
    st["clients"] = len(state.clients)
    st["round_trip_ms"] = state.rtt_summary()
    st["source"] = (
        state.server_source.state() if state.server_source is not None
        else {"mode": "browser", "running": False}
    )
    return st


# ---------------------------------------------------------------------------
# Control plane
# ---------------------------------------------------------------------------

@app.post("/api/filter")
async def set_filter(payload: dict = Body(...)) -> dict:
    ftype = payload.get("type")
    if ftype not in FILTER_TYPES:
        raise HTTPException(400, f"type must be one of {list(FILTER_TYPES)}")
    state.pipeline.set_filter(ftype)
    return {"filter_type": ftype}


@app.post("/api/framework")
async def set_framework(payload: dict = Body(...)) -> dict:
    name = payload.get("name")
    if name not in FRAMEWORKS:
        raise HTTPException(400, f"name must be one of {list(FRAMEWORKS)}")
    try:
        await asyncio.to_thread(state.pipeline.set_framework, name)
    except Exception as exc:
        raise HTTPException(
            503,
            f"Framework '{name}' could not be loaded: {type(exc).__name__}: {exc}",
        )
    return {"framework": name}


@app.post("/api/udp")
async def set_udp(payload: dict = Body(...)) -> dict:
    return state.pipeline.set_udp(
        enabled=bool(payload.get("enabled", True)),
        host=payload.get("host"),
        port=payload.get("port"),
        hz=payload.get("hz"),
    )


@app.post("/api/reset")
async def reset() -> dict:
    state.pipeline.reset_filters()
    return {"reset": True}


@app.post("/api/mirror")
async def set_mirror(payload: dict = Body(...)) -> dict:
    state.pipeline.config.mirror = bool(payload.get("enabled", True))
    return {"mirror": state.pipeline.config.mirror}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@app.post("/api/calibration/begin")
async def calibration_begin(payload: dict = Body(default={})) -> dict:
    side = payload.get("side", "right")
    try:
        return state.pipeline.calibration_begin(side)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/calibration/capture")
async def calibration_capture() -> dict:
    try:
        return state.pipeline.calibration_capture()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/calibration/cancel")
async def calibration_cancel() -> dict:
    return state.pipeline.calibration_cancel()


@app.post("/api/calibration/clear")
async def calibration_clear(payload: dict = Body(default={})) -> dict:
    return state.pipeline.calibration_clear(payload.get("side", "right"))


@app.get("/api/calibration")
async def calibration_state() -> dict:
    return state.pipeline.calibration_state()


# ---------------------------------------------------------------------------
# Session logging
# ---------------------------------------------------------------------------

@app.post("/api/logging/start")
async def logging_start(payload: dict = Body(default={})) -> dict:
    return state.pipeline.start_logging(str(payload.get("label", "")).strip())


@app.post("/api/logging/stop")
async def logging_stop() -> dict:
    return state.pipeline.stop_logging()


def _log_dir() -> Path:
    return WEB_OUTPUT_DIR / "logs"


def _resolve_session(name: str) -> Path:
    """Resolve a session filename inside the log directory, rejecting traversal."""
    log_dir = _log_dir().resolve()
    path = (log_dir / name).resolve()
    if log_dir not in path.parents or path.suffix != ".csv" or not path.exists():
        raise HTTPException(404, f"Session log '{name}' not found.")
    return path


@app.get("/api/sessions")
async def list_sessions() -> dict:
    log_dir = _log_dir()
    if not log_dir.exists():
        return {"sessions": []}
    rows = []
    for p in sorted(log_dir.glob("*.csv"), reverse=True):
        stat = p.stat()
        rows.append({
            "name": p.name,
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
        })
    return {"sessions": rows}


@app.get("/api/sessions/{name}")
async def download_session(name: str) -> FileResponse:
    path = _resolve_session(name)
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.get("/api/sessions/{name}/summary")
async def session_summary(name: str) -> dict:
    """Descriptive statistics computed from the recorded CSV itself."""
    import csv as _csv
    import math

    path = _resolve_session(name)
    channels = ["shoulder_flexion", "shoulder_abduction",
                "shoulder_rotation", "elbow_flexion"]
    data = {f"{s}_{c}": [] for s in ("right", "left") for c in channels}
    tracked = {"right": 0, "left": 0}
    rows = 0
    t_first: float | None = None
    t_last: float | None = None

    with open(path, newline="") as f:
        for row in _csv.DictReader(f):
            rows += 1
            try:
                ts = float(row.get("timestamp_s") or "nan")
            except ValueError:
                ts = float("nan")
            if not math.isnan(ts):
                t_first = ts if t_first is None else t_first
                t_last = ts
            for side in ("right", "left"):
                if row.get(f"{side}_tracked") == "1":
                    tracked[side] += 1
                for c in channels:
                    v = row.get(f"{side}_{c}")
                    if v not in (None, ""):
                        try:
                            data[f"{side}_{c}"].append(float(v))
                        except ValueError:
                            pass

    # Measured from the first logged frame, not from when logging was armed
    duration = (t_last - t_first) if t_first is not None else 0.0

    def stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0}
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        return {
            "n": n,
            "mean": round(mean, 3),
            "std": round(math.sqrt(var), 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "range": round(max(vals) - min(vals), 3),
        }

    return {
        "name": path.name,
        "rows": rows,
        "duration_s": round(duration, 2),
        # N rows span N-1 frame intervals between the first and last row
        "mean_rate_hz": round((rows - 1) / duration, 2) if duration > 0 else 0.0,
        "tracked_fraction": {
            s: round(tracked[s] / rows, 4) if rows else 0.0 for s in ("right", "left")
        },
        "channels": {k: stats(v) for k, v in data.items()},
    }


@app.get("/api/sessions/{name}/plot.png")
async def session_plot(name: str, side: str = Query("right")) -> Response:
    """Render the logged joint-angle time series for a recorded session."""
    import csv as _csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = _resolve_session(name)
    if side not in ("right", "left"):
        raise HTTPException(400, "side must be 'right' or 'left'")

    channels = ["shoulder_flexion", "shoulder_abduction",
                "shoulder_rotation", "elbow_flexion"]
    t: list[float] = []
    series: dict[str, list[float | None]] = {c: [] for c in channels}

    with open(path, newline="") as f:
        for row in _csv.DictReader(f):
            try:
                t.append(float(row.get("timestamp_s") or 0.0))
            except ValueError:
                continue
            for c in channels:
                v = row.get(f"{side}_{c}")
                series[c].append(float(v) if v not in (None, "") else np.nan)

    if not t:
        raise HTTPException(422, "Session log contains no usable rows.")

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, c in zip(axes, channels):
        ax.plot(t, series[c], linewidth=1.2)
        ax.set_ylabel(c.replace("shoulder_", "sh. ").replace("_", " ") + "\n[deg]",
                      fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{path.name} — {side} arm joint angles", fontsize=11)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return Response(buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# Server-side frame sources
# ---------------------------------------------------------------------------

@app.get("/api/sources")
async def list_sources() -> dict:
    videos = []
    if RAW_VIDEOS_DIR.exists():
        for p in sorted(RAW_VIDEOS_DIR.glob("*")):
            if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
                videos.append({"name": p.name, "path": str(p),
                               "size_bytes": p.stat().st_size})
    return {
        "cameras": await asyncio.to_thread(probe_cameras),
        "videos": videos,
        "video_dir": str(RAW_VIDEOS_DIR),
        "active": (state.server_source.state() if state.server_source
                   else {"mode": "browser", "running": False}),
    }


def _server_frame_callback(frame_bgr: np.ndarray, read_ms: float) -> None:
    """Process a server-sourced frame and push the result to all clients."""
    pipe = state.pipeline
    if pipe.config.mirror:
        frame_bgr = cv2.flip(frame_bgr, 1)
    result = pipe.process_frame(frame_bgr, transport_ms=0.0, decode_ms=read_ms)
    payload = result.as_dict()
    payload["source"] = "server"
    try:
        state.set_preview(encode_jpeg(frame_bgr, quality=75))
    except SourceError:
        pass
    loop = state.loop
    if loop is not None:
        asyncio.run_coroutine_threadsafe(state.broadcast(payload), loop)


@app.post("/api/source")
async def set_source(payload: dict = Body(...)) -> dict:
    mode = payload.get("mode", "browser")
    if state.server_source is not None:
        state.server_source.stop()
        state.server_source = None

    if mode == "browser":
        return {"mode": "browser", "running": False}

    if mode not in ("camera", "file"):
        raise HTTPException(400, "mode must be 'browser', 'camera' or 'file'")

    src = ServerFrameSource(
        mode=mode,
        camera_index=int(payload.get("camera_index", 0)),
        path=payload.get("path"),
        width=int(payload.get("width", Config.FRAME_WIDTH)),
        height=int(payload.get("height", Config.FRAME_HEIGHT)),
        loop=bool(payload.get("loop", True)),
    )
    try:
        await asyncio.to_thread(src.start, _server_frame_callback)
    except SourceError as exc:
        raise HTTPException(400, str(exc))
    state.server_source = src
    return src.state()


@app.get("/api/preview.mjpg")
async def preview() -> StreamingResponse:
    """Annotated MJPEG preview of the active server-side source."""
    async def gen():
        boundary = b"--frame\r\n"
        while True:
            jpeg = state.latest_preview()
            if jpeg is not None:
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(1 / 30)

    return StreamingResponse(
        gen(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------------------------------------------------
# Data plane
# ---------------------------------------------------------------------------

def _decode_jpeg(data: bytes) -> tuple[np.ndarray, float]:
    """Decode JPEG bytes into a BGR frame, returning the measured decode time."""
    t0 = time.perf_counter()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    dt = (time.perf_counter() - t0) * 1000.0
    if frame is None:
        raise ValueError("Frame could not be decoded as JPEG.")
    return frame, dt


@app.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    """
    Bidirectional stream.

    Client → server
        binary : one JPEG frame from the page's webcam
        text   : {"type": "rtt", "ms": <measured round trip>}
                 {"type": "ping"}

    Server → client
        text   : one JSON frame result per processed frame, plus periodic
                 status messages.
    """
    await ws.accept()
    state.clients.add(ws)
    await ws.send_text(json.dumps({"type": "hello", "status": await status()}))

    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            if (data := msg.get("bytes")) is not None:
                # Serialise pipeline access: one frame in flight at a time,
                # so timings reflect a single frame's true cost.
                async with state._frame_lock:
                    try:
                        frame, decode_ms = await asyncio.to_thread(_decode_jpeg, data)
                    except ValueError as exc:
                        await ws.send_text(json.dumps(
                            {"type": "error", "message": str(exc)}))
                        continue

                    if state.pipeline.config.mirror:
                        frame = cv2.flip(frame, 1)

                    try:
                        result = await asyncio.to_thread(
                            state.pipeline.process_frame, frame, 0.0, decode_ms
                        )
                    except Exception as exc:
                        await ws.send_text(json.dumps({
                            "type": "error",
                            "message": f"{type(exc).__name__}: {exc}",
                        }))
                        continue

                payload = result.as_dict()
                payload["source"] = "browser"
                payload["metrics"] = state.pipeline.metrics.snapshot(
                    state.pipeline.config.latency_budget_ms
                )
                payload["round_trip_ms"] = state.rtt_summary()
                await ws.send_text(json.dumps(payload))
                continue

            if (text := msg.get("text")) is not None:
                try:
                    cmd = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if cmd.get("type") == "rtt":
                    state.record_rtt(float(cmd.get("ms", 0.0)))
                elif cmd.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong", "t": time.time()}))
                elif cmd.get("type") == "status":
                    await ws.send_text(json.dumps(
                        {"type": "status", "status": await status()}))

    except WebSocketDisconnect:
        pass
    finally:
        state.clients.discard(ws)
