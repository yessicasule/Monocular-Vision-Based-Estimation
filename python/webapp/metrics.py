"""
metrics.py — Rolling performance and signal-quality statistics
===============================================================

Everything reported by the web dashboard is measured, never assumed:

* **Latency** is measured per pipeline stage with ``time.perf_counter()``
  around the actual call, and reported as mean / p50 / p95 / max over a
  rolling window.  The end-to-end figure is the sum of the measured stages
  for that frame, plus (for browser-sourced frames) the transport time the
  client stamped into the frame header.
* **Throughput** is derived from the arrival timestamps of processed frames,
  not from a nominal configured rate.
* **Static-pose stability** is the standard deviation of each joint-angle
  channel over a rolling window.  The project specification requires it to
  stay within ±3–5° while the subject holds still, so the dashboard shows
  the live value against that band.
* **Keypoint jitter** is the standard deviation of each tracked keypoint's
  pixel position over the same window — the raw noise the filters must
  remove.

A rolling window (default 90 frames ≈ 3 s at 30 Hz) is used everywhere so
the numbers respond to what the user is doing right now.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict


# Angle channels tracked for stability statistics
ANGLE_CHANNELS = (
    "shoulder_flexion",
    "shoulder_abduction",
    "shoulder_rotation",
    "elbow_flexion",
)

# Landmark indices whose pixel jitter is reported (both arms + torso anchors)
JITTER_LANDMARKS = (11, 12, 13, 14, 15, 16, 23, 24)


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(pos)]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def _std(values) -> float:
    """Population standard deviation of a sequence (0.0 for < 2 samples)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


@dataclass
class StageTimings:
    """Measured wall-clock duration of each pipeline stage, in milliseconds."""

    transport_ms: float = 0.0   # client → server frame transport (browser source)
    decode_ms:    float = 0.0   # JPEG decode / camera read
    pose_ms:      float = 0.0   # pose-estimator inference
    angles_ms:    float = 0.0   # torso frame + joint-angle solving
    filter_ms:    float = 0.0   # temporal filtering (all banks)
    calib_ms:     float = 0.0   # calibration mapping
    stream_ms:    float = 0.0   # UDP hand-off + CSV write
    total_ms:     float = 0.0   # sum of the above

    def finalise(self) -> "StageTimings":
        self.total_ms = (
            self.transport_ms + self.decode_ms + self.pose_ms + self.angles_ms
            + self.filter_ms + self.calib_ms + self.stream_ms
        )
        return self

    def as_dict(self) -> dict:
        return {k: round(v, 3) for k, v in asdict(self).items()}


class LatencyTracker:
    """Rolling per-stage latency statistics."""

    STAGES = (
        "transport_ms", "decode_ms", "pose_ms", "angles_ms",
        "filter_ms", "calib_ms", "stream_ms", "total_ms",
    )

    def __init__(self, window: int = 90) -> None:
        self._w = {s: deque(maxlen=window) for s in self.STAGES}

    def add(self, t: StageTimings) -> None:
        d = asdict(t)
        for stage in self.STAGES:
            self._w[stage].append(d[stage])

    def summary(self) -> dict:
        out = {}
        for stage, buf in self._w.items():
            if not buf:
                out[stage] = {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "n": 0}
                continue
            s = sorted(buf)
            out[stage] = {
                "mean": round(sum(buf) / len(buf), 2),
                "p50":  round(_percentile(s, 0.50), 2),
                "p95":  round(_percentile(s, 0.95), 2),
                "max":  round(max(buf), 2),
                "n":    len(buf),
            }
        return out

    def budget_compliance(self, budget_ms: float = 100.0) -> float:
        """Fraction of recent frames whose end-to-end latency met the budget."""
        buf = self._w["total_ms"]
        if not buf:
            return 0.0
        return round(sum(1 for v in buf if v <= budget_ms) / len(buf), 4)


class ThroughputTracker:
    """
    Rolling frame-rate measured from actual frame completion timestamps.

    A gap longer than ``gap_s`` means the stream stopped (the operator paused
    the camera, or a client disconnected). The window is restarted at that
    point so the reported rate describes the burst currently being processed
    rather than being dragged toward zero by idle time. Lifetime counters —
    total frames, detection rate, uptime — keep accumulating across gaps.
    """

    def __init__(self, window: int = 90, gap_s: float = 1.0) -> None:
        self._t = deque(maxlen=window)
        self._gap_s = gap_s
        self._frames = 0
        self._detected = 0
        self._segments = 1
        self._t0 = time.perf_counter()

    def add(self, detected: bool) -> None:
        now = time.perf_counter()
        if self._t and (now - self._t[-1]) > self._gap_s:
            self._t.clear()
            self._segments += 1
        self._t.append(now)
        self._frames += 1
        if detected:
            self._detected += 1

    @property
    def fps(self) -> float:
        if len(self._t) < 2:
            return 0.0
        span = self._t[-1] - self._t[0]
        return round((len(self._t) - 1) / span, 2) if span > 0 else 0.0

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def detection_rate(self) -> float:
        return round(self._detected / self._frames, 4) if self._frames else 0.0

    @property
    def uptime_s(self) -> float:
        return round(time.perf_counter() - self._t0, 1)


class StabilityTracker:
    """
    Rolling standard deviation of each joint-angle channel, per side and per
    signal (raw vs each filter), used to demonstrate filter effectiveness and
    to check the ±3–5° static-pose requirement.
    """

    def __init__(self, window: int = 90) -> None:
        self._window = window
        self._buf: dict[str, dict[str, deque]] = {}

    def add(self, signal: str, side: str, angles) -> None:
        if angles is None:
            return
        key = f"{signal}:{side}"
        chans = self._buf.setdefault(
            key, {c: deque(maxlen=self._window) for c in ANGLE_CHANNELS}
        )
        for c in ANGLE_CHANNELS:
            chans[c].append(getattr(angles, c))

    def summary(self) -> dict:
        out: dict[str, dict[str, float]] = {}
        for key, chans in self._buf.items():
            out[key] = {c: round(_std(buf), 3) for c, buf in chans.items()}
        return out

    def reset(self) -> None:
        self._buf.clear()


class KeypointJitterTracker:
    """
    Rolling per-keypoint pixel-position standard deviation.

    This is the input-side noise measurement: it quantifies how much the
    pose network's landmark estimates move while the subject holds still,
    which is the disturbance the temporal filters are designed to reject.
    """

    def __init__(self, window: int = 90) -> None:
        self._window = window
        self._x: dict[int, deque] = {i: deque(maxlen=window) for i in JITTER_LANDMARKS}
        self._y: dict[int, deque] = {i: deque(maxlen=window) for i in JITTER_LANDMARKS}

    def add(self, landmarks, width: int, height: int) -> None:
        if landmarks is None:
            return
        for i in JITTER_LANDMARKS:
            if i >= len(landmarks):
                continue
            self._x[i].append(landmarks[i].x * width)
            self._y[i].append(landmarks[i].y * height)

    def summary(self) -> dict:
        out = {}
        for i in JITTER_LANDMARKS:
            sx, sy = _std(self._x[i]), _std(self._y[i])
            out[str(i)] = {
                "std_x_px": round(sx, 3),
                "std_y_px": round(sy, 3),
                "rms_px":   round(math.sqrt(0.5 * (sx * sx + sy * sy)), 3),
            }
        vals = [v["rms_px"] for v in out.values()]
        out["mean_rms_px"] = round(sum(vals) / len(vals), 3) if vals else 0.0
        return out

    def reset(self) -> None:
        for i in JITTER_LANDMARKS:
            self._x[i].clear()
            self._y[i].clear()


@dataclass
class SessionMetrics:
    """Aggregate of every rolling tracker, serialised to the dashboard."""

    latency:   LatencyTracker      = field(default_factory=LatencyTracker)
    rate:      ThroughputTracker   = field(default_factory=ThroughputTracker)
    stability: StabilityTracker    = field(default_factory=StabilityTracker)
    jitter:    KeypointJitterTracker = field(default_factory=KeypointJitterTracker)

    def snapshot(self, latency_budget_ms: float = 100.0) -> dict:
        return {
            "fps":               self.rate.fps,
            "frames":            self.rate.frames,
            "detection_rate":    self.rate.detection_rate,
            "uptime_s":          self.rate.uptime_s,
            "latency":           self.latency.summary(),
            "latency_budget_ms": latency_budget_ms,
            "within_budget":     self.latency.budget_compliance(latency_budget_ms),
            "angle_std":         self.stability.summary(),
            "keypoint_jitter":   self.jitter.summary(),
        }
