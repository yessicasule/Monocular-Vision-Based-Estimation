"""
pipeline.py — Live frame-processing core behind the web dashboard
==================================================================

One :class:`LivePipeline` instance owns a complete real-time chain:

    frame (BGR)
      → pose estimator            (MediaPipe / MoveNet / PoseNet)
      → torso coordinate frame    (coordinate_frame.build_torso_frame)
      → bilateral joint angles    (angle_solver.compute_bilateral_angles)
      → temporal filter banks     (Kalman / moving average / Savitzky–Golay)
      → calibration mapping       (calibration.CalibrationManager, per side)
      → UDP packet to Unity       (streaming.udp_streamer.UdpAngleSender)
      → CSV log                   (processing.angle_logger)

All three filter families are evaluated on every frame, not just the active
one.  The extra cost is a few microseconds per channel and it lets the
dashboard show raw-versus-filtered traces for all filters simultaneously —
the live counterpart of the offline filter comparison required by the
project's Month-4 objectives.  Only the *selected* filter's output is sent
to Unity and written to the CSV log, so the exported reference signal is
unambiguous.

Nothing in this module fabricates data.  When no person is detected, or a
limb is occluded, the corresponding angles are ``None`` and are reported as
such; the UDP sender then holds the last valid pose (documented behaviour of
:class:`UdpAngleSender`) rather than snapping the avatar to zero.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.pose import angle_landmarks, load_estimator
from src.processing.angle_filter import BilateralFilterBank
from src.processing.angle_logger import BilateralCsvAngleLogger
from src.processing.angle_solver import (
    ArmAngles,
    BilateralArmAngles,
    ROTATION_RELIABLE_THRESHOLD,
    _SIDE_LANDMARKS,
    compute_bilateral_angles,
)
from src.processing.calibration import CalibrationManager, REQUIRED_POSES
from src.processing.coordinate_frame import (
    _normalize,
    _to_array,
    build_torso_frame,
    world_to_torso,
)
from src.streaming.udp_streamer import UdpAngleSender

from .metrics import (
    KeypointJitterTracker,
    LatencyTracker,
    SessionMetrics,
    StabilityTracker,
    StageTimings,
    ThroughputTracker,
)


# Filter families evaluated on every frame for the comparison view.
FILTER_TYPES = ("kalman", "ma", "sg")

# Number of consecutive filtered frames averaged when capturing a
# calibration reference pose (~0.5 s at 30 Hz).
CALIBRATION_SAMPLE_FRAMES = 15

# Landmarks forwarded to the browser for the skeleton overlay: both arms,
# the torso anchors that define the reference frame, and the face points
# that make the overlay readable.
OVERLAY_LANDMARKS = (0, 11, 12, 13, 14, 15, 16, 23, 24)

# The coordinates joint angles are computed in (see src.pose.angle_landmarks)
_COORDINATE_SPACES = {
    "world":  {"id": "world", "unit": "m",
               "label": "metric 3D world landmarks from the pose network, "
                        "hip-centred, metres"},
    "pixels": {"id": "pixels", "unit": "px",
               "label": "image keypoints scaled to pixels (x·width, y·height, "
                        "z·width) so all three axes share one unit"},
}

# Skeleton segments drawn by the front-end (pairs of landmark indices).
OVERLAY_EDGES = (
    (11, 12), (11, 23), (12, 24), (23, 24),      # torso quad
    (12, 14), (14, 16),                          # right arm
    (11, 13), (13, 15),                          # left arm
)

_ANGLE_FIELDS = (
    "shoulder_flexion",
    "shoulder_abduction",
    "shoulder_rotation",
    "elbow_flexion",
)


def angles_to_dict(a: ArmAngles | None) -> dict | None:
    """Serialise an ArmAngles (or None for an untracked limb)."""
    if a is None:
        return None
    return {
        "shoulder_flexion":   round(a.shoulder_flexion, 2),
        "shoulder_abduction": round(a.shoulder_abduction, 2),
        "shoulder_rotation":  round(a.shoulder_rotation, 2),
        "elbow_flexion":      round(a.elbow_flexion, 2),
        "rotation_reliable":  bool(a.rotation_reliable),
    }


def bilateral_to_dict(b: BilateralArmAngles | None) -> dict:
    """Serialise a BilateralArmAngles into {'right': ..., 'left': ...}."""
    if b is None:
        return {"right": None, "left": None}
    return {"right": angles_to_dict(b.right), "left": angles_to_dict(b.left)}


def _mean_angles(samples: list[ArmAngles]) -> ArmAngles:
    """Element-wise mean of a list of ArmAngles (all from the same side)."""
    n = len(samples)
    first = samples[0]
    return ArmAngles(
        shoulder_flexion   = sum(a.shoulder_flexion   for a in samples) / n,
        shoulder_abduction = sum(a.shoulder_abduction for a in samples) / n,
        shoulder_rotation  = sum(a.shoulder_rotation  for a in samples) / n,
        elbow_flexion      = sum(a.elbow_flexion      for a in samples) / n,
        rotation_reliable  = all(a.rotation_reliable for a in samples),
        side               = first.side,
    )


@dataclass
class PipelineConfig:
    """User-adjustable pipeline settings (all changeable while running)."""

    framework:    str   = "mediapipe"
    filter_type:  str   = "kalman"
    stream_hz:    float = 30.0
    udp_host:     str   = "127.0.0.1"
    udp_port:     int   = 9000
    udp_enabled:  bool  = True
    bilateral:    bool  = True
    mirror:       bool  = True
    detection_confidence: float = 0.5
    tracking_confidence:  float = 0.5
    model_complexity:     int   = 1
    latency_budget_ms:    float = 100.0
    metrics_window:       int   = 90

    def as_dict(self) -> dict:
        return {
            "framework": self.framework,
            "filter_type": self.filter_type,
            "stream_hz": self.stream_hz,
            "udp_host": self.udp_host,
            "udp_port": self.udp_port,
            "udp_enabled": self.udp_enabled,
            "bilateral": self.bilateral,
            "mirror": self.mirror,
            "detection_confidence": self.detection_confidence,
            "tracking_confidence": self.tracking_confidence,
            "model_complexity": self.model_complexity,
            "latency_budget_ms": self.latency_budget_ms,
            "metrics_window": self.metrics_window,
        }


@dataclass
class FrameResult:
    """Everything produced from a single frame, ready for JSON transport."""

    seq:        int
    timestamp:  float
    detected:   bool
    width:      int
    height:     int
    landmarks:  list = field(default_factory=list)
    raw:        dict = field(default_factory=dict)
    filtered:   dict = field(default_factory=dict)
    calibrated: dict = field(default_factory=dict)
    filter_bank: dict = field(default_factory=dict)
    trace:      dict = field(default_factory=dict)
    timings:    dict = field(default_factory=dict)
    status:     dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "type":       "frame",
            "seq":        self.seq,
            "t":          self.timestamp,
            "detected":   self.detected,
            "width":      self.width,
            "height":     self.height,
            "landmarks":  self.landmarks,
            "raw":        self.raw,
            "filtered":   self.filtered,
            "calibrated": self.calibrated,
            "filters":    self.filter_bank,
            "trace":      self.trace,
            "timings":    self.timings,
            "status":     self.status,
        }


class LivePipeline:
    """
    Thread-safe live processing chain.

    A single instance is shared by the HTTP control plane and the WebSocket
    data plane.  ``process_frame`` is called from a worker thread (one frame
    at a time, guarded by an internal lock); the control methods may be
    called concurrently from request handlers.
    """

    def __init__(self, config: PipelineConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._seq = 0
        self._started_at = time.time()

        # Pose estimator (constructed lazily so the server starts instantly)
        self._estimator = None
        self._estimator_name = config.framework
        self._estimator_error: str | None = None

        # One filter bank per family, all fed every frame
        self._banks = {
            ft: BilateralFilterBank(ft, stream_hz=config.stream_hz)
            for ft in FILTER_TYPES
        }

        # Per-side calibration (the manager models one arm at a time)
        self._calib = {"right": CalibrationManager(), "left": CalibrationManager()}
        self._calib_state = {
            "active": False, "side": "right", "pose": None,
            "captured": [], "remaining": list(REQUIRED_POSES),
            "samples": 0, "message": "",
        }
        self._load_calibration_if_present()

        # UDP transmit to Unity
        self._udp: UdpAngleSender | None = None
        if config.udp_enabled:
            self._start_udp()

        # CSV logging
        self._logger: BilateralCsvAngleLogger | None = None
        self._log_path: Path | None = None

        # Instrumentation, all rolling over the configured window
        win = max(int(config.metrics_window), 10)
        self.metrics = SessionMetrics(
            latency=LatencyTracker(win),
            rate=ThroughputTracker(win),
            stability=StabilityTracker(win),
            jitter=KeypointJitterTracker(win),
        )

        # Most recent calibrated pose per side (the "hold last value" source)
        self._last_good: dict[str, ArmAngles | None] = {"right": None, "left": None}
        # Short history of filtered angles per side, averaged when a
        # calibration reference pose is captured so a single noisy frame
        # cannot define the calibration.
        self._recent: dict[str, deque] = {
            "right": deque(maxlen=CALIBRATION_SAMPLE_FRAMES),
            "left":  deque(maxlen=CALIBRATION_SAMPLE_FRAMES),
        }

    # ------------------------------------------------------------------
    # Estimator lifecycle
    # ------------------------------------------------------------------

    def ensure_estimator(self):
        """Construct the pose estimator on first use; report load failures."""
        with self._lock:
            if self._estimator is not None:
                return self._estimator
            name = self.config.framework
            kwargs = {}
            if name == "mediapipe":
                kwargs = {
                    "detection_confidence": self.config.detection_confidence,
                    "tracking_confidence":  self.config.tracking_confidence,
                    "model_complexity":     self.config.model_complexity,
                }
            try:
                self._estimator = load_estimator(name, **kwargs)
                self._estimator_name = getattr(self._estimator, "name", name)
                self._estimator_error = None
            except Exception as exc:                      # pragma: no cover - env dependent
                self._estimator_error = f"{type(exc).__name__}: {exc}"
                raise
            return self._estimator

    def set_framework(self, name: str) -> None:
        """Swap the pose estimator, closing the previous one."""
        with self._lock:
            if name == self.config.framework and self._estimator is not None:
                return
            old = self._estimator
            self._estimator = None
            self.config.framework = name
            if old is not None:
                try:
                    old.close()
                except Exception:
                    pass
            self.reset_filters()
        self.ensure_estimator()

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def set_filter(self, filter_type: str) -> None:
        if filter_type not in FILTER_TYPES:
            raise ValueError(f"filter_type must be one of {FILTER_TYPES}")
        with self._lock:
            self.config.filter_type = filter_type

    def reset_filters(self) -> None:
        with self._lock:
            for bank in self._banks.values():
                bank.reset()
            self.metrics.stability.reset()
            self.metrics.jitter.reset()

    # ------------------------------------------------------------------
    # UDP transmit
    # ------------------------------------------------------------------

    def _start_udp(self) -> None:
        self._udp = UdpAngleSender(
            host=self.config.udp_host,
            port=self.config.udp_port,
            hz=self.config.stream_hz,
            bilateral=self.config.bilateral,
            verbose=False,
        )
        self._udp.start()

    def set_udp(self, enabled: bool, host: str | None = None,
                port: int | None = None, hz: float | None = None) -> dict:
        """Enable/disable or retarget the Unity UDP stream."""
        with self._lock:
            if host is not None:
                self.config.udp_host = host
            if port is not None:
                self.config.udp_port = int(port)
            if hz is not None:
                self.config.stream_hz = float(hz)
            self.config.udp_enabled = bool(enabled)

            if self._udp is not None:
                self._udp.stop()
                self._udp = None
            if self.config.udp_enabled:
                self._start_udp()
            return self.udp_state()

    def udp_state(self) -> dict:
        if self._udp is None:
            return {
                "enabled": False,
                "host": self.config.udp_host,
                "port": self.config.udp_port,
                "running": False,
                "packets_sent": 0,
                "send_errors": 0,
                "target_hz": self.config.stream_hz,
                "bilateral": self.config.bilateral,
                "last_packet": "",
                "history": [],
            }
        state = self._udp.wire_state()
        state["enabled"] = True
        return state

    # ------------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------------

    def start_logging(self, label: str = "") -> dict:
        with self._lock:
            if self._logger is not None:
                return self.logging_state()
            self._logger = BilateralCsvAngleLogger(
                output_dir=self.output_dir / "logs",
                filter_type=self.config.filter_type,
                session_label=label,
            )
            self._log_path = self._logger.start(
                calibrated=self._calib["right"].data.calibrated
            )
            return self.logging_state()

    def stop_logging(self) -> dict:
        with self._lock:
            if self._logger is None:
                return self.logging_state()
            self._logger.stop()
            self._logger = None
            state = self.logging_state()
            state["path"] = str(self._log_path) if self._log_path else None
            return state

    def logging_state(self) -> dict:
        return {
            "active": self._logger is not None,
            "path": str(self._log_path) if self._log_path else None,
            "filename": self._log_path.name if self._log_path else None,
        }

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _load_calibration_if_present(self) -> None:
        # Calibrations are saved per side as calibration_<side>.json
        for side in ("right", "left"):
            path = self.output_dir / f"calibration_{side}.json"
            if path.exists():
                try:
                    self._calib[side].load(path)
                except Exception:
                    pass

    def calibration_begin(self, side: str = "right") -> dict:
        with self._lock:
            if side not in ("right", "left"):
                raise ValueError("side must be 'right' or 'left'")
            mgr = self._calib[side]
            mgr.begin_static()
            pose = mgr.next_pose()
            self._calib_state = {
                "active": True,
                "side": side,
                "pose": pose,
                "captured": [],
                "remaining": [p for p in REQUIRED_POSES if p != pose],
                "samples": 0,
                "warnings": [],
                "message": f"Hold the '{pose}' reference pose, then capture.",
            }
            return self.calibration_state()

    def calibration_capture(self) -> dict:
        """
        Capture the current reference pose.

        The last :data:`CALIBRATION_SAMPLE_FRAMES` filtered frames for the
        selected arm are averaged, so the recorded reference reflects a held
        posture rather than one arbitrarily-timed frame. Capturing fails
        loudly when the arm is not currently tracked — an untracked limb has
        no measurement to record.
        """
        with self._lock:
            st = self._calib_state
            if not st["active"]:
                raise RuntimeError("Calibration is not active.")
            side = st["side"]
            history = list(self._recent[side])
            if not history:
                raise RuntimeError(
                    f"No {side} arm currently tracked — step into frame before capturing."
                )
            angles = _mean_angles(history)
            mgr = self._calib[side]
            mgr.capture_pose(angles)
            st["samples"] = len(history)
            st["captured"] = st["captured"] + [st["pose"]]
            nxt = mgr.next_pose()
            st["pose"] = nxt
            st["remaining"] = [p for p in REQUIRED_POSES if p not in st["captured"]]
            if nxt is None:
                mgr.finalise()
                mgr.save(self.output_dir / f"calibration_{side}.json")
                st["active"] = False
                st["warnings"] = list(mgr.warnings)
                st["message"] = (
                    f"Calibration complete for the {side} arm and saved."
                    if not mgr.warnings else
                    f"Calibration saved for the {side} arm, but "
                    f"{len(mgr.warnings)} reference pose(s) were unusable."
                )
            else:
                st["message"] = f"Hold the '{nxt}' reference pose, then capture."
            return self.calibration_state()

    def calibration_cancel(self) -> dict:
        with self._lock:
            self._calib_state.update(
                {"active": False, "pose": None, "message": "Calibration cancelled."}
            )
            return self.calibration_state()

    def calibration_clear(self, side: str = "right") -> dict:
        with self._lock:
            self._calib[side] = CalibrationManager()
            path = self.output_dir / f"calibration_{side}.json"
            if path.exists():
                path.unlink()
            self._calib_state["message"] = f"Calibration cleared for the {side} arm."
            return self.calibration_state()

    def calibration_state(self) -> dict:
        st = dict(self._calib_state)
        st["required_poses"] = list(REQUIRED_POSES)
        st["warnings"] = {
            side: list(self._calib[side].warnings) for side in ("right", "left")
        }
        st["calibrated"] = {
            side: bool(self._calib[side].data.calibrated) for side in ("right", "left")
        }
        st["parameters"] = {
            side: {
                dof: {
                    "offset": round(getattr(self._calib[side].data, dof).offset, 4),
                    "scale":  round(getattr(self._calib[side].data, dof).scale, 4),
                }
                for dof in ("flexion", "abduction", "rotation", "elbow")
            }
            for side in ("right", "left")
        }
        return st

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        transport_ms: float = 0.0,
        decode_ms: float = 0.0,
    ) -> FrameResult:
        """
        Run one frame through the whole chain and return everything measured.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Frame in OpenCV BGR order.
        transport_ms : float
            Measured client→server transport time for browser-sourced frames.
        decode_ms : float
            Measured JPEG decode (or camera read) time for this frame.
        """
        import cv2  # local import keeps module import cheap for tests

        t = StageTimings(transport_ms=transport_ms, decode_ms=decode_ms)
        h, w = frame_bgr.shape[:2]

        with self._lock:
            self._seq += 1
            seq = self._seq
            filter_type = self.config.filter_type

        estimator = self.ensure_estimator()

        # --- pose estimation -------------------------------------------------
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks = estimator.process(rgb)
        t.pose_ms = (time.perf_counter() - t0) * 1000.0

        # --- joint angles ----------------------------------------------------
        # Geometry runs on isotropic coordinates: metric world landmarks when
        # the network provides them, else image landmarks scaled to pixels.
        t0 = time.perf_counter()
        geo = angle_landmarks(estimator, landmarks, w, h)
        space = "world" if geo is not None and geo is getattr(estimator, "world_landmarks", None) else "pixels"
        raw = compute_bilateral_angles(geo) if geo is not None else None
        frame_obj = build_torso_frame(geo) if geo is not None else None
        t.angles_ms = (time.perf_counter() - t0) * 1000.0

        # --- temporal filtering (all families) -------------------------------
        t0 = time.perf_counter()
        bank_out: dict[str, BilateralArmAngles | None] = {}
        if raw is not None:
            with self._lock:
                for ft, bank in self._banks.items():
                    bank_out[ft] = bank.update(raw)
        else:
            bank_out = {ft: None for ft in FILTER_TYPES}
        active = bank_out.get(filter_type)
        t.filter_ms = (time.perf_counter() - t0) * 1000.0

        # --- calibration -----------------------------------------------------
        t0 = time.perf_counter()
        calibrated = None
        if active is not None:
            calibrated = BilateralArmAngles(
                right=self._calib["right"].apply(active.right) if active.right else None,
                left=self._calib["left"].apply(active.left) if active.left else None,
            )
            with self._lock:
                for side in ("right", "left"):
                    a = getattr(calibrated, side)
                    if a is not None:
                        self._last_good[side] = a
                    # Calibration references are captured from the *filtered
                    # but uncalibrated* signal, which is what the mapping is
                    # fitted against.
                    f = getattr(active, side)
                    if f is not None:
                        self._recent[side].append(f)
        t.calib_ms = (time.perf_counter() - t0) * 1000.0

        # --- transmit + log --------------------------------------------------
        t0 = time.perf_counter()
        if calibrated is not None:
            if self._udp is not None:
                if self.config.bilateral:
                    self._udp.update_bilateral(calibrated)
                elif calibrated.right is not None:
                    self._udp.update(calibrated.right)
            if self._logger is not None:
                self._logger.log(calibrated)
        t.stream_ms = (time.perf_counter() - t0) * 1000.0
        t.finalise()

        # --- instrumentation -------------------------------------------------
        detected = landmarks is not None
        self.metrics.latency.add(t)
        self.metrics.rate.add(detected)
        self.metrics.jitter.add(landmarks, w, h)
        if raw is not None:
            for side in ("right", "left"):
                self.metrics.stability.add("raw", side, getattr(raw, side))
            for ft, out in bank_out.items():
                if out is None:
                    continue
                for side in ("right", "left"):
                    self.metrics.stability.add(ft, side, getattr(out, side))

        return FrameResult(
            seq=seq,
            timestamp=time.time(),
            detected=detected,
            width=w,
            height=h,
            landmarks=self._serialise_landmarks(landmarks),
            raw=bilateral_to_dict(raw),
            filtered=bilateral_to_dict(active),
            calibrated=bilateral_to_dict(calibrated),
            filter_bank={ft: bilateral_to_dict(out) for ft, out in bank_out.items()},
            trace=self._build_trace(landmarks, geo, space, frame_obj, raw),
            timings=t.as_dict(),
            status={
                "framework":   self._estimator_name,
                "filter_type": filter_type,
                "logging":     self._logger is not None,
                "udp":         self.udp_state(),
                "calibrated":  {s: bool(self._calib[s].data.calibrated)
                                for s in ("right", "left")},
                "rotation_threshold_deg": ROTATION_RELIABLE_THRESHOLD,
            },
        )

    # ------------------------------------------------------------------
    # Explanatory trace
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise_landmarks(landmarks) -> list:
        """Forward the overlay landmarks with their detection confidence."""
        if landmarks is None:
            return []
        out = []
        for i in OVERLAY_LANDMARKS:
            if i >= len(landmarks):
                continue
            lm = landmarks[i]
            out.append({
                "i": i,
                "x": round(float(lm.x), 5),
                "y": round(float(lm.y), 5),
                "z": round(float(lm.z), 5),
                "v": round(float(getattr(lm, "visibility", 1.0)), 4),
            })
        return out

    def _build_trace(self, landmarks, geo, space, frame, raw) -> dict:
        """
        Expose the intermediate quantities the angles were derived from.

        The dashboard renders this as a step-by-step derivation so a reader
        can follow a single frame from the keypoints, through the torso
        reference frame and the segment vectors expressed in it, to the four
        reported joint angles — no step is a black box.

        ``landmarks`` are the image keypoints (for visibility); ``geo`` are
        the isotropic coordinates the angles were actually computed from.
        """
        if landmarks is None or geo is None or frame is None:
            return {}

        def _vec(v) -> list:
            return [round(float(c), 5) for c in v]

        trace = {
            "torso_frame": {
                "origin": _vec(frame.origin),
                "x_axis_lateral":  _vec(frame.x_axis),
                "y_axis_superior": _vec(frame.y_axis),
                "z_axis_anterior": _vec(frame.z_axis),
            },
            "coordinate_space": _COORDINATE_SPACES[space],
            "sides": {},
        }

        for side in ("right", "left"):
            i_sh, i_el, i_wr = _SIDE_LANDMARKS[side]
            try:
                p_sh = _to_array(geo[i_sh])
                p_el = _to_array(geo[i_el])
                p_wr = _to_array(geo[i_wr])
            except (IndexError, TypeError):
                continue

            v_upper_world = p_el - p_sh
            v_fore_world  = p_wr - p_el
            v_upper_torso = world_to_torso(v_upper_world, frame)
            v_fore_torso  = world_to_torso(v_fore_world, frame)
            if side == "left":
                mirror = np.array([-1.0, 1.0, 1.0])
                v_upper_torso = v_upper_torso * mirror
                v_fore_torso  = v_fore_torso * mirror

            u = _normalize(v_upper_torso)
            angles = getattr(raw, side) if raw is not None else None
            visibilities = {
                "shoulder": round(float(getattr(landmarks[i_sh], "visibility", 1.0)), 3),
                "elbow":    round(float(getattr(landmarks[i_el], "visibility", 1.0)), 3),
                "wrist":    round(float(getattr(landmarks[i_wr], "visibility", 1.0)), 3),
            }
            trace["sides"][side] = {
                "landmark_indices": {"shoulder": i_sh, "elbow": i_el, "wrist": i_wr},
                "visibility": visibilities,
                "min_visibility": round(min(visibilities.values()), 3),
                "upper_arm_world": _vec(v_upper_world),
                "forearm_world":   _vec(v_fore_world),
                "upper_arm_torso": _vec(v_upper_torso),
                "forearm_torso":   _vec(v_fore_torso),
                "upper_arm_unit_torso": _vec(u),
                "segment_lengths": {
                    "upper_arm": round(float(np.linalg.norm(v_upper_world)), 5),
                    "forearm":   round(float(np.linalg.norm(v_fore_world)), 5),
                },
                "angles": angles_to_dict(angles),
            }
        return trace

    # ------------------------------------------------------------------
    # Status / shutdown
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "config":      self.config.as_dict(),
            "framework":   self._estimator_name,
            "estimator_loaded": self._estimator is not None,
            "estimator_error":  self._estimator_error,
            "filter_type": self.config.filter_type,
            "filters_available": list(FILTER_TYPES),
            "udp":         self.udp_state(),
            "logging":     self.logging_state(),
            "calibration": self.calibration_state(),
            "metrics":     self.metrics.snapshot(self.config.latency_budget_ms),
            "started_at":  self._started_at,
            "frames":      self._seq,
        }

    def close(self) -> None:
        with self._lock:
            if self._logger is not None:
                self._logger.stop()
                self._logger = None
            if self._udp is not None:
                self._udp.stop()
                self._udp = None
            if self._estimator is not None:
                try:
                    self._estimator.close()
                except Exception:
                    pass
                self._estimator = None
