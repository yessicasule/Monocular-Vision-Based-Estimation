"""
mediapipe_runner.py — MediaPipe Pose Estimator
===============================================

Implements the PoseEstimator interface using Google MediaPipe Pose.

API Compatibility
-----------------
MediaPipe ships two pose inference APIs:

1. Legacy Solutions API (mediapipe < 0.10.x)
   mp.solutions.pose.Pose()
   Returns: results.pose_landmarks.landmark (direct list access)

2. Tasks API (mediapipe >= 0.10.x, recommended)
   vision.PoseLandmarker
   Returns: result.pose_landmarks[0] (NormalizedLandmark list)

This runner tries the Solutions API first and falls back to the Tasks API
if the solutions module is unavailable (it was removed in mediapipe 0.10.3x).
Both paths return the same Landmark interface to the rest of the pipeline,
and both honour ``model_complexity`` (Tasks API: lite / full / heavy model).

World Landmarks
---------------
Besides the normalised image landmarks, both APIs return *world* landmarks:
metric 3D coordinates in metres with the hip midpoint as origin and the same
axis orientation as the image (x right, y down, z away from the camera).
Unlike the image landmarks, their three axes share one unit, so they are the
correct input for joint-angle geometry. The latest set is exposed as
``self.world_landmarks``.

Timestamp Handling (Tasks API)
-------------------------------
The Tasks API requires a monotonically increasing timestamp in milliseconds.
We track elapsed wall-clock time from process() calls rather than adding a
fixed 33 ms per frame, which was incorrect in the original code and caused
drift at non-30 fps framerates.

Coordinate System
-----------------
MediaPipe returns landmark coordinates normalised to image dimensions:
    x ∈ [0, 1]  — horizontal, left-to-right
    y ∈ [0, 1]  — vertical, top-to-bottom
    z           — pseudo-depth relative to hip (same scale as x)

References:
    https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
    Bazarevsky et al. (2020). BlazePose: On-device Real-time Body Pose
    Tracking. arXiv:2006.10204.
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np

from .base import PoseEstimator, Landmark, N_LANDMARKS
from .tf_guard import hide_unloadable_tensorflow

# --------------------------------------------------------------------------
# Model download (Tasks API only)
# --------------------------------------------------------------------------
_MODEL_DIR      = Path(__file__).resolve().parent / "models"
_MODEL_VARIANTS = {0: "lite", 1: "full", 2: "heavy"}   # model_complexity → model


def _model_path(variant: str) -> Path:
    return _MODEL_DIR / f"pose_landmarker_{variant}.task"


def _model_url(variant: str) -> str:
    return (
        "https://storage.googleapis.com/mediapipe-models/"
        f"pose_landmarker/pose_landmarker_{variant}/float16/latest/"
        f"pose_landmarker_{variant}.task"
    )


def _ensure_model(variant: str) -> Path:
    """Return the .task file for `variant`, downloading it on first use."""
    path = _model_path(variant)
    if not path.exists():
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"[MediaPipe] Downloading {variant} model -> {path}")
        urllib.request.urlretrieve(_model_url(variant), tmp)
        tmp.replace(path)                # never leave a truncated model behind
        print("[MediaPipe] Download complete.")
    return path


class MediaPipeRunner(PoseEstimator):
    """
    MediaPipe Pose runner implementing the PoseEstimator interface.

    Returns 33 Landmark objects in MediaPipe's standard indexing.
    z coordinates are MediaPipe's pseudo-depth estimates (relative to hip).

    Parameters
    ----------
    detection_confidence : float
        Minimum confidence for person detection [0, 1].
    tracking_confidence : float
        Minimum confidence for landmark tracking [0, 1].
        When tracking confidence drops below this, detection re-runs.
    model_complexity : int
        0 = Lite (fastest, least accurate)
        1 = Full (balanced)  ← default
        2 = Heavy (most accurate, slowest)
    """

    @property
    def name(self) -> str:
        return "MediaPipe"

    def __init__(
        self,
        detection_confidence: float = 0.5,
        tracking_confidence:  float = 0.5,
        model_complexity:     int   = 1,
    ) -> None:
        self._use_tasks   = False
        self._start_time  = time.perf_counter()
        self._last_ts_ms  = -1   # last timestamp handed to detect_for_video()
        # Metric 3D landmarks of the most recent frame (None if not detected)
        self.world_landmarks: list[Landmark] | None = None
        # Which Tasks model actually loaded ("lite" / "full" / "heavy")
        self.model_variant: str | None = None

        # MediaPipe imports TensorFlow for doc helpers; a TF that is
        # installed but unloadable must not break MediaPipe.
        hide_unloadable_tensorflow()

        try:
            self._init_solutions(detection_confidence, tracking_confidence, model_complexity)
        except Exception as e:
            print(f"[MediaPipe] Solutions API unavailable ({e}), trying Tasks API...")
            self._use_tasks = True
            self._init_tasks(detection_confidence, tracking_confidence, model_complexity)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_solutions(
        self,
        det_conf:   float,
        track_conf: float,
        complexity: int,
    ) -> None:
        try:
            import mediapipe.python.solutions.pose as _pose_mod
        except ModuleNotFoundError:
            import importlib
            _pose_mod = importlib.import_module("mediapipe.solutions.pose")

        self._pose_mod = _pose_mod
        self.pose = _pose_mod.Pose(
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
            model_complexity=complexity,
        )

    def _init_tasks(self, det_conf: float, track_conf: float, complexity: int = 1) -> None:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        variant = _MODEL_VARIANTS.get(int(complexity), "full")
        try:
            model_path = _ensure_model(variant)
        except Exception as exc:
            if variant == "lite":
                raise
            # Offline and the requested model is not cached: the bundled
            # lite model still works, but say so rather than pretend.
            print(f"[MediaPipe] Could not obtain the {variant} model ({exc}); "
                  "falling back to the bundled lite model.")
            variant = "lite"
            model_path = _ensure_model(variant)
        self.model_variant = variant

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=det_conf,
            min_pose_presence_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.pose = vision.PoseLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # PoseEstimator interface
    # ------------------------------------------------------------------

    def process(self, image_rgb: np.ndarray) -> list[Landmark] | None:
        if self._use_tasks:
            return self._process_tasks(image_rgb)
        return self._process_solutions(image_rgb)

    def _process_solutions(self, image_rgb: np.ndarray) -> list[Landmark] | None:
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            self.world_landmarks = None
            return None
        world = getattr(results, "pose_world_landmarks", None)
        self.world_landmarks = (
            [Landmark(x=float(lm.x), y=float(lm.y), z=float(lm.z),
                      visibility=float(lm.visibility)) for lm in world.landmark]
            if world else None
        )
        return [
            Landmark(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility),
            )
            for lm in results.pose_landmarks.landmark
        ]

    def _process_tasks(self, image_rgb: np.ndarray) -> list[Landmark] | None:
        import mediapipe as mp

        # Use real elapsed time for the timestamp (rather than a hardcoded
        # per-frame increment, which drifts at non-30fps rates). detect_for_video()
        # requires STRICTLY increasing timestamps, but an int millisecond count
        # can repeat when two frames are processed inside the same millisecond
        # (fast offline loops over pre-extracted frames, or coarse timer
        # resolution), so step forward by 1ms whenever that happens.
        ts_ms = int((time.perf_counter() - self._start_time) * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.pose.detect_for_video(mp_img, ts_ms)

        if not result.pose_landmarks:
            self.world_landmarks = None
            return None
        image_lms = result.pose_landmarks[0]
        world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
        # World landmarks carry no visibility of their own; reuse the image one
        self.world_landmarks = (
            [Landmark(x=float(w.x), y=float(w.y), z=float(w.z),
                      visibility=float(i.visibility)) for w, i in zip(world, image_lms)]
            if world else None
        )
        return [
            Landmark(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility),
            )
            for lm in image_lms
        ]

    def close(self) -> None:
        try:
            self.pose.close()
        except Exception:
            pass
