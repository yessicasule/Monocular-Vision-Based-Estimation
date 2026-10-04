"""
base.py — Abstract Pose Estimator Interface
============================================

Defines the unified interface that MediaPipe, MoveNet, and PoseNet runners
must implement, enabling the angle solver and benchmarking pipeline to be
framework-agnostic.

All runners return landmarks as a list of 33 Landmark objects indexed with
MediaPipe's 33-keypoint convention. Frameworks that use a different keypoint
schema (e.g. COCO 17) must remap internally before returning.

MediaPipe 33-Keypoint Indices (used throughout the pipeline)
-------------------------------------------------------------
    0  — nose
    11 — left shoulder      12 — right shoulder
    13 — left elbow         14 — right elbow
    15 — left wrist         16 — right wrist
    23 — left hip           24 — right hip
    (full spec: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)

Landmark Coordinate System
---------------------------
    x, y  : normalised image coordinates [0, 1]
    z     : pseudo-depth relative to the hip (MediaPipe convention)
             Negative = closer to camera, positive = further.
             For frameworks without depth output, z should be set to 0.
    visibility : detection confidence in [0, 1].
                 Values < 0.5 indicate the joint is likely occluded.

Angle-Space Landmarks
---------------------
Normalised image coordinates are NOT isotropic: x is divided by the image
width, y by the height, so on a 16:9 frame one unit of y is 1.78× shorter
than one unit of x. Angles computed directly from them are distorted.
:func:`angle_landmarks` returns the coordinates joint-angle geometry must use:
the estimator's metric world landmarks when it provides them (MediaPipe),
otherwise the image landmarks rescaled to pixels (x·W, y·H, z·W — MediaPipe's
z shares x's scale).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Landmark:
    """Single 3D pose landmark in MediaPipe normalised image space."""
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


# Number of landmarks in the MediaPipe 33-keypoint schema
N_LANDMARKS = 33

# Convenience: build a list of N_LANDMARKS default (0,0,0) landmarks
def _default_landmarks() -> list[Landmark]:
    return [Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(N_LANDMARKS)]


def to_pixel_space(landmarks: list[Landmark], width: int, height: int) -> list[Landmark]:
    """Rescale normalised landmarks so x, y and z share one unit (pixels)."""
    return [
        Landmark(x=lm.x * width, y=lm.y * height, z=lm.z * width,
                 visibility=lm.visibility)
        for lm in landmarks
    ]


def angle_landmarks(
    estimator: "PoseEstimator",
    landmarks: list[Landmark] | None,
    width: int,
    height: int,
) -> list[Landmark] | None:
    """
    Landmarks in an isotropic space, ready for joint-angle geometry.

    Prefers the estimator's metric world landmarks for the frame just
    processed; falls back to the image landmarks rescaled to pixels.
    """
    if landmarks is None:
        return None
    world = getattr(estimator, "world_landmarks", None)
    if world is not None and len(world) == len(landmarks):
        return world
    return to_pixel_space(landmarks, width, height)


class PoseEstimator(ABC):
    """
    Abstract base class for all pose estimation backends.

    All subclasses must implement:
        process(image_rgb) → list[Landmark] | None
        close()            → None

    The process() return value must be a list of exactly 33 Landmark objects
    following MediaPipe's indexing convention. If the underlying framework
    uses a different keypoint schema, the subclass is responsible for remapping.

    If no person is detected, process() returns None.

    Estimators that also produce metric 3D landmarks set ``world_landmarks``
    for the frame just processed (None otherwise); see :func:`angle_landmarks`.
    """

    world_landmarks: list[Landmark] | None = None

    @property
    def name(self) -> str:
        """Human-readable framework name. Override in subclasses."""
        return self.__class__.__name__

    @abstractmethod
    def process(self, image_rgb: np.ndarray) -> list[Landmark] | None:
        """
        Run pose estimation on a single RGB frame.

        Parameters
        ----------
        image_rgb : np.ndarray, shape (H, W, 3), dtype uint8
            Input frame in RGB colour order.

        Returns
        -------
        list[Landmark] of length 33, or None if no person detected.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the estimator (e.g. TF sessions, MP graphs)."""
        ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
