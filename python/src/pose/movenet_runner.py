"""
movenet_runner.py — MoveNet Pose Estimator
==========================================

Implements the PoseEstimator interface using Google MoveNet via TensorFlow Hub.

Critical Design: COCO → MediaPipe Landmark Remapping
------------------------------------------------------
MoveNet outputs COCO 17-keypoint predictions, NOT MediaPipe's 33-keypoint
schema. The original code directly passed MoveNet output to compute_all()
which uses MediaPipe indices (shoulder=11, elbow=13, wrist=15, hip=23/24).
This caused WRONG body parts to be used for angle computation.

This implementation maps MoveNet's 17-joint COCO output to a padded
33-element Landmark list that exactly matches MediaPipe's indexing, so
the angle solver works identically regardless of which framework runs.

MoveNet COCO 17-Keypoint Indices
---------------------------------
    0  — nose
    1  — left eye          2  — right eye
    3  — left ear          4  — right ear
    5  — left shoulder     6  — right shoulder
    7  — left elbow        8  — right elbow
    9  — left wrist        10 — right wrist
    11 — left hip          12 — right hip
    13 — left knee         14 — right knee
    15 — left ankle        16 — right ankle

MoveNet Output Format
---------------------
    shape: [17, 3]   where each row is [y_norm, x_norm, score]
    Note the Y-BEFORE-X ordering (opposite to MediaPipe's x, y, z).

MediaPipe Landmark Indices (required by angle solver)
------------------------------------------------------
    11 — left shoulder     12 — right shoulder
    13 — left elbow        14 — right elbow
    15 — left wrist        16 — right wrist
    23 — left hip          24 — right hip

MoveNet → MediaPipe Index Mapping
-----------------------------------
    COCO 5  → MP 11  (left shoulder)
    COCO 6  → MP 12  (right shoulder)
    COCO 7  → MP 13  (left elbow)
    COCO 8  → MP 14  (right elbow)
    COCO 9  → MP 15  (left wrist)
    COCO 10 → MP 16  (right wrist)
    COCO 11 → MP 23  (left hip)
    COCO 12 → MP 24  (right hip)

All other MediaPipe indices are set to zero-visibility placeholders.

MoveNet Variants
----------------
    movenet_lightning : 192×192 input, fastest (~30+ FPS on CPU)
    movenet_thunder   : 256×256 input, more accurate, slower

CPU-Only Note
-------------
MoveNet uses TensorFlow. CPU inference is supported but significantly
slower than GPU/EdgeTPU. For CPU-only constraint: prefer `movenet_lightning`.

References:
    Bazarevsky et al. (2022). MoveNet: Towards a Model for Movement.
    Google AI Blog. https://ai.googleblog.com/2021/05/next-generation-pose-detection-with.html
    COCO keypoint annotations: https://cocodataset.org/#keypoints-2020
"""

from __future__ import annotations

import numpy as np

from .base import PoseEstimator, Landmark, N_LANDMARKS, _default_landmarks

# --------------------------------------------------------------------------
# COCO 17 → MediaPipe 33 index mapping
# --------------------------------------------------------------------------
# Format: { coco_index: mediapipe_index }
# Only the joints needed by the angle solver are mapped.
# All unmapped MediaPipe indices are left as zero-visibility landmarks.
_COCO_TO_MP = {
    5:  11,   # left shoulder
    6:  12,   # right shoulder
    7:  13,   # left elbow
    8:  14,   # right elbow
    9:  15,   # left wrist
    10: 16,   # right wrist
    11: 23,   # left hip
    12: 24,   # right hip
}

# Minimum COCO keypoint confidence to treat as detected
_MIN_SCORE = 0.2


class MoveNetRunner(PoseEstimator):
    """
    MoveNet single-person pose estimator implementing PoseEstimator.

    Internally uses TensorFlow Hub to load the MoveNet model and runs
    inference on CPU. Output is remapped from COCO 17 to MediaPipe 33
    keypoint format before returning.

    Parameters
    ----------
    variant : str
        "lightning" (faster, less accurate) or "thunder" (slower, more accurate).
    """

    _HUB_URLS = {
        "lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
        "thunder":   "https://tfhub.dev/google/movenet/singlepose/thunder/4",
    }
    _INPUT_SIZES = {
        "lightning": 192,
        "thunder":   256,
    }

    @property
    def name(self) -> str:
        return f"MoveNet-{self._variant.capitalize()}"

    def __init__(self, variant: str = "lightning") -> None:
        if variant not in self._HUB_URLS:
            raise ValueError(f"variant must be 'lightning' or 'thunder', got '{variant}'")
        self._variant = variant
        self._input_size = self._INPUT_SIZES[variant]

        from .tf_guard import hide_unloadable_tensorflow
        tf_error = hide_unloadable_tensorflow()
        if tf_error:
            raise RuntimeError(f"MoveNet needs TensorFlow, which is installed but cannot load: {tf_error}")

        print(f"[MoveNet] Loading {variant} from TensorFlow Hub...")
        import tensorflow_hub as hub
        module = hub.load(self._HUB_URLS[variant])
        self._model = module.signatures["serving_default"]
        print(f"[MoveNet] {variant} loaded. Input size: {self._input_size}×{self._input_size}")

    # ------------------------------------------------------------------
    # PoseEstimator interface
    # ------------------------------------------------------------------

    def process(self, image_rgb: np.ndarray) -> list[Landmark] | None:
        """
        Run MoveNet inference and return MediaPipe-format landmarks.

        The image is resized to the model's input size with padding to
        preserve aspect ratio (tf.image.resize_with_pad).

        Returns None if all required keypoints have score < _MIN_SCORE.
        """
        import tensorflow as tf

        # Resize and pad to square input (preserves aspect ratio)
        img_tensor = tf.image.resize_with_pad(image_rgb, self._input_size, self._input_size)
        img_tensor = tf.cast(img_tensor, dtype=tf.int32)
        img_tensor = tf.expand_dims(img_tensor, axis=0)   # [1, H, W, 3]

        outputs = self._model(img_tensor)
        # Output shape: [1, 1, 17, 3]  where last dim = [y, x, score]
        kps = outputs["output_0"].numpy()[0, 0]   # shape [17, 3]

        return self._remap_to_mediapipe(kps)

    # ------------------------------------------------------------------
    # Private: COCO → MediaPipe remapping
    # ------------------------------------------------------------------

    def _remap_to_mediapipe(self, kps: np.ndarray) -> list[Landmark] | None:
        """
        Remap MoveNet's [17, 3] COCO keypoints to a 33-element MediaPipe
        landmark list.

        Parameters
        ----------
        kps : np.ndarray, shape (17, 3)
            Each row: [y_norm, x_norm, confidence_score]

        Returns
        -------
        list[Landmark] of length 33, or None if required joints undetected.
        """
        # Check that at least the key arm/hip joints have acceptable confidence
        required_coco = [5, 6, 7, 8, 11, 12]    # shoulders, elbows, hips
        if not any(kps[i, 2] >= _MIN_SCORE for i in required_coco):
            return None

        landmarks = _default_landmarks()   # 33 zero-visibility placeholders

        for coco_idx, mp_idx in _COCO_TO_MP.items():
            y, x, score = float(kps[coco_idx, 0]), float(kps[coco_idx, 1]), float(kps[coco_idx, 2])
            landmarks[mp_idx] = Landmark(
                x=x,
                y=y,
                z=0.0,          # MoveNet has no depth estimate
                visibility=score,
            )

        return landmarks

    def close(self) -> None:
        # TF Hub models don't require explicit teardown
        pass
