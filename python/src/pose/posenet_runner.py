"""
posenet_runner.py — PoseNet Pose Estimator (CPU, TensorFlow Lite)
=================================================================

Implements the PoseEstimator interface using PoseNet via TensorFlow Lite.
Runs on CPU only (no GPU requirement), suitable for the paper's CPU-only
constraint.

Original Issue with Previous Implementation
--------------------------------------------
The original posenet_runner.py was a batch-processing wrapper around a
Node.js script (run_posenet.mjs), which:
    - Had no per-frame process() method (incompatible with the pipeline API)
    - Required Node.js to be installed
    - Only supported offline batch mode, not real-time streaming

This implementation replaces it with a pure-Python, CPU-only TFLite
runtime that:
    - Has a proper per-frame process(image_rgb) → list[Landmark] interface
    - Runs entirely in Python (no Node.js dependency)
    - Supports both ResNet-50 and MobileNetV1 backends

Architecture
------------
PoseNet uses a fully convolutional architecture (either MobileNetV1 or
ResNet-50) followed by a heatmap decoder. The model outputs:
    - heatmaps  : [1, H_out, W_out, 17]  — joint confidence at each position
    - offsets   : [1, H_out, W_out, 34]  — sub-cell offset refinement

Keypoint Detection
------------------
For each of 17 COCO joints, the argmax location of the heatmap slice
(after sigmoid) gives the rough joint position. The offset vectors are
then added to get sub-cell precision.

Keypoint Schema
---------------
PoseNet uses COCO 17-keypoint ordering (same as MoveNet). Output is
remapped to MediaPipe 33 via the same mapping table as MoveNetRunner.

Model Source
------------
TFLite PoseNet models are downloaded from the TensorFlow Lite model hub:
    https://tfhub.dev/tensorflow/lite-model/posenet/mobilenet/float/075/1/default/1

If the model file is not present, it will be downloaded automatically.

References:
    Papandreou et al. (2018). PersonLab: Person Pose Estimation and
    Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding
    Model. ECCV 2018.
    Google TFLite PoseNet guide:
    https://www.tensorflow.org/lite/examples/pose_estimation/overview
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from .base import PoseEstimator, Landmark, N_LANDMARKS, _default_landmarks

# ------------------------------------------------------------------
# COCO 17 → MediaPipe 33 remapping (shared with MoveNet)
# ------------------------------------------------------------------
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

_MIN_SCORE = 0.2

# ------------------------------------------------------------------
# Model download config
# ------------------------------------------------------------------
_MODEL_DIR  = Path(__file__).resolve().parent / "models"
_POSENET_MODEL_PATH = _MODEL_DIR / "posenet_mobilenet_v1_100_257x257.tflite"
_POSENET_MODEL_URL  = (
    "https://storage.googleapis.com/download.tensorflow.org/models/"
    "tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
)


def _ensure_posenet_model() -> Path:
    if not _POSENET_MODEL_PATH.exists():
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[PoseNet] Downloading TFLite model → {_POSENET_MODEL_PATH}")
        urllib.request.urlretrieve(_POSENET_MODEL_URL, _POSENET_MODEL_PATH)
        print("[PoseNet] Download complete.")
    return _POSENET_MODEL_PATH


class PoseNetRunner(PoseEstimator):
    """
    PoseNet per-frame pose estimator using TensorFlow Lite on CPU.

    This is a complete rewrite of the original batch-subprocess implementation.
    Uses TFLite Interpreter for lightweight, dependency-minimal inference.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to a .tflite PoseNet model. If None, the MobileNetV1-100
        257×257 model is downloaded automatically.

    Notes
    -----
    TFLite inference on CPU is intentionally slower than MediaPipe and
    MoveNet (which are optimised for CPU path). This runner is included
    for the framework comparison study, not as the production backend.
    """

    @property
    def name(self) -> str:
        return "PoseNet"

    def __init__(self, model_path: str | Path | None = None) -> None:
        try:
            from ai_edge_litert.interpreter import Interpreter
            _Interpreter = Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
                _Interpreter = Interpreter
            except ImportError:
                try:
                    import tensorflow as tf
                    _Interpreter = tf.lite.Interpreter
                except ImportError:
                    raise ImportError(
                        "PoseNet requires 'ai-edge-litert', 'tflite-runtime', or 'tensorflow'.\n"
                        "Install with: pip install ai-edge-litert"
                    )

        path = Path(model_path) if model_path else _ensure_posenet_model()
        self._interpreter = _Interpreter(model_path=str(path))
        self._interpreter.allocate_tensors()

        self._input_details  = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Input tensor shape: [1, H, W, 3]
        input_shape = self._input_details[0]["shape"]
        self._input_h = int(input_shape[1])
        self._input_w = int(input_shape[2])
        print(f"[PoseNet] TFLite model loaded. Input: {self._input_h}×{self._input_w}")

    # ------------------------------------------------------------------
    # PoseEstimator interface
    # ------------------------------------------------------------------

    def process(self, image_rgb: np.ndarray) -> list[Landmark] | None:
        """
        Run PoseNet inference on a single RGB frame.

        Algorithm
        ---------
        1. Resize image to model input dimensions (bilinear interpolation).
        2. Convert to float32 and normalise to [0, 1] (required by this model).
        3. Run TFLite inference.
        4. Decode heatmaps → COCO 17 keypoints using argmax + offset refinement.
        5. Remap COCO 17 → MediaPipe 33 landmark list.

        Parameters
        ----------
        image_rgb : np.ndarray, shape (H, W, 3), uint8

        Returns
        -------
        list[Landmark] of length 33, or None if all key joints are below
        confidence threshold.
        """
        import cv2

        # Step 1–2: Resize and normalise
        img = cv2.resize(image_rgb, (self._input_w, self._input_h))
        img_f = img.astype(np.float32) / 255.0                    # [0, 1]
        inp   = np.expand_dims(img_f, axis=0)                      # [1, H, W, 3]

        # Step 3: Inference
        self._interpreter.set_tensor(self._input_details[0]["index"], inp)
        self._interpreter.invoke()

        # Step 4: Decode outputs
        # heatmaps shape: [1, stride_h, stride_w, 17]
        # offsets  shape: [1, stride_h, stride_w, 34]
        heatmaps = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
        offsets  = self._interpreter.get_tensor(self._output_details[1]["index"])[0]

        kps = self._decode_keypoints(heatmaps, offsets)

        return self._remap_to_mediapipe(kps)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _decode_keypoints(
        self,
        heatmaps: np.ndarray,   # (H_out, W_out, 17)
        offsets:  np.ndarray,   # (H_out, W_out, 34)
    ) -> np.ndarray:
        """
        Decode heatmaps and offsets into 17 keypoint positions.

        For each joint j:
            - Find argmax position (row, col) in heatmaps[:,:,j]
            - Apply sigmoid to get confidence score
            - Add sub-cell refinement from offsets:
                y_refined = (row / H_out + offsets[row, col, j]     / H_input)
                x_refined = (col / W_out + offsets[row, col, j+17]  / W_input)

        Returns
        -------
        np.ndarray, shape (17, 3)
            Each row: [y_norm, x_norm, score]  (normalised image coords)
        """
        H_out, W_out, n_joints = heatmaps.shape
        assert n_joints == 17

        def _sigmoid(x: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

        keypoints = np.zeros((17, 3), dtype=np.float32)

        for j in range(17):
            hm_j = heatmaps[:, :, j]
            flat_idx = int(np.argmax(hm_j))
            row = flat_idx // W_out
            col = flat_idx  % W_out

            score = float(_sigmoid(hm_j[row, col]))

            # Sub-cell offset refinement (maps back to original image coords)
            y_offset = float(offsets[row, col, j])
            x_offset = float(offsets[row, col, j + 17])

            y_norm = (row / max(H_out - 1, 1) + y_offset / self._input_h)
            x_norm = (col / max(W_out - 1, 1) + x_offset / self._input_w)

            keypoints[j] = [
                float(np.clip(y_norm, 0.0, 1.0)),
                float(np.clip(x_norm, 0.0, 1.0)),
                score,
            ]

        return keypoints

    def _remap_to_mediapipe(self, kps: np.ndarray) -> list[Landmark] | None:
        """
        Remap COCO 17 keypoints to MediaPipe 33-landmark format.
        Identical logic to MoveNetRunner._remap_to_mediapipe().
        """
        required_coco = [5, 6, 7, 8, 11, 12]
        if not any(kps[i, 2] >= _MIN_SCORE for i in required_coco):
            return None

        landmarks = _default_landmarks()

        for coco_idx, mp_idx in _COCO_TO_MP.items():
            y, x, score = float(kps[coco_idx, 0]), float(kps[coco_idx, 1]), float(kps[coco_idx, 2])
            landmarks[mp_idx] = Landmark(x=x, y=y, z=0.0, visibility=score)

        return landmarks

    def close(self) -> None:
        # TFLite interpreter doesn't need explicit cleanup
        self._interpreter = None
