"""
verify_real_frame.py — End-to-End Pipeline Verification on a Real Photograph
==============================================================================

Runs the actual MediaPipe pose landmarker on a real, non-synthetic
photograph (not a dataset accuracy benchmark — no ground-truth angles are
compared here) to verify the full pipeline works end to end on real image
data: image -> pose detection -> torso frame -> bilateral joint angles ->
temporal filter.

The sample image is frame 50 of the CMU Panoptic Studio sequence
`171204_pose1_sample` (Joo et al., 2015), distributed by the dataset
maintainers as "freely available for non-commercial and research purpose
only." The subject is a Panoptic Studio capture participant included in
that public research release, not a participant recruited for this
project. We use it here strictly as a real-image smoke test of the code
path, not as an accuracy claim -- reproducing the accuracy numbers in the
associated paper would additionally require the sequence's 3D
ground-truth JSON and the temporal-alignment procedure in
src/evaluation/alignment.py, neither of which this script uses.

Usage
-----
    python -m scripts.verify_real_frame
    python -m scripts.verify_real_frame --image path/to/other_photo.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.pose.mediapipe_runner import MediaPipeRunner
from src.processing.angle_solver import compute_bilateral_angles
from src.processing.angle_filter import BilateralFilterBank

DEFAULT_IMAGE = (
    Path(__file__).resolve().parent.parent.parent
    / "docs" / "paper" / "figures" / "panoptic_171204_pose1_sample_frame50.jpg"
)

# MediaPipe 33-landmark skeleton edges relevant to the upper body, for the
# annotated overlay (shoulder/elbow/wrist/hip only — not the full body).
_SKELETON_EDGES = [
    (11, 12),  # shoulder-shoulder
    (11, 23), (12, 24),  # shoulder-hip
    (23, 24),  # hip-hip
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
]


def draw_overlay(bgr, landmarks) -> "cv2.Mat":
    out = bgr.copy()
    h, w = out.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _SKELETON_EDGES:
        cv2.line(out, pts[a], pts[b], (80, 220, 80), 3, cv2.LINE_AA)
    for idx in (11, 12, 13, 14, 15, 16, 23, 24):
        cv2.circle(out, pts[idx], 6, (60, 160, 255), -1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default=str(DEFAULT_IMAGE))
    ap.add_argument("--overlay_out", default="outputs/real_frame_overlay.png")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    print(f"[OK] Loaded {args.image}  shape={bgr.shape}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    runner = MediaPipeRunner()
    lms = runner.process(rgb)
    print(f"[OK] MediaPipe returned {'None (no person detected)' if lms is None else f'{len(lms)} landmarks'}")

    if lms is not None:
        bilateral = compute_bilateral_angles(lms)
        print(f"[OK] compute_bilateral_angles() returned: {bilateral is not None}")
        if bilateral is not None:
            for side, a in (("right", bilateral.right), ("left", bilateral.left)):
                if a is None:
                    print(f"  {side}: not tracked")
                else:
                    print(
                        f"  {side:5s}: flex={a.shoulder_flexion:7.2f}  "
                        f"abd={a.shoulder_abduction:7.2f}  "
                        f"rot={a.shoulder_rotation:7.2f} (reliable={a.rotation_reliable})  "
                        f"elbow={a.elbow_flexion:7.2f}"
                    )
            filt = BilateralFilterBank("kalman", stream_hz=30.0)
            filtered = filt.update(bilateral)
            print(f"[OK] Kalman filter update succeeded: {filtered is not None}")

        Path(args.overlay_out).parent.mkdir(parents=True, exist_ok=True)
        overlay = draw_overlay(bgr, lms)
        cv2.imwrite(args.overlay_out, overlay)
        print(f"[OK] Annotated overlay -> {args.overlay_out}")

    runner.close()
    print(
        "\n[NOTE] This is a code-path verification on one real photograph, "
        "not a dataset accuracy benchmark -- no ground truth is compared "
        "here. For measured accuracy against real 3D ground truth, run "
        "scripts/fetch_panoptic_sample.sh then scripts.evaluate_panoptic; "
        "see the paper's 'Accuracy Against 3D Motion-Capture Ground Truth'. "
        "These angles are also landmarker-version sensitive -- see that "
        "paper section on runtime dependencies."
    )


if __name__ == "__main__":
    main()
