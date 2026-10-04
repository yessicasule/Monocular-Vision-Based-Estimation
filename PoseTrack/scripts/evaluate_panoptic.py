"""
evaluate_panoptic.py — MonoArm Framework Validation on CMU Panoptic Studio
=============================================================================

Real (non-simulated) ground-truth validation of MediaPipe / MoveNet /
PoseNet, used as a substitute for Human3.6M when H3.6M account approval
is unavailable (registration at vision.imar.ro has been effectively
closed for an extended period).

Unlike evaluate_h36m.py's default "synthetic" mode (which injects
literature-calibrated noise into ground-truth angles to *simulate*
predictions), this script always runs the real pose estimators frame by
frame against real extracted video images and compares the result to
real Panoptic Studio 3D ground truth. There is no synthetic mode here.

Data layout expected
---------------------
    <sequence_dir>/
        hdPose3d_stage1_coco19/body3DScene_00000000.json, ...
        hdImgs/<camera>/<camera>_00000000.jpg, ...   (e.g. camera "00_00")

Usage
-----
    python -m scripts.evaluate_panoptic \\
        --sequence_dir ../outputs/panoptic-toolbox-master/171204_pose1_sample \\
        --camera 00_00 \\
        --frameworks mediapipe movenet_lightning posenet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

from src.evaluation.panoptic_loader import iter_panoptic_sequence
from src.evaluation.metrics import (
    evaluate_framework, print_metrics_table, metrics_to_dict,
    JOINTS, JOINTS_LEFT, JOINTS_BILATERAL, ROTATION_JOINTS,
)
from src.evaluation.alignment import check_alignment, apply_shift, shifted_pair
from src.evaluation.eval_plots import (
    plot_validation_summary, plot_bland_altman,
    plot_scatter_gt, plot_error_cdf, plot_timeseries_vs_gt,
)
from src.pose import load_estimator
from src.pose import angle_landmarks
from src.processing.angle_solver import compute_bilateral_angles
from src.processing.angle_filter import BilateralFilterBank

# GT/pred joint -> the GT reliability-flag field that gates it (own side).
_RELIABILITY_FIELD = {
    "shoulder_rotation":      "rotation_reliable",
    "left_shoulder_rotation": "left_rotation_reliable",
}


def load_gt(sequence_dir: Path, max_frames: int | None) -> tuple[list[int], dict[str, np.ndarray]]:
    """Load bilateral GT angle arrays plus per-side reliability flags (as float 0/1, joined into JOINTS_BILATERAL-keyed arrays)."""
    pose_dir = sequence_dir / "hdPose3d_stage1_coco19"
    gts = list(iter_panoptic_sequence(pose_dir, max_frames=max_frames))
    if not gts:
        print(f"[FAIL] No GT frames parsed from {pose_dir}")
        sys.exit(1)

    frame_idx = [g.frame_idx for g in gts]
    arrays = {j: np.array([getattr(g, j) for g in gts], dtype=np.float64) for j in JOINTS_BILATERAL}
    for field_name in _RELIABILITY_FIELD.values():
        arrays[field_name] = np.array([getattr(g, field_name) for g in gts], dtype=bool)
    return frame_idx, arrays


def run_live_predictions(
    sequence_dir: Path,
    camera: str,
    frame_idx: list[int],
    frameworks: list[str],
    stream_hz: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Run each real pose framework (both arms) on the exact frames that have GT, in order."""
    img_dir = sequence_dir / "hdImgs" / camera
    preds: dict[str, dict[str, np.ndarray]] = {}

    for fw_name in frameworks:
        print(f"\n  Running {fw_name} on {len(frame_idx)} frames...")
        try:
            runner = load_estimator(fw_name)
        except Exception as e:
            print(f"  [FAIL] Could not load {fw_name}: {e}")
            continue

        filt = BilateralFilterBank("kalman", stream_hz=stream_hz)
        angles_per_joint: dict[str, list[float]] = {j: [] for j in JOINTS_BILATERAL}
        n_missing = 0

        for idx in frame_idx:
            fp = img_dir / f"{camera}_{idx:08d}.jpg"
            bgr = cv2.imread(str(fp))
            if bgr is None:
                n_missing += 1
                for j in JOINTS_BILATERAL:
                    angles_per_joint[j].append(float("nan"))
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            lms = runner.process(rgb)
            bilateral = (compute_bilateral_angles(
                angle_landmarks(runner, lms, rgb.shape[1], rgb.shape[0]))
                if lms is not None else None)
            filt_bilateral = filt.update(bilateral) if bilateral is not None else None

            right = filt_bilateral.right if filt_bilateral else None
            left  = filt_bilateral.left  if filt_bilateral else None
            for j in JOINTS:
                angles_per_joint[j].append(getattr(right, j) if right is not None else float("nan"))
            for j in JOINTS_LEFT:
                side_field = j[len("left_"):]
                angles_per_joint[j].append(getattr(left, side_field) if left is not None else float("nan"))

        runner.close()

        for j in JOINTS_BILATERAL:
            arr = np.array(angles_per_joint[j])
            nan_mask = np.isnan(arr)
            if nan_mask.any() and not nan_mask.all():
                ix = np.arange(len(arr))
                valid = ix[~nan_mask]
                arr = np.interp(ix, valid, arr[valid])
            elif nan_mask.all():
                arr = np.zeros(len(arr))
            angles_per_joint[j] = arr

        preds[fw_name] = angles_per_joint
        det_rate = 100.0 * (len(frame_idx) - n_missing) / len(frame_idx)
        print(f"  [{fw_name}] done. Frames read: {det_rate:.0f}%")

    return preds


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Real (non-simulated) validation against CMU Panoptic Studio ground truth."
    )
    ap.add_argument("--sequence_dir", required=True,
                     help="Path to a downloaded+extracted Panoptic sequence folder")
    ap.add_argument("--camera", default="00_00",
                     help="HD camera id whose frames were extracted (default: 00_00)")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--frameworks", nargs="+",
                     default=["mediapipe", "movenet_lightning", "posenet"])
    ap.add_argument("--stream_hz", type=float, default=29.97,
                     help="Panoptic HD capture rate (used for the Kalman filter dt)")
    ap.add_argument("--output_dir", default="outputs/validation_panoptic")
    args = ap.parse_args()

    sequence_dir = Path(args.sequence_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[->] Loading Panoptic ground truth...")
    t0 = time.perf_counter()
    frame_idx, gt_raw = load_gt(sequence_dir, args.max_frames)
    print(f"[OK] {len(frame_idx)} GT frames loaded in {time.perf_counter()-t0:.1f}s "
          f"(real CMU Panoptic 3D mocap-derived ground truth, camera {args.camera})")

    pred_raw = run_live_predictions(
        sequence_dir, args.camera, frame_idx, args.frameworks, args.stream_hz,
    )
    if not pred_raw:
        print("[FAIL] No framework produced predictions.")
        sys.exit(1)

    # ── Automatic temporal-alignment pre-flight check ──────────────────────
    # Video and 3D-reconstruction are captured by separate Panoptic Studio
    # pipelines with no guaranteed common start time; pairing by filename
    # index alone can silently misalign the two streams (see paper Sec.
    # "Temporal alignment as a standard check"). Estimate and correct the
    # offset before any accuracy metric is computed, rather than after.
    ref_fw = "mediapipe" if "mediapipe" in pred_raw else next(iter(pred_raw))
    align = check_alignment(pred_raw, gt_raw, reference_framework=ref_fw)
    print(f"\n[->] Alignment check: shift={align['shift']:+d} frames "
          f"(r={align['reference_r']:.3f}, reference={ref_fw}/{align['reference_joint']})")
    for fw, c in align["consistency"].items():
        print(f"     consistency: {fw} independently implies shift="
              f"{c['independent_best_shift']:+d} (r={c['r_at_best_shift']:.3f})")

    pred_data, gt_arrays = apply_shift(pred_raw, gt_raw, align["shift"], JOINTS_BILATERAL)
    n_aligned = len(next(iter(gt_arrays.values())))

    # Reliability flags are boolean GT-only channels (no per-framework
    # counterpart) — shift them the same way so they stay frame-aligned
    # with gt_arrays. shifted_pair(x, x, shift) trims both copies to the
    # same aligned slice; only the (already-shifted) ground-truth half
    # is kept, mirroring how gt_arrays itself was produced above.
    reliability = {
        field_name: shifted_pair(gt_raw[field_name], gt_raw[field_name], align["shift"])[1][:n_aligned]
        for field_name in _RELIABILITY_FIELD.values()
    }
    print(f"[OK] Aligned overlap: n={n_aligned} frames (was {len(frame_idx)} before trimming for the shift)")

    all_results = []
    for fw in args.frameworks:
        if fw not in pred_data:
            continue
        reliable_masks = {
            joint: reliability[field_name].astype(bool)
            for joint, field_name in _RELIABILITY_FIELD.items()
        }
        result = evaluate_framework(framework=fw, pred_arrays=pred_data[fw], gt_arrays=gt_arrays,
                                     joints=JOINTS_BILATERAL, reliable_masks=reliable_masks)
        all_results.append(result)
        print(f"  [{fw}]  MPJAE={result.mpjae:.2f} deg  r={result.mean_r:.3f}  "
              f"PCK@5={result.mean_pck_5:.1f}%")

    print_metrics_table(all_results, dataset="CMU PANOPTIC")

    json_path = out_dir / "metrics_report.json"
    with open(json_path, "w") as f:
        json.dump({
            "alignment": {k: v for k, v in align.items()},
            "n_frames_aligned": n_aligned,
            "results": metrics_to_dict(all_results),
        }, f, indent=2)
    print(f"[OK] JSON metrics -> {json_path}")
    print("[NOTE] These numbers are measured from real inference on real "
          "CMU Panoptic Studio video + real 3D mocap ground truth -- not simulated. "
          "Bilateral (both arms); rotation DOF masked by each side's own GT reliability gate; "
          "flexion/rotation error uses circular (wraparound-safe) differencing.")

    print("\n[->] Generating figures (right-arm DOF, for continuity with prior figures)...")
    plot_validation_summary(all_results, str(out_dir / "validation_summary.png"))
    plot_bland_altman(all_results, pred_data, gt_arrays, str(out_dir / "bland_altman.png"))
    plot_scatter_gt(all_results, pred_data, gt_arrays, str(out_dir / "scatter_gt.png"))
    plot_error_cdf(all_results, pred_data, gt_arrays, str(out_dir / "error_cdf.png"))
    plot_timeseries_vs_gt(all_results, pred_data, gt_arrays,
                           str(out_dir / "timeseries_vs_gt.png"),
                           n_frames=min(500, n_aligned))

    print(f"\n[OK] All outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
