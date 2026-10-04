"""
metrics.py — Evaluation Metrics for MonoArm Validation
========================================================

Provides a complete set of metrics used to evaluate pose estimation accuracy
against Human3.6M ground truth, matching the reporting standards common in
IEEE conference papers on human pose estimation.

Metrics
-------
    MAE  — Mean Absolute Error (degrees): primary accuracy metric
    RMSE — Root Mean Squared Error (degrees): sensitive to outliers
    r    — Pearson correlation coefficient: linearity of tracking
    MPJAE— Mean Per-Joint Angle Error: per-DOF breakdown
    PCK  — Percentage of Correct Keypoints at threshold θ:
             fraction of frames where |pred - gt| < θ°
    Jitter— Mean absolute frame-to-frame change: proxy for stability
    Bias  — Mean signed error (degrees): systematic offset direction

All metrics are computed per-joint and then averaged (macro-average).
The primary reported metric is MPJAE following the convention of:
    Mehta et al. (2017), VNect: Real-time 3D Human Pose Estimation
    with a Single RGB Camera. ACM TOG 36(4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ── Data types ───────────────────────────────────────────────────────────────

JOINTS = ["shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion"]
JOINTS_LEFT = [f"left_{j}" for j in JOINTS]
JOINTS_BILATERAL = JOINTS + JOINTS_LEFT
JOINT_LABELS = {
    "shoulder_flexion":   "Shoulder Flexion",
    "shoulder_abduction": "Shoulder Abduction",
    "shoulder_rotation":  "Shoulder Rotation",
    "elbow_flexion":      "Elbow Flexion",
    "left_shoulder_flexion":   "L Shoulder Flexion",
    "left_shoulder_abduction": "L Shoulder Abduction",
    "left_shoulder_rotation":  "L Shoulder Rotation",
    "left_elbow_flexion":      "L Elbow Flexion",
}


@dataclass
class JointMetrics:
    """All metrics for a single joint DOF."""
    joint:     str
    n:         int
    mae:       float    # Mean Absolute Error (degrees)
    rmse:      float    # Root Mean Squared Error (degrees)
    bias:      float    # Mean signed error (degrees)  pred − gt
    r:         float    # Pearson correlation coefficient
    pck_5:     float    # % frames within ±5°
    pck_10:    float    # % frames within ±10°
    pck_15:    float    # % frames within ±15°
    jitter:    float    # Mean |Δangle| frame-to-frame (predicted signal)
    r2:        float = float("nan")   # Coefficient of determination
    mae_ci95_lo: float = float("nan") # 95% CI on MAE (normal approx.)
    mae_ci95_hi: float = float("nan")


@dataclass
class FrameworkMetrics:
    """Complete evaluation results for one pose estimation framework."""
    framework:       str
    n_frames:        int
    joints:          dict[str, JointMetrics] = field(default_factory=dict)
    # Aggregate (macro-average over joints)
    mpjae:           float = 0.0   # Mean Per-Joint Angle Error
    mean_rmse:       float = 0.0
    mean_r:          float = 0.0
    mean_r2:         float = float("nan")
    mean_pck_5:      float = 0.0
    mean_pck_10:     float = 0.0
    mean_jitter:     float = 0.0


# ── Per-joint computation ────────────────────────────────────────────────────

def circular_diff(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Signed angular difference on a ±180° circle: shortest-path (pred − gt),
    wrapped into [-180, 180).

    Naive subtraction overstates error whenever the true difference crosses
    the ±180° boundary (e.g. pred=179°, gt=-179° is actually 2° apart, not
    358°). This is a no-op away from the boundary, so applying it
    unconditionally to every DOF is always at least as correct as naive
    differencing. See the paper's "Metric corrections identified but not
    yet upstreamed" discussion — this was previously a post-hoc, flexion-only
    correction (docs/paper/analysis/sync_corrected_evaluation.py); it is now
    the default for all DOF.
    """
    d = pred - gt
    return (d + 180.0) % 360.0 - 180.0


def compute_joint_metrics(
    joint:   str,
    pred:    np.ndarray,
    gt:      np.ndarray,
    reliable_mask: np.ndarray | None = None,
) -> JointMetrics:
    """
    Compute all metrics for one joint given arrays of predicted and GT angles.

    Parameters
    ----------
    joint : str
        Joint name (e.g. "shoulder_flexion").
    pred : np.ndarray, shape (N,)
        Predicted angles in degrees.
    gt : np.ndarray, shape (N,)
        Ground-truth angles in degrees.
    reliable_mask : np.ndarray of bool, shape (N,), optional
        When given, frames where this is False are excluded in addition to
        NaN frames. Intended for the shoulder-rotation DOF, whose forearm-
        proxy estimate is only geometrically meaningful above a minimum
        elbow-flexion threshold (see angle_solver.ROT_RELIABLE_THRESHOLD) —
        without this, unconditional MAE mixes real errors with noise from
        frames where the estimate was never meant to be trusted.

    Returns
    -------
    JointMetrics
    """
    assert len(pred) == len(gt), f"Length mismatch: pred={len(pred)}, gt={len(gt)}"

    # Exclude frames where either signal is NaN (untracked frame or
    # missing GT), and — for DOF where it applies — frames flagged
    # unreliable — metrics must come only from genuinely observed,
    # geometrically meaningful pairs.
    valid = ~(np.isnan(pred) | np.isnan(gt))
    if reliable_mask is not None:
        valid = valid & np.asarray(reliable_mask, dtype=bool)
    pred_valid = np.asarray(pred, dtype=np.float64)[valid]
    gt_valid   = np.asarray(gt,   dtype=np.float64)[valid]
    pred, gt   = pred_valid, gt_valid

    n   = len(pred)
    if n == 0:
        nan = float("nan")
        return JointMetrics(joint=joint, n=0, mae=nan, rmse=nan, bias=nan,
                            r=nan, pck_5=nan, pck_10=nan, pck_15=nan, jitter=nan)
    err = circular_diff(pred, gt)

    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    # Pearson r (handle zero-variance case)
    if np.std(pred) > 1e-9 and np.std(gt) > 1e-9:
        r = float(np.corrcoef(pred, gt)[0, 1])
    else:
        r = float("nan")

    # Coefficient of determination: R² = 1 − SS_res / SS_tot
    ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 1e-9 else float("nan")

    # 95% CI on the MAE (normal approximation: mean ± 1.96·SE of |err|)
    if n > 1:
        se = float(np.std(np.abs(err), ddof=1) / np.sqrt(n))
        mae_ci_lo, mae_ci_hi = mae - 1.96 * se, mae + 1.96 * se
    else:
        mae_ci_lo = mae_ci_hi = float("nan")

    # PCK at 5°, 10°, 15°
    abs_err = np.abs(err)
    pck_5  = float(np.mean(abs_err <  5.0) * 100.0)
    pck_10 = float(np.mean(abs_err < 10.0) * 100.0)
    pck_15 = float(np.mean(abs_err < 15.0) * 100.0)

    # Jitter: mean absolute frame-to-frame change in predicted signal
    jitter = float(np.mean(np.abs(np.diff(pred)))) if n > 1 else 0.0

    return JointMetrics(
        joint=joint, n=n,
        mae=mae, rmse=rmse, bias=bias, r=r,
        pck_5=pck_5, pck_10=pck_10, pck_15=pck_15,
        jitter=jitter,
        r2=r2, mae_ci95_lo=mae_ci_lo, mae_ci95_hi=mae_ci_hi,
    )


# ── Full framework evaluation ─────────────────────────────────────────────────

# DOF whose estimate is only geometrically meaningful when the GT-reported
# reliability gate (elbow flexion above threshold) is satisfied; see
# angle_solver.ROT_RELIABLE_THRESHOLD and compute_joint_metrics. Right and
# left sides gate independently (each arm's own elbow flexion), so masks
# are supplied per joint rather than as one shared array.
ROTATION_JOINTS = {"shoulder_rotation", "left_shoulder_rotation"}


def evaluate_framework(
    framework: str,
    pred_arrays: dict[str, np.ndarray],   # joint → predicted angles
    gt_arrays:   dict[str, np.ndarray],   # joint → GT angles
    joints: list[str] | None = None,
    reliable_masks: dict[str, np.ndarray] | None = None,
) -> FrameworkMetrics:
    """
    Evaluate one framework against ground truth across all joints.

    Parameters
    ----------
    framework : str
        Framework display name.
    pred_arrays : dict[str, np.ndarray]
        Predicted angle arrays keyed by joint name.
    gt_arrays : dict[str, np.ndarray]
        Ground-truth angle arrays keyed by joint name.
    joints : list[str], optional
        Which joints to evaluate. Defaults to JOINTS.
    reliable_masks : dict[str, np.ndarray of bool], optional
        Per-joint reliability mask, keyed by joint name — only joints
        present in this dict are masked (typically just
        "shoulder_rotation" / "left_shoulder_rotation", each gated by its
        own side's GT elbow flexion; see ROT_RELIABLE_THRESHOLD). Absent
        joints are scored on every non-NaN frame, unchanged from before.

    Returns
    -------
    FrameworkMetrics
    """
    joints = joints or JOINTS
    n      = len(next(iter(gt_arrays.values())))
    result = FrameworkMetrics(framework=framework, n_frames=n)

    maes, rmses, rs, r2s, pck5s, jitters = [], [], [], [], [], []

    for j in joints:
        if j not in pred_arrays or j not in gt_arrays:
            continue
        mask = reliable_masks.get(j) if reliable_masks else None
        jm = compute_joint_metrics(j, pred_arrays[j], gt_arrays[j], reliable_mask=mask)
        result.joints[j] = jm
        maes.append(jm.mae)
        rmses.append(jm.rmse)
        if not np.isnan(jm.r):
            rs.append(jm.r)
        if not np.isnan(jm.r2):
            r2s.append(jm.r2)
        pck5s.append(jm.pck_5)
        jitters.append(jm.jitter)

    result.mpjae      = float(np.mean(maes))      if maes    else float("nan")
    result.mean_rmse  = float(np.mean(rmses))     if rmses   else float("nan")
    result.mean_r     = float(np.mean(rs))        if rs      else float("nan")
    result.mean_r2    = float(np.mean(r2s))       if r2s     else float("nan")
    result.mean_pck_5 = float(np.mean(pck5s))     if pck5s   else float("nan")
    result.mean_jitter = float(np.mean(jitters))  if jitters else float("nan")

    return result


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_metrics_table(all_results: list[FrameworkMetrics],
                        dataset_name: str = "H3.6M") -> None:
    """Print a formatted comparison table to stdout."""
    names = [r.framework for r in all_results]
    col_w = max(16, max(len(n) for n in names) + 2)
    sep   = "-" * (28 + col_w * len(names))

    def row(label, vals):
        print(f"  {label:<26}", end="")
        for v in vals:
            print(f"{str(v):>{col_w}}", end="")
        print()

    print("\n" + "=" * (28 + col_w * len(names)))
    print(f"  VALIDATION AGAINST {dataset_name} GROUND TRUTH")
    print("=" * (28 + col_w * len(names)))
    row("Framework",       names)
    row(sep[:26],          [sep[:col_w]] * len(names))
    row("N frames",        [str(r.n_frames)                  for r in all_results])
    row("MPJAE (deg) v",   [f"{r.mpjae:.2f}"                 for r in all_results])
    row("Mean RMSE (deg) v", [f"{r.mean_rmse:.2f}"            for r in all_results])
    row("Pearson r ^",     [f"{r.mean_r:.3f}"               for r in all_results])
    row("R2 ^",            [f"{r.mean_r2:.3f}"              for r in all_results])
    row("PCK@5deg ^",      [f"{r.mean_pck_5:.1f}%"          for r in all_results])
    row("Jitter (deg/fr) v", [f"{r.mean_jitter:.2f}"          for r in all_results])
    row(sep[:26],          [sep[:col_w]] * len(names))

    # Per-joint breakdown (all joints present in any result)
    seen: list[str] = []
    for r in all_results:
        for j in r.joints:
            if j not in seen:
                seen.append(j)
    for j in seen:
        label = JOINT_LABELS.get(j, j)
        row(f"{label[:24]} MAE",
            [f"{r.joints[j].mae:.2f}°" if j in r.joints else "N/A"
             for r in all_results])

    print("=" * (28 + col_w * len(names)))

    # Winner summary
    valid = [r for r in all_results if not np.isnan(r.mpjae)]
    if len(valid) >= 2:
        best = min(valid, key=lambda r: r.mpjae)
        print(f"\n  * Best overall MPJAE: {best.framework}  ({best.mpjae:.2f} deg)")
        best_r = max(valid, key=lambda r: r.mean_r)
        print(f"  * Best correlation:   {best_r.framework}  (r={best_r.mean_r:.3f})")
        best_pck = max(valid, key=lambda r: r.mean_pck_5)
        print(f"  * Best PCK@5deg:      {best_pck.framework}  ({best_pck.mean_pck_5:.1f}%)\n")


def metrics_to_dict(results: list[FrameworkMetrics]) -> list[dict]:
    """Serialise FrameworkMetrics to a plain dict for JSON export."""
    out = []
    for r in results:
        d: dict = {
            "framework":    r.framework,
            "n_frames":     r.n_frames,
            "mpjae":        r.mpjae,
            "mean_rmse":    r.mean_rmse,
            "mean_r":       r.mean_r,
            "mean_r2":      r.mean_r2 if not np.isnan(r.mean_r2) else None,
            "mean_pck_5":   r.mean_pck_5,
            "mean_pck_10":  max((jm.pck_10 for jm in r.joints.values()), default=0.0),
            "mean_jitter":  r.mean_jitter,
            "joints":       {},
        }
        for j, jm in r.joints.items():
            d["joints"][j] = {
                "mae":    jm.mae,
                "mae_ci95": [jm.mae_ci95_lo, jm.mae_ci95_hi]
                            if not np.isnan(jm.mae_ci95_lo) else None,
                "rmse":   jm.rmse,
                "bias":   jm.bias,
                "r":      jm.r if not np.isnan(jm.r) else None,
                "r2":     jm.r2 if not np.isnan(jm.r2) else None,
                "pck_5":  jm.pck_5,
                "pck_10": jm.pck_10,
                "pck_15": jm.pck_15,
                "jitter": jm.jitter,
            }
        out.append(d)
    return out


def compute_jitter(signal: list[float] | np.ndarray) -> float:
    """Compute jitter (mean absolute frame-to-frame change)."""
    n = len(signal)
    if n <= 1:
        return 0.0
    return float(np.mean(np.abs(np.diff(signal))))


def evaluate_predictions(
    framework: str,
    preds: np.ndarray,
    targets: np.ndarray,
    joint_names: list[str] | None = None
) -> FrameworkMetrics:
    """
    Evaluate predicted joint angles against ground truth.
    Flattens sequence dimensions if necessary.
    """
    if joint_names is None:
        joint_names = ["shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion"]
        
    # Flatten sequence dimensions if they are 3D
    if len(preds.shape) == 3:
        preds = preds.reshape(-1, preds.shape[-1])
    if len(targets.shape) == 3:
        targets = targets.reshape(-1, targets.shape[-1])
        
    pred_arrays = {joint_names[i]: preds[:, i] for i in range(min(len(joint_names), preds.shape[1]))}
    gt_arrays = {joint_names[i]: targets[:, i] for i in range(min(len(joint_names), targets.shape[1]))}
    
    # Map input names (like pitch/roll/yaw) to the canonical JOINTS list
    mapping = {
        "shoulder_pitch": "shoulder_flexion",
        "shoulder_roll": "shoulder_abduction",
        "shoulder_yaw": "shoulder_rotation",
        "elbow_flexion": "elbow_flexion",
        "shoulder_flexion": "shoulder_flexion",
        "shoulder_abduction": "shoulder_abduction",
        "shoulder_rotation": "shoulder_rotation",
    }
    
    mapped_preds = {}
    mapped_gts = {}
    for old_name, new_name in mapping.items():
        if old_name in pred_arrays:
            mapped_preds[new_name] = pred_arrays[old_name]
        if old_name in gt_arrays:
            mapped_gts[new_name] = gt_arrays[old_name]
            
    return evaluate_framework(framework, mapped_preds, mapped_gts, joints=JOINTS)


def metrics_to_model_dict(fm: FrameworkMetrics) -> dict:
    """Format FrameworkMetrics to matching dictionary format expected by training scripts."""
    mapping = {
        "shoulder_flexion": "shoulder_pitch",
        "shoulder_abduction": "shoulder_roll",
        "shoulder_rotation": "shoulder_yaw",
        "elbow_flexion": "elbow_flexion"
    }
    
    out = {}
    for j, jm in fm.joints.items():
        name = mapping.get(j, j)
        out[name] = {
            "MAE": jm.mae,
            "RMSE": jm.rmse,
            "Jitter": jm.jitter
        }
    out["average"] = {
        "MAE": fm.mpjae,
        "RMSE": fm.mean_rmse,
        "Jitter": fm.mean_jitter
    }
    return out
