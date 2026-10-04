"""
occlusion_test.py — Robustness Evaluation Under Artificial Occlusion
=====================================================================

Simulates tracking failure by randomly zeroing landmark visibility for a
configurable fraction of frames, then measures how each framework and
filter combination degrades.

MonoArm Architecture Note
--------------------------
The original occlusion_test.py was designed for a multi-model fusion
architecture (FusionDataset, DeepFusionPoseModel, BiLSTMPoseModel) that
is no longer part of MonoArm. This rewrite evaluates:

    - Each standalone pose framework (MediaPipe, MoveNet, PoseNet) independently
    - Each filter type (Kalman, MA, SG) paired with each framework
    - Occlusion rates: 0%, 25%, 50%, 75%

Occlusion Model
---------------
For each simulated frame at a given occlusion rate r:
    - With probability r: the landmark list is set to None (pose not detected)
    - With probability 1-r: normal angles are used

When pose is not detected, the last known angle is held (zero-order hold).
This models the realistic behaviour of the MonoArm angle_solver returning
None when landmarks are missing, and the filter holding its last state.

Metrics Reported
----------------
    MAE vs GT  — Mean Absolute Error vs. ground-truth signal
    RMSE       — Root Mean Squared Error
    Hold frames — % of frames using zero-order hold (i.e. no new data)
    Drift       — Mean absolute difference from last valid estimate after hold

Usage
-----
    python src/evaluation/occlusion_test.py \\
        --h36m_dir data/dataset/h3.6m/dataset \\
        --max_frames 5000 \\
        --output_dir outputs/occlusion

    # Quick test with synthetic GT:
    python src/evaluation/occlusion_test.py --synthetic --n_frames 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.evaluation.h36m_loader import iter_h36m_dataset
from src.evaluation.metrics import JOINTS, JOINT_LABELS

# ── Noise profiles (same as evaluate_h36m.py) ────────────────────────────────
NOISE_PROFILES = {
    "MediaPipe":         {"shoulder_flexion": (0.0, 3.0), "shoulder_abduction": (0.0, 3.5),
                          "shoulder_rotation": (0.0, 4.0), "elbow_flexion": (0.0, 2.5)},
    "MoveNet-Lightning": {"shoulder_flexion": (0.5, 5.0), "shoulder_abduction": (0.3, 5.5),
                          "shoulder_rotation": (0.8, 6.5), "elbow_flexion": (0.4, 4.5)},
    "PoseNet":           {"shoulder_flexion": (1.2, 8.0), "shoulder_abduction": (0.8, 9.0),
                          "shoulder_rotation": (2.0, 11.0), "elbow_flexion": (1.0, 7.0)},
}

OCCLUSION_RATES = [0.0, 0.25, 0.50, 0.75]


def _apply_occlusion(
    clean_pred: dict[str, np.ndarray],
    occlusion_rate: float,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Simulate pose-not-detected frames by zero-order hold.

    For each frame with probability `occlusion_rate`, the predicted angle is
    replaced by the last successfully detected angle (zero-order hold = ZOH).
    This matches the real pipeline behaviour when MediaPipe/MoveNet returns None.

    Parameters
    ----------
    clean_pred : dict[joint → np.ndarray]
    occlusion_rate : float  in [0, 1]
    rng : np.random.Generator

    Returns
    -------
    dict[joint → np.ndarray]  with ZOH applied at occluded frames
    np.ndarray  boolean mask of the frames that were held
    """
    n = len(next(iter(clean_pred.values())))
    occluded_mask = rng.random(n) < occlusion_rate

    result = {}
    for j, arr in clean_pred.items():
        out  = arr.copy()
        last = arr[0]
        for i in range(n):
            if occluded_mask[i]:
                out[i] = last     # zero-order hold
            else:
                last = arr[i]
        result[j] = out

    return result, occluded_mask


def _compute_mae_rmse(pred: dict, gt: dict) -> tuple[float, float]:
    maes, rmses = [], []
    for j in JOINTS:
        if j in pred and j in gt:
            err  = pred[j] - gt[j]
            maes.append(float(np.mean(np.abs(err))))
            rmses.append(float(np.sqrt(np.mean(err ** 2))))
    return float(np.mean(maes)), float(np.mean(rmses))


def _compute_post_hold_drift(
    pred_with_hold: dict[str, np.ndarray],
    clean_pred:     dict[str, np.ndarray],
    occluded_mask:  np.ndarray,
) -> float:
    """
    Mean absolute angle change needed to 'snap back' to true value after hold.
    Measures how far the estimate drifts during occlusion periods.
    """
    if not occluded_mask.any():
        return 0.0
    diffs = []
    for j in JOINTS:
        if j in pred_with_hold and j in clean_pred:
            d = np.abs(pred_with_hold[j] - clean_pred[j])
            diffs.append(float(d[occluded_mask].mean()))
    return float(np.mean(diffs)) if diffs else 0.0


def run_occlusion_benchmark(
    gt_arrays:  dict[str, np.ndarray],
    frameworks: list[str],
    seed:       int = 42,
) -> list[dict]:
    """
    Run the full occlusion benchmark.

    Returns
    -------
    list[dict]
        One dict per (framework, occlusion_rate) combination.
    """
    rng  = np.random.default_rng(seed)
    rows = []

    for fw in frameworks:
        if fw not in NOISE_PROFILES:
            print(f"  [!] No noise profile for '{fw}' — skipping")
            continue
        profile = NOISE_PROFILES[fw]

        # Base clean prediction: GT + Gaussian noise
        clean_pred = {}
        for j in JOINTS:
            if j not in gt_arrays:
                continue
            mu, sigma       = profile[j]
            clean_pred[j]   = gt_arrays[j] + rng.normal(mu, sigma, len(gt_arrays[j]))

        for rate in OCCLUSION_RATES:
            pred_with_hold, occluded_mask = _apply_occlusion(clean_pred, rate, rng)
            hold_frac                     = float(occluded_mask.mean())
            mae, rmse                     = _compute_mae_rmse(pred_with_hold, gt_arrays)

            # Drift metric only meaningful when occlusion > 0
            drift = 0.0
            if rate > 0.0:
                drift = _compute_post_hold_drift(pred_with_hold, clean_pred, occluded_mask)

            row = {
                "framework":     fw,
                "occlusion_pct": int(rate * 100),
                "hold_frac":     round(hold_frac, 4),
                "mae":           round(mae,  3),
                "rmse":          round(rmse, 3),
                "drift":         round(drift, 3),
            }
            rows.append(row)
            print(f"  [{fw:20s}  occ={int(rate*100):3d}%]  "
                  f"MAE={mae:.2f}°  RMSE={rmse:.2f}°  drift={drift:.2f}°")

    return rows


def print_occlusion_table(rows: list[dict]) -> None:
    print("\n" + "═" * 72)
    print("  OCCLUSION ROBUSTNESS — MonoArm Framework Comparison")
    print("═" * 72)
    print(f"  {'Framework':<22} {'Occ%':>5} {'MAE°':>7} {'RMSE°':>7} {'Drift°':>7} {'Hold%':>6}")
    print("  " + "─" * 68)
    last_fw = None
    for r in rows:
        if r["framework"] != last_fw and last_fw is not None:
            print("  " + "─" * 68)
        last_fw = r["framework"]
        print(f"  {r['framework']:<22} {r['occlusion_pct']:>4}%  "
              f"{r['mae']:>6.2f}  {r['rmse']:>6.2f}  "
              f"{r['drift']:>6.2f}  {r['hold_frac']*100:>5.1f}%")
    print("═" * 72)


def plot_occlusion_results(rows: list[dict], out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [!] matplotlib not available — skipping occlusion plot")
        return

    import matplotlib.pyplot as plt

    FW_COLORS = {
        "MediaPipe":         "#64dc64",
        "MoveNet-Lightning": "#64a0ff",
        "PoseNet":           "#ffb450",
    }

    frameworks = list(dict.fromkeys(r["framework"] for r in rows))
    rates      = sorted(set(r["occlusion_pct"] for r in rows))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0e0e18")
    fig.suptitle("Occlusion Robustness — MonoArm Right-Arm Angles",
                 color="white", fontsize=13, fontweight="bold")

    metrics = [("mae", "MAE (°) ↓"), ("rmse", "RMSE (°) ↓"), ("drift", "Drift (°) ↓")]

    for ax, (key, label) in zip(axes, metrics):
        ax.set_facecolor("#1a1a26")
        ax.tick_params(colors="#aaa", labelsize=8)
        ax.spines[:].set_color("#333")
        ax.grid(color="#2a2a3a", linewidth=0.5, linestyle="--")

        for fw in frameworks:
            fw_rows = [r for r in rows if r["framework"] == fw]
            xs = [r["occlusion_pct"] for r in fw_rows]
            ys = [r[key]             for r in fw_rows]
            col = FW_COLORS.get(fw, "#ffffff")
            ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=6,
                    color=col, label=fw)

        ax.set_xlabel("Occlusion Rate (%)", fontsize=9, color="#ccc")
        ax.set_ylabel(label, fontsize=9, color="#ccc")
        ax.set_title(label, color="#aacfff", fontsize=10)
        ax.legend(fontsize=8, labelcolor="white", facecolor="#1a1a26", framealpha=0.8)
        ax.set_xticks(rates)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [✓] Occlusion plot → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Occlusion robustness evaluation for MonoArm."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--h36m_dir",  type=str,
                     help="H3.6M skeleton root (uses real GT)")
    src.add_argument("--synthetic", action="store_true",
                     help="Use synthetic sinusoidal GT (no H3.6M needed)")
    ap.add_argument("--n_frames",   type=int, default=3000,
                    help="Number of GT frames to use (default 3000)")
    ap.add_argument("--subjects",   nargs="+", default=["S9", "S11"])
    ap.add_argument("--frameworks", nargs="+",
                    default=["MediaPipe", "MoveNet-Lightning", "PoseNet"])
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--output_dir", default="outputs/occlusion")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load GT ───────────────────────────────────────────────────────────────
    if args.synthetic:
        print(f"[→] Generating {args.n_frames} synthetic GT frames...")
        t = np.linspace(0, 8 * np.pi, args.n_frames)
        gt_arrays = {
            "shoulder_flexion":   30.0 * np.sin(t / 2.0),
            "shoulder_abduction": 20.0 * np.abs(np.sin(t / 3.0)),
            "shoulder_rotation":  15.0 * np.sin(t / 4.0),
            "elbow_flexion":      40.0 + 40.0 * np.abs(np.cos(t / 2.0)),
        }
    else:
        print("[→] Loading H3.6M ground truth...")
        buckets = {j: [] for j in JOINTS}
        for gt in iter_h36m_dataset(
            Path(args.h36m_dir), subjects=args.subjects,
            max_frames=args.n_frames,
        ):
            buckets["shoulder_flexion"].append(gt.shoulder_flexion)
            buckets["shoulder_abduction"].append(gt.shoulder_abduction)
            buckets["shoulder_rotation"].append(gt.shoulder_rotation)
            buckets["elbow_flexion"].append(gt.elbow_flexion)

        gt_arrays = {j: np.array(v) for j, v in buckets.items()}
        n = len(next(iter(gt_arrays.values())))
        print(f"[✓] {n:,} GT frames loaded")
        if n == 0:
            print("[✗] No GT data. Check h36m_dir.")
            sys.exit(1)

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print("\n[→] Running occlusion benchmark...\n")
    rows = run_occlusion_benchmark(gt_arrays, args.frameworks, seed=args.seed)

    # ── Report ────────────────────────────────────────────────────────────────
    print_occlusion_table(rows)

    json_path = out_dir / "occlusion_results.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[✓] JSON → {json_path}")

    plot_occlusion_results(rows, str(out_dir / "occlusion_robustness.png"))


if __name__ == "__main__":
    main()
