"""
evaluate_h36m.py — MonoArm Framework Validation on Human3.6M
=============================================================

Runs the complete evaluation pipeline comparing pose estimation frameworks
against Human3.6M ground-truth joint angles. Produces all figures and tables
needed for the IEEE conference paper's quantitative evaluation section.

Pipeline
--------
    1. Load H3.6M skeleton files → extract BILATERAL (left + right arm) GT
       angles using ZXY anatomical decomposition (same convention as the
       live MonoArm pipeline).
    2. Run each pose estimation framework over the corresponding H3.6M video
       frames (raw, unfiltered predictions).
    3. Compute MAE (±95% CI), RMSE, Pearson r, R², PCK@5°/10°/15°, jitter
       for each framework × joint combination.
    4. Statistical significance: pairwise paired t-tests, Wilcoxon
       signed-rank tests, Cohen's d, bootstrap CIs, Holm-Bonferroni
       correction (src/evaluation/statistics.py).
    5. Filter ablation: none / MA / Savitzky-Golay / 2-state Kalman applied
       causally to each framework's predictions and scored against GT.
    6. Export: Excel workbook (summary + ranking, per-joint metrics,
       statistical tests, ablation, frame-level data, reproducibility
       metadata), frame-level CSV, JSON reports, LaTeX table, and 5
       publication-quality figures.

Evaluation Modes (--mode is REQUIRED)
-------------------------------------
    --mode live
        Runs the real pose estimation frameworks on H3.6M video frames.
        This is the ONLY mode whose results are publication-grade.
        Requires:
            (a) The registration-gated H3.6M video + D3_Positions release
                (docs/DATA_H36M.md explains how to obtain it)
            (b) scripts/prepare_h36m.py run over it, then --manifest pointed
                at the manifest.json it writes

    --mode csv
        Loads previously computed real predictions from a dataset CSV
        (build_h36m_dataset.py). Publication-grade if the CSV was built
        from real framework output.

    --mode synthetic
        PIPELINE SMOKE TEST ONLY. Simulates predictions as GT + Gaussian
        noise. All outputs are tagged publication_grade=false and MUST
        NOT be reported as experimental results.

Protocols
---------
    --protocol test-split  (default)
        H3.6M standard split: parameters tuned on S1/S5/S6/S7, selection
        on S8, final evaluation on the held-out test subjects S9/S11.
    --protocol loso
        Leave-One-Subject-Out: per-subject metrics for each fold,
        aggregated mean ± std across folds (exported as a LOSO sheet).

Usage
-----
    # Publication run (requires the prepared video release)
    python scripts/prepare_h36m.py \\
        --h36m_root data/dataset/h3.6m/raw \\
        --out_dir   data/dataset/h3.6m/prepared \\
        --subjects  S9 S11 --stride 5

    python scripts/evaluate_h36m.py \\
        --manifest data/dataset/h3.6m/prepared/manifest.json \\
        --mode live \\
        --frameworks mediapipe movenet_lightning posenet

    # Pipeline smoke test (no video needed; NOT for publication)
    python scripts/evaluate_h36m.py \\
        --h36m_dir data/dataset/h3.6m/dataset \\
        --mode synthetic \\
        --max_frames 5000

Output
------
    outputs/validation/
        results.xlsx              — Multi-sheet Excel workbook
        frame_level.csv           — Full frame-level GT/pred/error table
        metadata.json             — Reproducibility record
        metrics_report.json       — Full per-joint metric tables
        statistical_tests.json    — Pairwise significance tests
        filter_ablation.json      — Filter ablation table
        paper_table.tex           — LaTeX table for the paper
        validation_summary.png    — Summary bar charts (paper-ready)
        bland_altman.png          — Bland-Altman agreement plots
        scatter_gt.png            — Predicted vs. GT scatter + regression
        error_cdf.png             — PCK-style error CDF curve
        timeseries_vs_gt.png      — Sample time-series overlay
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

import numpy as np
import pandas as pd

from src.evaluation.h36m_loader import iter_h36m_dataset, gt_angles_from_row
from src.evaluation.metrics import (
    evaluate_framework, print_metrics_table, metrics_to_dict,
    JOINTS, JOINTS_BILATERAL, FrameworkMetrics,
)
from src.evaluation.eval_plots import (
    plot_validation_summary, plot_bland_altman,
    plot_scatter_gt, plot_error_cdf, plot_timeseries_vs_gt,
)
from src.evaluation.statistics import compare_systems, comparison_report_to_dicts
from src.evaluation.ablation import run_filter_ablation
from src.evaluation.protocol import get_split, loso_folds, describe_protocol
from src.evaluation.report_export import (
    build_metadata, export_excel, export_frame_level_csv, export_metadata_json,
)

# ── Noise profiles for synthetic mode ────────────────────────────────────────
NOISE_PROFILES = {
    "MediaPipe": {
        "shoulder_flexion":   (0.0,  3.0),
        "shoulder_abduction": (0.0,  3.5),
        "shoulder_rotation":  (0.0,  4.0),
        "elbow_flexion":      (0.0,  2.5),
    },
    "MoveNet-Lightning": {
        "shoulder_flexion":   (0.5,  5.0),
        "shoulder_abduction": (0.3,  5.5),
        "shoulder_rotation":  (0.8,  6.5),
        "elbow_flexion":      (0.4,  4.5),
    },
    "PoseNet": {
        "shoulder_flexion":   (1.2,  8.0),
        "shoulder_abduction": (0.8,  9.0),
        "shoulder_rotation":  (2.0, 11.0),
        "elbow_flexion":      (1.0,  7.0),
    },
}


# ── Ground truth loading ──────────────────────────────────────────────────────

def load_gt_arrays(
    h36m_dir: Path,
    subjects: list[str],
    max_frames: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Load H3.6M bilateral ground truth angles into per-joint arrays.

    Returns
    -------
    (gt_arrays, meta)
        gt_arrays: joint name → shape (N,) float array, for both arms
                   (right joints unprefixed, left joints "left_"-prefixed).
        meta:      "subject" and "sequence" (subject/action) string arrays
                   aligned with the GT frames — used for LOSO folds and
                   filter-reset boundaries.
    """
    print("[→] Loading H3.6M ground truth (bilateral)...")
    t0 = time.perf_counter()

    buckets: dict[str, list[float]] = {j: [] for j in JOINTS_BILATERAL}
    subjects_col: list[str] = []
    sequence_col: list[str] = []
    n = 0

    for gt in iter_h36m_dataset(h36m_dir, subjects=subjects, max_frames=max_frames):
        buckets["shoulder_flexion"].append(gt.shoulder_flexion)
        buckets["shoulder_abduction"].append(gt.shoulder_abduction)
        buckets["shoulder_rotation"].append(gt.shoulder_rotation)
        buckets["elbow_flexion"].append(gt.elbow_flexion)
        buckets["left_shoulder_flexion"].append(gt.left_shoulder_flexion)
        buckets["left_shoulder_abduction"].append(gt.left_shoulder_abduction)
        buckets["left_shoulder_rotation"].append(gt.left_shoulder_rotation)
        buckets["left_elbow_flexion"].append(gt.left_elbow_flexion)
        subjects_col.append(gt.subject)
        sequence_col.append(f"{gt.subject}/{gt.action}")
        n += 1

    elapsed = time.perf_counter() - t0
    print(f"[✓] {n:,} GT frames loaded in {elapsed:.1f}s")

    if n == 0:
        print("[!] WARNING: No GT frames loaded. Check h36m_dir path and .txt file format.")

    gt_arrays = {j: np.array(v, dtype=np.float64) for j, v in buckets.items()}
    # Drop left-arm joints that are entirely NaN (e.g. malformed skeletons)
    gt_arrays = {j: a for j, a in gt_arrays.items() if not np.all(np.isnan(a))}

    meta = {
        "subject":  np.array(subjects_col),
        "sequence": np.array(sequence_col),
    }
    return gt_arrays, meta


def load_manifest_gt(
    manifest_path: Path,
    subjects:      list[str] | None,
    max_frames:    int | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[Path]]:
    """
    Load ground truth and the paired frame paths from a prepare_h36m manifest.

    This is the alignment-safe counterpart to load_gt_arrays() + a frame glob.
    Ground truth is read at the exact ``gt_row`` the manifest records for each
    frame, and a sample whose skeleton is degenerate is dropped from the GT
    array and the frame list together, so index i of the returned arrays always
    refers to index i of the returned frame paths.

    Returns
    -------
    (gt_arrays, meta, frame_paths)
    """
    print(f"[→] Loading manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent

    samples = manifest["samples"]
    if subjects:
        keep = set(subjects)
        samples = [s for s in samples if s["subject"] in keep]
    if not samples:
        raise ValueError(
            f"No manifest samples for subjects {subjects}. "
            f"Manifest contains {manifest.get('subjects')}."
        )

    # Group by (subject, action) so each positions file is read exactly once.
    by_seq: dict[tuple[str, str], list[dict]] = {}
    for s in samples:
        by_seq.setdefault((s["subject"], s["action"]), []).append(s)

    buckets: dict[str, list[float]] = {j: [] for j in JOINTS_BILATERAL}
    subjects_col: list[str] = []
    sequence_col: list[str] = []
    frame_paths: list[Path] = []
    n_dropped = 0

    positions_dir = root / manifest.get("positions_dir", "positions")

    for (subject, action), seq_samples in sorted(by_seq.items()):
        pos_path = positions_dir / subject / f"{action}.txt"
        if not pos_path.is_file():
            print(f"  [!] missing positions file {pos_path} — sequence skipped")
            continue
        rows = np.loadtxt(pos_path, delimiter=",", ndmin=2)

        for s in sorted(seq_samples, key=lambda x: x["gt_row"]):
            if max_frames is not None and len(frame_paths) >= max_frames:
                break
            gt_row = s["gt_row"]
            if gt_row >= len(rows):
                n_dropped += 1
                continue
            gt = gt_angles_from_row(rows[gt_row], gt_row, subject, action)
            if gt is None:
                n_dropped += 1
                continue

            buckets["shoulder_flexion"].append(gt.shoulder_flexion)
            buckets["shoulder_abduction"].append(gt.shoulder_abduction)
            buckets["shoulder_rotation"].append(gt.shoulder_rotation)
            buckets["elbow_flexion"].append(gt.elbow_flexion)
            buckets["left_shoulder_flexion"].append(gt.left_shoulder_flexion)
            buckets["left_shoulder_abduction"].append(gt.left_shoulder_abduction)
            buckets["left_shoulder_rotation"].append(gt.left_shoulder_rotation)
            buckets["left_elbow_flexion"].append(gt.left_elbow_flexion)
            subjects_col.append(subject)
            sequence_col.append(f"{subject}/{action}")
            frame_paths.append(root / s["frame_file"])

    if not frame_paths:
        raise ValueError("Manifest produced no usable samples.")

    print(f"[✓] {len(frame_paths):,} paired GT/frame samples"
          + (f" ({n_dropped} dropped: degenerate or out-of-range)" if n_dropped else ""))

    gt_arrays = {j: np.array(v, dtype=np.float64) for j, v in buckets.items()}
    gt_arrays = {j: a for j, a in gt_arrays.items() if not np.all(np.isnan(a))}
    meta = {
        "subject":  np.array(subjects_col),
        "sequence": np.array(sequence_col),
    }
    return gt_arrays, meta, frame_paths


# ── Synthetic mode ────────────────────────────────────────────────────────────

def synthetic_predictions(
    gt_arrays: dict[str, np.ndarray],
    frameworks: list[str],
    rng: np.random.Generator,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Generate synthetic framework predictions from GT + calibrated Gaussian noise.

    Parameters
    ----------
    gt_arrays : dict[joint → np.ndarray]
    frameworks : list of framework names to generate (must be in NOISE_PROFILES)
    rng : numpy random Generator

    Returns
    -------
    dict[framework → dict[joint → np.ndarray]]
    """
    preds: dict[str, dict[str, np.ndarray]] = {}

    for fw in frameworks:
        if fw not in NOISE_PROFILES:
            print(f"  [!] No noise profile for '{fw}' — skipping")
            continue
        profile  = NOISE_PROFILES[fw]
        preds[fw] = {}
        for j, arr in gt_arrays.items():
            # Left-arm joints reuse the base joint's noise profile
            base = j[len("left_"):] if j.startswith("left_") else j
            if base not in profile:
                preds[fw][j] = arr.copy()
                continue
            mu, sigma = profile[base]
            preds[fw][j] = arr + rng.normal(mu, sigma, size=len(arr))

    return preds


# ── Live mode ─────────────────────────────────────────────────────────────────

def live_predictions(
    gt_arrays:   dict[str, np.ndarray],
    frame_paths: list[Path],
    frameworks:  list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Run pose estimation frameworks on H3.6M video frames and collect angles.

    Parameters
    ----------
    gt_arrays : dict[joint → np.ndarray]
        GT angles. Element i must correspond to frame_paths[i]; the manifest
        path (load_manifest_gt) guarantees this, and it is the only supported
        way to build the pairing.
    frame_paths : list[Path]
        Frame images, index-aligned with the GT arrays.
    frameworks : list[str]
        Framework names to evaluate (passed to load_estimator()).

    Returns
    -------
    dict[framework → dict[joint → np.ndarray]]
    """
    import cv2
    from src.pose import load_estimator
    from src.processing.angle_solver import compute_bilateral_angles

    n_gt  = len(next(iter(gt_arrays.values())))
    preds: dict[str, dict[str, np.ndarray]] = {}
    joints = [j for j in JOINTS_BILATERAL if j in gt_arrays] or list(JOINTS)

    if len(frame_paths) != n_gt:
        raise ValueError(
            f"GT/frame misalignment: {n_gt} ground-truth samples but "
            f"{len(frame_paths)} frames. Rebuild the manifest with "
            f"scripts/prepare_h36m.py rather than pairing frames by filename."
        )

    n_frames = n_gt
    print(f"[→] Evaluating {n_frames} manifest-paired frames")

    for fw_name in frameworks:
        print(f"\n  Running {fw_name} on {n_frames} frames...")
        try:
            runner = load_estimator(fw_name)
        except Exception as e:
            print(f"  [✗] Could not load {fw_name}: {e}")
            continue

        # RAW predictions — no temporal filtering here. Filtering is a
        # separate, explicitly ablated stage (src/evaluation/ablation.py),
        # so framework accuracy and filter effects are never conflated.
        angles_per_joint: dict[str, list[float]] = {j: [] for j in joints}

        def _append_nan():
            for j in joints:
                angles_per_joint[j].append(float("nan"))

        for fp in frame_paths[:n_frames]:
            bgr = cv2.imread(str(fp))
            if bgr is None:
                _append_nan()
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            lms = runner.process(rgb)

            if lms is None:
                _append_nan()
                continue

            bl = compute_bilateral_angles(lms)
            if bl is None:
                _append_nan()
                continue

            side_angles = {"right": bl.right, "left": bl.left}
            for j in joints:
                side = "left" if j.startswith("left_") else "right"
                attr = j[len("left_"):] if j.startswith("left_") else j
                a = side_angles[side]
                angles_per_joint[j].append(
                    getattr(a, attr) if a is not None else float("nan")
                )

        runner.close()

        # NaN frames (no detection) are kept as NaN and excluded per-joint
        # by the metrics — never interpolated or fabricated.
        preds[fw_name] = {j: np.array(v, dtype=np.float64)
                          for j, v in angles_per_joint.items()}
        all_vals = np.concatenate(list(preds[fw_name].values()))
        det_rate = float(np.mean(~np.isnan(all_vals))) * 100 if len(all_vals) else 0.0
        print(f"  [{fw_name}] done. Detection rate ≈ {det_rate:.0f}%")

    return preds


# ── CSV mode ──────────────────────────────────────────────────────────────────

def csv_predictions(
    csv_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """
    Load a pre-built dataset CSV (from build_h36m_dataset.py) and split
    into GT arrays and per-framework prediction arrays.

    The CSV is expected to have columns:
        gt_shoulder_flexion, gt_shoulder_abduction, ...
        mp_shoulder_flexion, mp_shoulder_abduction, ...
        mv_shoulder_flexion, ...
        pn_shoulder_flexion, ...
    """
    df = pd.read_csv(csv_path)
    print(f"[OK] CSV loaded: {len(df):,} rows, columns: {list(df.columns)}")

    # Map CSV prefixes to display names
    PREFIX_MAP = {
        "mp": "MediaPipe",
        "mv": "MoveNet-Lightning",
        "pn": "PoseNet",
    }
    # Map old column names to new joint names
    JOINT_MAP = {
        "shoulder_pitch": "shoulder_flexion",
        "shoulder_roll":  "shoulder_abduction",
        "shoulder_yaw":   "shoulder_rotation",
        "elbow_flexion":  "elbow_flexion",
        # New names pass through directly
        "shoulder_flexion":   "shoulder_flexion",
        "shoulder_abduction": "shoulder_abduction",
        "shoulder_rotation":  "shoulder_rotation",
    }

    gt_arrays: dict[str, np.ndarray] = {}
    pred_data: dict[str, dict[str, np.ndarray]] = {}

    for j in JOINTS:
        # Try direct name first, then old pitch/roll/yaw names
        old_j = {
            "shoulder_flexion":   "shoulder_pitch",
            "shoulder_abduction": "shoulder_roll",
            "shoulder_rotation":  "shoulder_yaw",
            "elbow_flexion":      "elbow_flexion",
        }.get(j, j)

        gt_col = f"gt_{j}" if f"gt_{j}" in df.columns else f"gt_{old_j}"
        if gt_col in df.columns:
            gt_arrays[j] = df[gt_col].to_numpy(dtype=np.float64)

    for prefix, fw_name in PREFIX_MAP.items():
        pred_data[fw_name] = {}
        for j in JOINTS:
            old_j = {
                "shoulder_flexion":   "shoulder_pitch",
                "shoulder_abduction": "shoulder_roll",
                "shoulder_rotation":  "shoulder_yaw",
                "elbow_flexion":      "elbow_flexion",
            }.get(j, j)
            col = f"{prefix}_{j}" if f"{prefix}_{j}" in df.columns else f"{prefix}_{old_j}"
            if col in df.columns:
                pred_data[fw_name][j] = df[col].to_numpy(dtype=np.float64)

    return gt_arrays, pred_data


# ── LaTeX table generator ─────────────────────────────────────────────────────

def export_latex_table(
    all_results: list[FrameworkMetrics],
    out_path: str,
    publication_grade: bool = True,
) -> None:
    """
    Generate a LaTeX table for the IEEE paper evaluation section.
    """
    caption_note = (
        "" if publication_grade else
        " SYNTHETIC SIMULATION — pipeline verification only, NOT experimental results."
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison of pose estimation frameworks on Human3.6M (bilateral arms, ZXY decomposition). "
        r"MPJAE = Mean Per-Joint Angle Error. PCK@5° = \% frames within 5\textdegree. "
        r"Best result per column in \textbf{bold}." + caption_note + r"}",
        r"\label{tab:framework_comparison}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Framework & MPJAE $\downarrow$ & RMSE $\downarrow$ & Pearson $r$ $\uparrow$ "
        r"& PCK@5° $\uparrow$ & Jitter $\downarrow$ \\",
        r" & (°) & (°) & & (\%) & (°/fr) \\",
        r"\hline",
    ]

    valid = [r for r in all_results if not np.isnan(r.mpjae)]

    # Find bests
    best_mpjae  = min(valid, key=lambda r: r.mpjae).framework
    best_rmse   = min(valid, key=lambda r: r.mean_rmse).framework
    best_r      = max(valid, key=lambda r: r.mean_r).framework
    best_pck    = max(valid, key=lambda r: r.mean_pck_5).framework
    best_jitter = min(valid, key=lambda r: r.mean_jitter).framework

    def bf(fw, metric_name, val_str):
        if fw == metric_name:
            return r"\textbf{" + val_str + r"}"
        return val_str

    for r in valid:
        fw = r.framework
        row = (
            f"{fw} & "
            f"{bf(fw, best_mpjae,  f'{r.mpjae:.2f}')} & "
            f"{bf(fw, best_rmse,   f'{r.mean_rmse:.2f}')} & "
            f"{bf(fw, best_r,      f'{r.mean_r:.3f}')} & "
            f"{bf(fw, best_pck,    f'{r.mean_pck_5:.1f}')} & "
            f"{bf(fw, best_jitter, f'{r.mean_jitter:.2f}')} \\\\"
        )
        lines.append(row)

    # Per-joint MAE sub-table
    lines += [
        r"\hline",
        r"\multicolumn{6}{l}{\textit{Per-joint MAE (degrees)}} \\",
        r"\hline",
        r" & Flex. & Abd. & Rot. & Elbow & \multicolumn{1}{c}{---} \\",
        r"\hline",
    ]
    for r in valid:
        joint_maes = [
            f"{r.joints[j].mae:.2f}" if j in r.joints else "---"
            for j in JOINTS
        ]
        lines.append(f"{r.framework} & " + " & ".join(joint_maes) + r" & \\ ")

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [✓] LaTeX table → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

SYNTHETIC_WARNING = """
================================================================================
  WARNING: --mode synthetic simulates predictions as GT + Gaussian noise.
  This is a PIPELINE SMOKE TEST. Every output will be tagged
  publication_grade=false. Do NOT report these numbers as experimental
  results — use --mode live on real H3.6M frames for publishable metrics.
================================================================================
"""


def loso_evaluation(
    gt_arrays: dict[str, np.ndarray],
    pred_data: dict[str, dict[str, np.ndarray]],
    subject_col: np.ndarray,
    joints: list[str],
) -> list[dict]:
    """
    Leave-One-Subject-Out evaluation: metrics per held-out subject,
    plus mean ± std aggregate rows per framework.
    """
    rows: list[dict] = []
    subjects_present = [s for s in dict.fromkeys(subject_col.tolist()) if s]
    if len(subjects_present) < 2:
        print(f"  [!] LOSO needs ≥2 subjects in the loaded data; found "
              f"{subjects_present}. Increase --max_frames or add --subjects.")
        return rows

    for fold in loso_folds(subjects_present):
        mask = subject_col == fold.test_subject
        if not mask.any():
            continue
        gt_fold = {j: a[mask] for j, a in gt_arrays.items()}
        for fw, fw_preds in pred_data.items():
            pred_fold = {j: a[mask] for j, a in fw_preds.items() if len(a) == len(mask)}
            if not pred_fold:
                continue
            m = evaluate_framework(fw, pred_fold, gt_fold, joints=joints)
            rows.append({
                "fold":          fold.fold_idx,
                "test_subject":  fold.test_subject,
                "framework":     fw,
                "n_frames":      m.n_frames,
                "MPJAE_deg":     m.mpjae,
                "RMSE_deg":      m.mean_rmse,
                "Pearson_r":     m.mean_r,
                "R2":            m.mean_r2,
                "PCK@5_pct":     m.mean_pck_5,
                "jitter_deg_per_frame": m.mean_jitter,
            })

    # Aggregate mean ± std across folds per framework
    for fw in pred_data:
        fw_rows = [r for r in rows if r["framework"] == fw and isinstance(r["fold"], int)]
        if not fw_rows:
            continue
        for stat_name, stat_fn in (("MEAN", np.mean), ("STD", np.std)):
            rows.append({
                "fold":          stat_name,
                "test_subject":  "ALL",
                "framework":     fw,
                "n_frames":      int(np.sum([r["n_frames"] for r in fw_rows])),
                "MPJAE_deg":     float(stat_fn([r["MPJAE_deg"] for r in fw_rows])),
                "RMSE_deg":      float(stat_fn([r["RMSE_deg"] for r in fw_rows])),
                "Pearson_r":     float(stat_fn([r["Pearson_r"] for r in fw_rows])),
                "R2":            float(stat_fn([r["R2"] for r in fw_rows])),
                "PCK@5_pct":     float(stat_fn([r["PCK@5_pct"] for r in fw_rows])),
                "jitter_deg_per_frame":
                    float(stat_fn([r["jitter_deg_per_frame"] for r in fw_rows])),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate MonoArm frameworks against H3.6M ground truth."
    )
    ap.add_argument("--h36m_dir",   default="data/dataset/h3.6m/dataset",
                    help="Root of H3.6M skeleton .txt files")
    ap.add_argument("--manifest",   default=None,
                    help="manifest.json from scripts/prepare_h36m.py, pairing "
                         "each extracted video frame with its ground-truth "
                         "pose row (required for --mode live)")
    ap.add_argument("--csv",        default=None,
                    help="Pre-built dataset CSV from build_h36m_dataset.py")
    ap.add_argument("--mode",       choices=["live", "csv", "synthetic"],
                    required=True,
                    help="Prediction source. 'live' runs real frameworks on "
                         "H3.6M frames (publication-grade); 'csv' loads "
                         "precomputed real predictions; 'synthetic' is a "
                         "pipeline smoke test (NOT for publication).")
    ap.add_argument("--protocol",   choices=["test-split", "loso"],
                    default="test-split",
                    help="Experimental protocol (default: H3.6M standard "
                         "test split S9/S11; 'loso' = Leave-One-Subject-Out)")
    ap.add_argument("--subjects",   nargs="+", default=None,
                    help="H3.6M subjects to evaluate (default: protocol's "
                         "subject set)")
    ap.add_argument("--max_frames", type=int, default=10000,
                    help="Maximum GT frames to load (default 10000)")
    ap.add_argument("--frameworks", nargs="+",
                    default=["MediaPipe", "MoveNet-Lightning", "PoseNet"],
                    help="Frameworks to compare")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--alpha",      type=float, default=0.05,
                    help="Significance level for statistical tests")
    ap.add_argument("--n_boot",     type=int, default=5000,
                    help="Bootstrap resamples for confidence intervals")
    ap.add_argument("--skip_ablation", action="store_true",
                    help="Skip the filter ablation study")
    ap.add_argument("--output_dir", default="outputs/validation",
                    help="Directory for all output files")
    args = ap.parse_args()

    publication_grade = args.mode != "synthetic"
    if not publication_grade:
        print(SYNTHETIC_WARNING)

    if args.subjects is None:
        from src.evaluation.protocol import H36M_ALL_SUBJECTS
        args.subjects = (H36M_ALL_SUBJECTS if args.protocol == "loso"
                         else get_split("test"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ── Load data ─────────────────────────────────────────────────────────────
    meta: dict[str, np.ndarray] | None = None

    if args.mode == "csv":
        if not args.csv:
            print("[✗] --mode csv requires --csv <path>")
            sys.exit(1)
        gt_arrays, pred_data = csv_predictions(Path(args.csv))

    elif args.mode == "live":
        if not args.manifest:
            print("[✗] --mode live requires --manifest <prepared manifest.json>")
            print("    Build one with:  python scripts/prepare_h36m.py "
                  "--h36m_root <raw release> --out_dir <prepared>")
            print("    See docs/DATA_H36M.md for obtaining the video release.")
            sys.exit(1)
        gt_arrays, meta, frame_paths = load_manifest_gt(
            Path(args.manifest), args.subjects, args.max_frames
        )
        pred_data = live_predictions(gt_arrays, frame_paths, args.frameworks)

    else:   # synthetic (explicit opt-in smoke test)
        gt_arrays, meta = load_gt_arrays(Path(args.h36m_dir), args.subjects, args.max_frames)
        if not any(len(v) > 0 for v in gt_arrays.values()):
            print("[X] No GT data loaded. Check that h36m_dir exists and contains .txt files.")
            print(f"    Expected path: {Path(args.h36m_dir).resolve()}")
            print("    Example structure: data/dataset/h3.6m/dataset/S9/Directions 1.txt")
            sys.exit(1)
        pred_data = synthetic_predictions(gt_arrays, args.frameworks, rng)

    joints   = list(gt_arrays.keys())
    n_frames = len(next(iter(gt_arrays.values())))
    print(f"[OK] {n_frames:,} frames × {len(joints)} joints ready for evaluation\n")

    # ── Compute metrics ───────────────────────────────────────────────────────
    all_results: list[FrameworkMetrics] = []
    for fw in args.frameworks:
        if fw not in pred_data:
            print(f"  [!] No predictions for {fw} — skipping")
            continue
        result = evaluate_framework(
            framework=fw,
            pred_arrays=pred_data[fw],
            gt_arrays=gt_arrays,
            joints=joints,
        )
        all_results.append(result)
        print(f"  [{fw}]  MPJAE={result.mpjae:.2f}°  "
              f"r={result.mean_r:.3f}  PCK@5={result.mean_pck_5:.1f}%")

    if not all_results:
        print("[✗] No results produced.")
        sys.exit(1)

    # ── Console table ─────────────────────────────────────────────────────────
    print_metrics_table(all_results)

    # ── Statistical significance testing ─────────────────────────────────────
    print("[→] Statistical significance tests (paired t, Wilcoxon, Cohen's d)...")
    stats_report = compare_systems(
        pred_data, gt_arrays, joints=joints,
        alpha=args.alpha, n_boot=args.n_boot, seed=args.seed,
    )
    stats_rows = comparison_report_to_dicts(stats_report)
    n_sig = sum(1 for r in stats_rows if r["significant"])
    print(f"[✓] {len(stats_rows)} pairwise tests; {n_sig} significant "
          f"after Holm-Bonferroni (α={args.alpha})")

    # ── Filter ablation vs ground truth ───────────────────────────────────────
    ablation_rows: list[dict] = []
    if not args.skip_ablation:
        print("[→] Filter ablation study (none / MA / SG / Kalman)...")
        seq_ids = meta["sequence"] if meta is not None else None
        ablation_rows = run_filter_ablation(
            pred_data, gt_arrays, joints=joints, seq_ids=seq_ids,
        )
        print(f"[✓] Ablation grid complete ({len(ablation_rows)} rows)")

    # ── LOSO protocol ─────────────────────────────────────────────────────────
    loso_rows: list[dict] = []
    if args.protocol == "loso":
        if meta is None:
            print("[!] LOSO requires per-frame subject labels — not available "
                  "in csv mode. Skipping LOSO aggregation.")
        else:
            print("[→] Leave-One-Subject-Out evaluation...")
            loso_rows = loso_evaluation(gt_arrays, pred_data, meta["subject"], joints)
            print(f"[✓] LOSO complete ({len(loso_rows)} rows)")

    # ── Reproducibility metadata ──────────────────────────────────────────────
    metadata = build_metadata(
        dataset="Human3.6M (Ionescu et al., 2014)",
        subjects=args.subjects,
        frameworks=args.frameworks,
        mode=args.mode,
        protocol=describe_protocol(args.protocol, args.subjects),
        seed=args.seed,
        extra={
            "n_frames": n_frames,
            "joints": joints,
            "max_frames": args.max_frames,
            "alpha": args.alpha,
            "n_boot": args.n_boot,
        },
    )

    # ── Exports ───────────────────────────────────────────────────────────────
    with open(out_dir / "metrics_report.json", "w") as f:
        json.dump({"metadata": metadata,
                   "results": metrics_to_dict(all_results)}, f, indent=2)
    with open(out_dir / "statistical_tests.json", "w") as f:
        json.dump(stats_rows, f, indent=2)
    if ablation_rows:
        with open(out_dir / "filter_ablation.json", "w") as f:
            json.dump(ablation_rows, f, indent=2)
    export_metadata_json(out_dir / "metadata.json", metadata)
    print(f"[✓] JSON reports → {out_dir}/")

    export_frame_level_csv(out_dir / "frame_level.csv", gt_arrays, pred_data,
                           meta_columns=meta)
    xlsx = export_excel(
        out_dir / "results.xlsx",
        all_results, gt_arrays, pred_data, metadata,
        stats_rows=stats_rows,
        ablation_rows=ablation_rows,
        meta_columns=meta,
        extra_sheets={"LOSO": loso_rows} if loso_rows else None,
    )
    print(f"[✓] Excel workbook → {xlsx}")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    export_latex_table(all_results, str(out_dir / "paper_table.tex"),
                       publication_grade=publication_grade)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n[→] Generating figures...")
    plot_validation_summary(all_results, str(out_dir / "validation_summary.png"))
    plot_bland_altman(all_results, pred_data, gt_arrays, str(out_dir / "bland_altman.png"))
    plot_scatter_gt(all_results, pred_data, gt_arrays, str(out_dir / "scatter_gt.png"))
    plot_error_cdf(all_results, pred_data, gt_arrays, str(out_dir / "error_cdf.png"))
    plot_timeseries_vs_gt(all_results, pred_data, gt_arrays,
                          str(out_dir / "timeseries_vs_gt.png"), n_frames=min(500, n_frames))

    if not publication_grade:
        print(SYNTHETIC_WARNING)
    print(f"\n[✓] All outputs saved to {out_dir}/")
    print("    Run 'python scripts/compare_frameworks.py' for runtime benchmarks")
    print("    Open outputs/validation/paper_table.tex for the LaTeX table\n")


if __name__ == "__main__":
    main()
