"""
report_export.py — Structured Result Export for Reproducible Reporting
========================================================================

Exports every experimental result to machine-readable files so that all
reported numbers can be reproduced from the exported artefacts without
rerunning the experiments:

    results.xlsx            — multi-sheet Excel workbook:
        Summary             — per-framework aggregate metrics + ranking
        Per-Joint Metrics   — full per-joint metric table
        Statistical Tests   — pairwise significance tests
        Filter Ablation     — per-filter accuracy/jitter/latency (optional)
        Frame-Level Data    — GT + predictions for every frame
        Metadata            — reproducibility record

    frame_level.csv         — same frame-level table as CSV
    metadata.json           — reproducibility record as JSON

Reproducibility metadata includes dataset identifier and subjects, model
and library versions, evaluation protocol, git commit, random seed, and
timestamps — everything required by the benchmarking specification.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ── Metadata ─────────────────────────────────────────────────────────────────

def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "not-installed"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def build_metadata(
    dataset:    str,
    subjects:   list[str],
    frameworks: list[str],
    mode:       str,
    protocol:   str,
    seed:       int,
    extra:      dict | None = None,
) -> dict:
    """
    Assemble the reproducibility metadata record.

    `mode` is the prediction source ("live", "csv", "synthetic").
    Synthetic results are explicitly flagged as not publication-grade.
    """
    meta = {
        "timestamp_utc":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset":           dataset,
        "subjects":          subjects,
        "frameworks":        frameworks,
        "evaluation_mode":   mode,
        "publication_grade": mode != "synthetic",
        "protocol":          protocol,
        "random_seed":       seed,
        "git_commit":        _git_commit(),
        "python_version":    sys.version.split()[0],
        "platform":          platform.platform(),
        "library_versions": {
            "numpy":      _package_version("numpy"),
            "scipy":      _package_version("scipy"),
            "pandas":     _package_version("pandas"),
            "mediapipe":  _package_version("mediapipe"),
            "tensorflow": _package_version("tensorflow"),
            "opencv":     _package_version("opencv-python"),
        },
    }
    if mode == "synthetic":
        meta["warning"] = (
            "Predictions were SIMULATED from ground truth plus a noise model. "
            "These results are for pipeline verification only and MUST NOT "
            "be reported as experimental findings."
        )
    if extra:
        meta.update(extra)
    return meta


# ── Table builders ───────────────────────────────────────────────────────────

def summary_dataframe(all_results) -> pd.DataFrame:
    """
    Per-framework aggregate metrics with per-metric and overall ranks.

    `all_results` is a list of metrics.FrameworkMetrics.
    """
    rows = []
    for r in all_results:
        rows.append({
            "framework":   r.framework,
            "n_frames":    r.n_frames,
            "MPJAE_deg":   r.mpjae,
            "RMSE_deg":    r.mean_rmse,
            "Pearson_r":   r.mean_r,
            "R2":          r.mean_r2,
            "PCK@5_pct":   r.mean_pck_5,
            "Jitter_deg_per_frame": r.mean_jitter,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Ranks: 1 = best. Lower-is-better for errors/jitter, higher for r/R²/PCK.
    df["rank_MPJAE"]  = df["MPJAE_deg"].rank(method="min")
    df["rank_RMSE"]   = df["RMSE_deg"].rank(method="min")
    df["rank_r"]      = df["Pearson_r"].rank(method="min", ascending=False)
    df["rank_PCK@5"]  = df["PCK@5_pct"].rank(method="min", ascending=False)
    df["rank_jitter"] = df["Jitter_deg_per_frame"].rank(method="min")
    rank_cols = ["rank_MPJAE", "rank_RMSE", "rank_r", "rank_PCK@5", "rank_jitter"]
    df["mean_rank"]    = df[rank_cols].mean(axis=1)
    df["overall_rank"] = df["mean_rank"].rank(method="min")
    return df.sort_values("overall_rank").reset_index(drop=True)


def per_joint_dataframe(all_results) -> pd.DataFrame:
    """Flat per-joint metric table across all frameworks."""
    rows = []
    for r in all_results:
        for j, jm in r.joints.items():
            rows.append({
                "framework":   r.framework,
                "joint":       j,
                "n":           jm.n,
                "MAE_deg":     jm.mae,
                "MAE_ci95_lo": jm.mae_ci95_lo,
                "MAE_ci95_hi": jm.mae_ci95_hi,
                "RMSE_deg":    jm.rmse,
                "bias_deg":    jm.bias,
                "Pearson_r":   jm.r,
                "R2":          jm.r2,
                "PCK@5_pct":   jm.pck_5,
                "PCK@10_pct":  jm.pck_10,
                "PCK@15_pct":  jm.pck_15,
                "jitter_deg_per_frame": jm.jitter,
            })
    return pd.DataFrame(rows)


def frame_level_dataframe(
    gt_arrays: dict[str, np.ndarray],
    pred_data: dict[str, dict[str, np.ndarray]],
    meta_columns: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """
    One row per frame: ground truth, every framework's prediction, and
    the signed error for each framework × joint.
    """
    n = len(next(iter(gt_arrays.values())))
    data: dict[str, np.ndarray] = {"frame": np.arange(n)}

    if meta_columns:
        for name, arr in meta_columns.items():
            data[name] = np.asarray(arr[:n])

    for j, arr in gt_arrays.items():
        data[f"gt_{j}"] = arr[:n]

    for fw, joints in pred_data.items():
        tag = fw.lower().replace("-", "_").replace(" ", "_")
        for j, arr in joints.items():
            m = min(len(arr), n)
            pred = np.full(n, np.nan)
            pred[:m] = arr[:m]
            data[f"{tag}_{j}"] = pred
            if j in gt_arrays:
                data[f"{tag}_{j}_err"] = pred - gt_arrays[j][:n]

    return pd.DataFrame(data)


# ── Export drivers ───────────────────────────────────────────────────────────

def export_excel(
    out_path:      str | Path,
    all_results,
    gt_arrays:     dict[str, np.ndarray],
    pred_data:     dict[str, dict[str, np.ndarray]],
    metadata:      dict,
    stats_rows:    list[dict] | None = None,
    ablation_rows: list[dict] | None = None,
    meta_columns:  dict[str, np.ndarray] | None = None,
    extra_sheets:  dict[str, list[dict]] | None = None,
    max_frame_rows: int = 100_000,
) -> Path:
    """
    Write the multi-sheet Excel workbook. Returns the written path.

    Frame-level data is truncated to `max_frame_rows` in the workbook
    (Excel practical limits); the full table is always exported to CSV
    by export_frame_level_csv().
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_df = frame_level_dataframe(gt_arrays, pred_data, meta_columns)

    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        summary_dataframe(all_results).to_excel(xl, sheet_name="Summary", index=False)
        per_joint_dataframe(all_results).to_excel(xl, sheet_name="Per-Joint Metrics", index=False)

        if stats_rows:
            pd.DataFrame(stats_rows).to_excel(xl, sheet_name="Statistical Tests", index=False)

        if ablation_rows:
            pd.DataFrame(ablation_rows).to_excel(xl, sheet_name="Filter Ablation", index=False)

        if extra_sheets:
            for sheet_name, sheet_rows in extra_sheets.items():
                if sheet_rows:
                    pd.DataFrame(sheet_rows).to_excel(xl, sheet_name=sheet_name[:31], index=False)

        frame_df.head(max_frame_rows).to_excel(xl, sheet_name="Frame-Level Data", index=False)

        meta_df = pd.DataFrame(
            [(k, json.dumps(v) if isinstance(v, (dict, list)) else v)
             for k, v in metadata.items()],
            columns=["key", "value"],
        )
        meta_df.to_excel(xl, sheet_name="Metadata", index=False)

    return out_path


def export_frame_level_csv(
    out_path:     str | Path,
    gt_arrays:    dict[str, np.ndarray],
    pred_data:    dict[str, dict[str, np.ndarray]],
    meta_columns: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write the complete frame-level table (untruncated) as CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_level_dataframe(gt_arrays, pred_data, meta_columns).to_csv(out_path, index=False)
    return out_path


def export_metadata_json(out_path: str | Path, metadata: dict) -> Path:
    """Write the reproducibility metadata record as JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return out_path
