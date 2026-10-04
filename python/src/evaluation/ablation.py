"""
ablation.py — Filter Ablation Study Against Ground Truth
=========================================================

Quantifies the effect of each temporal filter in the AngleFilterBank
(2-state Kalman, Savitzky-Golay, Moving Average) on accuracy, jitter
reduction, temporal stability, and latency — all measured against
dataset ground truth rather than against the unfiltered signal alone.

For every (framework × filter) combination the study applies the filter
CAUSALLY over the predicted angle time-series (frame by frame, exactly
as the real-time pipeline would), resetting filter state at sequence
boundaries so filtering never crosses subject/action transitions, then
computes the standard metric set versus ground truth. Per-update
compute latency is measured with perf_counter over the actual filter
update calls.

The "none" condition (raw predictions) is always included as the
baseline for the ablation deltas.
"""

from __future__ import annotations

import time

import numpy as np

from ..processing.angle_filter import (
    MovingAverageFilter, SavitzkyGolayFilter, KalmanFilter2State,
)
from .metrics import compute_joint_metrics

FILTER_CONDITIONS = ["none", "ma", "sg", "kalman"]


def _make_filter(condition: str, stream_hz: float):
    if condition == "ma":
        return MovingAverageFilter()
    if condition == "sg":
        return SavitzkyGolayFilter()
    if condition == "kalman":
        return KalmanFilter2State(dt=1.0 / stream_hz)
    return None


def _apply_causal(
    series:    np.ndarray,
    condition: str,
    seq_ids:   np.ndarray | None,
    stream_hz: float,
) -> tuple[np.ndarray, float]:
    """
    Run one filter causally over a series.

    Returns (filtered_series, mean_update_latency_us). NaN samples pass
    through unfiltered and reset the filter (tracking loss semantics).
    """
    if condition == "none":
        return series.copy(), 0.0

    filt = _make_filter(condition, stream_hz)
    out  = np.empty_like(series)
    total_t = 0.0
    n_updates = 0
    prev_seq = None

    for i, v in enumerate(series):
        if seq_ids is not None:
            seq = seq_ids[i]
            if prev_seq is not None and seq != prev_seq:
                filt.reset()
            prev_seq = seq

        if np.isnan(v):
            filt.reset()
            out[i] = np.nan
            continue

        t0 = time.perf_counter()
        out[i] = filt.update(float(v))
        total_t += time.perf_counter() - t0
        n_updates += 1

    latency_us = (total_t / n_updates * 1e6) if n_updates else float("nan")
    return out, latency_us


def run_filter_ablation(
    pred_data: dict[str, dict[str, np.ndarray]],
    gt_arrays: dict[str, np.ndarray],
    joints:    list[str] | None = None,
    seq_ids:   np.ndarray | None = None,
    stream_hz: float = 50.0,
) -> list[dict]:
    """
    Run the full ablation grid: framework × filter condition × joint.

    Parameters
    ----------
    pred_data : dict[framework → dict[joint → RAW predicted angles]]
        Predictions must be unfiltered; the ablation applies each filter
        itself so conditions are directly comparable.
    gt_arrays : dict[joint → ground-truth angles]
    joints : joints to include (default: gt_arrays keys).
    seq_ids : per-frame sequence identifiers (e.g. "S9/Walking 1") used
        to reset filter state at sequence boundaries.
    stream_hz : assumed frame rate for the Kalman dt.

    Returns
    -------
    list[dict]
        One row per (framework × filter × joint) plus per-condition
        aggregate rows (joint = "ALL"), ready for DataFrame/Excel export.
    """
    joints = joints or list(gt_arrays.keys())
    rows: list[dict] = []

    for fw, fw_preds in pred_data.items():
        for condition in FILTER_CONDITIONS:
            maes, rmses, jitters, pck5s = [], [], [], []
            latencies = []

            for j in joints:
                if j not in fw_preds or j not in gt_arrays:
                    continue
                filtered, lat_us = _apply_causal(
                    np.asarray(fw_preds[j], dtype=np.float64),
                    condition, seq_ids, stream_hz,
                )
                gt   = gt_arrays[j]
                n    = min(len(filtered), len(gt))
                mask = ~(np.isnan(filtered[:n]) | np.isnan(gt[:n]))
                if mask.sum() < 2:
                    continue
                jm = compute_joint_metrics(j, filtered[:n][mask], gt[:n][mask])

                rows.append({
                    "framework":   fw,
                    "filter":      condition,
                    "joint":       j,
                    "n":           jm.n,
                    "MAE_deg":     jm.mae,
                    "RMSE_deg":    jm.rmse,
                    "Pearson_r":   jm.r,
                    "R2":          jm.r2,
                    "PCK@5_pct":   jm.pck_5,
                    "jitter_deg_per_frame": jm.jitter,
                    "latency_us_per_update": lat_us,
                })
                maes.append(jm.mae)
                rmses.append(jm.rmse)
                jitters.append(jm.jitter)
                pck5s.append(jm.pck_5)
                if condition != "none":
                    latencies.append(lat_us)

            if maes:
                rows.append({
                    "framework":   fw,
                    "filter":      condition,
                    "joint":       "ALL",
                    "n":           int(np.sum([r["n"] for r in rows
                                               if r["framework"] == fw
                                               and r["filter"] == condition
                                               and r["joint"] != "ALL"])),
                    "MAE_deg":     float(np.mean(maes)),
                    "RMSE_deg":    float(np.mean(rmses)),
                    "Pearson_r":   float("nan"),
                    "R2":          float("nan"),
                    "PCK@5_pct":   float(np.mean(pck5s)),
                    "jitter_deg_per_frame": float(np.mean(jitters)),
                    "latency_us_per_update": float(np.mean(latencies)) if latencies else 0.0,
                })

    return rows
