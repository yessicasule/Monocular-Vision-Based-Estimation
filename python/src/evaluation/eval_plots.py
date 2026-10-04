"""
eval_plots.py — Publication-Quality Figures for H3.6M Validation
=================================================================

Generates all figures needed for the IEEE conference paper's evaluation
section from FrameworkMetrics objects produced by evaluate_h36m.py.

Figures Generated
-----------------
    validation_summary.png   — 2×4 grid: MPJAE, RMSE, r, PCK bars + per-joint breakdown
    bland_altman.png         — Bland-Altman agreement plots for each framework/joint
    scatter_gt.png           — Predicted vs. GT scatter with regression line per joint
    error_cdf.png            — CDF of absolute error per framework (PCK-style curve)
    timeseries_vs_gt.png     — Sample angle time-series overlaid with ground truth

All figures use a consistent dark theme and MonoArm colour palette.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from .metrics import FrameworkMetrics, JOINTS, JOINT_LABELS

# ── Colour palette ────────────────────────────────────────────────────────────
FW_COLORS = {
    "MediaPipe":           "#64dc64",
    "MoveNet-Lightning":   "#64a0ff",
    "PoseNet":             "#ffb450",
    "MoveNet":             "#64a0ff",
}
JOINT_COLORS = {
    "shoulder_flexion":   "#64dc64",
    "shoulder_abduction": "#64a0ff",
    "shoulder_rotation":  "#ffb450",
    "elbow_flexion":      "#dc50dc",
}

BG_DARK  = "#0e0e18"
BG_PANEL = "#1a1a26"
GRID_COL = "#2a2a3a"
TEXT_COL = "#cccccc"
TITLE_COL = "#aacfff"


def _fw_color(name: str) -> str:
    for k, c in FW_COLORS.items():
        if k.lower() in name.lower():
            return c
    return "#ffffff"


def _style(ax, title: str = "") -> None:
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.spines[:].set_color(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.5, linestyle="--")
    if title:
        ax.set_title(title, color=TITLE_COL, fontsize=9, pad=5)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)


def _save(fig, path: str, dpi: int = 150) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [✓] {path}")


# ── 1. Validation summary bar chart ──────────────────────────────────────────

def plot_validation_summary(
    all_results: list[FrameworkMetrics],
    out_path: str,
) -> None:
    """
    2×4 panel figure:
    Row 0: MPJAE, RMSE, Pearson r, PCK@5° (aggregate bars)
    Row 1: Per-joint MAE bars for each metric
    """
    valid  = [r for r in all_results if not np.isnan(r.mpjae)]
    names  = [r.framework for r in valid]
    n_fw   = len(valid)
    colors = [_fw_color(n) for n in names]
    x      = np.arange(n_fw)
    bar_kw = dict(width=0.55, edgecolor=GRID_COL, linewidth=0.6)

    fig = plt.figure(figsize=(18, 9), facecolor=BG_DARK)
    fig.suptitle(
        "MonoArm — Validation Against Human3.6M Ground Truth",
        color="white", fontsize=14, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.6, wspace=0.4,
                           left=0.05, right=0.97, top=0.94, bottom=0.07)

    # ── Row 0: aggregate metrics ──────────────────────────────────────────────
    aggregate_panels = [
        ("MPJAE (°)  ↓ lower better",  [r.mpjae         for r in valid], True),
        ("RMSE (°)   ↓ lower better",  [r.mean_rmse     for r in valid], True),
        ("Pearson r  ↑ higher better", [r.mean_r        for r in valid], False),
        ("PCK @ 5°   ↑ higher better", [r.mean_pck_5    for r in valid], False),
    ]

    for col, (title, values, lower_better) in enumerate(aggregate_panels):
        ax = fig.add_subplot(gs[0, col])
        bars = ax.bar(x, values, color=colors, **bar_kw)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7, color=TEXT_COL)
        _style(ax, title)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{v:.2f}",
                ha="center", va="bottom", color="white", fontsize=8,
            )
        # Best marker
        best_idx = (np.argmin if lower_better else np.argmax)(values)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.0)

    # ── Row 1: per-joint MAE breakdown ────────────────────────────────────────
    for col, j in enumerate(JOINTS):
        ax = fig.add_subplot(gs[1, col])
        mae_vals = [
            r.joints[j].mae if j in r.joints else float("nan")
            for r in valid
        ]
        bars = ax.bar(x, mae_vals, color=colors, **bar_kw)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7, color=TEXT_COL)
        ax.axhline(5.0, color="#ff6666", linestyle="--", linewidth=0.8, label="5° spec")
        _style(ax, f"{JOINT_LABELS[j]}\nMAE (°)")
        ax.legend(fontsize=6, labelcolor="white", facecolor=BG_PANEL, framealpha=0.7)
        for bar, v in zip(bars, mae_vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v * 1.03,
                    f"{v:.1f}",
                    ha="center", va="bottom", color="white", fontsize=7,
                )

    _save(fig, out_path)


# ── 2. Bland-Altman agreement plots ──────────────────────────────────────────

def plot_bland_altman(
    all_results: list[FrameworkMetrics],
    pred_data:   dict[str, dict[str, np.ndarray]],   # fw → joint → predicted
    gt_data:     dict[str, np.ndarray],               # joint → GT
    out_path:    str,
) -> None:
    """
    Bland-Altman plot (mean vs. difference) for each framework × joint.
    Shows systematic bias and limits of agreement (mean ± 1.96σ).

    Parameters
    ----------
    pred_data : dict[fw_name → dict[joint → np.ndarray]]
    gt_data   : dict[joint → np.ndarray]
    """
    valid  = [r for r in all_results if r.framework in pred_data]
    n_fw   = len(valid)
    n_j    = len(JOINTS)

    fig, axes = plt.subplots(
        n_fw, n_j,
        figsize=(5 * n_j, 4 * n_fw),
        facecolor=BG_DARK, squeeze=False,
    )
    fig.suptitle(
        "Bland-Altman Agreement: Predicted vs. H3.6M Ground Truth",
        color="white", fontsize=13, fontweight="bold",
    )

    for fi, r in enumerate(valid):
        fw   = r.framework
        col  = _fw_color(fw)
        preds = pred_data.get(fw, {})
        for ji, j in enumerate(JOINTS):
            ax = axes[fi][ji]
            if j not in preds or j not in gt_data:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", color="grey")
                _style(ax)
                continue

            p  = preds[j]
            g  = gt_data[j]
            mn = (p + g) / 2.0
            di = p - g

            bias     = float(np.mean(di))
            loa      = 1.96 * float(np.std(di))

            ax.scatter(mn, di, alpha=0.15, s=4, color=col)
            ax.axhline(bias,        color=col,      linewidth=1.5, linestyle="-",
                       label=f"bias={bias:+.2f}°")
            ax.axhline(bias + loa,  color="#ff6666", linewidth=1.0, linestyle="--",
                       label=f"+1.96σ={bias+loa:.2f}°")
            ax.axhline(bias - loa,  color="#ff6666", linewidth=1.0, linestyle="--",
                       label=f"-1.96σ={bias-loa:.2f}°")
            ax.axhline(0,           color="#555",   linewidth=0.6, linestyle=":")

            _style(ax, f"{fw} — {JOINT_LABELS[j]}")
            ax.set_xlabel("Mean of predicted & GT (°)", fontsize=7)
            ax.set_ylabel("Predicted − GT (°)",         fontsize=7)
            ax.legend(fontsize=6, labelcolor="white", facecolor=BG_PANEL,
                      framealpha=0.7, loc="upper right")

    plt.tight_layout()
    _save(fig, out_path)


# ── 3. Predicted vs. GT scatter ──────────────────────────────────────────────

def plot_scatter_gt(
    all_results: list[FrameworkMetrics],
    pred_data:   dict[str, dict[str, np.ndarray]],
    gt_data:     dict[str, np.ndarray],
    out_path:    str,
) -> None:
    """
    Predicted vs. GT scatter with linear regression line and r² annotation.
    Layout: N_fw rows × 4 joint columns.
    """
    valid  = [r for r in all_results if r.framework in pred_data]
    n_fw   = len(valid)
    n_j    = len(JOINTS)

    fig, axes = plt.subplots(
        n_fw, n_j,
        figsize=(5 * n_j, 4 * n_fw),
        facecolor=BG_DARK, squeeze=False,
    )
    fig.suptitle(
        "Predicted vs. Ground Truth — Per Framework and Joint",
        color="white", fontsize=13, fontweight="bold",
    )

    for fi, r in enumerate(valid):
        fw    = r.framework
        col   = _fw_color(fw)
        preds = pred_data.get(fw, {})
        for ji, j in enumerate(JOINTS):
            ax = axes[fi][ji]
            if j not in preds or j not in gt_data:
                _style(ax)
                continue

            p = preds[j]
            g = gt_data[j]
            ax.scatter(g, p, alpha=0.12, s=3, color=col)

            # Regression line
            lo, hi = g.min(), g.max()
            if hi > lo:
                m, b     = np.polyfit(g, p, 1)
                r2       = float(np.corrcoef(g, p)[0, 1]) ** 2
                x_line   = np.linspace(lo, hi, 200)
                ax.plot(x_line, m * x_line + b, color=col, linewidth=1.5,
                        label=f"y={m:.2f}x+{b:.1f}")
                # Identity line
                ax.plot([lo, hi], [lo, hi], color="#888", linewidth=0.8,
                        linestyle="--", label="y=x")
                ax.text(0.05, 0.92, f"r²={r2:.3f}",
                        transform=ax.transAxes, color="white", fontsize=8)

            _style(ax, f"{fw} — {JOINT_LABELS[j]}")
            ax.set_xlabel("Ground Truth (°)", fontsize=7)
            ax.set_ylabel("Predicted (°)",    fontsize=7)
            ax.legend(fontsize=6, labelcolor="white", facecolor=BG_PANEL,
                      framealpha=0.7)

    plt.tight_layout()
    _save(fig, out_path)


# ── 4. CDF of absolute error ─────────────────────────────────────────────────

def plot_error_cdf(
    all_results: list[FrameworkMetrics],
    pred_data:   dict[str, dict[str, np.ndarray]],
    gt_data:     dict[str, np.ndarray],
    out_path:    str,
) -> None:
    """
    CDF of |pred - GT| for each framework, aggregated across all 4 joints.
    Shows the proportion of frames within a given error threshold (PCK curve).

    Vertical dashed lines at 5° and 10° indicate common reporting thresholds.
    """
    valid = [r for r in all_results if r.framework in pred_data]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor=BG_DARK)

    thresholds = np.linspace(0, 30, 300)

    for r in valid:
        fw    = r.framework
        col   = _fw_color(fw)
        preds = pred_data.get(fw, {})

        all_abs_errs = []
        for j in JOINTS:
            if j in preds and j in gt_data:
                all_abs_errs.append(np.abs(preds[j] - gt_data[j]))

        if not all_abs_errs:
            continue

        err_flat = np.concatenate(all_abs_errs)
        cdf      = np.array([np.mean(err_flat < t) * 100.0 for t in thresholds])
        ax.plot(thresholds, cdf, color=col, linewidth=2.0, label=fw)

    ax.axvline(5.0,  color="#ff6666", linewidth=0.9, linestyle="--", label="5° threshold")
    ax.axvline(10.0, color="#ffaa44", linewidth=0.9, linestyle="--", label="10° threshold")
    ax.axhline(80.0, color="#555",    linewidth=0.6, linestyle=":")
    ax.set_xlabel("Absolute Error Threshold (°)", fontsize=10)
    ax.set_ylabel("% Frames Within Threshold",    fontsize=10)
    _style(ax, "Error CDF (PCK Curve) — All Joints Combined")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, labelcolor="white", facecolor=BG_PANEL, framealpha=0.8)

    _save(fig, out_path)


# ── 5. Sample time-series with GT overlay ────────────────────────────────────

def plot_timeseries_vs_gt(
    all_results: list[FrameworkMetrics],
    pred_data:   dict[str, dict[str, np.ndarray]],
    gt_data:     dict[str, np.ndarray],
    out_path:    str,
    n_frames:    int = 500,
) -> None:
    """
    Angle time-series for a sample window, one panel per joint.
    Each framework plotted with its assigned colour; GT in white dashed.
    """
    valid = [r for r in all_results if r.framework in pred_data]
    fig, axes = plt.subplots(
        len(JOINTS), 1,
        figsize=(14, 3.5 * len(JOINTS)),
        facecolor=BG_DARK, sharex=True,
    )
    fig.suptitle(
        f"Joint Angle Time Series vs. Ground Truth  (first {n_frames} frames)",
        color="white", fontsize=13, fontweight="bold",
    )

    for ji, j in enumerate(JOINTS):
        ax = axes[ji]
        t  = np.arange(n_frames)

        # Ground truth
        if j in gt_data:
            ax.plot(t, gt_data[j][:n_frames],
                    color="white", linewidth=1.5, linestyle="--",
                    alpha=0.8, label="Ground Truth", zorder=10)

        # Predictions
        for r in valid:
            fw    = r.framework
            col   = _fw_color(fw)
            preds = pred_data.get(fw, {})
            if j in preds:
                ax.plot(t, preds[j][:n_frames],
                        color=col, linewidth=1.0, alpha=0.75, label=fw)

        _style(ax, JOINT_LABELS[j])
        ax.set_ylabel("degrees (°)", fontsize=8)
        if ji == len(JOINTS) - 1:
            ax.set_xlabel("frame", fontsize=8)
        ax.legend(fontsize=7.5, labelcolor="white", facecolor=BG_PANEL,
                  framealpha=0.7, loc="upper right")

    plt.tight_layout()
    _save(fig, out_path)
