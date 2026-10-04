"""
statistics.py — Statistical Significance Testing for Framework Comparison
==========================================================================

Determines whether performance differences between pose estimation
frameworks (or filter configurations) are statistically meaningful
rather than incidental, as required for publication-grade evaluation.

Tests provided
--------------
Paired t-test
    Parametric test on the per-frame absolute-error differences between
    two systems evaluated on the SAME frames (paired samples).
    H0: mean(|err_A| − |err_B|) = 0.

Wilcoxon signed-rank test
    Non-parametric alternative that does not assume normality of the
    error differences — appropriate because pose-error distributions are
    typically heavy-tailed.

Cohen's d (paired / d_z)
    Effect size for paired samples:
        d_z = mean(diff) / std(diff, ddof=1)
    Interpretation thresholds (Cohen, 1988): 0.2 small, 0.5 medium, 0.8 large.

Bootstrap confidence intervals
    Percentile bootstrap CI on any statistic of a sample (default: mean),
    resampling frames with replacement.

Multiple-comparison correction
    Holm-Bonferroni step-down correction applied across the family of
    pairwise framework comparisons per joint.

References
----------
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral
        Sciences, 2nd ed. Lawrence Erlbaum.
    Wilcoxon, F. (1945). Individual comparisons by ranking methods.
        Biometrics Bulletin 1(6):80-83.
    Holm, S. (1979). A simple sequentially rejective multiple test
        procedure. Scand. J. Statist. 6(2):65-70.
    Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

import numpy as np
from scipy import stats as _st

from .metrics import circular_diff


# ── Data types ───────────────────────────────────────────────────────────────

@dataclass
class PairedComparison:
    """Statistical comparison of two systems on one joint."""
    system_a:    str
    system_b:    str
    joint:       str
    n:           int
    mean_abs_err_a: float     # MAE of system A on this joint (degrees)
    mean_abs_err_b: float     # MAE of system B (degrees)
    mean_diff:   float        # mean(|err_A| − |err_B|); negative → A better
    diff_ci95:   tuple[float, float]   # bootstrap 95% CI on mean_diff
    t_stat:      float
    t_p:         float
    wilcoxon_stat: float
    wilcoxon_p:  float
    cohens_d:    float
    significant: bool = False   # after Holm-Bonferroni at alpha
    p_corrected: float = float("nan")


@dataclass
class ComparisonReport:
    """Family of pairwise comparisons with correction metadata."""
    alpha:       float
    correction:  str
    comparisons: list[PairedComparison] = field(default_factory=list)


# ── Core tests ───────────────────────────────────────────────────────────────

def paired_t_test(err_a: np.ndarray, err_b: np.ndarray) -> tuple[float, float]:
    """
    Paired t-test on two same-length error arrays.

    Returns (t_statistic, p_value). NaN pairs are dropped.
    """
    a, b = _drop_nan_pairs(err_a, err_b)
    if len(a) < 2:
        return float("nan"), float("nan")
    t, p = _st.ttest_rel(a, b)
    return float(t), float(p)


def wilcoxon_test(err_a: np.ndarray, err_b: np.ndarray) -> tuple[float, float]:
    """
    Wilcoxon signed-rank test on paired error arrays.

    Returns (statistic, p_value). Identical pairs (zero differences) are
    discarded per the standard 'wilcox' zero-handling policy.
    """
    a, b = _drop_nan_pairs(err_a, err_b)
    diff = a - b
    diff = diff[diff != 0.0]
    if len(diff) < 2:
        return float("nan"), float("nan")
    try:
        w, p = _st.wilcoxon(diff)
    except ValueError:
        return float("nan"), float("nan")
    return float(w), float(p)


def cohens_d_paired(err_a: np.ndarray, err_b: np.ndarray) -> float:
    """
    Cohen's d for paired samples (d_z): mean(diff) / std(diff).
    """
    a, b = _drop_nan_pairs(err_a, err_b)
    diff = a - b
    if len(diff) < 2:
        return float("nan")
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)


def bootstrap_ci(
    values:  np.ndarray,
    stat:    Callable[[np.ndarray], float] = np.mean,
    n_boot:  int = 5000,
    alpha:   float = 0.05,
    seed:    int = 0,
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval on `stat` of `values`.

    Returns (lo, hi) at the (1 − alpha) level.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng  = np.random.default_rng(seed)
    idx  = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = np.apply_along_axis(stat, 1, values[idx])
    lo   = float(np.percentile(boot, 100.0 * (alpha / 2)))
    hi   = float(np.percentile(boot, 100.0 * (1 - alpha / 2)))
    return lo, hi


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """
    Holm-Bonferroni step-down correction.

    Returns (reject_flags, corrected_p_values) in the ORIGINAL order.
    NaN p-values are never rejected and stay NaN.
    """
    m       = len(p_values)
    order   = sorted(range(m), key=lambda i: (np.isnan(p_values[i]), p_values[i]))
    reject  = [False] * m
    p_corr  = [float("nan")] * m

    running_max = 0.0
    stopped = False
    n_valid = sum(1 for p in p_values if not np.isnan(p))
    rank = 0
    for i in order:
        p = p_values[i]
        if np.isnan(p):
            continue
        adj = min(1.0, (n_valid - rank) * p)
        running_max = max(running_max, adj)
        p_corr[i] = running_max
        if not stopped and running_max <= alpha:
            reject[i] = True
        else:
            stopped = True
        rank += 1

    return reject, p_corr


# ── Framework comparison driver ──────────────────────────────────────────────

def compare_systems(
    pred_data: dict[str, dict[str, np.ndarray]],
    gt_arrays: dict[str, np.ndarray],
    joints:    list[str] | None = None,
    alpha:     float = 0.05,
    n_boot:    int = 5000,
    seed:      int = 0,
) -> ComparisonReport:
    """
    Run all pairwise statistical comparisons between systems.

    Parameters
    ----------
    pred_data : dict[system → dict[joint → predicted angles]]
        Frame-aligned predictions for each system (frameworks or filter
        variants) evaluated on the SAME frames.
    gt_arrays : dict[joint → ground-truth angles]
    joints : list of joints to test. Defaults to keys of gt_arrays.
    alpha : family-wise significance level.
    n_boot : bootstrap resamples for the CI on the mean difference.

    Returns
    -------
    ComparisonReport
        One PairedComparison per (system pair × joint), with
        Holm-Bonferroni corrected significance flags per joint family.
    """
    joints  = joints or list(gt_arrays.keys())
    systems = list(pred_data.keys())
    report  = ComparisonReport(alpha=alpha, correction="holm-bonferroni")

    for joint in joints:
        if joint not in gt_arrays:
            continue
        gt = gt_arrays[joint]

        joint_comparisons: list[PairedComparison] = []
        for sys_a, sys_b in combinations(systems, 2):
            if joint not in pred_data[sys_a] or joint not in pred_data[sys_b]:
                continue
            # circular_diff (not naive subtraction) so wraparound-affected
            # frames don't distort the paired comparison the same way they
            # would distort a naive MAE (see metrics.circular_diff).
            err_a = np.abs(circular_diff(pred_data[sys_a][joint], gt))
            err_b = np.abs(circular_diff(pred_data[sys_b][joint], gt))

            t, t_p = paired_t_test(err_a, err_b)
            w, w_p = wilcoxon_test(err_a, err_b)
            d      = cohens_d_paired(err_a, err_b)

            a, b   = _drop_nan_pairs(err_a, err_b)
            diff   = a - b
            ci     = bootstrap_ci(diff, np.mean, n_boot=n_boot, alpha=alpha, seed=seed)

            joint_comparisons.append(PairedComparison(
                system_a=sys_a, system_b=sys_b, joint=joint, n=len(a),
                mean_abs_err_a=float(np.mean(a)) if len(a) else float("nan"),
                mean_abs_err_b=float(np.mean(b)) if len(b) else float("nan"),
                mean_diff=float(np.mean(diff)) if len(diff) else float("nan"),
                diff_ci95=ci,
                t_stat=t, t_p=t_p,
                wilcoxon_stat=w, wilcoxon_p=w_p,
                cohens_d=d,
            ))

        # Holm-Bonferroni within this joint's family (Wilcoxon p-values,
        # the more conservative non-parametric test, drive significance)
        flags, p_corr = holm_bonferroni([c.wilcoxon_p for c in joint_comparisons], alpha)
        for c, f, pc in zip(joint_comparisons, flags, p_corr):
            c.significant = f
            c.p_corrected = pc

        report.comparisons.extend(joint_comparisons)

    return report


def comparison_report_to_dicts(report: ComparisonReport) -> list[dict]:
    """Serialise a ComparisonReport for JSON/Excel export."""
    rows = []
    for c in report.comparisons:
        rows.append({
            "system_a":       c.system_a,
            "system_b":       c.system_b,
            "joint":          c.joint,
            "n":              c.n,
            "mae_a":          c.mean_abs_err_a,
            "mae_b":          c.mean_abs_err_b,
            "mean_diff":      c.mean_diff,
            "diff_ci95_lo":   c.diff_ci95[0],
            "diff_ci95_hi":   c.diff_ci95[1],
            "t_stat":         c.t_stat,
            "t_p":            c.t_p,
            "wilcoxon_stat":  c.wilcoxon_stat,
            "wilcoxon_p":     c.wilcoxon_p,
            "cohens_d":       c.cohens_d,
            "p_corrected":    c.p_corrected,
            "significant":    c.significant,
            "alpha":          report.alpha,
            "correction":     report.correction,
        })
    return rows


# ── Helpers ──────────────────────────────────────────────────────────────────

def _drop_nan_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop indices where either array is NaN; truncate to common length."""
    n = min(len(a), len(b))
    a = np.asarray(a[:n], dtype=np.float64)
    b = np.asarray(b[:n], dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]
