"""
alignment.py — Automatic Temporal-Alignment Check for Real-Data Validation
=============================================================================

Upstreams the lag-estimation procedure originally developed as a one-off
post-hoc fix (docs/paper/analysis/sync_corrected_evaluation.py) into the
shipped evaluation pipeline, per the MonoArm paper's own recommendation
(Section "Discussion and Limitations" — Temporal alignment as a standard
check): an evaluation harness that pairs video and motion-capture streams
by filename/frame index should verify alignment automatically before
reporting any accuracy numbers, rather than relying on manual post-hoc
inspection.

Procedure
---------
For candidate integer shifts s in `search_range`, pair predicted frame i
with ground-truth frame i+s and compute the Pearson correlation between
the two resulting series; the chosen shift maximises that correlation.
The reference channel should be a DOF free of both the rotation-
reliability gate and the ±180° wraparound (shoulder abduction is the
natural choice for MonoArm's angle convention).
"""

from __future__ import annotations

import numpy as np

DEFAULT_REFERENCE_JOINT = "shoulder_abduction"
DEFAULT_SEARCH_RANGE = range(-30, 31)
DEFAULT_MIN_OVERLAP = 40


def shifted_pair(pred: np.ndarray, gt: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred, gt) arrays aligned so that pred[i] <-> gt[i+shift]."""
    if shift >= 0:
        p = pred[: len(pred) - shift] if shift > 0 else pred
        g = gt[shift:]
    else:
        p = pred[-shift:]
        g = gt[: len(gt) + shift]
    n = min(len(p), len(g))
    return p[:n], g[:n]


def best_correlation_shift(
    pred: np.ndarray,
    gt: np.ndarray,
    search_range=DEFAULT_SEARCH_RANGE,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
) -> tuple[int, float]:
    """Return (shift, pearson_r) maximising correlation over `search_range`."""
    best_shift, best_r = 0, -np.inf
    for shift in search_range:
        p, g = shifted_pair(pred, gt, shift)
        if len(p) < min_overlap:
            continue
        if np.std(p) < 1e-9 or np.std(g) < 1e-9:
            continue
        r = float(np.corrcoef(p, g)[0, 1])
        if r > best_r:
            best_shift, best_r = shift, r
    return best_shift, best_r


def check_alignment(
    preds: dict[str, dict[str, np.ndarray]],
    gt: dict[str, np.ndarray],
    reference_framework: str,
    reference_joint: str = DEFAULT_REFERENCE_JOINT,
    search_range=DEFAULT_SEARCH_RANGE,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
) -> dict:
    """
    Estimate the video/ground-truth frame offset before any accuracy metric
    is computed, and report a consistency check from a second framework.

    Parameters
    ----------
    preds : dict[framework -> dict[joint -> array]]
        Raw (unshifted) per-framework predictions, one entry per framework
        that was run (e.g. "mediapipe", "movenet_lightning").
    gt : dict[joint -> array]
        Raw (unshifted) ground truth, aligned by frame index to `preds`.
    reference_framework : str
        The framework whose reference-joint channel fixes the shift applied
        to ALL frameworks and joints. Chosen once, not per framework/joint.
    reference_joint : str
        DOF used for the correlation search. Must be free of the rotation-
        reliability gate and the ±180° wraparound.

    Returns
    -------
    dict with keys: shift, reference_r, reference_framework, reference_joint,
    consistency (per-framework independently-implied shift/r, for sanity
    checking — NOT used to choose the applied shift).
    """
    shift, ref_r = best_correlation_shift(
        preds[reference_framework][reference_joint], gt[reference_joint],
        search_range, min_overlap,
    )

    consistency = {}
    for fw in preds:
        s, r = best_correlation_shift(
            preds[fw][reference_joint], gt[reference_joint], search_range, min_overlap,
        )
        consistency[fw] = {"independent_best_shift": s, "r_at_best_shift": r}

    return {
        "shift": shift,
        "reference_r": ref_r,
        "reference_framework": reference_framework,
        "reference_joint": reference_joint,
        "consistency": consistency,
    }


def apply_shift(
    preds: dict[str, dict[str, np.ndarray]],
    gt: dict[str, np.ndarray],
    shift: int,
    joints: list[str],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Apply one fixed shift to every framework/joint, trimming to the overlap."""
    preds_aligned: dict[str, dict[str, np.ndarray]] = {fw: {} for fw in preds}
    gt_aligned: dict[str, np.ndarray] = {}
    for j in joints:
        for fw in preds:
            if j not in preds[fw] or j not in gt:
                continue
            p, g = shifted_pair(preds[fw][j], gt[j], shift)
            preds_aligned[fw][j] = p
            gt_aligned[j] = g
    return preds_aligned, gt_aligned
