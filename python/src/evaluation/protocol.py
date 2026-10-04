"""
protocol.py — Reproducible Experimental Protocol Definitions
=============================================================

Defines the subject-level data partitions used by the evaluation
pipeline so that parameter tuning, model/filter selection, and final
testing are performed on disjoint subjects — eliminating data leakage.

Human3.6M standard protocol (Ionescu et al., 2014; used by essentially
all published H3.6M evaluations):

    train      S1, S5, S6, S7      — parameter optimisation only
                                     (e.g. Kalman Q/R, filter windows)
    validation S8                  — model and filter selection
    test       S9, S11             — final evaluation, touched exactly once

Leave-One-Subject-Out (LOSO) cross-validation is provided as an
alternative protocol to establish generalisation across unseen
participants: each fold holds out one subject for testing and leaves
the remainder for tuning/selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

H36M_ALL_SUBJECTS = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]

H36M_SPLITS = {
    "train": ["S1", "S5", "S6", "S7"],
    "val":   ["S8"],
    "test":  ["S9", "S11"],
}


@dataclass
class LosoFold:
    """One Leave-One-Subject-Out fold."""
    fold_idx:       int
    test_subject:   str
    train_subjects: list[str]


def get_split(name: str) -> list[str]:
    """
    Return the subject list for a named split ("train" | "val" | "test").
    """
    if name not in H36M_SPLITS:
        raise ValueError(f"Unknown split {name!r}; expected one of {sorted(H36M_SPLITS)}")
    return list(H36M_SPLITS[name])


def loso_folds(subjects: list[str] | None = None) -> Iterator[LosoFold]:
    """
    Yield Leave-One-Subject-Out folds over `subjects`
    (default: all seven standard H3.6M subjects).
    """
    subjects = list(subjects or H36M_ALL_SUBJECTS)
    if len(subjects) < 2:
        raise ValueError("LOSO requires at least two subjects")
    for i, test_subject in enumerate(subjects):
        yield LosoFold(
            fold_idx=i,
            test_subject=test_subject,
            train_subjects=[s for s in subjects if s != test_subject],
        )


def assert_no_leakage(tuning_subjects: list[str], test_subjects: list[str]) -> None:
    """
    Raise if any subject used for tuning/selection also appears in the
    test set. Call this before reporting final numbers.
    """
    overlap = set(tuning_subjects) & set(test_subjects)
    if overlap:
        raise RuntimeError(
            f"Data leakage: subjects {sorted(overlap)} appear in both the "
            f"tuning and test sets. Final metrics would be invalid."
        )


def describe_protocol(name: str, subjects: list[str] | None = None) -> str:
    """Human-readable protocol description for reproducibility metadata."""
    if name == "loso":
        subs = subjects or H36M_ALL_SUBJECTS
        return (
            f"Leave-One-Subject-Out cross-validation over {subs}; metrics "
            f"reported per held-out subject and aggregated as mean ± std "
            f"across folds."
        )
    return (
        f"H3.6M standard split — train {H36M_SPLITS['train']} (parameter "
        f"optimisation), val {H36M_SPLITS['val']} (filter/model selection), "
        f"test {H36M_SPLITS['test']} (final evaluation only)."
    )
