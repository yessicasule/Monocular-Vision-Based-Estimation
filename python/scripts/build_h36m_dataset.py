"""
build_h36m_dataset.py — Synthetic Framework-Error Simulator
=============================================================

SIMULATOR. This script does not run MediaPipe, MoveNet or PoseNet, and no
image is ever read. It parses Human3.6M skeletons, computes ground-truth
joint angles, and then *fabricates* each framework's "prediction" as

    prediction = ground_truth + bias + N(0, sigma)

with bias and sigma taken from the hardcoded NOISE_PROFILES below. The
profiles are calibrated to published error rates, which makes the output
plausible-looking and therefore easy to mistake for a measurement.

What this output is valid for
-----------------------------
Exercising the shape of the evaluation pipeline — column names, joins,
metric plumbing — when no real data is available.

What it is NOT valid for
------------------------
* Any accuracy or comparison number about a pose-estimation framework.
  Such a number describes NOISE_PROFILES, not MediaPipe or MoveNet or
  PoseNet.
* Training a model. A network fitted to this CSV learns to remove
  zero-mean Gaussian noise of known constant sigma from ground truth.
  That problem has a closed-form optimum and nothing to do with monocular
  pose estimation. Because the three frameworks differ only in sigma,
  "multi-framework fusion" on this data collapses to averaging three
  noisy copies of one signal, and any fusion gain is an artifact of the
  generator.

For real measurements, obtain the Human3.6M video release and run the
frameworks on actual frames — see docs/DATA_H36M.md:

    python scripts/prepare_h36m.py  --h36m_root ... --out_dir ...
    python scripts/evaluate_h36m.py --mode live --manifest .../manifest.json

Guardrails
----------
Running this script requires the explicit --i-understand-this-is-simulated
flag, it prints a banner, every row carries a ``data_kind=simulated``
column, and a ``.provenance.json`` sidecar is written next to the CSV so
the marking survives the file being copied elsewhere.

Angle convention
----------------
Ground-truth angles use the ZXY anatomical decomposition (shoulder_flexion,
shoulder_abduction, shoulder_rotation, elbow_flexion) matching the live
MonoArm pipeline, on the RIGHT arm (RShoulder=25, RElbow=26, RWrist=27),
with the torso frame built by the same Gram-Schmidt construction as
coordinate_frame.py.

Output CSV columns
------------------
    frame, subject, action, data_kind,
    gt_shoulder_flexion, gt_shoulder_abduction, gt_shoulder_rotation, gt_elbow_flexion,
    mp_shoulder_flexion, mp_shoulder_abduction, mp_shoulder_rotation, mp_elbow_flexion,
    mv_shoulder_flexion, mv_shoulder_abduction, mv_shoulder_rotation, mv_elbow_flexion,
    pn_shoulder_flexion, pn_shoulder_abduction, pn_shoulder_rotation, pn_elbow_flexion

The mp_/mv_/pn_ columns are generated, not measured.

Usage
-----
    python scripts/build_h36m_dataset.py \\
        --h36m_dir  data/dataset/h3.6m/dataset \\
        --output    outputs/synthetic_noise_dataset.csv \\
        --subjects  S1 S5 S6 S7 S8 \\
        --seed      42 \\
        --i-understand-this-is-simulated
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from src.evaluation.h36m_loader import parse_h36m_file, GTAngles

# ── Synthetic noise profiles (per-joint, right arm) ──────────────────────────
# Tuples of (bias_deg, std_deg).
# Calibrated to published cross-validation error rates:
#   MediaPipe: Bazarevsky et al. (2020), BlazePose
#   MoveNet:   Google internal benchmarks (2021)
#   PoseNet:   Papandreou et al. (2018), PersonLab
NOISE_PROFILES = {
    "mp": {
        "shoulder_flexion":   (0.0,  3.0),
        "shoulder_abduction": (0.0,  3.5),
        "shoulder_rotation":  (0.0,  4.0),
        "elbow_flexion":      (0.0,  2.5),
    },
    "mv": {
        "shoulder_flexion":   (0.5,  5.0),
        "shoulder_abduction": (0.3,  5.5),
        "shoulder_rotation":  (0.8,  6.5),
        "elbow_flexion":      (0.4,  4.5),
    },
    "pn": {
        "shoulder_flexion":   (1.2,  8.0),
        "shoulder_abduction": (0.8,  9.0),
        "shoulder_rotation":  (2.0, 11.0),
        "elbow_flexion":      (1.0,  7.0),
    },
}

GT_JOINTS = ["shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion"]
FW_PREFIXES = ["mp", "mv", "pn"]


SIMULATION_BANNER = """
================================================================================
  SIMULATED DATA — NOT A MEASUREMENT
================================================================================
  The mp_/mv_/pn_ columns this script writes are ground truth plus Gaussian
  noise drawn from hardcoded per-joint profiles. No pose estimator is run.
  MediaPipe, MoveNet and PoseNet never see an image here.

  Consequences, if this output is used as if it were real:
    * Accuracy tables measure the noise generator, not any framework.
    * A model trained on it learns to remove zero-mean Gaussian noise of
      known constant sigma — a task with a closed-form optimum, unrelated
      to monocular pose estimation.
    * The three frameworks differ only in sigma, so "fusion" across them
      reduces to averaging three noisy copies of one signal.

  For real framework accuracy see docs/DATA_H36M.md:
      scripts/prepare_h36m.py  then  evaluate_h36m.py --mode live
================================================================================
"""


def _write_provenance(output_csv: Path, args) -> None:
    """
    Write a sidecar JSON marking the CSV as simulated.

    The CSV can outlive this terminal session, get copied into a notebook, or
    be uploaded to a training environment on its own. A sidecar travels with it
    and records, in machine-readable form, that nothing in it was measured.
    """
    sidecar = output_csv.with_suffix(".provenance.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({
        "data_kind":         "simulated",
        "publication_grade": False,
        "trainable":         False,
        "generator":         "scripts/build_h36m_dataset.py",
        "description": (
            "Framework columns are ground truth plus Gaussian noise from "
            "hardcoded profiles. No pose estimator was executed. Not valid "
            "for accuracy reporting or model training."
        ),
        "ground_truth_source": "Human3.6M skeletons via src/evaluation/h36m_loader.py",
        "noise_profiles":      NOISE_PROFILES,
        "seed":                args.seed,
        "subjects":            args.subjects,
        "real_alternative":    "docs/DATA_H36M.md",
    }, indent=2), encoding="utf-8")
    print(f"[i] Provenance marker written: {sidecar}")


def _enrich_with_noise(
    gt: GTAngles,
    rng: np.random.Generator,
) -> dict:
    """
    Add calibrated per-joint Gaussian noise to simulate each framework's error.

    Parameters
    ----------
    gt : GTAngles
        Ground-truth angles for one frame.
    rng : np.random.Generator
        Seeded random generator for reproducibility.

    Returns
    -------
    dict
        All GT and synthetic framework angle columns for one CSV row.
    """
    gt_vals = {
        "shoulder_flexion":   gt.shoulder_flexion,
        "shoulder_abduction": gt.shoulder_abduction,
        "shoulder_rotation":  gt.shoulder_rotation,
        "elbow_flexion":      gt.elbow_flexion,
    }
    row = {f"gt_{j}": v for j, v in gt_vals.items()}

    for prefix, profile in NOISE_PROFILES.items():
        for j in GT_JOINTS:
            mu, sigma = profile[j]
            noise          = rng.normal(mu, sigma)
            row[f"{prefix}_{j}"] = round(gt_vals[j] + noise, 4)

    # Self-marking: a row lifted out of this CSV and pasted elsewhere still
    # carries the fact that its framework columns were generated, not measured.
    row["data_kind"] = "simulated"

    return row


def build_dataset(
    h36m_dir:              Path,
    output_csv:            Path,
    subjects:              list[str] | None = None,
    max_frames_per_action: int | None = None,
    seed:                  int = 42,
) -> pd.DataFrame:
    """
    Walk all subject/action .txt files, compute anatomical GT angles,
    add per-framework synthetic noise, and write the unified CSV.

    Parameters
    ----------
    h36m_dir : Path
        Root directory containing S1/, S5/, ... subject folders.
    output_csv : Path
        Destination CSV path.
    subjects : list[str], optional
        Which H3.6M subjects to include. Defaults to all 7 standard subjects.
    max_frames_per_action : int, optional
        Cap frames per action (useful for quick smoke tests).
    seed : int
        Random seed for reproducible synthetic noise.

    Returns
    -------
    pd.DataFrame
        The built dataset (also written to output_csv).
    """
    rng      = np.random.default_rng(seed)
    h36m_dir = Path(h36m_dir)
    subjects = subjects or ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]

    all_rows      = []
    global_frame  = 0
    skipped_files = 0

    for subject in subjects:
        subj_dir = h36m_dir / subject
        if not subj_dir.exists():
            print(f"  [SKIP] Subject {subject} not found at {subj_dir}")
            continue

        txt_files = sorted(subj_dir.glob("*.txt"))
        print(f"\nSubject {subject}: {len(txt_files)} action files found")

        for txt_path in txt_files:
            action_name = txt_path.stem
            gt_frames   = parse_h36m_file(txt_path)

            if not gt_frames:
                print(f"  [SKIP] {action_name} — no parseable frames")
                skipped_files += 1
                continue

            if max_frames_per_action and len(gt_frames) > max_frames_per_action:
                gt_frames = gt_frames[:max_frames_per_action]

            for i, gt in enumerate(gt_frames):
                row = {
                    "frame":   global_frame + i,
                    "subject": subject,
                    "action":  action_name,
                }
                row.update(_enrich_with_noise(gt, rng))
                all_rows.append(row)

            global_frame += len(gt_frames)
            print(f"  OK  {action_name:<30} {len(gt_frames):>5} frames  "
                  f"[total: {global_frame:,}]")

    if not all_rows:
        print("\n[!] No data rows generated. Check h36m_dir path and .txt format.")
        return pd.DataFrame()

    # Column order. data_kind leads so the simulated provenance is the first
    # thing visible in a spreadsheet, a `head`, or a DataFrame repr.
    cols = (
        ["data_kind", "frame", "subject", "action"]
        + [f"gt_{j}" for j in GT_JOINTS]
        + [f"{p}_{j}" for p in FW_PREFIXES for j in GT_JOINTS]
    )
    df = pd.DataFrame(all_rows)[cols]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*64}")
    print(f"  Dataset saved  →  {output_csv}")
    print(f"  Total frames   :  {len(df):,}")
    print(f"  Subjects       :  {df['subject'].nunique()}")
    print(f"  Actions        :  {df['action'].nunique()}")
    print(f"  Columns ({len(df.columns)}):  {list(df.columns)}")
    if skipped_files:
        print(f"  Skipped files  :  {skipped_files}")
    print(f"\n  Ground-Truth Angle Statistics:")
    print(
        df[[f"gt_{j}" for j in GT_JOINTS]]
        .rename(columns={f"gt_{j}": j for j in GT_JOINTS})
        .describe()
        .round(2)
        .to_string()
    )
    print(f"{'='*64}\n")

    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SIMULATOR: build a synthetic H3.6M CSV whose framework "
                    "columns are ground truth plus Gaussian noise. Not a "
                    "measurement of any pose estimator."
    )
    ap.add_argument(
        "--h36m_dir", default="data/dataset/h3.6m/dataset",
        help="Root directory containing S1/, S5/, ... subject folders",
    )
    ap.add_argument(
        "--output", default="outputs/synthetic_noise_dataset.csv",
        help="Output CSV path (default: outputs/synthetic_noise_dataset.csv). "
             "The name must make the simulated provenance obvious to anyone "
             "who later finds the file on its own.",
    )
    ap.add_argument(
        "--subjects", nargs="*",
        default=["S1", "S5", "S6", "S7", "S8", "S9", "S11"],
        help="Subjects to include",
    )
    ap.add_argument(
        "--max_frames_per_action", type=int, default=None,
        help="Cap frames per action file (for quick testing)",
    )
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for synthetic noise (default: 42)")
    ap.add_argument(
        "--i-understand-this-is-simulated", dest="acknowledged",
        action="store_true",
        help="Required. Confirms you know the mp_/mv_/pn_ columns are "
             "generated, not measured, and will not be used to train a model "
             "or report an accuracy number.",
    )
    args = ap.parse_args()

    if not args.acknowledged:
        print(SIMULATION_BANNER)
        print("[x] Refusing to run without --i-understand-this-is-simulated.")
        print("    To measure real framework accuracy instead, see "
              "docs/DATA_H36M.md and use:")
        print("        python scripts/prepare_h36m.py  ...")
        print("        python scripts/evaluate_h36m.py --mode live --manifest ...")
        raise SystemExit(2)

    print(SIMULATION_BANNER)

    output_csv = Path(args.output)
    build_dataset(
        h36m_dir              = Path(args.h36m_dir),
        output_csv            = output_csv,
        subjects              = args.subjects,
        max_frames_per_action = args.max_frames_per_action,
        seed                  = args.seed,
    )
    _write_provenance(output_csv, args)


if __name__ == "__main__":
    main()
