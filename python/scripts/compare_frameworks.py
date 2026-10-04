"""
compare_frameworks.py — Pose Estimation Framework Benchmark
============================================================

Evaluates MediaPipe, MoveNet, and PoseNet on a video file or webcam,
producing a quantitative comparison of their performance on the right-arm
joint angle estimation task.

Metrics reported per framework
-------------------------------
    FPS                  : Mean inference throughput (frames / second)
    ms / frame           : Mean per-frame inference time
    Detection rate       : Fraction of frames with a valid detected pose
    Per-joint mean       : Mean estimated angle across all detected frames
    Per-joint σ (jitter) : Angle standard deviation — lower = more stable
    Latency P95          : 95th percentile per-frame inference time (ms)

Angle outputs use the anatomically consistent ZXY decomposition from
angle_solver.py. Keys: shoulder_flexion, shoulder_abduction,
shoulder_rotation, elbow_flexion.

Usage
-----
    # On a video file (recommended for reproducible results)
    python scripts/compare_frameworks.py --video path/to/clip.mp4

    # From webcam (records max_frames frames live)
    python scripts/compare_frameworks.py --webcam --max_frames 300

    # Compare only specific frameworks
    python scripts/compare_frameworks.py --video clip.mp4 \\
        --frameworks mediapipe movenet_lightning

    # Use a specific filter
    python scripts/compare_frameworks.py --video clip.mp4 --filter kalman

Output files (in --output_dir)
-------------------------------
    comparison_report.json  — Full metrics (importable for paper tables)
    comparison_report.png   — Publication-quality figure (dark theme)
    per_joint_timeseries.png — Raw vs. filtered angle traces per framework
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.pose import load_estimator
from src.pose import angle_landmarks
from src.processing.angle_solver import compute_arm_angles, ArmAngles
from src.processing.angle_filter import AngleFilterBank

# ── Joint metadata ──────────────────────────────────────────────────────────
JOINTS = ["shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion"]
JOINT_LABELS = {
    "shoulder_flexion":   "Shoulder Flexion (°)",
    "shoulder_abduction": "Shoulder Abduction (°)",
    "shoulder_rotation":  "Shoulder Rotation (°)",
    "elbow_flexion":      "Elbow Flexion (°)",
}
JOINT_COLORS = {
    "shoulder_flexion":   "#64dc64",   # green
    "shoulder_abduction": "#64a0ff",   # blue
    "shoulder_rotation":  "#ffb450",   # orange
    "elbow_flexion":      "#dc50dc",   # purple
}

# ── Available frameworks ─────────────────────────────────────────────────────
ALL_FRAMEWORKS = ["mediapipe", "movenet_lightning", "posenet"]


# ── Framework loading ────────────────────────────────────────────────────────

def load_runners(framework_names: list[str]) -> dict[str, object]:
    """
    Load requested pose estimators. Frameworks that fail to import are
    skipped with a warning (e.g. TF not installed → MoveNet/PoseNet skipped).
    """
    runners = {}
    for name in framework_names:
        try:
            runner = load_estimator(name)
            runners[name] = runner
            print(f"  [OK] {runner.name} loaded")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")
    return runners


# ── Per-framework evaluation ─────────────────────────────────────────────────

def _angles_to_dict(a: ArmAngles) -> dict[str, float]:
    return {
        "shoulder_flexion":   a.shoulder_flexion,
        "shoulder_abduction": a.shoulder_abduction,
        "shoulder_rotation":  a.shoulder_rotation,
        "elbow_flexion":      a.elbow_flexion,
    }


def run_framework(
    name:         str,
    runner,
    frames_rgb:   list[np.ndarray],
    filter_type:  str = "kalman",
    stream_hz:    float = 30.0,
    verbose:      bool = True,
) -> dict:
    """
    Run one framework over a pre-loaded list of RGB frames and collect metrics.

    Parameters
    ----------
    name : str
        Display name of the framework.
    runner : PoseEstimator
        Instantiated estimator.
    frames_rgb : list[np.ndarray]
        List of (H, W, 3) RGB uint8 frames.
    filter_type : str
        Temporal filter to apply to angle signals.
    stream_hz : float
        Expected frame rate (used to initialise Kalman dt).
    verbose : bool
        Print per-frame progress.

    Returns
    -------
    dict
        Metrics dict with per-joint arrays and summary statistics.
    """
    filt = AngleFilterBank(filter_type=filter_type, stream_hz=stream_hz)

    inference_times_ms = []
    detected_flags     = []
    raw_angles:  dict[str, list[float]] = {j: [] for j in JOINTS}
    filt_angles: dict[str, list[float]] = {j: [] for j in JOINTS}

    n = len(frames_rgb)
    if verbose:
        print(f"\n  [{name}] Evaluating {n} frames...")

    for i, frame_rgb in enumerate(frames_rgb):
        t0 = time.perf_counter()
        lms = runner.process(frame_rgb)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        inference_times_ms.append(elapsed_ms)

        if lms is not None:
            angles = compute_arm_angles(angle_landmarks(
                runner, lms, frame_rgb.shape[1], frame_rgb.shape[0]))
        else:
            angles = None

        if angles is not None:
            detected_flags.append(True)
            filtered = filt.update(angles)
            for j in JOINTS:
                raw_angles[j].append(_angles_to_dict(angles)[j])
                filt_angles[j].append(_angles_to_dict(filtered)[j])
        else:
            detected_flags.append(False)
            # Append last known filtered value to keep arrays aligned
            for j in JOINTS:
                filt_angles[j].append(filt_angles[j][-1] if filt_angles[j] else 0.0)
                raw_angles[j].append(float("nan"))

        if verbose and (i + 1) % 50 == 0:
            rate = sum(detected_flags) / len(detected_flags)
            print(f"    frame {i+1:>4}/{n}  det={rate*100:.0f}%  "
                  f"ms={np.mean(inference_times_ms):.1f}")

    times_arr = np.array(inference_times_ms)
    n_det     = int(sum(detected_flags))

    # Per-joint statistics (NaN-safe)
    joint_stats = {}
    for j in JOINTS:
        raw_arr  = np.array(raw_angles[j])
        filt_arr = np.array(filt_angles[j])
        raw_valid = raw_arr[~np.isnan(raw_arr)]

        joint_stats[j] = {
            "mean_raw":  float(np.mean(raw_valid))  if len(raw_valid) else 0.0,
            "std_raw":   float(np.std(raw_valid))   if len(raw_valid) else 0.0,
            "mean_filt": float(np.nanmean(filt_arr)),
            "std_filt":  float(np.nanstd(filt_arr)),
            "raw":       raw_angles[j],       # kept for plotting
            "filtered":  filt_angles[j],
        }

    metrics = {
        "name":            name,
        "n_frames":        n,
        "n_detected":      n_det,
        "detection_rate":  n_det / n if n > 0 else 0.0,
        "mean_ms":         float(np.mean(times_arr)),
        "fps":             float(1000.0 / np.mean(times_arr)) if len(times_arr) else 0.0,
        "p95_ms":          float(np.percentile(times_arr, 95)),
        "filter_type":     filter_type,
        "joints":          joint_stats,
    }

    if verbose:
        print(f"    -> FPS: {metrics['fps']:.1f}  "
              f"det: {metrics['detection_rate']*100:.1f}%  "
              f"P95: {metrics['p95_ms']:.1f}ms")

    return metrics


# ── Console report table ─────────────────────────────────────────────────────

def print_table(all_metrics: list[dict]) -> None:
    names  = [m["name"] for m in all_metrics]
    col_w  = max(18, max(len(n) + 2 for n in names))

    sep = "-" * (24 + col_w * len(names))

    def hdr(label, vals):
        print(f"  {label:<22}", end="")
        for v in vals:
            print(f"{str(v):>{col_w}}", end="")
        print()

    print("\n" + "=" * (24 + col_w * len(names)))
    print("  FRAMEWORK COMPARISON - MONOARM")
    print("=" * (24 + col_w * len(names)))
    hdr("Framework",      names)
    hdr(sep[:22],         [sep[:col_w]] * len(names))
    hdr("FPS",            [f"{m['fps']:.1f}"                 for m in all_metrics])
    hdr("ms / frame",     [f"{m['mean_ms']:.1f}"             for m in all_metrics])
    hdr("P95 latency",    [f"{m['p95_ms']:.1f} ms"          for m in all_metrics])
    hdr("Detection rate", [f"{m['detection_rate']*100:.1f}%" for m in all_metrics])
    hdr(sep[:22],         [sep[:col_w]] * len(names))

    for j in JOINTS:
        label = JOINT_LABELS[j].replace(" (°)", "")
        hdr(f"{label} mean",
            [f"{m['joints'][j]['mean_raw']:+.1f}°"  if m["n_detected"] else "N/A"
             for m in all_metrics])
        hdr(f"{label} std",
            [f"{m['joints'][j]['std_raw']:.2f} deg"   if m["n_detected"] else "N/A"
             for m in all_metrics])

    print("=" * (24 + col_w * len(names)))

    # Overall winners
    valid = [m for m in all_metrics if m["n_detected"] > 0]
    if len(valid) >= 2:
        best_fps   = max(valid, key=lambda m: m["fps"])
        best_det   = max(valid, key=lambda m: m["detection_rate"])
        # Use elbow σ as stability proxy (most sensitive to jitter)
        best_stab  = min(valid, key=lambda m: m["joints"]["elbow_flexion"]["std_raw"])
        print(f"\n  [BEST] Fastest:       {best_fps['name']}  ({best_fps['fps']:.1f} FPS)")
        print(f"  [BEST] Most stable:   {best_stab['name']}"
              f"  (std={best_stab['joints']['elbow_flexion']['std_raw']:.2f} deg elbow)")
        print(f"  [BEST] Best detection:{best_det['name']}"
              f"  ({best_det['detection_rate']*100:.1f}%)\n")


# ── Plotting ─────────────────────────────────────────────────────────────────

def _style_ax(ax, title: str) -> None:
    ax.set_facecolor("#1a1a26")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.spines[:].set_color("#333")
    ax.set_title(title, color="#aacfff", fontsize=9, pad=5)
    for lbl in [ax.yaxis.label, ax.xaxis.label]:
        lbl.set_color("#ccc")


def plot_summary(all_metrics: list[dict], out_path: str) -> None:
    """
    Generate publication-quality summary figure (dark theme).
    Panels: FPS, Detection Rate, P95 Latency, Elbow Jitter.
    """
    valid  = [m for m in all_metrics if m["n_detected"] > 0]
    if not valid:
        print("[!] No valid data to plot.")
        return

    names  = [m["name"] for m in valid]
    n_fw   = len(valid)
    colors = [plt.cm.tab10(i / 10) for i in range(n_fw)]
    x      = np.arange(n_fw)
    bar_kw = dict(width=0.55, edgecolor="#333", linewidth=0.6)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor="#0e0e18")
    fig.suptitle(
        "Pose Framework Comparison — MonoArm Right-Arm Angle Estimation",
        fontsize=13, color="white", fontweight="bold", y=1.01,
    )

    panels = [
        ("FPS  (higher ↑)", [m["fps"]                              for m in valid], "frames/s"),
        ("Detection Rate (%)",  [m["detection_rate"] * 100        for m in valid], "%"),
        ("P95 Latency  (lower ↓)", [m["p95_ms"]                   for m in valid], "ms"),
        ("Elbow σ Jitter  (lower ↓)", [m["joints"]["elbow_flexion"]["std_raw"] for m in valid], "°"),
    ]

    for ax, (title, values, ylabel) in zip(axes, panels):
        bars = ax.bar(x, values, color=colors, **bar_kw)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8, color="#ccc")
        ax.set_ylabel(ylabel, fontsize=8, color="#ccc")
        _style_ax(ax, title)
        # Value labels on bars
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{v:.1f}",
                ha="center", va="bottom", color="white", fontsize=8,
            )

    # Reference line: ±5° jitter spec on elbow jitter panel
    axes[3].axhline(5.0, color="#ff6666", linestyle="--", linewidth=0.9,
                    label="±5° spec")
    axes[3].legend(fontsize=7, labelcolor="white", facecolor="#1a1a26",
                   framealpha=0.7)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Summary figure saved -> {out_path}")


def plot_timeseries(all_metrics: list[dict], out_path: str) -> None:
    """
    Per-joint raw vs. filtered angle traces for each framework.
    4 rows (joints) × N cols (frameworks).
    """
    valid = [m for m in all_metrics if m["n_detected"] > 0]
    if not valid:
        return

    n_fw     = len(valid)
    n_joints = len(JOINTS)

    fig, axes = plt.subplots(
        n_joints, n_fw,
        figsize=(6 * n_fw, 3 * n_joints),
        facecolor="#0e0e18",
        squeeze=False,
    )
    fig.suptitle("Joint Angle Time Series: Raw vs. Filtered",
                 fontsize=13, color="white", fontweight="bold")

    for ji, jkey in enumerate(JOINTS):
        color = JOINT_COLORS[jkey]
        for fi, m in enumerate(valid):
            ax = axes[ji][fi]
            raw  = np.array(m["joints"][jkey]["raw"],      dtype=float)
            filt = np.array(m["joints"][jkey]["filtered"], dtype=float)
            t    = np.arange(len(raw))

            ax.plot(t, raw,  alpha=0.25, linewidth=0.8, color=color, label="raw")
            ax.plot(t, filt, alpha=0.90, linewidth=1.5, color=color, label="filtered")
            ax.axhline(0, color="#444", linewidth=0.5, linestyle="--")
            _style_ax(ax, f"{m['name']} — {JOINT_LABELS[jkey]}")
            ax.set_xlabel("frame", fontsize=7, color="#aaa")
            ax.set_ylabel("°", fontsize=8, color="#aaa")
            if ji == 0:
                ax.legend(fontsize=6.5, labelcolor="white", facecolor="#1a1a26",
                          loc="upper right", framealpha=0.6)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Time-series figure saved -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare pose estimation frameworks for MonoArm."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",  type=str, help="Path to video file")
    src.add_argument("--webcam", action="store_true", help="Capture from webcam")
    ap.add_argument("--max_frames",   type=int,   default=300,
                    help="Max frames to evaluate (default 300)")
    ap.add_argument("--frameworks",   nargs="+",  default=ALL_FRAMEWORKS,
                    choices=ALL_FRAMEWORKS,
                    help="Frameworks to include (default: all)")
    ap.add_argument("--filter",       type=str,   default="kalman",
                    choices=["kalman", "ma", "sg"],
                    help="Temporal filter type (default: kalman)")
    ap.add_argument("--output_dir",   type=str,   default="outputs",
                    help="Directory for saved reports and figures")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load frames ──────────────────────────────────────────────────────────
    if args.webcam:
        cap = cv2.VideoCapture(0)
        print(f"[->] Recording {args.max_frames} frames from webcam...")
    else:
        vpath = Path(args.video)
        if not vpath.exists():
            print(f"[FAIL] Video not found: {vpath}")
            sys.exit(1)
        cap = cv2.VideoCapture(str(vpath))
        print(f"[->] Loading from: {vpath}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or args.max_frames
    step  = max(1, total // args.max_frames)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames_rgb = []
    idx = 0
    while len(frames_rgb) < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()

    if not frames_rgb:
        print("[FAIL] No frames loaded.")
        sys.exit(1)
    print(f"[OK] {len(frames_rgb)} frames loaded  (source FPS ~ {video_fps:.0f})")

    # ── Load runners ─────────────────────────────────────────────────────────
    print("\n[->] Loading frameworks...")
    runners = load_runners(args.frameworks)
    if not runners:
        print("[FAIL] No frameworks available. Install mediapipe and/or tensorflow.")
        sys.exit(1)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n[->] Evaluating frameworks...")
    all_metrics = []
    for fw_name, runner in runners.items():
        m = run_framework(
            name=fw_name,
            runner=runner,
            frames_rgb=frames_rgb,
            filter_type=args.filter,
            stream_hz=video_fps,
        )
        all_metrics.append(m)
        runner.close()

    # ── Report ───────────────────────────────────────────────────────────────
    print_table(all_metrics)

    # Save JSON (strip raw arrays to keep file small)
    json_metrics = []
    for m in all_metrics:
        m_clean = {k: v for k, v in m.items() if k != "joints"}
        m_clean["joints"] = {
            j: {k2: v2 for k2, v2 in jd.items() if k2 not in ("raw", "filtered")}
            for j, jd in m["joints"].items()
        }
        json_metrics.append(m_clean)

    report_json = out_dir / "comparison_report.json"
    with open(report_json, "w") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"[OK] JSON report -> {report_json}")

    # Figures
    plot_summary(all_metrics, str(out_dir / "comparison_report.png"))
    plot_timeseries(all_metrics, str(out_dir / "per_joint_timeseries.png"))


if __name__ == "__main__":
    main()
